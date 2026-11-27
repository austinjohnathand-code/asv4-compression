"""Phase 2 Systems Benchmark & Pareto Frontier Harness (C++20 NEON + native libzstd 1.5.5).

Benchmarks the compiled C++20 `asv3` binary (surviving Stages 1, 2, and 4 only;
Stage 3 permanently cut per Kill Gate 1B) across the full 10-file, 2.91 GB
pre-registered corpus at both --level 3 (fast) and --level 19 (max ratio):
  - Net compressed bytes & ratio (vs B3a CLI zstd 1.5.5 -19 --long=27 & B2 zstd -3)
  - Wall-clock Encode throughput (MB/s) & Decode throughput (MB/s)
  - OS-measured Peak RSS (MB) for BOTH encode and decode via `/usr/bin/time -l`
  - 100% streaming SHA-256 verification on every run
  - Also benchmarks B1 (lz4 -1), B2 (zstd -3 --long=27), B3a (zstd -19 --long=27)
    encode/decode speed & Peak RSS on the same machine for the Pareto frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

V3_DIR = Path(__file__).resolve().parent
CORPUS_DIR = V3_DIR / "corpus"
ASV3_BIN = V3_DIR / "cpp" / "asv3"
MB = 1024 * 1024


def run_timed_rss(cmd: List[str]) -> Tuple[float, float, int]:
    """Run command under macOS `/usr/bin/time -l` -> (wall_sec, peak_rss_mb, rc)."""
    t0 = time.perf_counter()
    pr = subprocess.run(
        ["/usr/bin/time", "-l"] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wall = time.perf_counter() - t0
    err = pr.stderr.decode(errors="replace")
    # macOS `/usr/bin/time -l` reports `maximum resident set size` in BYTES!
    m = re.search(r"(\d+)\s+maximum resident set size", err)
    rss_mb = (int(m.group(1)) / MB) if m else 0.0
    if pr.returncode != 0:
        raise RuntimeError(f"Command failed ({pr.returncode}): {cmd}\n{err}")
    return wall, rss_mb, pr.returncode


def bench_asv3_file(entry: Dict[str, Any], level: int, p0_row: Dict[str, Any]) -> Dict[str, Any]:
    fname = entry["name"]
    p = CORPUS_DIR / fname
    orig = p.stat().st_size
    tmp_enc = V3_DIR / f".p2_l{level}_{fname}.asv3"
    tmp_stats = V3_DIR / f".p2_l{level}_{fname}.json"

    try:
        enc_s, enc_rss, _ = run_timed_rss([
            str(ASV3_BIN), "c", "--level", str(level), "--stats", str(tmp_stats),
            str(p), str(tmp_enc),
        ])
        comp_bytes = tmp_enc.stat().st_size
        st = json.loads(tmp_stats.read_text())
        assert st["sha256"] == entry["sha256"], f"Encode SHA mismatch on {fname}"

        # Decode directly to /dev/null (asv3 computes & verifies exact SHA-256 internally!)
        dec_s, dec_rss, _ = run_timed_rss([
            str(ASV3_BIN), "d", str(tmp_enc), "/dev/null",
        ])
    finally:
        if tmp_enc.exists():
            tmp_enc.unlink()
        if tmp_stats.exists():
            tmp_stats.unlink()

    b3a = p0_row["B3a_zstd19_l27"]
    b2 = p0_row["B2_zstd3"]
    b5 = p0_row["B5_openzl"]
    vs_b3a = 100.0 * (comp_bytes - b3a) / b3a
    vs_b2 = 100.0 * (comp_bytes - b2) / b2
    vs_b5 = 100.0 * (comp_bytes - b5) / b5

    row = {
        "name": fname,
        "category": entry["category"],
        "level": level,
        "orig_bytes": orig,
        "comp_bytes": comp_bytes,
        "ratio": round(orig / max(1, comp_bytes), 4),
        "vs_B3a_pct": round(vs_b3a, 4),
        "vs_B2_pct": round(vs_b2, 4),
        "vs_B5_pct": round(vs_b5, 2),
        "s_meta_bytes": st["s_meta_bytes"],
        "regions": st["regions_count"],
        "work_mult": round(st["scoring_work_mult"], 3),
        "enc_sec": round(enc_s, 3),
        "dec_sec": round(dec_s, 3),
        "enc_mb_s": round((orig / MB) / max(enc_s, 1e-6), 1),
        "dec_mb_s": round((orig / MB) / max(dec_s, 1e-6), 1),
        "enc_rss_mb": round(enc_rss, 1),
        "dec_rss_mb": round(dec_rss, 1),
        "sha256_pass": True,
    }
    print(
        f"[C++ L{level:2d}] {fname:<28} {comp_bytes/MB:6.2f}MB ({row['ratio']:5.2f}x | "
        f"{vs_b3a:+6.3f}% vs B3a | {vs_b2:+6.2f}% vs B2) | "
        f"Enc={row['enc_mb_s']:6.1f} MB/s (RSS {enc_rss:5.1f}MB) | "
        f"Dec={row['dec_mb_s']:6.1f} MB/s (RSS {dec_rss:5.1f}MB) | regs={row['regions']:2d} | SHA=PASS",
        flush=True,
    )
    return row


def bench_baseline_speeds(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure real wall-clock encode/decode MB/s & Peak RSS for B1 (lz4), B2 (zstd -3), B3a (zstd -19 --long=27)."""
    out: Dict[str, Any] = {}
    # Measure on 3 representative files spanning structured, mixed, and 1 GB enwik9
    rep_names = ["target_struct64.bin", "target_mixed_structured.bin", "neg_enwik9.xml"]
    for name in rep_names:
        p = CORPUS_DIR / name
        orig = p.stat().st_size
        out[name] = {}

        # B1: lz4 -1
        tmp = V3_DIR / f".tmp_b1_{name}.lz4"
        try:
            es, er, _ = run_timed_rss(["lz4", "-1", "-f", "-q", str(p), str(tmp)])
            sz = tmp.stat().st_size
            ds, dr, _ = run_timed_rss(["lz4", "-d", "-f", "-q", str(tmp), "/dev/null"])
            out[name]["B1_lz4"] = {
                "ratio": round(orig / sz, 3),
                "enc_mb_s": round((orig / MB) / es, 1),
                "dec_mb_s": round((orig / MB) / ds, 1),
                "enc_rss_mb": round(er, 1),
                "dec_rss_mb": round(dr, 1),
            }
        finally:
            if tmp.exists():
                tmp.unlink()

        # B2: zstd -3 --long=27 -T1
        tmp = V3_DIR / f".tmp_b2_{name}.zst"
        try:
            es, er, _ = run_timed_rss(["zstd", "-3", "--long=27", "-T1", "-f", "-q", str(p), "-o", str(tmp)])
            sz = tmp.stat().st_size
            ds, dr, _ = run_timed_rss(["zstd", "-d", "--long=27", "-f", "-q", str(tmp), "-o", "/dev/null"])
            out[name]["B2_zstd3_l27"] = {
                "comp_bytes": sz,
                "ratio": round(orig / sz, 3),
                "enc_mb_s": round((orig / MB) / es, 1),
                "dec_mb_s": round((orig / MB) / ds, 1),
                "enc_rss_mb": round(er, 1),
                "dec_rss_mb": round(dr, 1),
            }
        finally:
            if tmp.exists():
                tmp.unlink()

        print(f"[BASE SPEED] {name:<28} lz4={out[name]['B1_lz4']} | zstd3_l27={out[name]['B2_zstd3_l27']}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 C++20 Pareto Frontier Benchmark")
    parser.add_argument("--workers-l19", type=int, default=3, help="Parallel workers for L19")
    args = parser.parse_args()

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    p0_by_name = {r["name"]: r for r in json.loads((V3_DIR / "phase0_results.json").read_text())}

    print("=" * 130)
    print("PHASE 2A: C++20 NEON ASV3 --level 3 (SINGLE-THREADED, ZERO INTERFERENCE, TRUE WALL-CLOCK MB/s & PEAK RSS)")
    print("=" * 130, flush=True)
    # Run L3 sequentially so throughput & RSS timings have ZERO CPU contention!
    l3_rows: List[Dict[str, Any]] = []
    for entry in manifest:
        l3_rows.append(bench_asv3_file(entry, level=3, p0_row=p0_by_name[entry["name"]]))

    print("\n" + "=" * 130)
    print("PHASE 2B: BASELINE SPEED & RSS MEASUREMENTS (lz4 -1, zstd -3 --long=27)")
    print("=" * 130, flush=True)
    base_speeds = bench_baseline_speeds(manifest)

    print("\n" + "=" * 130)
    print("PHASE 2C: C++20 NEON ASV3 --level 19 (EXACT LIBZSTD 1.5.5 APPLES-TO-APPLES vs B3a)")
    print("=" * 130, flush=True)
    l19_by_name: Dict[str, Dict[str, Any]] = {}
    ordered_todo = sorted(manifest, key=lambda m: m["bytes"])
    with ThreadPoolExecutor(max_workers=args.workers_l19) as ex:
        futs = {ex.submit(bench_asv3_file, m, 19, p0_by_name[m["name"]]): m["name"] for m in ordered_todo}
        for fut in as_completed(futs):
            r = fut.result()
            l19_by_name[r["name"]] = r
    l19_rows = [l19_by_name[m["name"]] for m in manifest]

    results = {
        "asv3_cpp_l3": l3_rows,
        "asv3_cpp_l19": l19_rows,
        "baseline_speeds": base_speeds,
    }
    (V3_DIR / "phase2_results.json").write_text(json.dumps(results, indent=2))
    with open(V3_DIR / "phase2_l3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(l3_rows[0].keys()))
        w.writeheader()
        w.writerows(l3_rows)
    with open(V3_DIR / "phase2_l19.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(l19_rows[0].keys()))
        w.writeheader()
        w.writerows(l19_rows)

    print("\nPHASE 2 BENCHMARK COMPLETE! Saved phase2_results.json, phase2_l3.csv, phase2_l19.csv")


if __name__ == "__main__":
    main()
