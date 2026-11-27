"""Audit Harness: Exact In-Process libzstd 1.5.5 C API Baselines (B2_capi & B3a_capi)
vs. Hardened C++20 ASV3 (--level 3 and --level 19) across ALL 14 datasets:
  - 10 original pre-registered corpus files (`v3/corpus/`)
  - 4 real public dumps (`v3/corpus_real/`):
      * real_gpt2_f32.safetensors     (522.7 MB - OpenAI GPT-2 production IEEE-754 float32 weights)
      * real_nyc_taxi_struct104.bin   (256.0 MB - 2.58M real NYC TLC Yellow Taxi 104B binary structs)
      * real_nyc_taxi_columnar.arrow  (803.1 MB - 5.97M real NYC TLC Yellow Taxi Apache Arrow IPC)
      * real_mixed_public.bin         (283.9 MB - 100% real public multi-regime concatenated file)

Every `% vs B2_capi` and `% vs B3a_capi` uses the EXACT SAME single-threaded
libzstd 1.5.5 C API (`ZSTD_c_enableLongDistanceMatching=1, ZSTD_c_windowLog=27`)
that `asv3.cpp` calls internally.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

V3_DIR = Path(__file__).resolve().parent
CORPUS_DIR = V3_DIR / "corpus"
REAL_DIR = V3_DIR / "corpus_real"
ASV3_BIN = V3_DIR / "cpp" / "asv3"
CAPI_BIN = V3_DIR / "cpp" / "zstd_capi_bench"
MB = 1024 * 1024


def run_capi(level: int, path: Path) -> Dict[str, Any]:
    pr = subprocess.run(
        [str(CAPI_BIN), str(level), str(path), "27"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(pr.stdout.decode())


def run_asv3(level: int, path: Path, tag: str) -> Dict[str, Any]:
    tmp_enc = V3_DIR / f".audit_{tag}_l{level}.asv3"
    tmp_st = V3_DIR / f".audit_{tag}_l{level}.json"
    try:
        subprocess.run(
            [str(ASV3_BIN), "c", "--level", str(level), "--stats", str(tmp_st), str(path), str(tmp_enc)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        st = json.loads(tmp_st.read_text())
        # Verify single-cursor decode + hardware SHA-256
        subprocess.run(
            [str(ASV3_BIN), "d", str(tmp_enc), "/dev/null"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        st["sha256_pass"] = True
        return st
    finally:
        if tmp_enc.exists():
            tmp_enc.unlink()
        if tmp_st.exists():
            tmp_st.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact C API libzstd 1.5.5 Audit Benchmark")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    datasets: List[Dict[str, Any]] = [
        # Real public targets
        {"name": "real_gpt2_f32.safetensors", "cat": "real_public_target", "path": REAL_DIR / "real_gpt2_f32.safetensors"},
        {"name": "real_nyc_taxi_struct104.bin", "cat": "real_public_target", "path": REAL_DIR / "real_nyc_taxi_struct104.bin"},
        {"name": "real_nyc_taxi_columnar.arrow", "cat": "real_public_target", "path": REAL_DIR / "real_nyc_taxi_columnar.arrow"},
        {"name": "real_mixed_public.bin", "cat": "real_public_mixed", "path": REAL_DIR / "real_mixed_public.bin"},
        # Original 10 pre-registered corpus
        {"name": "target_numeric_f32.bin", "cat": "synth_target", "path": CORPUS_DIR / "target_numeric_f32.bin"},
        {"name": "target_struct64.bin", "cat": "synth_target", "path": CORPUS_DIR / "target_struct64.bin"},
        {"name": "target_sqlite.db", "cat": "real_fmt_sqlite", "path": CORPUS_DIR / "target_sqlite.db"},
        {"name": "target_mixed_structured.bin", "cat": "mixed_target", "path": CORPUS_DIR / "target_mixed_structured.bin"},
        {"name": "target_snapshots.bin", "cat": "synth_snapshot", "path": CORPUS_DIR / "target_snapshots.bin"},
        {"name": "neg_enwik9.xml", "cat": "real_negative", "path": CORPUS_DIR / "neg_enwik9.xml"},
        {"name": "neg_source_tar.tar", "cat": "real_negative", "path": CORPUS_DIR / "neg_source_tar.tar"},
        {"name": "adv_binaries.bin", "cat": "real_adversarial", "path": CORPUS_DIR / "adv_binaries.bin"},
        {"name": "adv_compressed_media.tar", "cat": "mixed_adversarial", "path": CORPUS_DIR / "adv_compressed_media.tar"},
        {"name": "adv_random.bin", "cat": "real_adversarial", "path": CORPUS_DIR / "adv_random.bin"},
    ]

    cache_path = V3_DIR / "audit_capi_results.json"
    cache: Dict[str, Dict[str, Any]] = {}
    if args.resume and cache_path.exists():
        for r in json.loads(cache_path.read_text()):
            cache[r["name"]] = r

    todo = [d for d in datasets if d["name"] not in cache]
    # Run smaller files first
    todo.sort(key=lambda d: d["path"].stat().st_size)

    lock = threading.Lock()
    print("=" * 130)
    print(f"AUDIT: EXACT IN-PROCESS libzstd 1.5.5 C API (B2_capi L3 & B3a_capi L19) vs HARDENED C++20 ASV3 | todo={len(todo)}/14")
    print("=" * 130, flush=True)

    def _run_one(d: Dict[str, Any]) -> Dict[str, Any]:
        name = d["name"]
        p = d["path"]
        orig = p.stat().st_size
        b2 = run_capi(3, p)
        a3 = run_asv3(3, p, name)
        b3a = run_capi(19, p)
        a19 = run_asv3(19, p, name)

        d_l3 = 100.0 * (a3["comp_bytes"] - b2["comp_bytes"]) / b2["comp_bytes"]
        d_l19 = 100.0 * (a19["comp_bytes"] - b3a["comp_bytes"]) / b3a["comp_bytes"]
        row = {
            "name": name,
            "category": d["cat"],
            "orig_bytes": orig,
            "B2_capi_L3": b2["comp_bytes"],
            "ASV3_L3": a3["comp_bytes"],
            "L3_vs_B2_capi_pct": round(d_l3, 4),
            "B3a_capi_L19": b3a["comp_bytes"],
            "ASV3_L19": a19["comp_bytes"],
            "L19_vs_B3a_capi_pct": round(d_l19, 4),
            "L19_regions": a19["regions_count"],
            "L19_s_meta_bytes": a19["s_meta_bytes"],
            "L19_s_xform_bytes": a19["s_xform_bytes"],
            "L19_work_mult": round(a19["scoring_work_mult"], 3),
            "L3_sha_pass": a3["sha256_pass"],
            "L19_sha_pass": a19["sha256_pass"],
        }
        print(
            f"[AUDIT] {name:<30} ({orig/MB:5.1f}M) | "
            f"L3: {b2['comp_bytes']/MB:6.2f}->{a3['comp_bytes']/MB:6.2f}M ({d_l3:+7.3f}%) | "
            f"L19: {b3a['comp_bytes']/MB:6.2f}->{a19['comp_bytes']/MB:6.2f}M ({d_l19:+7.3f}%) | "
            f"regs={a19['regions_count']:2d} | work={a19['scoring_work_mult']:.2f}x | SHA=PASS",
            flush=True,
        )
        return row

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_one, d) for d in todo]
            for fut in as_completed(futs):
                row = fut.result()
                with lock:
                    cache[row["name"]] = row
                    ordered = [cache[d["name"]] for d in datasets if d["name"] in cache]
                    cache_path.write_text(json.dumps(ordered, indent=2))

    ordered = [cache[d["name"]] for d in datasets]
    cache_path.write_text(json.dumps(ordered, indent=2))
    csv_path = V3_DIR / "audit_capi_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ordered[0].keys()))
        w.writeheader()
        w.writerows(ordered)

    print("\n" + "=" * 130)
    print(f"{'Dataset':<30} {'Cat':<19} {'OrigMB':>7} {'B2_capi':>8} {'ASV3_L3':>8} {'L3 vs B2':>9} {'B3a_capi':>8} {'ASV3_L19':>8} {'L19 vs B3a':>10} {'Regs':>4} {'Work':>5}")
    print("=" * 130)
    for r in ordered:
        print(
            f"{r['name']:<30} {r['category']:<19} {r['orig_bytes']/MB:7.1f} "
            f"{r['B2_capi_L3']/MB:8.2f} {r['ASV3_L3']/MB:8.2f} {r['L3_vs_B2_capi_pct']:+8.3f}% "
            f"{r['B3a_capi_L19']/MB:8.2f} {r['ASV3_L19']/MB:8.2f} {r['L19_vs_B3a_capi_pct']:+9.3f}% "
            f"{r['L19_regions']:4d} {r['L19_work_mult']:4.2f}x"
        )
    print("=" * 130)
    neg_adv = [r for r in ordered if "negative" in r["category"] or "adversarial" in r["category"]]
    worst_l3 = max(r["L3_vs_B2_capi_pct"] for r in neg_adv)
    worst_l19 = max(r["L19_vs_B3a_capi_pct"] for r in neg_adv)
    print(f"APPLES-TO-APPLES WORST-CASE TAX (neg/adv): L3={worst_l3:+.4f}% | L19={worst_l19:+.4f}% (cap <= +0.1500%)")


if __name__ == "__main__":
    main()
