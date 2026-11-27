"""Final Post-Fix Phase 2 Refresh: Throughput (MB/s), OS Peak RSS (MB), Work Mult,
and Exact In-Process libzstd 1.5.5 C API Deltas across all 15 datasets (10 pre-registered +
4 real public GPT-2/NYC Taxi dumps + 1 real NASA/ESA Hubble FITS astronomy raster).
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

V3_DIR = Path(__file__).resolve().parent
CORPUS_DIR = V3_DIR / "corpus"
REAL_DIR = V3_DIR / "corpus_real"
ASV3_BIN = V3_DIR / "cpp" / "asv3"
CAPI_BIN = V3_DIR / "cpp" / "zstd_capi_bench"
MB = 1024 * 1024


def run_timed_rss(cmd: List[str]) -> Tuple[float, float, bytes]:
    t0 = time.perf_counter()
    pr = subprocess.run(["/usr/bin/time", "-l"] + cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall = time.perf_counter() - t0
    err = pr.stderr.decode(errors="replace")
    m = re.search(r"(\d+)\s+maximum resident set size", err)
    rss_mb = (int(m.group(1)) / MB) if m else 0.0
    if pr.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{err}")
    return wall, rss_mb, pr.stdout


def main() -> None:
    datasets = [
        ("real_gpt2_f32.safetensors", "real_public_target", REAL_DIR / "real_gpt2_f32.safetensors"),
        ("real_smollm2_bf16.safetensors", "real_public_target", REAL_DIR / "real_smollm2_bf16.safetensors"),
        ("real_hubble_eagle_f32.fits", "real_public_raster", REAL_DIR / "real_hubble_eagle_f32.fits"),
        ("real_mixed_public.bin", "real_public_mixed", REAL_DIR / "real_mixed_public.bin"),
        ("real_nyc_taxi_struct104.bin", "real_public_target", REAL_DIR / "real_nyc_taxi_struct104.bin"),
        ("real_nyc_taxi_columnar.arrow", "real_public_target", REAL_DIR / "real_nyc_taxi_columnar.arrow"),
        ("target_numeric_f32.bin", "synth_target", CORPUS_DIR / "target_numeric_f32.bin"),
        ("target_struct64.bin", "synth_target", CORPUS_DIR / "target_struct64.bin"),
        ("target_sqlite.db", "real_fmt_sqlite", CORPUS_DIR / "target_sqlite.db"),
        ("target_mixed_structured.bin", "mixed_target", CORPUS_DIR / "target_mixed_structured.bin"),
        ("target_snapshots.bin", "synth_snapshot", CORPUS_DIR / "target_snapshots.bin"),
        ("neg_enwik9.xml", "real_negative", CORPUS_DIR / "neg_enwik9.xml"),
        ("neg_source_tar.tar", "real_negative", CORPUS_DIR / "neg_source_tar.tar"),
        ("adv_binaries.bin", "real_adversarial", CORPUS_DIR / "adv_binaries.bin"),
        ("adv_compressed_media.tar", "mixed_adversarial", CORPUS_DIR / "adv_compressed_media.tar"),
        ("adv_random.bin", "real_adversarial", CORPUS_DIR / "adv_random.bin"),
    ]

    # Reuse exact B2_capi & B3a_capi already measured (and measure Hubble FITS CAPI)
    prev_audit = {r["name"]: r for r in json.loads((V3_DIR / "audit_capi_results.json").read_text())}

    rows: List[Dict[str, Any]] = []
    print("=" * 155)
    print("FINAL REFRESHED C++20 NEON ASV4 v4.1 --level 3 & --level 19 (SAME BINARY ON ALL 16 DATASETS, SINGLE-THREADED MB/s, OS PEAK RSS, EXACT C-API DELTA)")
    print("=" * 155, flush=True)

    for name, cat, path in datasets:
        orig = path.stat().st_size
        if name in prev_audit:
            b2_bytes = prev_audit[name]["B2_capi_L3"]
            b3a_bytes = prev_audit[name]["B3a_capi_L19"]
        else:
            _, _, out3 = run_timed_rss([str(CAPI_BIN), "3", str(path), "27"])
            _, _, out19 = run_timed_rss([str(CAPI_BIN), "19", str(path), "27"])
            b2_bytes = json.loads(out3.decode())["comp_bytes"]
            b3a_bytes = json.loads(out19.decode())["comp_bytes"]

        # Run --level 3 encode + SHA-256 decode
        tmp_enc3 = V3_DIR / f".final3_{name}.asv3"
        tmp_st3 = V3_DIR / f".final3_{name}.json"
        try:
            enc3_s, enc3_rss, _ = run_timed_rss([
                str(ASV3_BIN), "c", "--level", "3", "--stats", str(tmp_st3), str(path), str(tmp_enc3)
            ])
            st3 = json.loads(tmp_st3.read_text())
            dec3_s, dec3_rss, _ = run_timed_rss([str(ASV3_BIN), "d", str(tmp_enc3), "/dev/null"])
        finally:
            if tmp_enc3.exists():
                tmp_enc3.unlink()
            if tmp_st3.exists():
                tmp_st3.unlink()

        # Run --level 19 encode + SHA-256 decode from the exact same binary on ALL 15 datasets
        tmp_enc19 = V3_DIR / f".final19_{name}.asv3"
        tmp_st19 = V3_DIR / f".final19_{name}.json"
        try:
            enc19_s, enc19_rss, _ = run_timed_rss([
                str(ASV3_BIN), "c", "--level", "19", "--stats", str(tmp_st19), str(path), str(tmp_enc19)
            ])
            st19 = json.loads(tmp_st19.read_text())
            dec19_s, dec19_rss, _ = run_timed_rss([str(ASV3_BIN), "d", str(tmp_enc19), "/dev/null"])
        finally:
            if tmp_enc19.exists():
                tmp_enc19.unlink()
            if tmp_st19.exists():
                tmp_st19.unlink()

        d_b2 = 100.0 * (st3["comp_bytes"] - b2_bytes) / b2_bytes
        d_b3a = 100.0 * (st19["comp_bytes"] - b3a_bytes) / b3a_bytes
        enc3_mbs = (orig / MB) / max(enc3_s, 1e-6)
        dec3_mbs = (orig / MB) / max(dec3_s, 1e-6)
        dec19_mbs = (orig / MB) / max(dec19_s, 1e-6)

        row = {
            "name": name,
            "category": cat,
            "orig_bytes": orig,
            "B2_capi_L3": b2_bytes,
            "ASV3_L3": st3["comp_bytes"],
            "L3_vs_B2_capi_pct": round(d_b2, 4),
            "B3a_capi_L19": b3a_bytes,
            "ASV3_L19": st19["comp_bytes"],
            "L19_vs_B3a_capi_pct": round(d_b3a, 4),
            "enc_mb_s_L3": round(enc3_mbs, 1),
            "dec_mb_s_L3": round(dec3_mbs, 1),
            "dec_mb_s_L19": round(dec19_mbs, 1),
            "enc_rss_mb_L3": round(enc3_rss, 1),
            "dec_rss_mb_L3": round(dec3_rss, 1),
            "dec_rss_mb_L19": round(dec19_rss, 1),
            "regions_L3": st3["regions_count"],
            "regions_L19": st19["regions_count"],
            "work_mult_L3": round(st3["scoring_work_mult"], 3),
            "work_mult_L19": round(st19["scoring_work_mult"], 3),
            "sha256_pass": True,
        }
        rows.append(row)
        print(
            f"[FINAL] {name:<30} ({orig/MB:5.1f}M) | L3={st3['comp_bytes']/MB:6.2f}M ({d_b2:+7.3f}% vs B2) | "
            f"L19={st19['comp_bytes']/MB:6.2f}M ({d_b3a:+7.3f}% vs B3a) | Enc3={enc3_mbs:5.1f} Dec3={dec3_mbs:6.1f} (RSS {dec3_rss:5.1f}M) | "
            f"Dec19={dec19_mbs:6.1f} | regs={st3['regions_count']:2d}/{st19['regions_count']:2d} | work={st3['scoring_work_mult']:.2f}x/{st19['scoring_work_mult']:.2f}x",
            flush=True,
        )

    (V3_DIR / "final_refresh_results.json").write_text(json.dumps(rows, indent=2))
    with open(V3_DIR / "final_refresh_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    max_work_l3 = max(r["work_mult_L3"] for r in rows)
    avg_work_l3 = sum(r["work_mult_L3"] for r in rows) / len(rows)
    max_work_l19 = max(r["work_mult_L19"] for r in rows)
    avg_work_l19 = sum(r["work_mult_L19"] for r in rows) / len(rows)
    neg_adv = [r for r in rows if "negative" in r["category"] or "adversarial" in r["category"]]
    worst_tax_l3 = max(r["L3_vs_B2_capi_pct"] for r in neg_adv)
    worst_tax_l19 = max(r["L19_vs_B3a_capi_pct"] for r in neg_adv)
    print("\n" + "=" * 120)
    print(
        f"SUMMARY: L3 avg/max Work Mult = {avg_work_l3:.2f}x / {max_work_l3:.2f}x | "
        f"L19 avg/max Work Mult = {avg_work_l19:.2f}x / {max_work_l19:.2f}x | "
        f"worst neg/adv tax = {worst_tax_l3:+.4f}% (L3) / {worst_tax_l19:+.4f}% (L19)"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
