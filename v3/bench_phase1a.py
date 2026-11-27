"""Phase 1A Benchmark & Kill Gate 1A Verification Harness (A2: Coupled Stage 1+2 + ASV3).

Runs ASV3 (`codec.py`) with `zstd -19 --long=27` across ALL 10 pre-registered files,
verifies streaming SHA-256 bit-exact round-trip on every file + a Unix pipe test,
runs the decoupled block-size sweep (1 MB / 4 MB / 16 MB) on target_mixed_structured.bin,
and evaluates all 3 hard conditions of Kill Gate 1A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from codec import ASV3Decoder, ASV3Encoder, EncoderConfig
from transforms import (
    T1_TRANSPOSE,
    T2_BITSHUFFLE,
    T2_ELEM_DELTA,
    T2_XOR_TRANSPOSE,
    T3_2D_PAETH,
    TransformSpec,
    TwoTierScorer,
    apply_transform_forward,
    apply_transform_inverse,
)

V3_DIR = Path(__file__).resolve().parent
CORPUS_DIR = V3_DIR / "corpus"
MB = 1024 * 1024


class _NullSink(io.RawIOBase):
    """Streaming sink that discards decoded bytes while ASV3Decoder verifies SHA-256."""

    def writable(self) -> bool:
        return True

    def write(self, b: bytes | bytearray | memoryview) -> int:
        return len(b)


def run_one_file_phase1a(
    entry: Dict[str, Any],
    phase0_row: Dict[str, Any],
    level: int = 19,
    block_mb: int = 4,
) -> Dict[str, Any]:
    fname = entry["name"]
    cat = entry["category"]
    p = CORPUS_DIR / fname
    expected_sha = entry["sha256"]

    cfg = EncoderConfig(
        zstd_level=level,
        window_log=27,
        flush_group_bytes=16 * MB,
        block_size_bytes=block_mb * MB,
    )

    tmp_enc = V3_DIR / f".tmp_{fname}.asv3"
    t0 = time.perf_counter()
    try:
        with open(p, "rb") as fin, open(tmp_enc, "wb") as fout:
            enc = ASV3Encoder(fout, cfg)
            st = enc.encode_stream(fin)
        enc_sec = time.perf_counter() - t0
        assert st.sha256 == expected_sha, f"Encode SHA mismatch on {fname}"

        # Single-cursor streaming decode + full SHA-256 verification
        t1 = time.perf_counter()
        with open(tmp_enc, "rb") as fin:
            dec = ASV3Decoder()
            dec_bytes, dec_sha = dec.decode_stream(fin, _NullSink())
        dec_sec = time.perf_counter() - t1

        sha_ok = (dec_bytes == st.orig_bytes) and (dec_sha == expected_sha)
        assert sha_ok, f"Decode SHA-256 verification FAILED on {fname}!"
    finally:
        if tmp_enc.exists():
            tmp_enc.unlink()

    b3a = phase0_row["B3a_zstd19_l27"]
    b5 = phase0_row["B5_openzl"]
    a0 = phase0_row["A0_global"]
    a1 = phase0_row["A1_fixed4m"]

    vs_b3a = 100.0 * (st.comp_bytes - b3a) / b3a
    vs_b5 = 100.0 * (st.comp_bytes - b5) / b5
    vs_a0 = 100.0 * (st.comp_bytes - a0) / a0
    vs_a1 = 100.0 * (st.comp_bytes - a1) / a1

    bt = st.bytes_by_transform
    row = {
        "name": fname,
        "category": cat,
        "orig_bytes": st.orig_bytes,
        "A2_comp_bytes": st.comp_bytes,
        "A2_ratio": round(st.orig_bytes / max(1, st.comp_bytes), 4),
        "B3a_bytes": b3a,
        "B5_openzl_bytes": b5,
        "A0_bytes": a0,
        "A1_bytes": a1,
        "delta_vs_B3a_pct": round(vs_b3a, 4),
        "delta_vs_B5_pct": round(vs_b5, 2),
        "delta_vs_A0_pct": round(vs_a0, 3),
        "delta_vs_A1_pct": round(vs_a1, 3),
        "S_META_bytes": st.s_meta_bytes,
        "S_META_pct_out": round(st.meta_pct_of_out, 4),
        "S_RAW_bytes": st.s_raw_bytes,
        "S_XFORM_bytes": st.s_xform_bytes,
        "S_RESID_bytes": st.s_resid_bytes,
        "framing_bytes": st.framing_bytes,
        "regions_count": st.regions_count,
        "scoring_work_mult": round(st.scoring_work_multiplier, 3),
        "T0_bytes": bt.get("T0", 0),
        "T1_bytes": bt.get("T1", 0),
        "T2_ELEM_bytes": bt.get("T2_ELEM", 0),
        "T2_XOR_bytes": bt.get("T2_XOR", 0),
        "T2_BIT_bytes": bt.get("T2_BIT", 0),
        "T3_2D_bytes": bt.get("T3_2D", 0),
        "enc_sec": round(enc_sec, 2),
        "dec_sec": round(dec_sec, 2),
        "sha256_pass": sha_ok,
    }
    print(
        f"[DONE] {fname:<28} A2={st.comp_bytes/MB:6.2f}MB ({vs_b3a:+6.3f}% vs B3a, {vs_a1:+6.2f}% vs A1, "
        f"{vs_b5:+6.2f}% vs OZL) | meta={st.s_meta_bytes}B ({st.meta_pct_of_out:.4f}%) | "
        f"regs={st.regions_count:2d} | work={st.scoring_work_multiplier:.2f}x | SHA=PASS ({enc_sec:.1f}s/{dec_sec:.1f}s)",
        flush=True,
    )
    return row


def run_pipe_test() -> bool:
    """Verify strict stdin/stdout pipe compatibility: cat in | codec.py c | codec.py d."""
    src = CORPUS_DIR / "target_mixed_structured.bin"
    # Stream 48 MB through real OS pipes (3 FlushGroups, crossing multiple region boundaries!)
    n_bytes = 48 * MB
    with open(src, "rb") as f:
        sample = f.read(n_bytes)
    expected = hashlib.sha256(sample).hexdigest()

    codec_py = str(V3_DIR / "codec.py")
    p_enc = subprocess.Popen(
        ["python3", codec_py, "c", "--level", "3", "--flush-mb", "16"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p_dec = subprocess.Popen(
        ["python3", codec_py, "d"],
        stdin=p_enc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert p_enc.stdout is not None and p_enc.stdin is not None and p_dec.stdout is not None
    p_enc.stdout.close()  # Allow SIGPIPE if decoder exits early

    sha = hashlib.sha256()
    dec_len = [0]

    def _read_dec():
        while True:
            b = p_dec.stdout.read(4 * MB)
            if not b:
                break
            sha.update(b)
            dec_len[0] += len(b)

    th = threading.Thread(target=_read_dec, daemon=True)
    th.start()
    p_enc.stdin.write(sample)
    p_enc.stdin.close()
    th.join()
    err_enc = p_enc.stderr.read().decode(errors="replace") if p_enc.stderr else ""
    err_dec = p_dec.stderr.read().decode(errors="replace") if p_dec.stderr else ""
    p_enc.wait()
    p_dec.wait()

    ok = (p_enc.returncode == 0 and p_dec.returncode == 0 and dec_len[0] == n_bytes and sha.hexdigest() == expected)
    if not ok:
        raise RuntimeError(f"Pipe test FAILED! enc_rc={p_enc.returncode} dec_rc={p_dec.returncode} err={err_enc} {err_dec}")
    print(f"[PASS] Pipe-safe stdin/stdout streaming test (48 MB across 3 FlushGroups): SHA-256={expected[:16]}...", flush=True)
    return True


def run_block_size_sweep(phase0_by_name: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sweep decoupled processing block size (1 MB / 4 MB / 16 MB) on target_mixed_structured.bin at L3."""
    p = CORPUS_DIR / "target_mixed_structured.bin"
    out = []
    for bmb in (1, 4, 16):
        cfg = EncoderConfig(zstd_level=3, window_log=27, flush_group_bytes=16 * MB, block_size_bytes=bmb * MB)
        sink = _NullSink()
        with open(p, "rb") as fin:
            st = ASV3Encoder(sink, cfg).encode_stream(fin)
        out.append({
            "block_mb": bmb,
            "comp_bytes": st.comp_bytes,
            "s_meta_bytes": st.s_meta_bytes,
            "regions": st.regions_count,
        })
        print(f"  [BLOCK SWEEP L3] target_mixed_structured.bin block={bmb:2d}MB -> {st.comp_bytes/MB:.3f}MB (meta={st.s_meta_bytes}B)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1A Benchmark & Kill Gate 1A Harness")
    parser.add_argument("--workers", type=int, default=3, help="Parallel files (default 3 to prevent LLC thrashing)")
    parser.add_argument("--resume", action="store_true", help="Reuse cached phase1a_results.json")
    args = parser.parse_args()

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    phase0 = json.loads((V3_DIR / "phase0_results.json").read_text())
    p0_by_name = {r["name"]: r for r in phase0}

    # Warmup Numba JIT
    scorer = TwoTierScorer()
    dummy = np.arange(65536, dtype=np.uint8)
    scorer.score_probe(dummy[:16384], dummy)
    for sp in (
        TransformSpec(T1_TRANSPOSE, 4),
        TransformSpec(T2_XOR_TRANSPOSE, 4),
        TransformSpec(T2_ELEM_DELTA, 4),
        TransformSpec(T2_BITSHUFFLE, 4),
        TransformSpec(T3_2D_PAETH, 4, 64),
    ):
        assert np.array_equal(apply_transform_inverse(apply_transform_forward(dummy, sp), sp), dummy)

    print("=" * 120)
    print("PHASE 1A: COUPLED STAGE 1+2 + ASV3 STREAMING CONTAINER (zstd -19 --long=27)")
    print("=" * 120, flush=True)

    pipe_ok = run_pipe_test()
    sweep = run_block_size_sweep(p0_by_name)

    out_json = V3_DIR / "phase1a_results.json"
    done_by_name: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_json.exists():
        for r in json.loads(out_json.read_text()):
            done_by_name[r["name"]] = r

    todo = [m for m in manifest if m["name"] not in done_by_name]
    # Run smaller files first so progress reports rapidly, then 256MB-1GB files
    todo.sort(key=lambda m: m["bytes"])

    lock = threading.Lock()
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one_file_phase1a, m, p0_by_name[m["name"]]): m["name"] for m in todo}
            for fut in as_completed(futs):
                row = fut.result()
                with lock:
                    done_by_name[row["name"]] = row
                    ordered = [done_by_name[m["name"]] for m in manifest if m["name"] in done_by_name]
                    out_json.write_text(json.dumps(ordered, indent=2))

    rows = [done_by_name[m["name"]] for m in manifest]
    out_json.write_text(json.dumps(rows, indent=2))
    (V3_DIR / "phase1a_block_sweep.json").write_text(json.dumps(sweep, indent=2))

    out_csv = V3_DIR / "phase1a_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Print summary table & Kill Gate 1A
    print("\n" + "=" * 134)
    print(
        f"{'Dataset':<28} {'Cat':<14} {'A2 MB':>7} {'Ratio':>6} {'vs B3a':>8} {'vs A1':>8} "
        f"{'vs OZL':>8} {'MetaB':>6} {'Regs':>5} {'Work':>6} {'SHA':>4}"
    )
    print("=" * 134)
    for r in rows:
        print(
            f"{r['name']:<28} {r['category']:<14} {r['A2_comp_bytes']/MB:7.2f} {r['A2_ratio']:5.2f}x "
            f"{r['delta_vs_B3a_pct']:+7.3f}% {r['delta_vs_A1_pct']:+7.2f}% {r['delta_vs_B5_pct']:+7.2f}% "
            f"{r['S_META_bytes']:6d} {r['regions_count']:5d} {r['scoring_work_mult']:5.2f}x {'PASS' if r['sha256_pass'] else 'FAIL':>4}"
        )
    print("=" * 134)

    avg_work = sum(r["scoring_work_mult"] for r in rows) / len(rows)
    all_sha = pipe_ok and all(r["sha256_pass"] for r in rows)
    neg_adv = [r for r in rows if r["category"] in ("negative", "adversarial")]
    max_tax = max(r["delta_vs_B3a_pct"] for r in neg_adv)
    zero_reg_ok = all(r["delta_vs_B3a_pct"] <= 0.15 for r in neg_adv)

    target_wins = [r for r in rows if r["category"].startswith("target") and r["delta_vs_B3a_pct"] <= -3.0]
    stage1_wins = [
        r for r in rows
        if r["category"] in ("target", "target_mixed")
        and r["delta_vs_A0_pct"] <= -1.0
        and r["delta_vs_A1_pct"] <= -1.0
    ]

    print("\nKILL GATE 1A EVALUATION:")
    print(f"  1. 100% Bit-Exact SHA-256 (10 files + pipe):  {'PASS' if all_sha else 'FAIL'}")
    print(f"  2. Zero Regression (<= +0.15% on neg/adv):    {'PASS' if zero_reg_ok else 'FAIL'} (worst tax = {max_tax:+.4f}%)")
    print(f"  3. Target Net Gain (>= 3.0% vs B3a):          {'PASS' if target_wins else 'FAIL'} ({len(target_wins)} files: {[r['name'] for r in target_wins]})")
    print(f"  4. Stage 1 Beats A0 & A1 (>= 1.0% on mixed):  {'PASS' if stage1_wins else 'FAIL'} ({[(r['name'], r['delta_vs_A1_pct']) for r in stage1_wins]})")
    print(f"  5. Avg Scoring Work Multiplier (<= 3.0x):     {'PASS' if avg_work <= 3.0 else 'FAIL'} (avg = {avg_work:.2f}x)")

    if all_sha and zero_reg_ok and target_wins and stage1_wins:
        print("\n>>> KILL GATE 1A: ALL CONDITIONS PASSED! Proceed to Phase 1B (Stage 3 Ablation).")
    else:
        print("\n>>> KILL GATE 1A: CHECK FAILED CONDITIONS ABOVE.")


if __name__ == "__main__":
    main()
