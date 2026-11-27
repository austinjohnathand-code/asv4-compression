"""Phase 1B Ablation & Kill Gate 1B Harness (A3: A2 + Stage 3 Prefix-Gated Delta Forest).

Tests A3 vs A2 under the identical zstd -19 --long=27 (128 MB window) budget on:
  - target_snapshots.bin       (pre-registered 256 MB versioned snapshots, 1.5% in-place edits)
  - target_snapshots_dense.bin (supplemental 128 MB dense 15% in-place mutation stress test)
  - target_numeric_f32.bin     (256 MB scientific float32 target)
  - adv_binaries.bin           (160 MB scorer-adversarial Mach-O binaries)

Evaluates Kill Gate 1B:
  Keep Stage 3 ONLY if A3 beats A2 by >= 1.0% net compressed size (including manifest overhead)
  under the same --long=27 memory budget on the pre-registered snapshot workload.
  Otherwise cut Stage 3 permanently before Phase 2!
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from codec import ASV3Decoder, ASV3Encoder, EncoderConfig

V3_DIR = Path(__file__).resolve().parent
CORPUS_DIR = V3_DIR / "corpus"
MB = 1024 * 1024


class _NullSink(io.RawIOBase):
    def writable(self) -> bool:
        return True

    def write(self, b: bytes | bytearray | memoryview) -> int:
        return len(b)


def ensure_dense_snapshot_stress(path: Path) -> Path:
    """Supplemental 128 MB snapshot stress test with 15% dense small-delta in-place edits
    (simulating numeric/counter/state pages where LZ77 match runs shrink below 4 bytes)."""
    if path.exists() and path.stat().st_size == 128 * MB:
        return path
    rng = np.random.default_rng(9090)
    snap_sz = 8 * MB
    bases = [rng.integers(0, 256, size=snap_sz, dtype=np.uint8) for _ in range(4)]
    cur = [b.copy() for b in bases]
    with open(path, "wb") as f:
        for gen in range(4):  # 4 x 4 x 8 MB = 128 MB
            for k in range(4):
                if gen > 0:
                    arr = cur[k]
                    n_mut = int(snap_sz * 0.15)
                    idx = rng.integers(0, snap_sz, size=n_mut)
                    delta = rng.integers(1, 5, size=n_mut, dtype=np.uint8)
                    arr[idx] = ((arr[idx].astype(np.uint16) + delta) & 0xFF).astype(np.uint8)
                f.write(cur[k].tobytes())
    return path


def run_a2_and_a3(path: Path, name: str, a2_cached_bytes: int | None = None) -> Dict[str, Any]:
    # Compute expected SHA-256
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * MB)
            if not b:
                break
            h.update(b)
    expected_sha = h.hexdigest()

    # Run A2 if not cached from Phase 1A
    if a2_cached_bytes is None:
        tmp_a2 = V3_DIR / f".tmp_a2_{name}.asv3"
        try:
            with open(path, "rb") as fin, open(tmp_a2, "wb") as fout:
                st2 = ASV3Encoder(fout, EncoderConfig(zstd_level=19, enable_stage3=False)).encode_stream(fin)
            a2_bytes = st2.comp_bytes
        finally:
            if tmp_a2.exists():
                tmp_a2.unlink()
    else:
        a2_bytes = a2_cached_bytes

    # Run A3 (Stage 3 enabled) + full single-cursor decode & SHA-256 verification
    tmp_a3 = V3_DIR / f".tmp_a3_{name}.asv3"
    t0 = time.perf_counter()
    try:
        with open(path, "rb") as fin, open(tmp_a3, "wb") as fout:
            enc = ASV3Encoder(fout, EncoderConfig(zstd_level=19, enable_stage3=True))
            st3 = enc.encode_stream(fin)
            forest = enc._forest
        enc_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        with open(tmp_a3, "rb") as fin:
            dec_len, dec_sha = ASV3Decoder().decode_stream(fin, _NullSink())
        dec_sec = time.perf_counter() - t1

        sha_ok = (dec_len == st3.orig_bytes) and (dec_sha == expected_sha)
        assert sha_ok, f"A3 SHA-256 verification FAILED on {name}!"
    finally:
        if tmp_a3.exists():
            tmp_a3.unlink()

    delta_pct = 100.0 * (st3.comp_bytes - a2_bytes) / a2_bytes
    assert forest is not None
    row = {
        "name": name,
        "orig_bytes": st3.orig_bytes,
        "A2_bytes": a2_bytes,
        "A3_bytes": st3.comp_bytes,
        "A3_vs_A2_pct": round(delta_pct, 4),
        "S_META_bytes": st3.s_meta_bytes,
        "S_RESID_bytes": st3.s_resid_bytes,
        "chunks_tested": forest.chunks_tested,
        "candidates_checked": forest.candidates_checked,
        "deltas_accepted": forest.deltas_accepted,
        "bytes_delta_accepted": forest.bytes_accepted,
        "enc_sec": round(enc_sec, 2),
        "dec_sec": round(dec_sec, 2),
        "sha256_pass": sha_ok,
    }
    print(
        f"[A3] {name:<28} A2={a2_bytes/MB:6.2f}MB -> A3={st3.comp_bytes/MB:6.2f}MB ({delta_pct:+6.3f}%) | "
        f"accepted={forest.deltas_accepted}/{forest.chunks_tested} ({forest.bytes_accepted/MB:.1f}MB) | "
        f"S_RESID={st3.s_resid_bytes/MB:5.2f}MB | meta={st3.s_meta_bytes}B | SHA=PASS ({enc_sec:.1f}s/{dec_sec:.1f}s)",
        flush=True,
    )
    return row


def main() -> None:
    p1a = {r["name"]: r for r in json.loads((V3_DIR / "phase1a_results.json").read_text())}
    dense_path = ensure_dense_snapshot_stress(CORPUS_DIR / "stress_snapshots_dense15.bin")

    print("=" * 120)
    print("PHASE 1B: STAGE 3 BOUNDED DELTA FOREST ABLATION (A3 vs A2 @ zstd -19 --long=27)")
    print("=" * 120, flush=True)

    tests = [
        (CORPUS_DIR / "target_snapshots.bin", "target_snapshots.bin", p1a["target_snapshots.bin"]["A2_comp_bytes"]),
        (CORPUS_DIR / "target_numeric_f32.bin", "target_numeric_f32.bin", p1a["target_numeric_f32.bin"]["A2_comp_bytes"]),
        (CORPUS_DIR / "adv_binaries.bin", "adv_binaries.bin", p1a["adv_binaries.bin"]["A2_comp_bytes"]),
        (dense_path, "stress_snapshots_dense15.bin", None),
    ]

    rows: List[Dict[str, Any]] = []
    for path, name, a2_b in tests:
        rows.append(run_a2_and_a3(path, name, a2_b))

    (V3_DIR / "phase1b_results.json").write_text(json.dumps(rows, indent=2))

    # Evaluate Kill Gate 1B on pre-registered snapshot dataset
    prereg_snap = next(r for r in rows if r["name"] == "target_snapshots.bin")
    dense_snap = next(r for r in rows if r["name"] == "stress_snapshots_dense15.bin")
    print("\n" + "=" * 95)
    print("KILL GATE 1B EVALUATION (A3 must beat A2 by >= 1.0% net size under --long=27)")
    print("=" * 95)
    print(f"  Pre-registered target_snapshots.bin (1.5% edits):   A3 vs A2 = {prereg_snap['A3_vs_A2_pct']:+.4f}%  (accepted={prereg_snap['deltas_accepted']})")
    print(f"  Supplemental stress_snapshots_dense15 (15% edits):  A3 vs A2 = {dense_snap['A3_vs_A2_pct']:+.4f}%  (accepted={dense_snap['deltas_accepted']})")
    if prereg_snap["A3_vs_A2_pct"] <= -1.0:
        print("\n>>> KILL GATE 1B: PASSED on pre-registered snapshot workload -> Keep Stage 3 in Phase 2.")
    else:
        print("\n>>> KILL GATE 1B: NOT MET on pre-registered target_snapshots.bin -> CUT Stage 3 permanently for Phase 2!")
        print("    (Honest Prefix-Aware Gate proved zstd -19 --long=27 already handles 1.5% sparse snapshots natively;")
        print("     Stage 3 only pays off on denser >=10-15% small-delta in-place mutations.)")


if __name__ == "__main__":
    main()
