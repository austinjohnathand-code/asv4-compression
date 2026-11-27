"""Unit & bit-exact round-trip verification tests for FTCC."""

from __future__ import annotations

import hashlib
import numpy as np

from ftcc import (
    CodecConfig,
    ClumpingConfig,
    ForestConfig,
    LAYOUT_1D_IDENTITY,
    LAYOUT_1D_DELTA,
    LAYOUT_2D_TRANSPOSE,
    LAYOUT_2D_TRANSPOSE_DELTA,
    LAYOUT_3D_CUBE_PERMUTE,
    LAYOUT_3D_CUBE_PERMUTE_DELTA,
    LayoutDescriptor,
    apply_layout_forward,
    apply_layout_inverse,
    encode_buffer,
    decode_buffer,
)
from ftcc.stage3_forest import coarse_fine_encode, coarse_fine_decode
from corpus import (
    gen_tabular_telemetry,
    gen_volumetric_3d,
    gen_distant_records,
    gen_mixed_text_logs,
    gen_mixed_binary_entropy,
)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_all_layout_inverses_arbitrary_lengths():
    rng = np.random.default_rng(1)
    layouts = [
        LayoutDescriptor(LAYOUT_1D_IDENTITY),
        LayoutDescriptor(LAYOUT_1D_DELTA, 1),
        LayoutDescriptor(LAYOUT_1D_DELTA, 4),
        LayoutDescriptor(LAYOUT_2D_TRANSPOSE, 13),
        LayoutDescriptor(LAYOUT_2D_TRANSPOSE, 64),
        LayoutDescriptor(LAYOUT_2D_TRANSPOSE_DELTA, 17),
        LayoutDescriptor(LAYOUT_2D_TRANSPOSE_DELTA, 64),
        LayoutDescriptor(LAYOUT_3D_CUBE_PERMUTE, 2, 64, 0),
        LayoutDescriptor(LAYOUT_3D_CUBE_PERMUTE, 2, 64, 1),
        LayoutDescriptor(LAYOUT_3D_CUBE_PERMUTE, 4, 32, 2),
        LayoutDescriptor(LAYOUT_3D_CUBE_PERMUTE_DELTA, 2, 64, 0),
        LayoutDescriptor(LAYOUT_3D_CUBE_PERMUTE_DELTA, 2, 64, 1),
        LayoutDescriptor(LAYOUT_3D_CUBE_PERMUTE_DELTA, 4, 32, 2),
    ]
    # Test non-divisible lengths (including primes and small sizes)
    for n in (0, 1, 7, 63, 257, 4099, 16411):
        arr = rng.integers(0, 256, size=n, dtype=np.uint8)
        for lay in layouts:
            fwd = apply_layout_forward(arr, lay)
            inv = apply_layout_inverse(fwd, lay)
            assert inv.shape == arr.shape, f"Shape mismatch for {lay}, n={n}"
            assert np.array_equal(inv, arr), f"Bit mismatch for {lay}, n={n}"
    print("[PASS] All N-D layout forward/inverse transforms bit-exact on arbitrary lengths")


def test_coarse_fine_roundtrip():
    t = np.linspace(0, 10, 50000)
    sig = ((128 + 80 * np.sin(t) + 15 * np.cos(7 * t)) % 256).astype(np.uint8)
    cfg = ForestConfig(tree3_min_improvement=1.5, tree3_level1_improvement=1.5)
    ok, n_lvl, cb, fine = coarse_fine_encode(sig, cfg)
    assert ok and n_lvl in (1, 2)
    rec = coarse_fine_decode(fine, cb, n_lvl)
    assert np.array_equal(rec, sig), "Coarse-to-fine roundtrip mismatch"
    print(f"[PASS] Coarse-to-fine roundtrip bit-exact (levels={n_lvl})")


def test_edge_cases_and_finishers():
    rng = np.random.default_rng(9)
    cases = {
        "empty": b"",
        "tiny_13b": b"hello world!!",
        "small_prime": rng.bytes(7919),
        "random_1mb": rng.bytes(1024 * 1024),
        "tabular_2mb": gen_tabular_telemetry(2 * 1024 * 1024),
        "volumetric_2mb": gen_volumetric_3d(2 * 1024 * 1024),
        "logs_2mb": gen_mixed_text_logs(2 * 1024 * 1024),
        "mixed_2mb": gen_mixed_binary_entropy(2 * 1024 * 1024),
    }

    for finisher in ("zstd", "zstd_ldm", "lz4"):
        cfg = CodecConfig(
            finisher=finisher,
            clumping=ClumpingConfig(min_clump_size=131072, max_clump_size=524288),
        )
        for name, raw in cases.items():
            enc, st = encode_buffer(raw, cfg)
            dec = decode_buffer(enc)
            assert sha256(dec) == sha256(raw), f"Roundtrip failed on {name} ({finisher})"
            # Verify random_1mb escapes cleanly to identity with tiny overhead (<0.1%)
            if name == "random_1mb":
                assert st.overhead_pct < 0.1, f"Random overhead too high: {st.overhead_pct:.4f}%"
                assert st.mode_counts.get("IDENTITY", 0) == st.num_clumps, (
                    f"Random data did not fully escape to IDENTITY: {st.mode_counts}"
                )
        print(f"[PASS] Full codec bit-exact roundtrip on all edge cases ({finisher})")


def test_distant_records_tree2():
    raw = gen_distant_records(16 * 1024 * 1024, seed=77)
    cfg = CodecConfig(
        finisher="zstd",
        clumping=ClumpingConfig(min_clump_size=262144, max_clump_size=1048576),
    )
    enc, st = encode_buffer(raw, cfg)
    dec = decode_buffer(enc)
    assert sha256(dec) == sha256(raw)
    assert st.tree2_edges_used >= 3, f"Expected Tree 2 edges on distant records, got {st.tree2_edges_used}"
    print(f"[PASS] Distant records roundtrip: modes={st.mode_counts}, t2_edges={st.tree2_edges_used}")


if __name__ == "__main__":
    test_all_layout_inverses_arbitrary_lengths()
    test_coarse_fine_roundtrip()
    test_edge_cases_and_finishers()
    test_distant_records_tree2()
    print("ALL TESTS PASSED!")
