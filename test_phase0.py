#!/usr/bin/env python3
"""Automated verification suite for Phase 0 falsification harness (v2.2).

Run with:
    python3 -m unittest -v test_phase0.py
or:
    pytest -v test_phase0.py
"""
import itertools
import os
import unittest
from pathlib import Path

import numpy as np

import phase0_harness as p0

DATA_DIR = Path(__file__).resolve().parent / "phase0_data"


class TestIntegerTransforms(unittest.TestCase):
    def test_dwt53_roundtrip_odd_and_even_shapes(self):
        rng = np.random.default_rng(101)
        for shp in ((1, 1, 1), (3, 5, 7), (7, 13, 19), (16, 24, 32), (17, 18, 19)):
            C = rng.integers(-15000, 15000, size=shp, dtype=p0.I64)
            for axes in ((0,), (1, 2), (0, 1, 2)):
                y, plan = p0.dwt53_axes(C, axes, 3)
                rec = p0.dwt53_inv(y, plan, axes)
                np.testing.assert_array_equal(rec, C)

    def test_all_reversible_spectral_programs_roundtrip(self):
        C = p0.synth_hyper(B=24, H=32, W=32, seed=11).astype(p0.I64)
        for kind in ("none", "dwt53", "pot", "clustered_serm", "corr_serm", "clustered_givens", "givens"):
            Y, hdr, ctx = p0.spectral_forward(C, kind, L=3)
            self.assertGreaterEqual(hdr, 0)
            rec = p0.spectral_inverse(Y, ctx)
            np.testing.assert_array_equal(rec, C, err_msg=f"Round-trip failed for {kind}")

    def test_folded_rklt_roundtrip(self):
        C = p0.synth_hyper(B=16, H=32, W=32, seed=13).astype(p0.I64)
        for spec_kind in ("pot", "clustered_serm", "corr_serm"):
            Z, hdr, fctx = p0.folded_forward(C, fold=4, spec_kind=spec_kind)
            self.assertGreater(hdr, 0)
            rec = p0.folded_inverse(Z, fctx)
            np.testing.assert_array_equal(rec, C)

    def test_deadzone_quantizer_lossless_at_step1(self):
        rng = np.random.default_rng(17)
        X = rng.integers(-2000, 2000, size=(8, 16, 16), dtype=p0.I64)
        Q = p0.dz_quant(X, 1)
        np.testing.assert_array_equal(p0.dz_dequant(Q, 1), X)


class TestCCSDS123AndRiceBitstream(unittest.TestCase):
    def test_ccsds123_exact_roundtrip_multiple_shapes(self):
        for shp in ((1, 8, 8), (5, 9, 11), (16, 32, 32)):
            C = p0.synth_hyper(B=shp[0], H=shp[1], W=shp[2], seed=23).astype(p0.I64)
            for P in (0, 1, 3):
                bits, res = p0.row_ccsds123(C, P=P, verify=True)
                self.assertGreater(bits, 0.0)
                rec = p0._ccsds123_reconstruct_numba(res, P=P)
                np.testing.assert_array_equal(rec, C)

    def test_rice_bitstream_exact_roundtrip_normal_and_escape(self):
        rng = np.random.default_rng(29)
        # Small & zero-heavy residuals
        B0 = np.zeros((4, 8, 8), dtype=p0.I64)
        bits0 = p0.real_rice_block_bits(B0, verify=True)
        self.assertGreater(bits0, 0)

        # Heavy-tailed with negative & large escape values (> 32 unary threshold)
        B1 = rng.integers(-20, 20, size=(4, 16, 16), dtype=p0.I64)
        B1[0, 0, 0] = -1234567
        B1[1, 5, 7] = 9876543
        B1[3, 15, 15] = -(1 << 20)
        bits1 = p0.real_rice_block_bits(B1, verify=True)
        self.assertGreater(bits1, 0)

        # Verify Rice tracks idealized Laplace estimator within reasonable margin
        C = p0.synth_hyper(B=16, H=32, W=32, seed=31).astype(p0.I64)
        est_bits, res = p0.row_ccsds123(C, P=3, verify=True)
        rice_bits = p0.real_rice_block_bits(res, verify=True)
        diff_bps = abs(rice_bits - est_bits) / C.size
        self.assertLess(diff_bps, 0.8, f"Rice ({rice_bits/C.size:.3f}) diverged from est ({est_bits/C.size:.3f})")


class TestDataLoaders(unittest.TestCase):
    def test_raw_bsq_bil_bip_roundtrip(self):
        rng = np.random.default_rng(37)
        B, H, W = 5, 7, 9
        C = rng.integers(-3000, 3000, size=(B, H, W), dtype=np.int16)
        for dtype_str in (">i2", "<i2", "<i4"):
            for order, arr in [
                ("BSQ", C),
                ("BIL", np.moveaxis(C, 0, 1)),
                ("BIP", np.moveaxis(C, 0, 2)),
            ]:
                buf = np.ascontiguousarray(arr).astype(dtype_str).tobytes()
                parsed, isz = p0.parse_raw_buffer(buf, dtype_str, order, (B, H, W))
                self.assertEqual(isz, np.dtype(dtype_str).itemsize)
                np.testing.assert_array_equal(parsed, C.astype(p0.I64))


class TestBlindStrideRecovery(unittest.TestCase):
    def test_acf_dc_invariance_and_constant_guard(self):
        rng = np.random.default_rng(41)
        x = rng.standard_normal(4096)
        r0 = p0.acf_fft_pearson(x, 256)
        r1 = p0.acf_fft_pearson(x + 1e7, 256)
        np.testing.assert_allclose(r0, r1, atol=1e-6)

        const = np.full(8192, 32767.0)
        rc = p0.acf_fft_pearson(const, 512)
        self.assertEqual(rc[0], 1.0)
        self.assertTrue(np.all(rc[1:] == 0.0))
        self.assertEqual(p0.recover_strides(const, max_axes=5)[0], [1])

    def test_small_grid_synth_no_collapse(self):
        for fn in (p0.synth_hyper, p0.synth_smooth):
            C = fn(B=16, H=32, W=32, seed=43)
            self.assertGreater(float(C.std()), 10.0)

    def test_4d_and_noise_stride_recovery(self):
        x4 = p0.synth_4d()
        rec4, _ = p0.recover_strides(x4.ravel(), max_axes=5)
        self.assertEqual(rec4, [1, 32, 1024, 16384])

        noise = p0.synth_noise(B=16, H=32, W=32, seed=47).astype(np.float64).ravel()
        self.assertEqual(p0.recover_strides(noise, max_axes=5)[0], [1])

    def test_synth_all_6_permutations(self):
        for fn in (p0.synth_hyper, p0.synth_smooth):
            C = fn(B=32, H=96, W=128, seed=0).astype(np.float64)
            for perm in itertools.permutations(range(3)):
                A = np.ascontiguousarray(np.transpose(C, perm))
                truth = [1, A.shape[2], A.shape[1] * A.shape[2]]
                rec, _ = p0.recover_strides(A.ravel(), max_axes=5)
                self.assertEqual(rec, truth, f"{fn.__name__} perm={perm} failed: {rec} != {truth}")

    @unittest.skipUnless((DATA_DIR / "Indian_pines.mat").exists(), "Real Indian_pines.mat not downloaded")
    def test_real_indian_pines_6_permutations(self):
        from scipy.io import loadmat
        arr = np.asarray(loadmat(str(DATA_DIR / "Indian_pines.mat"))["indian_pines_corrected"])
        C = np.ascontiguousarray(np.moveaxis(arr, 2, 0))[:, :144, :144].astype(np.float64)
        for perm in itertools.permutations(range(3)):
            A = np.ascontiguousarray(np.transpose(C, perm))
            truth = [1, A.shape[2], A.shape[1] * A.shape[2]]
            rec, _ = p0.recover_strides(A.ravel(), max_axes=5)
            self.assertEqual(rec, truth, f"Indian_pines perm={perm}: {rec} != {truth}")

    @unittest.skipUnless((DATA_DIR / "Salinas.mat").exists(), "Real Salinas.mat not downloaded")
    def test_real_salinas_6_permutations(self):
        from scipy.io import loadmat
        arr = np.asarray(loadmat(str(DATA_DIR / "Salinas.mat"))["salinas_corrected"])
        C = np.ascontiguousarray(np.moveaxis(arr, 2, 0))[:64, :256, :216].astype(np.float64)
        for perm in itertools.permutations(range(3)):
            A = np.ascontiguousarray(np.transpose(C, perm))
            truth = [1, A.shape[2], A.shape[1] * A.shape[2]]
            rec, _ = p0.recover_strides(A.ravel(), max_axes=5)
            self.assertEqual(rec, truth, f"Salinas perm={perm}: {rec} != {truth}")

    @unittest.skipUnless((DATA_DIR / "PaviaU.mat").exists(), "Real PaviaU.mat not downloaded")
    def test_real_paviau_6_permutations(self):
        from scipy.io import loadmat
        arr = np.asarray(loadmat(str(DATA_DIR / "PaviaU.mat"))["paviaU"])
        C = np.ascontiguousarray(np.moveaxis(arr, 2, 0))[:, :256, :256].astype(np.float64)
        for perm in itertools.permutations(range(3)):
            A = np.ascontiguousarray(np.transpose(C, perm))
            truth = [1, A.shape[2], A.shape[1] * A.shape[2]]
            rec, _ = p0.recover_strides(A.ravel(), max_axes=5)
            self.assertEqual(rec, truth, f"PaviaU perm={perm}: {rec} != {truth}")


if __name__ == "__main__":
    unittest.main()
