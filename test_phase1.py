#!/usr/bin/env python3
"""Automated verification suite for Phase 1 .hnd1 end-to-end binary codec.

Run with:
    python3 -m unittest -v test_phase1.py
"""
import math
import unittest
from pathlib import Path

import numpy as np

import phase0_harness as p0
import phase1_codec as p1

DATA_DIR = Path(__file__).resolve().parent / "phase0_data"


class TestLduRklt(unittest.TestCase):
    def test_complete_pivoted_ldu_exact_roundtrip_and_serialization(self):
        rng = np.random.default_rng(101)
        for M in (2, 5, 8, 16, 33, 64):
            A = rng.standard_normal((M, M))
            Q, _ = np.linalg.qr(A)
            f = p1.complete_pivoted_ldu(Q)
            X = rng.integers(-8000, 8000, size=(M, 512), dtype=np.int64)
            Y = p1._ldu_fwd_numba(X, f["rperm"], f["cperm"], f["sign0"], f["dsgn"], f["dperm"],
                                  f["Lq"], f["Uq"], f["cq"], f["inv_cq_m1"], f["cq_m1"])
            # Pack & unpack binary stage, then invert
            grp = list(range(M))
            raw = p1.pack_ldu_stage(grp, f)
            grp2, f2, end_pos = p1.unpack_ldu_stage(memoryview(raw), 0)
            self.assertEqual(end_pos, len(raw))
            np.testing.assert_array_equal(grp2, grp)
            rec = p1._ldu_inv_numba(Y, f2["rperm"], f2["cperm"], f2["sign0"], f2["dsgn"], f2["dperm"],
                                    f2["Lq"], f2["Uq"], f2["cq"], f2["inv_cq_m1"], f2["cq_m1"])
            np.testing.assert_array_equal(rec, X)


class TestRans3D(unittest.TestCase):
    def test_rans_3d_roundtrip_normal_sparse_and_escape(self):
        rng = np.random.default_rng(202)
        # Sparse zeros
        z = np.zeros((4, 16, 16), dtype=np.int64)
        blob_z = p1.pack_rans_subband(z)
        rec_z, pos_z = p1.unpack_rans_subband(memoryview(blob_z), 0, z.shape)
        self.assertEqual(pos_z, len(blob_z))
        np.testing.assert_array_equal(rec_z, z)
        # Sparse zeros should compress well below 1 bit/sample (beating Rice floor!)
        self.assertLess(len(blob_z) * 8 / z.size, 0.25)

        # Heavy-tailed + escape values + parent context
        par = rng.integers(-500, 500, size=(6, 8, 8), dtype=np.int64)
        blk = rng.integers(-40, 40, size=(6, 16, 16), dtype=np.int64)
        blk[0, 0, 0] = -1234567
        blk[3, 7, 11] = 2345678
        blk[5, 15, 15] = -(1 << 21)
        blob = p1.pack_rans_subband(blk, parent=par)
        rec, pos = p1.unpack_rans_subband(memoryview(blob), 0, blk.shape, parent=par)
        self.assertEqual(pos, len(blob))
        np.testing.assert_array_equal(rec, blk)


class TestHnd1Container(unittest.TestCase):
    def test_arbitrary_odd_shapes_and_dtypes_lossless(self):
        rng = np.random.default_rng(303)
        for shp, dt in [
            ((17, 29, 37), np.int16),
            ((12, 33, 41), np.uint16),
            ((9, 25, 27), np.int32),
        ]:
            lo = 0 if dt == np.uint16 else -4000
            hi = 8000
            C = rng.integers(lo, hi, size=shp, dtype=dt)
            for mode in ("full", "clustered", "auto"):
                blob = p1.encode_cube(C, spec_mode=mode, L=3, q_base=1)
                rec = p1.decode_cube(blob, verify_sha=True)
                self.assertEqual(rec.dtype, C.dtype)
                np.testing.assert_array_equal(rec, C)

    def test_prefix_progressive_and_l2_lossy_monotonic_psnr(self):
        C = p0.synth_hyper(B=24, H=64, W=64, seed=5)
        blob = p1.encode_cube(C, spec_mode="auto", L=3, q_base=1)
        peak = float(int(C.max()) - int(C.min()))

        # Prefix progressive decode monotonically improves MSE to exact 0 at level 3
        mses = []
        for lv in range(4):
            rec = p1.decode_cube(blob, max_level=lv, verify_sha=(lv == 3))
            mses.append(float(np.mean((C.astype(np.float64) - rec.astype(np.float64)) ** 2)))
        self.assertEqual(mses[-1], 0.0)
        self.assertLess(mses[2], mses[0])

        # Lossy q_base sweep: strictly monotonic rate & PSNR
        rates, psnrs = [], []
        for q in (2, 8, 32):
            qlob = p1.encode_cube(C, spec_mode="auto", L=3, q_base=q)
            qrec = p1.decode_cube(qlob, verify_sha=False)
            mse = float(np.mean((C.astype(np.float64) - qrec.astype(np.float64)) ** 2))
            rates.append(len(qlob) * 8 / C.size)
            psnrs.append(10.0 * math.log10(peak * peak / mse))
        self.assertGreater(rates[0], rates[1])
        self.assertGreater(rates[1], rates[2])
        self.assertGreater(psnrs[0], psnrs[1])
        self.assertGreater(psnrs[1], psnrs[2])

    def test_blind_1d_trackA_all_permutations_and_noise_fallback(self):
        C = p0.synth_hyper(B=24, H=48, W=64, seed=9)
        for perm in ((0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0)):
            A = np.ascontiguousarray(np.transpose(C, perm))
            flat = A.ravel()
            blob, rec_s, sp_ax = p1.encode_blind_1d(flat, spec_mode="auto", L=3)
            self.assertEqual(rec_s, [1, A.shape[2], A.shape[1] * A.shape[2]])
            self.assertEqual(("B", "H", "W")[perm[sp_ax]], "B")
            rec = p1.decode_blind_1d(blob, verify_sha=True)
            np.testing.assert_array_equal(rec, flat)

        # Incompressible noise -> clean raw fallback with tiny framing overhead (< 0.1%)
        noise = p0.synth_noise(B=16, H=32, W=32, seed=11).ravel()
        blob_n, rec_n, sp_n = p1.encode_blind_1d(noise)
        self.assertEqual(rec_n, [1])
        self.assertIsNone(sp_n)
        self.assertLess(len(blob_n) / (noise.size * 2), 1.002)
        np.testing.assert_array_equal(p1.decode_blind_1d(blob_n, verify_sha=True), noise)

    @unittest.skipUnless((DATA_DIR / "Indian_pines.mat").exists(), "Real Indian_pines.mat not downloaded")
    def test_real_indian_pines_native_odd_145x145_exact_roundtrip(self):
        # Full native (200, 145, 145) -- odd spatial dimensions!
        C = p1._load_mat_cube(DATA_DIR / "Indian_pines.mat")
        self.assertEqual(C.shape, (200, 145, 145))
        blob = p1.encode_cube(C, spec_mode="auto", L=3, q_base=1)
        rec = p1.decode_cube(blob, verify_sha=True)
        np.testing.assert_array_equal(rec, C)
        # Must beat standard CCSDS-123 Rice (6.734 b/s) cleanly!
        self.assertLess(len(blob) * 8 / C.size, 6.50)


if __name__ == "__main__":
    unittest.main()
