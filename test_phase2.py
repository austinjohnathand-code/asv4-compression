#!/usr/bin/env python3
"""Cross-language Python <-> C++20 NEON verification suite for Phase 2 (.hnd2).

Run with:
    python3 -m unittest -v test_phase2.py
"""
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import phase0_harness as p0
import phase1_codec as p1
import phase2_codec as p2

ROOT = Path(__file__).resolve().parent
HND2_BIN = ROOT / "hnd2"
DATA_DIR = ROOT / "phase0_data"


def _dt_flag(dt):
    dt = np.dtype(dt).newbyteorder("<")
    return {np.dtype("<i2"): "i16", np.dtype("<u2"): "u16", np.dtype("<i4"): "i32"}[dt]


class TestPhase2Hnd2(unittest.TestCase):
    def test_python_hnd2_embedded_and_oneshot_roundtrip(self):
        C = p0.synth_hyper(B=24, H=45, W=53, seed=19)  # odd spatial dims!
        peak = float(int(C.max()) - int(C.min()))

        # One-shot (base_k=0)
        b0 = p2.encode_hnd2(C, spec_mode="auto", L=3, base_k=0)
        np.testing.assert_array_equal(p2.decode_hnd2(b0, verify_sha=True), C)

        # 7-layer quality-embedded (base_k=6): monotonic PSNR from layer 6 -> 0 -> exact
        b6 = p2.encode_hnd2(C, spec_mode="auto", L=3, base_k=6)
        psnrs = []
        for k in (6, 4, 2, 0):
            rec = p2.decode_hnd2(b6, stop_layer=k, verify_sha=(k == 0))
            mse = float(np.mean((C.astype(np.float64) - rec.astype(np.float64)) ** 2))
            if k == 0:
                self.assertEqual(mse, 0.0)
                np.testing.assert_array_equal(rec, C)
            else:
                psnrs.append(10.0 * math.log10(peak * peak / mse))
        self.assertLess(psnrs[0], psnrs[1])
        self.assertLess(psnrs[1], psnrs[2])

    @unittest.skipUnless(HND2_BIN.exists(), "Compile ./hnd2 first via `make hnd2`")
    def test_cross_language_python_to_cpp_and_cpp_to_python(self):
        rng = np.random.default_rng(77)
        cases = [
            (p0.synth_hyper(B=20, H=37, W=43, seed=3).astype("<i2"), 0, 0),
            (p0.synth_hyper(B=20, H=37, W=43, seed=4).astype("<i2"), 6, 0),
            (rng.integers(0, 12000, size=(16, 29, 31), dtype="<u2"), 6, 1),
            (rng.integers(-50000, 50000, size=(12, 25, 27), dtype="<i4"), 6, 2),
        ]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for idx, (C, base_k, sp_ax) in enumerate(cases):
                # 1. Python encode -> C++ decode (verify SHA-256 + exact raw bytes!)
                py_hnd2 = td / f"py_{idx}.hnd2"
                cpp_raw = td / f"cpp_out_{idx}.raw"
                blob = p2.encode_hnd2(C, spec_mode="auto", L=3, base_k=base_k, spec_axis=sp_ax)
                py_hnd2.write_bytes(blob)
                subprocess.run([str(HND2_BIN), "dec", str(py_hnd2), str(cpp_raw)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(cpp_raw.read_bytes(), C.tobytes(), f"Py->C++ mismatch case {idx}")

                # 2. C++ encode -> Python decode (verify SHA-256 + exact array!)
                in_raw = td / f"in_{idx}.raw"
                cpp_hnd2 = td / f"cpp_{idx}.hnd2"
                in_raw.write_bytes(C.tobytes())
                shp_s = f"{C.shape[0]},{C.shape[1]},{C.shape[2]}"
                subprocess.run([
                    str(HND2_BIN), "enc", str(in_raw), str(cpp_hnd2),
                    "--shape", shp_s, "--dtype", _dt_flag(C.dtype),
                    "--base-k", str(base_k), "--spec-axis", str(sp_ax), "--mode", "auto",
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                rec_py = p2.decode_hnd2(cpp_hnd2.read_bytes(), stop_layer=0, verify_sha=True)
                np.testing.assert_array_equal(rec_py, C, err_msg=f"C++->Py mismatch case {idx}")

                # 3. Progressive truncated stop_layer match between Python & C++!
                if base_k == 6:
                    cpp_mid = td / f"cpp_mid_{idx}.raw"
                    subprocess.run([str(HND2_BIN), "dec", str(py_hnd2), str(cpp_mid), "--stop-layer", "3"],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    py_mid = p2.decode_hnd2(blob, stop_layer=3, verify_sha=False)
                    self.assertEqual(cpp_mid.read_bytes(), py_mid.tobytes(), f"Stop-layer 3 mismatch case {idx}")

    @unittest.skipUnless(HND2_BIN.exists(), "Compile ./hnd2 first via `make hnd2`")
    def test_blind_1d_python_encode_cpp_decode_and_noise_fallback(self):
        C = p0.synth_hyper(B=24, H=48, W=64, seed=13).astype("<i2")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for perm in ((0, 1, 2), (1, 2, 0)):
                A = np.ascontiguousarray(np.transpose(C, perm))
                blob, rec_s, _ax = p2.encode_hnd2_blind_1d(A.ravel(), base_k=6)
                self.assertEqual(rec_s[:3], [1, A.shape[2], A.shape[1] * A.shape[2]])
                p = td / "blind.hnd2"
                o = td / "blind.raw"
                p.write_bytes(blob)
                subprocess.run([str(HND2_BIN), "dec", str(p), str(o)], check=True, stdout=subprocess.PIPE)
                self.assertEqual(o.read_bytes(), A.tobytes())

            noise = p0.synth_noise(B=16, H=32, W=32, seed=5).astype("<i2").ravel()
            blob_n, rec_n, _ = p2.encode_hnd2_blind_1d(noise, base_k=6)
            self.assertEqual(rec_n, [1])
            p = td / "noise.hnd2"
            o = td / "noise.raw"
            p.write_bytes(blob_n)
            subprocess.run([str(HND2_BIN), "dec", str(p), str(o)], check=True, stdout=subprocess.PIPE)
            self.assertEqual(o.read_bytes(), noise.tobytes())

    @unittest.skipUnless(HND2_BIN.exists() and (DATA_DIR / "Indian_pines.mat").exists(),
                         "Requires ./hnd2 and phase0_data/Indian_pines.mat")
    def test_real_indian_pines_python_cpp_interop(self):
        C = p1._load_mat_cube(DATA_DIR / "Indian_pines.mat").astype("<u2")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "ip.raw"
            f_cpp = td / "ip_cpp.hnd2"
            out_cpp = td / "ip_dec.raw"
            raw.write_bytes(C.tobytes())
            subprocess.run([
                str(HND2_BIN), "enc", str(raw), str(f_cpp),
                "--shape", f"{C.shape[0]},{C.shape[1]},{C.shape[2]}",
                "--dtype", "u16", "--base-k", "6", "--mode", "auto",
            ], check=True, stdout=subprocess.PIPE)
            # Decode with BOTH C++ and Python!
            subprocess.run([str(HND2_BIN), "dec", str(f_cpp), str(out_cpp)], check=True, stdout=subprocess.PIPE)
            self.assertEqual(out_cpp.read_bytes(), C.tobytes())
            rec_py = p2.decode_hnd2(f_cpp.read_bytes(), stop_layer=0, verify_sha=True)
            np.testing.assert_array_equal(rec_py, C)


if __name__ == "__main__":
    unittest.main()
