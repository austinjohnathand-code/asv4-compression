#!/usr/bin/env python3
"""Phase 3 Cross-Language & Native C++20 Blind Track A Verification Suite (.hnd3).

Run with:
    python3 -m unittest -v test_phase3.py
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import phase0_harness as p0
import phase1_codec as p1
import phase3_codec as p3

ROOT = Path(__file__).resolve().parent
HND3_BIN = ROOT / "hnd3"
DATA_DIR = ROOT / "phase0_data"


def _dt_flag(dt):
    dt = np.dtype(dt).newbyteorder("<")
    return {np.dtype("<i2"): "i16", np.dtype("<u2"): "u16", np.dtype("<i4"): "i32"}[dt]


class TestPhase3Hnd3(unittest.TestCase):
    @unittest.skipUnless(HND3_BIN.exists(), "Compile ./hnd3 first via `make hnd3`")
    def test_cross_language_striped_py_to_cpp_and_cpp_to_py(self):
        rng = np.random.default_rng(303)
        cases = [
            (p0.synth_hyper(B=32, H=41, W=47, seed=1).astype("<i2"), 0, 16, 0),
            (p0.synth_hyper(B=32, H=41, W=47, seed=2).astype("<i2"), 6, 16, 0),
            (rng.integers(0, 10000, size=(24, 31, 35), dtype="<u2"), 6, 8, 1),
            (rng.integers(-60000, 60000, size=(12, 23, 29), dtype="<i4"), 6, 4, 2),
        ]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for idx, (C, bk, ns, sp_ax) in enumerate(cases):
                # 1. Python encode -> C++ decode
                py_f = td / f"py_{idx}.hnd3"
                cpp_out = td / f"cpp_out_{idx}.raw"
                blob = p3.encode_hnd3(C, spec_mode="auto", L=3, base_k=bk, n_stripes=ns, spec_axis=sp_ax)
                py_f.write_bytes(blob)
                subprocess.run([str(HND3_BIN), "dec", str(py_f), str(cpp_out)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(cpp_out.read_bytes(), C.tobytes(), f"Py->C++ mismatch case {idx}")

                # 2. C++ encode -> Python decode
                in_raw = td / f"in_{idx}.raw"
                cpp_f = td / f"cpp_{idx}.hnd3"
                in_raw.write_bytes(C.tobytes())
                subprocess.run([
                    str(HND3_BIN), "enc", str(in_raw), str(cpp_f),
                    "--shape", f"{C.shape[0]},{C.shape[1]},{C.shape[2]}",
                    "--dtype", _dt_flag(C.dtype), "--base-k", str(bk),
                    "--stripes", str(ns), "--spec-axis", str(sp_ax),
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                rec_py = p3.decode_hnd3(cpp_f.read_bytes(), stop_layer=0, verify_sha=True)
                np.testing.assert_array_equal(rec_py, C, err_msg=f"C++->Py mismatch case {idx}")

                # 3. Byte-identical stop_layer=3 preview between Python & C++
                if bk == 6:
                    cpp_mid = td / f"mid_{idx}.raw"
                    subprocess.run([str(HND3_BIN), "dec", str(py_f), str(cpp_mid), "--stop-layer", "3"],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    py_mid = p3.decode_hnd3(blob, stop_layer=3, verify_sha=False)
                    self.assertEqual(cpp_mid.read_bytes(), py_mid.tobytes())

    @unittest.skipUnless(HND3_BIN.exists(), "Compile ./hnd3 first via `make hnd3`")
    def test_native_cpp_blind_trackA_bsq_bil_bip_and_noise(self):
        C = p0.synth_hyper(B=32, H=64, W=96, seed=0).astype("<i2")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for tag, perm in [("BSQ", (0, 1, 2)), ("BIL", (1, 0, 2)), ("BIP", (1, 2, 0))]:
                A = np.ascontiguousarray(np.transpose(C, perm))
                in_raw = td / f"{tag}.raw"
                h3 = td / f"{tag}.hnd3"
                out_raw = td / f"{tag}_dec.raw"
                in_raw.write_bytes(A.tobytes())
                pr = subprocess.run([
                    str(HND3_BIN), "enc", str(in_raw), str(h3),
                    "--blind", "--dtype", "i16", "--base-k", "0",
                ], check=True, stdout=subprocess.PIPE, text=True)
                # Must discover exact 3D strides & compress cleanly (< 7 b/s vs 16 b/s raw!)
                self.assertIn(f"shape=({A.shape[0]},{A.shape[1]},{A.shape[2]})", pr.stdout)
                self.assertLess(h3.stat().st_size * 8 / A.size, 7.0)
                subprocess.run([str(HND3_BIN), "dec", str(h3), str(out_raw)],
                               check=True, stdout=subprocess.PIPE)
                self.assertEqual(out_raw.read_bytes(), A.tobytes())
                # Also decode in Python!
                rec_py = p3.decode_hnd3(h3.read_bytes(), verify_sha=True)
                np.testing.assert_array_equal(rec_py, A)

            # Incompressible noise -> pure C++ clean raw fallback (< 0.2% tax)
            noise = p0.synth_noise(B=16, H=32, W=32, seed=7).astype("<i2")
            nr = td / "noise.raw"; nh = td / "noise.hnd3"; no = td / "noise_dec.raw"
            nr.write_bytes(noise.tobytes())
            pr_n = subprocess.run([str(HND3_BIN), "enc", str(nr), str(nh), "--blind", "--dtype", "i16"],
                                  check=True, stdout=subprocess.PIPE, text=True)
            self.assertIn("strides=[1]", pr_n.stdout)
            self.assertLess(nh.stat().st_size / nr.stat().st_size, 1.003)
            subprocess.run([str(HND3_BIN), "dec", str(nh), str(no)], check=True, stdout=subprocess.PIPE)
            self.assertEqual(no.read_bytes(), noise.tobytes())

    @unittest.skipUnless(HND3_BIN.exists() and (DATA_DIR / "Indian_pines.mat").exists(),
                         "Requires ./hnd3 and phase0_data/Indian_pines.mat")
    def test_real_indian_pines_native_cpp_blind_and_interop(self):
        C = p1._load_mat_cube(DATA_DIR / "Indian_pines.mat").astype("<u2")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "ip.raw"; h3 = td / "ip.hnd3"; out = td / "ip_dec.raw"
            raw.write_bytes(C.tobytes())
            pr = subprocess.run([
                str(HND3_BIN), "enc", str(raw), str(h3),
                "--blind", "--dtype", "u16", "--base-k", "0",
            ], check=True, stdout=subprocess.PIPE, text=True)
            self.assertIn("shape=(200,145,145)", pr.stdout)
            self.assertIn("spec_ax=0", pr.stdout)
            subprocess.run([str(HND3_BIN), "dec", str(h3), str(out)], check=True, stdout=subprocess.PIPE)
            self.assertEqual(out.read_bytes(), raw.read_bytes())
            np.testing.assert_array_equal(p3.decode_hnd3(h3.read_bytes(), verify_sha=True), C)


if __name__ == "__main__":
    unittest.main()
