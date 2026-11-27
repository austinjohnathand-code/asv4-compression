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

    @unittest.skipUnless(HND3_BIN.exists(), "Compile ./hnd3 first via `make hnd3`")
    def test_regression_full_range_int32_constant_cube_and_degenerate_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # B1 + B8: Smooth full-range int32 cube (spanning +-1.9e9, post-transform > 2^31)
            rng = np.random.default_rng(3)
            B, H, W = 32, 64, 64
            base = np.cumsum(np.cumsum(rng.normal(0, 1, (H, W)), 0), 1)
            base = base / np.abs(base).max()
            c32 = np.clip(
                np.rint(np.stack([base * (1.9e9 * (0.5 + 0.5 * np.cos(b / 5))) for b in range(B)])
                        + rng.normal(0, 1e6, (B, H, W))),
                -2**31, 2**31 - 1,
            ).astype("<i4")
            r32 = td / "full32.raw"
            r32.write_bytes(c32.tobytes())
            for bk in (0, 6):
                h3 = td / f"full32_k{bk}.hnd3"
                o32 = td / f"full32_k{bk}.raw"
                subprocess.run([
                    str(HND3_BIN), "enc", str(r32), str(h3),
                    "--shape", f"{B},{H},{W}", "--dtype", "i32", "--base-k", str(bk),
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run([str(HND3_BIN), "dec", str(h3), str(o32)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(o32.read_bytes(), c32.tobytes())
                np.testing.assert_array_equal(p3.decode_hnd3(h3.read_bytes(), verify_sha=True), c32)

            # B2: Constant all-zero cube (>600x compression must decode cleanly!)
            z = np.zeros((64, 256, 256), dtype="<i2")
            rz = td / "zero.raw"; hz = td / "zero.hnd3"; oz = td / "zero_d.raw"
            rz.write_bytes(z.tobytes())
            subprocess.run([
                str(HND3_BIN), "enc", str(rz), str(hz),
                "--shape", "64,256,256", "--dtype", "i16", "--base-k", "0",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertGreater(z.nbytes / hz.stat().st_size, 256.0)
            subprocess.run([str(HND3_BIN), "dec", str(hz), str(oz)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(oz.read_bytes(), z.tobytes())

            # Degenerate shapes
            for shp in [(1, 32, 32), (16, 1, 64), (16, 3, 3), (2, 5, 7)]:
                arr = rng.integers(0, 1000, size=shp, dtype=np.int64).astype("<i2")
                ra = td / "deg.raw"; ha = td / "deg.hnd3"; oa = td / "deg_d.raw"
                ra.write_bytes(arr.tobytes())
                subprocess.run([
                    str(HND3_BIN), "enc", str(ra), str(ha),
                    "--shape", ",".join(map(str, shp)), "--dtype", "i16", "--base-k", "6",
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run([str(HND3_BIN), "dec", str(ha), str(oa)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(oa.read_bytes(), arr.tobytes())

    @unittest.skipUnless(HND3_BIN.exists(), "Compile ./hnd3 first via `make hnd3`")
    def test_regression_cli_validation_preview_sha_truncation_and_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            C = p0.synth_hyper(B=32, H=40, W=40, seed=11).astype("<i2")
            raw = td / "c.raw"; raw.write_bytes(C.tobytes())
            h3_mt = td / "c_mt.hnd3"; h3_1t = td / "c_1t.hnd3"

            # B3-B5 & B7: bad CLI arguments must exit cleanly with rc=1 (no crash)
            bad_cases = [
                ["enc", str(raw), str(td / "x.hnd3"), "--shape", "64,256,256", "--dtype", "i16"],
                ["enc", str(raw), str(td / "x.hnd3"), "--shape", "32,40,40", "--dtype", "i16", "--spec-axis", "5"],
                ["enc", str(raw), str(td / "x.hnd3"), "--shape", "32,40,40"],
                ["enc", str(raw), str(td / "x.hnd3"), "--shape", "32,40,40", "--dtype", "i16", "--base-k", "20"],
                ["enc", str(raw), str(td / "x.hnd3"), "--shape", "32,40,40", "--dtype", "i16", "--levels", "12"],
                ["fuzz", str(raw), "10"],
            ]
            for args in bad_cases:
                r = subprocess.run([str(HND3_BIN)] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(r.returncode, 1, f"Expected clean rc=1 for {args}, got {r.returncode}")

            # Thread determinism (1T == MT bit-exact on both encode & decode)
            subprocess.run([
                str(HND3_BIN), "enc", str(raw), str(h3_mt),
                "--shape", "32,40,40", "--dtype", "i16", "--base-k", "6",
            ], check=True, stdout=subprocess.PIPE)
            subprocess.run([
                str(HND3_BIN), "enc", str(raw), str(h3_1t),
                "--shape", "32,40,40", "--dtype", "i16", "--base-k", "6", "--threads", "1",
            ], check=True, stdout=subprocess.PIPE)
            blob = h3_mt.read_bytes()
            self.assertEqual(blob, h3_1t.read_bytes(), "1T vs MT encode mismatch")

            # B6: Preview decode must print SHA256=N/A (preview), NOT SHA256=PASS, and match 1T == MT
            p_mt = td / "p_mt.raw"; p_1t = td / "p_1t.raw"
            pr = subprocess.run([str(HND3_BIN), "dec", str(h3_mt), str(p_mt), "--stop-layer", "6"],
                                check=True, stdout=subprocess.PIPE, text=True)
            self.assertIn("SHA256=N/A (preview)", pr.stdout)
            subprocess.run([str(HND3_BIN), "dec", str(h3_mt), str(p_1t), "--stop-layer", "6", "--threads", "1"],
                           check=True, stdout=subprocess.PIPE)
            self.assertEqual(p_mt.read_bytes(), p_1t.read_bytes(), "1T vs MT preview decode mismatch")

            # Truncated-at-layer-boundary decode (core progressive promise)
            # Walk base layer byte boundary via header + spec_len + base stripes
            import struct
            (_, _, _, _, _, n_lv, sp_ax, base_k, n_st, _, d0, d1, d2, _, spec_len) = struct.unpack_from(p3.HDR_FMT, blob, 0)
            wd = [d0, d1, d2]; B0 = wd.pop(sp_ax)
            plan = p1._build_plan_from_shape((B0, wd[0], wd[1]), n_lv)
            sched = p1._c2f_subband_schedule(plan, (B0, wd[0], wd[1]))
            pos = p3.HDR_SIZE + spec_len
            for _ in range(len(sched) * n_st):
                plen = struct.unpack_from("<I", blob, pos)[0]
                pos += 4 + plen
            trunc_f = td / "trunc_k6.hnd3"; trunc_out = td / "trunc_k6.raw"
            trunc_f.write_bytes(blob[:pos])
            subprocess.run([str(HND3_BIN), "dec", str(trunc_f), str(trunc_out), "--stop-layer", "6"],
                           check=True, stdout=subprocess.PIPE)
            self.assertEqual(trunc_out.read_bytes(), p_mt.read_bytes())


if __name__ == "__main__":
    unittest.main()
