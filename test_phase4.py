import hashlib
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HND3_BIN = ROOT / "hnd3"
LIB_A = ROOT / "libhnd3.a"
DATA_DIR = ROOT / "phase0_data"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class TestPhase4Hnd4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HND3_BIN.exists() or not LIB_A.exists():
            subprocess.run(["make", "hnd3"], cwd=ROOT, check=True)

    def test_hnd4_tiled_lossless_bsq_bil_bip_odd_shapes_and_dtypes(self):
        rng = np.random.default_rng(404)
        cases = [
            ((19, 73, 85), "<i2", "i16", 0, 32, 6),
            ((67, 17, 79), "<u2", "u16", 1, 32, 0),
            ((71, 63, 13), "<i4", "i32", 2, 32, 6),
            ((32, 129, 130), "<i2", "i16", 0, 64, 6),
        ]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for idx, (shape, npdt, dt_str, axis, tsz, bk) in enumerate(cases):
                if dt_str == "i16":
                    cube = rng.integers(-6000, 6000, size=shape, dtype=npdt)
                elif dt_str == "u16":
                    cube = rng.integers(0, 60000, size=shape, dtype=npdt)
                else:
                    cube = rng.integers(-500000, 500000, size=shape, dtype=npdt)
                raw_in = td / f"in_{idx}.raw"
                hnd4_f = td / f"out_{idx}.hnd4"
                raw_out = td / f"dec_{idx}.raw"
                cube.tofile(raw_in)

                enc = subprocess.run(
                    [
                        str(HND3_BIN), "enc", str(raw_in), str(hnd4_f),
                        "--shape", ",".join(map(str, shape)),
                        "--dtype", dt_str,
                        "--spec-axis", str(axis),
                        "--base-k", str(bk),
                        "--tile-size", str(tsz),
                        "--threads", "4",
                    ],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("ENC4", enc.stdout)
                self.assertIn("VERIFY=PASS", enc.stdout)
                self.assertEqual(hnd4_f.read_bytes()[:4], b"HND4")

                info = subprocess.run(
                    [str(HND3_BIN), "info", str(hnd4_f)],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("HND4 shape=", info.stdout)
                self.assertIn("tiles=", info.stdout)

                dec = subprocess.run(
                    [str(HND3_BIN), "dec", str(hnd4_f), str(raw_out), "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("SHA256=PASS", dec.stdout)
                self.assertIn("CRC32C=PASS", dec.stdout)
                self.assertEqual(raw_in.read_bytes(), raw_out.read_bytes())

    def test_hnd4_random_access_single_tile_and_preview(self):
        rng = np.random.default_rng(77)
        B, H, W = 24, 100, 110
        # Smooth correlated cube so progressive preview has high PSNR
        base = np.linspace(0, 1000, H, dtype=np.float64)[:, None] + np.linspace(0, 800, W, dtype=np.float64)[None, :]
        cube = np.zeros((B, H, W), dtype="<i2")
        for b in range(B):
            cube[b] = np.round((0.5 + 0.05 * b) * base + rng.normal(0, 8.0, (H, W))).astype("<i2")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_in = td / "in.raw"
            hnd4_f = td / "tiled.hnd4"
            cube.tofile(raw_in)

            subprocess.run(
                [
                    str(HND3_BIN), "enc", str(raw_in), str(hnd4_f),
                    "--shape", f"{B},{H},{W}", "--dtype", "i16",
                    "--base-k", "6", "--tile-size", "48", "--threads", "4",
                ],
                check=True, stdout=subprocess.DEVNULL,
            )
            # Grid is ceil(100/48)=3 x ceil(110/48)=3; test interior (1,1) AND ragged corner (2,2)!
            for ty, tx in [(0, 0), (1, 1), (2, 2)]:
                y0, y1 = ty * 48, min(H, (ty + 1) * 48)
                x0, x1 = tx * 48, min(W, (tx + 1) * 48)
                tile_out = td / f"tile_{ty}_{tx}.raw"
                dec = subprocess.run(
                    [str(HND3_BIN), "dec", str(hnd4_f), str(tile_out), "--tile", f"{ty},{tx}"],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("CRC32C=PASS", dec.stdout)
                self.assertIn(f"DEC4 ({B},{y1 - y0},{x1 - x0})", dec.stdout)
                got = np.fromfile(tile_out, dtype="<i2").reshape(B, y1 - y0, x1 - x0)
                np.testing.assert_array_equal(got, cube[:, y0:y1, x0:x1])

                # Single-tile progressive preview (--stop-layer 6) also verifies CRC32C!
                prev_out = td / f"prev_{ty}_{tx}.raw"
                dec_p = subprocess.run(
                    [
                        str(HND3_BIN), "dec", str(hnd4_f), str(prev_out),
                        "--tile", f"{ty},{tx}", "--stop-layer", "6",
                    ],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("CRC32C=PASS", dec_p.stdout)
                self.assertIn("SHA256=N/A (preview)", dec_p.stdout)
                got_p = np.fromfile(prev_out, dtype="<i2").reshape(B, y1 - y0, x1 - x0)
                err = np.abs(got_p.astype(np.int64) - cube[:, y0:y1, x0:x1].astype(np.int64))
                self.assertLess(np.mean(err), 120.0)

            # Out-of-range tile must fail cleanly
            bad = subprocess.run(
                [str(HND3_BIN), "dec", str(hnd4_f), str(td / "bad.raw"), "--tile", "3,0"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(bad.returncode, 0)

    def test_hnd4_per_header_and_per_layer_crc32c_tamper_and_truncation(self):
        rng = np.random.default_rng(123)
        cube = rng.integers(-3000, 3000, size=(16, 48, 48), dtype="<i2")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_in = td / "in.raw"
            hnd4_f = td / "clean.hnd4"
            cube.tofile(raw_in)

            subprocess.run(
                [
                    str(HND3_BIN), "enc", str(raw_in), str(hnd4_f),
                    "--shape", "16,48,48", "--dtype", "i16",
                    "--base-k", "6", "--crc", "--threads", "2",
                ],
                check=True, stdout=subprocess.DEVNULL,
            )
            blob = bytearray(hnd4_f.read_bytes())
            self.assertEqual(blob[:4], b"HND4")

            # Parse HND4 header to locate Tile Index Table + layer offsets
            spec_len = struct.unpack_from("<I", blob, 76)[0]
            hdr4_size = 84
            tidx_off = hdr4_size + spec_len
            tile_off = struct.unpack_from("<Q", blob, tidx_off)[0]
            layer_lens = struct.unpack_from("<16I", blob, tidx_off + 12)
            k6_off = tile_off
            k6_len = layer_lens[6]
            self.assertGreater(k6_len, 10)
            k3_off = tile_off + layer_lens[6] + layer_lens[5] + layer_lens[4]
            k3_len = layer_lens[3]
            self.assertGreater(k3_len, 10)

            # 1) Tamper header byte (inside SHA field, offset 50) & Tile Index -> rejected by header/index CRC32C!
            for label, off in [("hdr", 50), ("tidx", tidx_off + 80)]:
                b1 = bytearray(blob)
                b1[off] ^= 0x01
                p1 = td / f"bad_{label}.hnd4"
                p1.write_bytes(b1)
                r1 = subprocess.run([str(HND3_BIN), "dec", str(p1), str(td / f"o1_{label}.raw")], capture_output=True, text=True)
                self.assertNotEqual(r1.returncode, 0)
                self.assertIn("CRC32C", r1.stderr)

            # 2) Tamper base layer k=6 -> rejected EVEN for --stop-layer 6 preview!
            b2 = bytearray(blob)
            b2[k6_off + k6_len // 2] ^= 0x80
            p2 = td / "bad_k6.hnd4"
            p2.write_bytes(b2)
            r2 = subprocess.run(
                [str(HND3_BIN), "dec", str(p2), str(td / "o2.raw"), "--stop-layer", "6"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(r2.returncode, 0)
            self.assertIn("CRC32C", r2.stderr)

            # 3) Tamper refinement layer k=3 -> --stop-layer 6 SUCCEEDS (CRC32C=PASS!), --stop-layer 3 REJECTS!
            b3 = bytearray(blob)
            b3[k3_off + k3_len // 2] ^= 0x40
            p3 = td / "bad_k3.hnd4"
            p3.write_bytes(b3)
            r3_prev = subprocess.run(
                [str(HND3_BIN), "dec", str(p3), str(td / "o3p.raw"), "--stop-layer", "6"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("CRC32C=PASS", r3_prev.stdout)
            r3_k3 = subprocess.run(
                [str(HND3_BIN), "dec", str(p3), str(td / "o3k3.raw"), "--stop-layer", "3"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(r3_k3.returncode, 0)
            self.assertIn("CRC32C", r3_k3.stderr)

            # 4) Truncate right after k=6 -> --stop-layer 6 SUCCEEDS with CRC32C=PASS, full decode fails cleanly
            p4 = td / "trunc_k6.hnd4"
            p4.write_bytes(blob[: k6_off + k6_len])
            r4_prev = subprocess.run(
                [str(HND3_BIN), "dec", str(p4), str(td / "o4p.raw"), "--stop-layer", "6"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("CRC32C=PASS", r4_prev.stdout)
            r4_full = subprocess.run(
                [str(HND3_BIN), "dec", str(p4), str(td / "o4f.raw")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(r4_full.returncode, 0)

    def test_hnd4_guaranteed_linf_near_lossless_and_extremes(self):
        rng = np.random.default_rng(999)
        # Correlated i16 cube with boundary extremes injected!
        B, H, W = 16, 64, 64
        base = np.sin(np.linspace(0, 6, H))[:, None] * 1000.0 + np.cos(np.linspace(0, 6, W))[None, :] * 1000.0
        cube_i16 = np.zeros((B, H, W), dtype="<i2")
        for b in range(B):
            cube_i16[b] = np.clip(np.round((1.0 + 0.1 * b) * base + rng.normal(0, 40, (H, W))), -32768, 32767).astype("<i2")
        cube_i16[0, 0, 0] = -32768
        cube_i16[0, 0, 1] = 32767
        cube_i16[0, 0, 2] = -32767
        cube_i16[0, 0, 3] = 32766

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_in = td / "i16.raw"
            cube_i16.tofile(raw_in)

            prev_size = None
            for E in (0, 1, 2, 4, 8, 16):
                out_f = td / f"nl_e{E}.hnd4"
                dec_f = td / f"nl_e{E}.raw"
                enc = subprocess.run(
                    [
                        str(HND3_BIN), "enc", str(raw_in), str(out_f),
                        "--shape", f"{B},{H},{W}", "--dtype", "i16",
                        "--base-k", "0", "--tile-size", "32",
                        "--max-error", str(E), "--threads", "4",
                    ],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("VERIFY=PASS", enc.stdout)
                sz = out_f.stat().st_size
                if prev_size is not None:
                    self.assertLess(sz, prev_size, f"Expected E={E} ({sz} B) < previous ({prev_size} B)")
                prev_size = sz

                dec = subprocess.run(
                    [str(HND3_BIN), "dec", str(out_f), str(dec_f), "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("CRC32C=PASS", dec.stdout)
                rec = np.fromfile(dec_f, dtype="<i2").reshape(B, H, W)
                max_err = int(np.max(np.abs(cube_i16.astype(np.int64) - rec.astype(np.int64))))
                self.assertLessEqual(max_err, E, f"L_inf violated for E={E}: got {max_err}")
                if E == 0:
                    self.assertEqual(max_err, 0)

            # Also verify u16 and i32 extremes (0, 65535, INT32_MIN, INT32_MAX) under near-lossless!
            for dt_str, npdt, vals, E in [
                ("u16", "<u2", [0, 1, 65534, 65535], 5),
                ("i32", "<i4", [-2147483648, -2147483647, 2147483646, 2147483647], 100),
            ]:
                c = rng.integers(0, 5000, size=(8, 16, 16), dtype=npdt)
                for i, v in enumerate(vals):
                    c[0, 0, i] = v
                ri = td / f"{dt_str}.raw"
                hf = td / f"{dt_str}.hnd4"
                ro = td / f"{dt_str}_dec.raw"
                c.tofile(ri)
                subprocess.run(
                    [
                        str(HND3_BIN), "enc", str(ri), str(hf),
                        "--shape", "8,16,16", "--dtype", dt_str,
                        "--max-error", str(E), "--threads", "2",
                    ],
                    check=True, stdout=subprocess.DEVNULL,
                )
                subprocess.run([str(HND3_BIN), "dec", str(hf), str(ro)], check=True, stdout=subprocess.DEVNULL)
                rec = np.fromfile(ro, dtype=npdt).reshape(8, 16, 16)
                linf = int(np.max(np.abs(c.astype(np.int64) - rec.astype(np.int64))))
                self.assertLessEqual(linf, E, f"{dt_str} extreme L_inf violated: {linf} > {E}")

    def test_c_abi_roundtrip_and_error_handling(self):
        c_src = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hnd3.h"

int main(void) {
    const uint32_t B = 8, H = 32, W = 32;
    size_t n_samp = (size_t)B * H * W;
    int16_t* cube = (int16_t*)malloc(n_samp * sizeof(int16_t));
    for (uint32_t b = 0; b < B; ++b)
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x)
                cube[(b * H + y) * W + x] = (int16_t)((b + 1) * 10 + y * 3 + x * 2);

    char err[256] = {0};
    uint8_t* enc = NULL;
    size_t enc_len = 0;

    // 1) HND4 tiled + lossless via C ABI
    hnd3_c_encode_opts eopt = {0};
    eopt.d0 = B; eopt.d1 = H; eopt.d2 = W;
    eopt.dtype = 1; // i16
    eopt.levels = 3;
    eopt.spec_axis = 0;
    eopt.base_k = 6;
    eopt.stripes = 4;
    eopt.per_layer_crc = 1;
    eopt.verify = 1;
    eopt.tile_size = 16;
    eopt.max_error = 0;
    eopt.threads = 2;

    int rc = hnd3_encode_buffer((const uint8_t*)cube, n_samp * 2, &eopt,
                                &enc, &enc_len, err, sizeof(err));
    if (rc != 0 || !enc || enc_len < 84 || memcmp(enc, "HND4", 4) != 0) {
        fprintf(stderr, "C ABI encode failed rc=%d len=%zu err=%s\n", rc, enc_len, err);
        return 1;
    }

    hnd3_c_decode_opts dopt = {0};
    dopt.stop_layer = 0;
    dopt.max_level = -1;
    dopt.tile_y = -1;
    dopt.tile_x = -1;
    dopt.verify_sha = 1;
    dopt.verify_crc = 1;
    dopt.threads = 2;

    uint8_t* dec = NULL;
    size_t dec_len = 0;
    uint32_t od0 = 0, od1 = 0, od2 = 0;
    uint8_t odt = 255;
    rc = hnd3_decode_buffer(enc, enc_len, &dopt, &dec, &dec_len,
                            &od0, &od1, &od2, &odt, err, sizeof(err));
    if (rc != 0 || dec_len != n_samp * 2 || od0 != B || od1 != H || od2 != W || odt != 1) {
        fprintf(stderr, "C ABI decode failed rc=%d len=%zu err=%s\n", rc, dec_len, err);
        return 2;
    }
    if (memcmp(cube, dec, n_samp * 2) != 0) {
        fprintf(stderr, "C ABI lossless mismatch\n");
        return 3;
    }
    hnd3_free(dec);

    // 2) Corrupt 1 byte -> must fail cleanly with non-zero rc and populated err
    enc[enc_len / 2] ^= 0xFF;
    dec = NULL; dec_len = 0; err[0] = '\0';
    rc = hnd3_decode_buffer(enc, enc_len, &dopt, &dec, &dec_len,
                            &od0, &od1, &od2, &odt, err, sizeof(err));
    if (rc == 0 || err[0] == '\0') {
        fprintf(stderr, "C ABI failed to reject corrupted HND4 buffer\n");
        return 4;
    }
    hnd3_free(enc);

    // 3) Near-lossless E=3 via C ABI
    eopt.base_k = 0;
    eopt.tile_size = 0;
    eopt.max_error = 3;
    enc = NULL; enc_len = 0;
    rc = hnd3_encode_buffer((const uint8_t*)cube, n_samp * 2, &eopt,
                            &enc, &enc_len, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "C ABI near-lossless encode failed rc=%d err=%s\n", rc, err);
        return 5;
    }
    dec = NULL; dec_len = 0;
    rc = hnd3_decode_buffer(enc, enc_len, &dopt, &dec, &dec_len,
                            &od0, &od1, &od2, &odt, err, sizeof(err));
    if (rc != 0 || dec_len != n_samp * 2) {
        fprintf(stderr, "C ABI near-lossless decode failed rc=%d err=%s\n", rc, err);
        return 6;
    }
    const int16_t* rec = (const int16_t*)dec;
    for (size_t i = 0; i < n_samp; ++i) {
        int diff = (int)cube[i] - (int)rec[i];
        if (diff < -3 || diff > 3) {
            fprintf(stderr, "C ABI L_inf violation at %zu: %d\n", i, diff);
            return 7;
        }
    }
    hnd3_free(dec);
    hnd3_free(enc);
    free(cube);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "test_c_abi.c"
            exe = td / "test_c_abi"
            src.write_text(c_src)
            # Compile strictly as C99 (`cc -std=c99`) linking C++ static libhnd3.a!
            subprocess.run(
                ["cc", "-std=c99", "-O2", "-Wall", "-Wextra", f"-I{ROOT / 'include'}",
                 str(src), str(LIB_A), "-lstdc++", "-lm", "-pthread", "-o", str(exe)],
                check=True,
            )
            subprocess.run([str(exe)], check=True)

    def test_real_indian_pines_hnd4_tiled_and_near_lossless(self):
        mat_path = DATA_DIR / "Indian_pines.mat"
        if not mat_path.exists():
            self.skipTest("Indian_pines.mat not present")
        import scipy.io
        m = scipy.io.loadmat(mat_path)
        arr = next(v for k, v in m.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3)
        cube = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)).astype("<i2"))  # (200, 145, 145)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_in = td / "ip.raw"
            cube.tofile(raw_in)

            # Tiled lossless (--tile-size 64 -> 3x3 grid including ragged 17-px edges)
            h4 = td / "ip_t64.hnd4"
            dec = td / "ip_t64.raw"
            subprocess.run(
                [
                    str(HND3_BIN), "enc", str(raw_in), str(h4),
                    "--shape", "200,145,145", "--dtype", "i16",
                    "--base-k", "6", "--tile-size", "64", "--threads", "4",
                ],
                check=True, stdout=subprocess.DEVNULL,
            )
            subprocess.run([str(HND3_BIN), "dec", str(h4), str(dec), "--threads", "4"], check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(raw_in.read_bytes(), dec.read_bytes())

            # Random-access ragged corner tile (2, 2) -> (200, 17, 17)
            t22 = td / "ip_t22.raw"
            subprocess.run(
                [str(HND3_BIN), "dec", str(h4), str(t22), "--tile", "2,2"],
                check=True, stdout=subprocess.DEVNULL,
            )
            got22 = np.fromfile(t22, dtype="<i2").reshape(200, 17, 17)
            np.testing.assert_array_equal(got22, cube[:, 128:145, 128:145])

            # Guaranteed L_inf E=2 near-lossless
            h4_e2 = td / "ip_e2.hnd4"
            dec_e2 = td / "ip_e2.raw"
            subprocess.run(
                [
                    str(HND3_BIN), "enc", str(raw_in), str(h4_e2),
                    "--shape", "200,145,145", "--dtype", "i16",
                    "--base-k", "0", "--max-error", "2", "--threads", "4",
                ],
                check=True, stdout=subprocess.DEVNULL,
            )
            subprocess.run([str(HND3_BIN), "dec", str(h4_e2), str(dec_e2), "--threads", "4"], check=True, stdout=subprocess.DEVNULL)
            rec_e2 = np.fromfile(dec_e2, dtype="<i2").reshape(200, 145, 145)
            self.assertLessEqual(int(np.max(np.abs(cube.astype(np.int64) - rec_e2.astype(np.int64)))), 2)
            # E=2 must be substantially smaller than lossless
            self.assertLess(h4_e2.stat().st_size, int(h4.stat().st_size * 0.72))


if __name__ == "__main__":
    unittest.main()
