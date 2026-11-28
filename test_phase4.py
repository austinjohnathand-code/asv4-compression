import hashlib
import os
import struct
import subprocess
import sys
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
                self.assertIn("HeaderCRC32C=PASS", info.stdout)

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
            # Compile strictly as C99 (`cc -std=c99`) linking C++ static libhnd3.a on both macOS and Linux!
            cpp_rt = "-lc++" if sys.platform == "darwin" else "-lstdc++"
            subprocess.run(
                ["cc", "-std=c99", "-O2", "-Wall", "-Wextra", f"-I{ROOT / 'include'}",
                 str(src), str(LIB_A), cpp_rt, "-lm", "-pthread", "-o", str(exe)],
                check=True,
            )
            subprocess.run([str(exe)], check=True)

    def test_dense_full_range_int32_multiband_and_strict_cli_and_info_crc(self):
        # 1) Dense full-range ±1.9e9 int32 multi-band cube (specifically exercises unsigned __int128
        #    cross-band EMA product in BOTH pack_rans_stripe and unpack_rans_stripe without int64 overflow!)
        rng = np.random.default_rng(2026)
        B, H, W = 12, 32, 32
        base = rng.integers(-1900000000, 1900000000, size=(H, W), dtype="<i4")
        cube = np.stack([
            np.clip(base.astype(np.int64) + rng.integers(-50000000, 50000000, size=(H, W)),
                    -2147483648, 2147483647).astype("<i4")
            for _ in range(B)
        ], axis=0)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ri = td / "dense_i32.raw"
            cube.tofile(ri)
            for tag, extra in [("v3_k0", ["--base-k", "0"]), ("v3_k6", ["--base-k", "6"]),
                               ("v4_t16", ["--base-k", "6", "--tile-size", "16"])]:
                hf = td / f"{tag}.hnd"
                ro = td / f"{tag}.raw"
                subprocess.run(
                    [str(HND3_BIN), "enc", str(ri), str(hf), "--shape", f"{B},{H},{W}",
                     "--dtype", "i32", "--threads", "2"] + extra,
                    check=True, stdout=subprocess.DEVNULL,
                )
                subprocess.run([str(HND3_BIN), "dec", str(hf), str(ro), "--threads", "2"],
                               check=True, stdout=subprocess.DEVNULL)
                self.assertEqual(ri.read_bytes(), ro.read_bytes(), f"Dense int32 mismatch in {tag}")

            # 2) Blind HND4 preserves FLAG_BLIND (0x01) and info reports HeaderCRC32C=PASS / rejects tampered header
            c16 = np.zeros((16, 32, 32), dtype="<i2")
            g = np.linspace(0, 1000, 32, dtype=np.float64)[:, None] + np.linspace(0, 800, 32)[None, :]
            for b in range(16):
                c16[b] = np.round((0.5 + 0.05 * b) * g + rng.normal(0, 4, (32, 32))).astype("<i2")
            r16 = td / "c16.raw"
            h4b = td / "c16_blind.hnd4"
            d16 = td / "c16_dec.raw"
            c16.tofile(r16)
            subprocess.run(
                [str(HND3_BIN), "enc", str(r16), str(h4b), "--blind", "--dtype", "i16",
                 "--tile-size", "16", "--crc", "--threads", "2"],
                check=True, stdout=subprocess.DEVNULL,
            )
            info_ok = subprocess.run([str(HND3_BIN), "info", str(h4b)], capture_output=True, text=True, check=True)
            self.assertIn("HeaderCRC32C=PASS", info_ok.stdout)
            self.assertIn("flags=0x19", info_ok.stdout)  # FLAG_BLIND(0x01)|FLAG_TILED(0x08)|FLAG_CRC32C(0x10)
            subprocess.run([str(HND3_BIN), "dec", str(h4b), str(d16)], check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(r16.read_bytes(), d16.read_bytes())

            # Tampered HND4 header -> hnd3 info must report HeaderCRC32C=FAIL and return non-zero!
            bad = bytearray(h4b.read_bytes())
            bad[50] ^= 0x01
            h4_bad = td / "bad.hnd4"
            h4_bad.write_bytes(bad)
            info_bad = subprocess.run([str(HND3_BIN), "info", str(h4_bad)], capture_output=True, text=True)
            self.assertNotEqual(info_bad.returncode, 0)
            self.assertIn("HeaderCRC32C=FAIL", info_bad.stdout)

            # 3) Strict CLI validation on hnd3 and asv4
            for bad_args in [
                [str(HND3_BIN), "enc", str(r16), str(td / "x.hnd3"), "--shape", "-1,32,32", "--dtype", "i16"],
                [str(HND3_BIN), "enc", str(r16), str(td / "x.hnd3"), "--shape", "16,32,32xyz", "--dtype", "i16"],
                [str(HND3_BIN), "enc", str(r16), str(td / "x.hnd3"), "--shape", "16,32,32", "--dtype", "i16", "--base-k", "abc"],
                [str(HND3_BIN), "dec", str(h4b), str(td / "x.raw"), "--tile", "0,abc"],
            ]:
                r = subprocess.run(bad_args, capture_output=True, text=True)
                self.assertNotEqual(r.returncode, 0, f"Expected CLI error for {bad_args}")
            asv4_bin = ROOT / "asv4"
            if asv4_bin.exists():
                for bad_a in [
                    [str(asv4_bin), "badmode", str(r16), str(td / "a.asv4")],
                    [str(asv4_bin), "c", str(r16), str(td / "a.asv4"), "--block-mb", "0"],
                    [str(asv4_bin), "c", str(r16), str(td / "a.asv4"), "--level", "99"],
                ]:
                    r = subprocess.run(bad_a, capture_output=True, text=True)
                    self.assertNotEqual(r.returncode, 0, f"Expected asv4 CLI error for {bad_a}")

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

    def test_hnd4_sparse_hist_lut_fmt1_fmt2_progressive_blind_and_real_holdouts(self):
        rng = np.random.default_rng(1337)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            # 1) fmt=1 sparse LUT (K <= 4096): smooth k^2 u16 & signed i16 (like Brain_MRI), untiled + tiled + progressive + blind
            B, H, W = 12, 80, 84
            grid = np.linspace(0, 120, H)[:, None] + np.linspace(0, 100, W)[None, :]
            idx_u16 = np.clip(np.round(grid[None, :, :] + np.arange(B)[:, None, None] * 2 + rng.normal(0, 1.5, (B, H, W))), 0, 225).astype(np.int64)
            sq_u16 = (idx_u16 * idx_u16).astype("<u2")
            palette_u16 = set(np.unique(sq_u16).tolist())
            # Signed i16 sparse values spanning negative & positive range: -30000 + 250 * k
            sp_i16 = (-30000 + 250 * idx_u16).astype("<i2")
            palette_i16 = set(np.unique(sp_i16).tolist())

            for name, arr, dt_str, npdt, pal in [
                ("sq_u16", sq_u16, "u16", "<u2", palette_u16),
                ("sp_i16", sp_i16, "i16", "<i2", palette_i16),
            ]:
                ri = td / f"{name}.raw"
                arr.tofile(ri)
                # Auto LUT ON vs OFF comparison + tiled + progressive preview
                h_on = td / f"{name}_on.hnd4"
                h_off = td / f"{name}_off.hnd3"
                h_tiled = td / f"{name}_tiled.hnd4"
                dec_on = td / f"{name}_dec_on.raw"
                dec_t = td / f"{name}_dec_t.raw"
                dec_p = td / f"{name}_prev.raw"

                e_on = subprocess.run(
                    [str(HND3_BIN), "enc", str(ri), str(h_on), "--shape", f"{B},{H},{W}",
                     "--dtype", dt_str, "--base-k", "6", "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("ENC4", e_on.stdout)
                self.assertNotIn("lut=0", e_on.stdout)
                self.assertEqual(h_on.read_bytes()[:4], b"HND4")
                info_on = subprocess.run([str(HND3_BIN), "info", str(h_on)], capture_output=True, text=True, check=True)
                self.assertIn("HeaderCRC32C=PASS", info_on.stdout)
                # FLAG_HIST_LUT = 0x40 must be set
                flags_val = int(info_on.stdout.split("flags=0x")[1].split()[0], 16)
                self.assertTrue(flags_val & 0x40)

                e_off = subprocess.run(
                    [str(HND3_BIN), "enc", str(ri), str(h_off), "--shape", f"{B},{H},{W}",
                     "--dtype", dt_str, "--base-k", "6", "--hist-lut", "off", "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                self.assertEqual(h_off.read_bytes()[:4], b"HND3")
                # LUT must drastically compress sparse-histogram cubes!
                self.assertLess(h_on.stat().st_size, int(h_off.stat().st_size * 0.65))

                d_on = subprocess.run([str(HND3_BIN), "dec", str(h_on), str(dec_on), "--threads", "4"],
                                      capture_output=True, text=True, check=True)
                self.assertIn("SHA256=PASS", d_on.stdout)
                self.assertIn("CRC32C=PASS", d_on.stdout)
                self.assertEqual(ri.read_bytes(), dec_on.read_bytes())

                # Progressive preview (--stop-layer 6) maps cleanly back onto exact LUT palette!
                d_p = subprocess.run([str(HND3_BIN), "dec", str(h_on), str(dec_p), "--stop-layer", "6", "--threads", "4"],
                                     capture_output=True, text=True, check=True)
                self.assertIn("CRC32C=PASS", d_p.stdout)
                prev = np.fromfile(dec_p, dtype=npdt).reshape(B, H, W)
                self.assertTrue(set(np.unique(prev).tolist()).issubset(pal))

                # Tiled streaming encode + multi-column strip decode with LUT active
                subprocess.run(
                    [str(HND3_BIN), "enc", str(ri), str(h_tiled), "--shape", f"{B},{H},{W}",
                     "--dtype", dt_str, "--base-k", "6", "--tile-size", "32", "--threads", "4"],
                    check=True, stdout=subprocess.DEVNULL,
                )
                d_t = subprocess.run([str(HND3_BIN), "dec", str(h_tiled), str(dec_t), "--threads", "4"],
                                     capture_output=True, text=True, check=True)
                self.assertIn("SHA256=PASS", d_t.stdout)
                self.assertEqual(ri.read_bytes(), dec_t.read_bytes())

            # Blind mode auto-activates LUT on sparse sq_u16!
            h_blind = td / "sq_blind.hnd4"
            d_blind = td / "sq_blind.raw"
            eb = subprocess.run(
                [str(HND3_BIN), "enc", str(td / "sq_u16.raw"), str(h_blind), "--blind", "--dtype", "u16", "--threads", "4"],
                capture_output=True, text=True, check=True,
            )
            self.assertNotIn("lut=0", eb.stdout)
            subprocess.run([str(HND3_BIN), "dec", str(h_blind), str(d_blind), "--threads", "4"],
                           check=True, stdout=subprocess.DEVNULL)
            self.assertEqual((td / "sq_u16.raw").read_bytes(), d_blind.read_bytes())

            # 2) fmt=2 bitmap LUT (K > 4096, e.g. K=5000 scattered multiples of 13 in u16)
            B2, H2, W2 = 16, 128, 128  # 262,144 samples -> 8193 B bitmap = 0.25 b/s > 0.15 auto, so also test 524,288 samples!
            B2 = 32  # 524,288 samples -> 8193*8/524288 = 0.125 b/s < 0.15 -> AUTO triggers fmt=2!
            base2 = (np.linspace(0, 2400, H2)[:, None] + np.linspace(0, 2400, W2)[None, :])
            idx2 = np.clip(np.round(base2[None, :, :] + np.arange(B2)[:, None, None] * 6 + rng.normal(0, 10, (B2, H2, W2))), 0, 4999).astype(np.int64)
            # Ensure > 4096 unique multiples of 13 (0, 13, 26, ..., 64987)
            idx2.flat[:5000] = np.arange(5000)
            bm_u16 = (idx2 * 13).astype("<u2")
            self.assertGreater(len(np.unique(bm_u16)), 4096)
            ri2 = td / "bm_u16.raw"
            h2 = td / "bm_u16.hnd4"
            d2 = td / "bm_u16_dec.raw"
            bm_u16.tofile(ri2)
            e2 = subprocess.run(
                [str(HND3_BIN), "enc", str(ri2), str(h2), "--shape", f"{B2},{H2},{W2}",
                 "--dtype", "u16", "--base-k", "0", "--tile-size", "64", "--threads", "4"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("lut=5000", e2.stdout)
            d2r = subprocess.run([str(HND3_BIN), "dec", str(h2), str(d2), "--threads", "4"],
                                 capture_output=True, text=True, check=True)
            self.assertIn("SHA256=PASS", d2r.stdout)
            self.assertEqual(ri2.read_bytes(), d2.read_bytes())

            # 3) Real holdouts: Brain_MRI and Cells3D (if present) must trigger auto-LUT & hit target b/s!
            brain_p = DATA_DIR / "brain.tiff"
            cells_p = DATA_DIR / "cells3d.tif"
            if brain_p.exists() and cells_p.exists():
                import tifffile
                brain = np.ascontiguousarray(tifffile.imread(brain_p).astype("<u2"))
                cells = np.ascontiguousarray(tifffile.imread(cells_p)[:, 0, :, :].astype("<u2"))
                for tag, arr, max_bps in [("brain", brain, 3.0), ("cells", cells, 9.1)]:
                    ri_r = td / f"{tag}.raw"
                    h_r = td / f"{tag}.hnd4"
                    d_r = td / f"{tag}_dec.raw"
                    arr.tofile(ri_r)
                    sh = ",".join(map(str, arr.shape))
                    er = subprocess.run(
                        [str(HND3_BIN), "enc", str(ri_r), str(h_r), "--shape", sh,
                         "--dtype", "u16", "--base-k", "0", "--threads", "4"],
                        capture_output=True, text=True, check=True,
                    )
                    self.assertIn("ENC4", er.stdout)
                    self.assertNotIn("lut=0", er.stdout)
                    bps = h_r.stat().st_size * 8.0 / arr.size
                    self.assertLess(bps, max_bps, f"{tag} got {bps:.3f} b/s >= {max_bps}")
                    dr = subprocess.run([str(HND3_BIN), "dec", str(h_r), str(d_r), "--threads", "4"],
                                        capture_output=True, text=True, check=True)
                    self.assertIn("SHA256=PASS", dr.stdout)
                    self.assertEqual(ri_r.read_bytes(), d_r.read_bytes())


if __name__ == "__main__":
    unittest.main()
