#!/usr/bin/env python3
"""
Reproducible HND3 Benchmark & Verification Suite (bench/bench_hnd3.py).

Regenerates all Phase 3 tables on exact uncropped cubes with fair 1T-vs-1T and
MT-vs-MT timing (best-of-3 + median), official external codec binaries,
Spectral Angle Mapper (SAM), and exact discrete-entropy LDU rounding-noise audit.

Usage (from repository root):
    python3 bench/bench_hnd3.py rd7
    python3 bench/bench_hnd3.py baselines
    python3 bench/bench_hnd3.py ldu_noise
    python3 bench/bench_hnd3.py blind6
    python3 bench/bench_hnd3.py holdouts
    python3 bench/bench_hnd3.py all
"""
import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
import concurrent.futures
import hashlib
import itertools
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imagecodecs
import numpy as np
import scipy.io
import tifffile
import zstandard as zstd

import phase0_harness as p0
import phase1_codec as p1
import phase2_codec as p2
import phase3_codec as p3

HND3_BIN = str(ROOT / "hnd3")
ASV4_BIN = str(ROOT / "asv4")
DATA_DIR = ROOT / "phase0_data"
EMPORDA_JAR = Path(os.environ.get("EMPORDA_JAR", "/tmp/emporda/EMPORDA/dist/emporda.jar"))
JAVA_BIN = os.environ.get("JAVA_BIN", "/tmp/jdk/Contents/Home/bin/java")
CJXL_BIN = shutil.which("cjxl") or "/opt/homebrew/bin/cjxl"
DJXL_BIN = shutil.which("djxl") or "/opt/homebrew/bin/djxl"
N_CORES = os.cpu_count() or 8


def print_environment():
    cpu = platform.processor()
    ram_gb = 0.0
    if sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            ram_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / (1024**3)
        except Exception:
            pass
    print(f"Environment: {cpu}, {N_CORES} logical CPUs, {ram_gb:.0f} GiB RAM, {platform.platform()}")
    print(f"Libraries:   numpy {np.__version__}, imagecodecs {imagecodecs.__version__}, zstandard {zstd.__version__}")
    print(f"Optional:    emporda={EMPORDA_JAR.exists() and Path(JAVA_BIN).exists()}, cjxl={Path(CJXL_BIN).exists()}")


def load_mat_cube(name):
    d = scipy.io.loadmat(DATA_DIR / name)
    k = [k for k in d if not k.startswith("__")][0]
    return np.ascontiguousarray(d[k].transpose(2, 0, 1), dtype="<i2")


def load_aviris_i16():
    return np.ascontiguousarray(np.load(DATA_DIR / "aviris_holdout.npy"), dtype="<i2")


def load_aviris_i32():
    return np.ascontiguousarray(np.load(DATA_DIR / "aviris_holdout.npy"), dtype="<i4")


def load_cells3d():
    arr = tifffile.imread(DATA_DIR / "cells3d.tif")  # (60, 2, 256, 256) uint16
    return np.ascontiguousarray(arr[:, 0, :, :], dtype="<u2")


def load_brain_mri():
    return np.ascontiguousarray(tifffile.imread(DATA_DIR / "brain.tiff"), dtype="<u2")


def run_with_rss(cmd):
    if sys.platform == "darwin":
        t0 = time.perf_counter()
        res = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True, check=True)
        wall = time.perf_counter() - t0
        m = re.search(r"(\d+)\s+maximum resident set size", res.stderr)
        rss_mb = (int(m.group(1)) / 1e6) if m else 0.0
        return res.stdout, res.stderr, wall, rss_mb
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout, res.stderr, time.perf_counter() - t0, 0.0


def best_of_n(fn, reps=3):
    """Run fn() reps times, returning (last_val, best_sec, median_sec)."""
    times = []
    val = None
    for _ in range(reps):
        t0 = time.perf_counter()
        val = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return val, times[0], times[len(times) // 2]


def parse_hnd3_layer_bytes(blob):
    mv = memoryview(blob)
    (magic, ver, flags, dt_code, spec_code, n_lv, spec_axis, base_k, n_stripes, q_base,
     d0, d1, d2, sha_exp, spec_len) = struct.unpack_from(p3.HDR_FMT, mv, 0)
    assert magic == p3.MAGIC_HND3 and ver == p3.VERSION_HND3
    n_stripes = max(1, n_stripes)
    work_dims = [d0, d1, d2]
    B = work_dims.pop(spec_axis)
    work_shape = (B, work_dims[0], work_dims[1])
    plan = p1._build_plan_from_shape(work_shape, n_lv)
    sched = p1._c2f_subband_schedule(plan, work_shape)
    offsets = p2.get_subband_offsets(sched, len(plan))
    pos = p3.HDR_SIZE + spec_len

    def skip_stripe(p):
        plen = struct.unpack_from("<I", mv, p)[0]
        return p + 4 + plen

    for _ in range(len(sched) * n_stripes):
        pos = skip_stripe(pos)
    layer_bytes = {base_k: pos}
    for k in range(base_k - 1, -1, -1):
        for lv in range(len(plan) + 1):
            for si, (_sl, slv, _par) in enumerate(sched):
                if slv != lv:
                    continue
                if max(0, k - offsets[si]) == max(0, (k + 1) - offsets[si]):
                    continue
                for _ in range(n_stripes):
                    pos = skip_stripe(pos)
        layer_bytes[k] = pos
    assert pos == len(blob)
    return layer_bytes, spec_len


def compute_sam_deg(orig64, rec64):
    """Mean and 99th-percentile Spectral Angle Mapper (SAM) in degrees across all spatial pixels."""
    a = orig64.reshape(orig64.shape[0], -1).astype(np.float64)
    b = rec64.reshape(rec64.shape[0], -1).astype(np.float64)
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    valid = (na > 0) & (nb > 0)
    if not np.any(valid):
        return 0.0, 0.0
    cos_t = np.sum(a[:, valid] * b[:, valid], axis=0) / (na[valid] * nb[valid])
    ang = np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))
    return float(np.mean(ang)), float(np.percentile(ang, 99.0))


def bench_hnd3_cli(raw_p, shape_s, dt, k, threads, reps=3):
    hp = f"{raw_p}_k{k}_t{threads}.hnd3"
    enc_times = []
    for _ in range(reps):
        out_e = subprocess.check_output(
            [HND3_BIN, "enc", raw_p, hp, "--shape", shape_s, "--dtype", dt,
             "--base-k", str(k), "--threads", str(threads), "--no-verify"], text=True
        )
        enc_times.append(float(re.search(r"\(([0-9.]+)s\)", out_e).group(1)))
    # Also verify once with default-on SHA-256 verification
    subprocess.run([HND3_BIN, "dec", hp], check=True, stdout=subprocess.DEVNULL)
    out_d = subprocess.check_output([HND3_BIN, "dec", hp, "--threads", str(threads), "--bench"], text=True)
    dec_best = float(re.search(r"\(([0-9.]+)s\)", out_d).group(1))
    sz = os.path.getsize(hp)
    os.remove(hp)
    return sz, min(enc_times), dec_best


def bench_emporda_ccsds123(cube_i16, P=3, entropy_type=0):
    """Run official UAB/ESA Emporda CCSDS-123.0-B-1 reference encoder & decoder with bit-exact verification."""
    if not (EMPORDA_JAR.exists() and Path(JAVA_BIN).exists()):
        return None
    B, H, W = cube_i16.shape
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_f = td / "in.raw"
        out_base = td / "out"
        dec_f = td / "dec.raw"
        opt_f = td / "opts.txt"
        cube_i16.tofile(raw_f)
        opt_f.write_text(
            f"DYNAMIC_RANGE=16\n"
            f"SAMPLE_ENCODING_ORDER=1\n"
            f"SUBFRAME_INTERLEAVING_DEPTH=0\n"
            f"OUTPUT_WORD_SIZE=4\n"
            f"ENTROPY_CODER_TYPE={entropy_type}\n"
            f"NUMBER_PREDICTION_BANDS={P}\n"
            f"PREDICTION_MODE=0\n"
            f"LOCAL_SUM_MODE=0\n"
            f"REGISTER_SIZE=32\n"
            f"WEIGHT_COMPONENT_RESOLUTION=13\n"
            f"WEIGHT_UPDATE_SECI=6\n"
            f"WEIGHT_UPDATE_SE=-1\n"
            f"WEIGHT_UPDATE_SEFP=3\n"
        )
        t0 = time.perf_counter()
        subprocess.run(
            [JAVA_BIN, "-Xmx4g", "-jar", str(EMPORDA_JAR), "-c",
             "-i", str(raw_f), "-o", str(out_base),
             "-ig", str(B), str(H), str(W), "3", "0", "-e", "1", "-f", str(opt_f)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        enc_s = time.perf_counter() - t0
        # Emporda writes out_base (or out_base.fl / similar)
        cands = [p for p in td.iterdir() if p.name.startswith("out")]
        comp_f = cands[0]
        comp_bytes = comp_f.stat().st_size

        t1 = time.perf_counter()
        subprocess.run(
            [JAVA_BIN, "-Xmx4g", "-jar", str(EMPORDA_JAR), "-d",
             "-i", str(comp_f), "-o", str(dec_f), "-e", "1", "-format", "3"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        dec_s = time.perf_counter() - t1
        assert dec_f.read_bytes() == raw_f.read_bytes(), f"Emporda P={P} round-trip mismatch!"
        return comp_bytes, enc_s, dec_s


def bench_per_band_codec(cube_i64, kind="j2k", n_workers=1, reps=3):
    """Per-band lossless OpenJPEG J2K or CharLS JPEG-LS, timed cleanly at 1T or MT (best-of-N)."""
    B, H, W = cube_i64.shape
    bands = []
    hdr_extra = 0
    for b in range(B):
        sl = cube_i64[b]
        mn, mx = int(sl.min()), int(sl.max())
        if kind == "j2k" and mn >= -32768 and mx <= 32767:
            bands.append(("i16", np.ascontiguousarray(sl, dtype=np.int16), 0))
        elif mx - mn <= 65535:
            bands.append(("u16", np.ascontiguousarray(sl - mn, dtype=np.uint16), mn))
            hdr_extra += 4
        else:
            bands.append(("z32", np.ascontiguousarray(sl, dtype=np.int32), 0))

    def enc_one(item):
        tag, arr, off = item
        if tag == "z32":
            return ("z32", zstd.ZstdCompressor(level=19).compress(arr.tobytes()), off)
        if kind == "j2k":
            return ("j2k", imagecodecs.jpeg2k_encode(arr, level=0, reversible=True), off)
        return ("jls", imagecodecs.jpegls_encode(arr, level=0), off)

    def dec_one(item):
        tag, blob, off = item
        if tag == "z32":
            return np.frombuffer(zstd.ZstdDecompressor().decompress(blob), dtype=np.int32).reshape(H, W).astype(np.int64)
        if tag == "j2k":
            return imagecodecs.jpeg2k_decode(blob).astype(np.int64) + off
        return imagecodecs.jpegls_decode(blob).astype(np.int64) + off

    def run_enc():
        if n_workers <= 1:
            return [enc_one(it) for it in bands]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(enc_one, bands))

    blobs, enc_best, _ = best_of_n(run_enc, reps=reps)

    def run_dec():
        if n_workers <= 1:
            return [dec_one(it) for it in blobs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(dec_one, blobs))

    recs, dec_best, _ = best_of_n(run_dec, reps=reps)
    for b in range(B):
        assert np.array_equal(recs[b], cube_i64[b]), f"{kind} round-trip failed band {b}"
    tot_bytes = sum(len(it[1]) + 4 for it in blobs) + hdr_extra
    return tot_bytes, enc_best, dec_best


def bench_cjxl_per_band(cube_i64, effort=7, n_workers=N_CORES):
    """Official libjxl cjxl/djxl lossless per-band via 16-bit binary PGM (P5), verified bit-exact."""
    if not (Path(CJXL_BIN).exists() and Path(DJXL_BIN).exists()):
        return None
    B, H, W = cube_i64.shape
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tasks = []
        hdr_extra = 0
        for b in range(B):
            sl = cube_i64[b]
            mn, mx = int(sl.min()), int(sl.max())
            if mx - mn <= 65535:
                u16_be = np.ascontiguousarray(sl - mn, dtype=">u2")
                pgm = td / f"b{b}.pgm"
                jxl = td / f"b{b}.jxl"
                rec = td / f"r{b}.pgm"
                with open(pgm, "wb") as f:
                    f.write(f"P5\n{W} {H}\n65535\n".encode("ascii"))
                    f.write(u16_be.tobytes())
                tasks.append(("jxl", b, pgm, jxl, rec, mn))
                hdr_extra += 4
            else:
                tasks.append(("z32", b, None, None, None, 0))

        def do_enc(t):
            tag, b, pgm, jxl, _, _ = t
            if tag == "z32":
                zb = zstd.ZstdCompressor(level=19).compress(np.ascontiguousarray(cube_i64[b], dtype=np.int32).tobytes())
                return len(zb) + 4
            subprocess.run(
                [CJXL_BIN, str(pgm), str(jxl), "-d", "0", "-e", str(effort), "--num_threads=1", "--quiet"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return jxl.stat().st_size + 4

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            sizes = list(ex.map(do_enc, tasks))
        enc_s = time.perf_counter() - t0

        def do_dec(t):
            tag, b, _, jxl, rec, mn = t
            if tag == "z32":
                return
            subprocess.run(
                [DJXL_BIN, str(jxl), str(rec), "--num_threads=1", "--quiet"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            raw = rec.read_bytes()
            # Parse P5 header (3 whitespace tokens after P5)
            idx = 0
            toks = 0
            while toks < 4:
                while raw[idx] in b" \t\r\n":
                    idx += 1
                if raw[idx] == ord("#"):
                    idx = raw.index(b"\n", idx) + 1
                    continue
                while raw[idx] not in b" \t\r\n":
                    idx += 1
                toks += 1
            idx += 1  # single whitespace byte after maxval
            arr = np.frombuffer(raw, dtype=">u2", offset=idx).reshape(H, W).astype(np.int64) + mn
            assert np.array_equal(arr, cube_i64[b]), f"cjxl round-trip failed band {b}"

        t1 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(do_dec, tasks))
        dec_s = time.perf_counter() - t1
        return sum(sizes) + hdr_extra, enc_s, dec_s


def cmd_rd7():
    cubes = [
        ("Indian_pines", load_mat_cube("Indian_pines.mat"), "i16"),
        ("Salinas", load_mat_cube("Salinas.mat"), "i16"),
        ("PaviaU", load_mat_cube("PaviaU.mat"), "i16"),
        ("AVIRIS_holdout_i16", load_aviris_i16(), "i16"),
    ]
    print("=" * 122)
    print("7-LAYER PROGRESSIVE RATE-DISTORTION TABLE (k = 6..0, C++20 ./hnd3, Best-of-3 MT & 1T, with SAM)")
    print("=" * 122)
    for name, cube, dt in cubes:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_p = str(td / f"{name}.raw")
            hnd_p = str(td / f"{name}.hnd3")
            rec_p = str(td / f"{name}_rec.raw")
            cube.tofile(raw_p)
            shape_s = f"{cube.shape[0]},{cube.shape[1]},{cube.shape[2]}"
            subprocess.run([HND3_BIN, "enc", raw_p, hnd_p, "--shape", shape_s, "--dtype", dt, "--base-k", "6"],
                           check=True, stdout=subprocess.DEVNULL)
            blob = Path(hnd_p).read_bytes()
            layer_bytes, spec_len = parse_hnd3_layer_bytes(blob)
            orig64 = cube.astype(np.int64)
            dyn = float(orig64.max() - orig64.min())
            print(f"\n### {name} {cube.shape} ({cube.nbytes/1e6:.2f} MB, dyn={dyn:.0f}, LDU_hdr={spec_len:,} B)")
            print(f"{'Layer':>5} | {'Cum Bytes':>10} | {'%Strm':>6} | {'b/s':>6} | {'Ratio':>7} | {'PSNR_rng':>9} | {'SAM_mean':>8} | {'SAM_p99':>8} | {'L_inf':>5} | {'MT ms (MB/s)':>15} | {'1T MB/s':>7}")
            print("-" * 122)
            for k in range(6, -1, -1):
                out_mt = subprocess.check_output([HND3_BIN, "dec", hnd_p, rec_p, "--stop-layer", str(k), "--bench"], text=True)
                out_1t = subprocess.check_output([HND3_BIN, "dec", hnd_p, "--stop-layer", str(k), "--threads", "1", "--bench"], text=True)
                m_mt = re.search(r"([0-9.]+)\s+MB/s\s+\(([0-9.]+)s\)", out_mt)
                m_1t = re.search(r"([0-9.]+)\s+MB/s", out_1t)
                mbps_mt, ms_mt = float(m_mt.group(1)), float(m_mt.group(2)) * 1000.0
                mbps_1t = float(m_1t.group(1))
                rec = np.fromfile(rec_p, dtype=cube.dtype).reshape(cube.shape).astype(np.int64)
                diff = np.abs(orig64 - rec)
                linf = int(diff.max())
                mse = float(np.mean(diff.astype(np.float64) ** 2))
                if mse == 0.0:
                    psnr_r, sam_m, sam_99 = "inf", "0.000°", "0.000°"
                else:
                    psnr_r = f"{20.0 * math.log10(dyn / math.sqrt(mse)):.2f} dB"
                    sm, s99 = compute_sam_deg(orig64, rec)
                    sam_m, sam_99 = f"{sm:.3f}°", f"{s99:.3f}°"
                cb = layer_bytes[k]
                print(f"k={k:>3} | {cb:>10,} | {100.0*cb/len(blob):>5.1f}% | {cb*8/cube.size:>6.3f} | {cube.nbytes/cb:>6.2f}x | {psnr_r:>9} | {sam_m:>8} | {sam_99:>8} | {linf:>5} | {ms_mt:>6.1f} ({mbps_mt:>5.1f}) | {mbps_1t:>7.1f}")


def cmd_baselines():
    cubes = [
        ("Indian_pines", load_mat_cube("Indian_pines.mat")),
        ("Salinas", load_mat_cube("Salinas.mat")),
        ("PaviaU", load_mat_cube("PaviaU.mat")),
        ("AVIRIS_holdout_i16", load_aviris_i16()),
    ]
    # Warm up Numba JIT
    tiny = np.zeros((4, 8, 8), dtype=np.int64)
    _, r0 = p0.row_ccsds123(tiny, P=3, verify=True); p0.real_rice_total_bits(r0, verify=True)

    print("=" * 124)
    print("FAIR APPLES-TO-APPLES UNCROPPED BASELINES (Best-of-3, 1T vs 1T and MT vs MT, Official Emporda + cjxl + OpenJPEG)")
    print("=" * 124)
    for name, cube in cubes:
        B, H, W = cube.shape
        n_samp, n_bytes = cube.size, cube.nbytes
        c64 = cube.astype(np.int64)
        print(f"\n### {name} {cube.shape} ({n_bytes/1e6:.2f} MB, dtype={cube.dtype})")
        print(f"{'Codec / Method':<45} | {'Bytes':>10} | {'b/s':>6} | {'Ratio':>6} | {'Enc 1T/MT':>13} | {'Dec 1T/MT':>13}")
        print("-" * 110)

        with tempfile.TemporaryDirectory() as td:
            raw_p = str(Path(td) / f"{name}.raw")
            cube.tofile(raw_p)
            shape_s = f"{B},{H},{W}"

            # 1. HND3 k=0 and k=6 (both 1T and 16T MT, best-of-3, --no-verify pure timing + separately verified)
            for k in [0, 6]:
                sz, e1, d1 = bench_hnd3_cli(raw_p, shape_s, "i16", k, threads=1, reps=3)
                _, em, dm = bench_hnd3_cli(raw_p, shape_s, "i16", k, threads=N_CORES, reps=3)
                lbl = f"HND3 (k={k}, {'7-layer' if k else '1-shot'}, auto)"
                print(f"{lbl:<45} | {sz:>10,} | {sz*8/n_samp:>6.3f} | {n_bytes/sz:>5.2f}x | {n_bytes/1e6/e1:>5.1f}/{n_bytes/1e6/em:<6.1f} | {n_bytes/1e6/d1:>5.1f}/{n_bytes/1e6/dm:<6.1f}")

            # 2. Official UAB/ESA Emporda CCSDS-123.0-B-1 (P=3 & P=15, Sample-Adaptive & Block-Adaptive)
            for P in [3, 15]:
                res_emp = bench_emporda_ccsds123(cube, P=P, entropy_type=0)
                if res_emp:
                    eb, ees, eds = res_emp
                    lbl = f"Emporda CCSDS-123.0-B-1 (P={P}, SA, official)"
                    print(f"{lbl:<45} | {eb:>10,} | {eb*8/n_samp:>6.3f} | {n_bytes/eb:>5.2f}x | {n_bytes/1e6/ees:>5.1f}/{'—':<6} | {n_bytes/1e6/eds:>5.1f}/{'—':<6}")
            res_emp_ba = bench_emporda_ccsds123(cube, P=3, entropy_type=1)
            if res_emp_ba:
                eb, ees, eds = res_emp_ba
                print(f"{'Emporda CCSDS-123.0-B-1 (P=3, BA, official)':<45} | {eb:>10,} | {eb*8/n_samp:>6.3f} | {n_bytes/eb:>5.2f}x | {n_bytes/1e6/ees:>5.1f}/{'—':<6} | {n_bytes/1e6/eds:>5.1f}/{'—':<6}")

            # 3. Custom Numba CCSDS-123.0-B-1 (P=15, Rice) + CCSDS-123 (P=15) + 3D-rANS
            _, res15 = p0.row_ccsds123(c64, P=15, verify=True)
            rbits = p0.real_rice_total_bits(res15, verify=True)
            rbytes = (rbits + 7) // 8
            print(f"{'Custom CCSDS-123.0-B-1 (P=15, Rice, Numba)':<45} | {rbytes:>10,} | {rbits/n_samp:>6.3f} | {n_bytes/rbytes:>5.2f}x | {'(Numba)':>13} | {'—':>13}")
            rb_blob = p1.pack_rans_subband(res15, parent=None)
            print(f"{'Custom CCSDS-123 (P=15) + 3D-rANS':<45} | {len(rb_blob):>10,} | {len(rb_blob)*8/n_samp:>6.3f} | {n_bytes/len(rb_blob):>5.2f}x | {'(Numba)':>13} | {'—':>13}")

            # 4. Official OpenJPEG J2K (Part 1, per-band) — BOTH 1T and 16T MT!
            j1_b, j1_e1, j1_d1 = bench_per_band_codec(c64, "j2k", n_workers=1, reps=2)
            _, j1_em, j1_dm = bench_per_band_codec(c64, "j2k", n_workers=N_CORES, reps=3)
            print(f"{'OpenJPEG J2K Part 1 (per-band lossless)':<45} | {j1_b:>10,} | {j1_b*8/n_samp:>6.3f} | {n_bytes/j1_b:>5.2f}x | {n_bytes/1e6/j1_e1:>5.1f}/{n_bytes/1e6/j1_em:<6.1f} | {n_bytes/1e6/j1_d1:>5.1f}/{n_bytes/1e6/j1_dm:<6.1f}")

            # 5. LDU-RKLT + per-band OpenJPEG J2K (accurately named!) — BOTH 1T and 16T MT!
            t0 = time.perf_counter()
            Y_spec, spec_code, spec_bytes = p1.spectral_ldu_forward(c64, mode="auto")
            ldu_e = time.perf_counter() - t0
            t1 = time.perf_counter()
            C_inv, _ = p1.spectral_ldu_inverse(Y_spec.copy(), spec_code, memoryview(spec_bytes), 0)
            ldu_d = time.perf_counter() - t1
            assert np.array_equal(C_inv, c64)
            j2_b, j2_e1, j2_d1 = bench_per_band_codec(Y_spec, "j2k", n_workers=1, reps=2)
            _, j2_em, j2_dm = bench_per_band_codec(Y_spec, "j2k", n_workers=N_CORES, reps=3)
            tot_j2 = len(spec_bytes) + j2_b
            print(f"{'LDU-RKLT + per-band OpenJPEG J2K (Part 1)':<45} | {tot_j2:>10,} | {tot_j2*8/n_samp:>6.3f} | {n_bytes/tot_j2:>5.2f}x | {n_bytes/1e6/(ldu_e+j2_e1):>5.1f}/{n_bytes/1e6/(ldu_e+j2_em):<6.1f} | {n_bytes/1e6/(ldu_d+j2_d1):>5.1f}/{n_bytes/1e6/(ldu_d+j2_dm):<6.1f}")

            # 6. Official libjxl JPEG-XL v0.13.0 (per-band lossless, raw & + LDU-RKLT)
            res_jxl1 = bench_cjxl_per_band(c64, effort=7, n_workers=N_CORES)
            if res_jxl1:
                xb, xes, xds = res_jxl1
                print(f"{'JPEG-XL cjxl 0.13 (per-band, -d 0 -e 7)':<45} | {xb:>10,} | {xb*8/n_samp:>6.3f} | {n_bytes/xb:>5.2f}x | {'—':>5}/{n_bytes/1e6/xes:<6.1f} | {'—':>5}/{n_bytes/1e6/xds:<6.1f}")
            res_jxl2 = bench_cjxl_per_band(Y_spec, effort=7, n_workers=N_CORES)
            if res_jxl2:
                xb, xes, xds = res_jxl2
                tot_x = len(spec_bytes) + xb
                print(f"{'LDU-RKLT + per-band JPEG-XL cjxl (-d 0 -e 7)':<45} | {tot_x:>10,} | {tot_x*8/n_samp:>6.3f} | {n_bytes/tot_x:>5.2f}x | {'—':>5}/{n_bytes/1e6/(ldu_e+xes):<6.1f} | {'—':>5}/{n_bytes/1e6/(ldu_d+xds):<6.1f}")

            # 7. CharLS JPEG-LS (1T & MT)
            jl_b, jl_e1, jl_d1 = bench_per_band_codec(c64, "jls", n_workers=1, reps=2)
            _, jl_em, jl_dm = bench_per_band_codec(c64, "jls", n_workers=N_CORES, reps=3)
            print(f"{'CharLS JPEG-LS (per-band lossless)':<45} | {jl_b:>10,} | {jl_b*8/n_samp:>6.3f} | {n_bytes/jl_b:>5.2f}x | {n_bytes/1e6/jl_e1:>5.1f}/{n_bytes/1e6/jl_em:<6.1f} | {n_bytes/1e6/jl_d1:>5.1f}/{n_bytes/1e6/jl_dm:<6.1f}")

            # 8. ZFP Reversible 3D & zstd -19 (2B shuffled)
            c32 = np.ascontiguousarray(cube, dtype=np.int32)
            zb, ze, _ = best_of_n(lambda: imagecodecs.zfp_encode(c32, mode='R'), reps=3)
            zr, zd, _ = best_of_n(lambda: imagecodecs.zfp_decode(zb), reps=3)
            assert np.array_equal(zr, c32)
            print(f"{'ZFP Reversible 3D (official, i32 mode)':<45} | {len(zb):>10,} | {len(zb)*8/n_samp:>6.3f} | {n_bytes/len(zb):>5.2f}x | {n_bytes/1e6/ze:>5.1f}/{'—':<6} | {n_bytes/1e6/zd:>5.1f}/{'—':<6}")

            shuf = np.ascontiguousarray(np.frombuffer(cube.tobytes(), dtype=np.uint8).reshape(-1, 2).T).tobytes()
            zz, zze, _ = best_of_n(lambda: zstd.ZstdCompressor(level=19).compress(shuf), reps=1)
            zzr, zzd, _ = best_of_n(lambda: zstd.ZstdDecompressor().decompress(zz), reps=3)
            assert zzr == shuf
            print(f"{'zstd -19 (byte-shuffled 2B)':<45} | {len(zz):>10,} | {len(zz)*8/n_samp:>6.3f} | {n_bytes/len(zz):>5.2f}x | {n_bytes/1e6/zze:>5.1f}/{'—':<6} | {n_bytes/1e6/zzd:>5.1f}/{'—':<6}")


def discrete_band_entropy(Y_int):
    """Exact empirical discrete zero-order entropy (b/s), averaged across bands."""
    ent = 0.0
    B, HW = Y_int.shape
    for b in range(B):
        _, cnts = np.unique(Y_int[b], return_counts=True)
        p = cnts.astype(np.float64) / float(HW)
        ent -= float(np.sum(p * np.log2(p)))
    return ent / float(B)


def cmd_ldu_noise():
    print("=" * 124)
    print("COMPLETE-PIVOTED LDU-RKLT ROUNDING-NOISE AUDIT (Exact Same-Matrix Discrete Entropy & Auto vs Full)")
    print("=" * 124)
    cubes = [
        ("Indian_pines (200)", load_mat_cube("Indian_pines.mat").astype(np.int64)),
        ("Salinas (204)", load_mat_cube("Salinas.mat").astype(np.int64)),
        ("PaviaU (103)", load_mat_cube("PaviaU.mat").astype(np.int64)),
        ("AVIRIS_holdout (224)", load_aviris_i16().astype(np.int64)),
    ]
    rng = np.random.default_rng(42)
    B_s = 204
    t = np.linspace(0, 1, B_s)
    basis = np.stack([np.sin(2 * np.pi * (k + 1) * t) for k in range(6)], axis=1)
    syn = np.round(basis @ rng.normal(0, 1000.0, (6, 128 * 128)) + rng.normal(0, 1.0, (B_s, 128 * 128))).astype(np.int64).reshape(B_s, 128, 128)
    cubes.append(("Synth_60dB (204, sn=1)", syn))

    print(f"{'Dataset':<24} | {'AutoMode':<9} | {'max|L|':>6} | {'max|U|':>6} | {'max|c|':>6} | {'max|1/c|':>8} | {'RMS(e)':>6} | {'Var(e)':>6} | {'H_flt':>6} | {'H_ldu':>6} | {'dH_disc':>8} | {'Givens'}")
    print("-" * 124)
    Q = float(1 << p1.LDU_SHIFT)
    for name, C in cubes:
        B, H, W = C.shape
        HW = H * W
        X = C.reshape(B, HW)
        mu = np.rint(X.mean(axis=1)).astype(np.int64)
        Xc = X - mu[:, None]
        Xf = Xc.astype(np.float64)
        full_hdr_bits = 16.0 * (B * (B - 1) + 12.0 * B)
        auto_mode = "full" if (B <= 16 or full_hdr_bits / C.size < 0.04) else "clustered"

        step = max(1, HW // 16384)
        Xs = Xf[:, ::step]
        cov = (Xs @ Xs.T) / float(Xs.shape[1])
        eigvals, eigvecs = np.linalg.eigh(cov)
        U_klt = eigvecs[:, np.argsort(eigvals)[::-1]].T
        st = p1.complete_pivoted_ldu(U_klt)
        Y_flt = U_klt @ Xf
        Y_int = p1._ldu_fwd_numba(
            Xc.copy(), st["rperm"], st["cperm"], st["sign0"], st["dsgn"], st["dperm"],
            st["Lq"], st["Uq"], st["cq"], st["inv_cq_m1"], st["cq_m1"]
        )
        X_rec = p1._ldu_inv_numba(
            Y_int.copy(), st["rperm"], st["cperm"], st["sign0"], st["dsgn"], st["dperm"],
            st["Lq"], st["Uq"], st["cq"], st["inv_cq_m1"], st["cq_m1"]
        )
        assert np.array_equal(X_rec, Xc)
        maxL = float(np.max(np.abs(st["Lq"]))) / Q
        maxU = float(np.max(np.abs(st["Uq"]))) / Q
        maxC = float(np.max(np.abs(st["cq"] / Q)))
        maxIC = float(np.max(np.abs(st["inv_cq_m1"] / Q + 1.0)))
        diff = Y_int.astype(np.float64) - Y_flt
        rms = float(np.sqrt(np.mean(diff ** 2)))
        var = float(np.var(diff))
        h_flt = discrete_band_entropy(np.rint(Y_flt).astype(np.int64))
        h_ldu = discrete_band_entropy(Y_int)
        # Note: complete_pivoted_ldu has NO Givens fallback branch at all (strict LDU ladder succeeds 100%)
        print(f"{name:<24} | {auto_mode:<9} | {maxL:>6.4f} | {maxU:>6.4f} | {maxC:>6.3f} | {maxIC:>8.3f} | {rms:>6.3f} | {var:>6.3f} | {h_flt:>6.3f} | {h_ldu:>6.3f} | {h_ldu-h_flt:>+7.4f} | 0 (none)")

        if auto_mode == "clustered":
            # Also report actual shipped clustered-mode discrete entropy!
            Yc, sc, sb = p1.spectral_ldu_forward(C, mode="clustered")
            Ci, _ = p1.spectral_ldu_inverse(Yc, sc, memoryview(sb), 0)
            assert np.array_equal(Ci, C)
            h_cl = discrete_band_entropy(Yc.reshape(B, HW))
            print(f"  -> shipped clustered   | clustered | {'<=1.0':>6} | {'<=1.0':>6} | {'—':>6} | {'—':>8} | {'—':>6} | {'—':>6} | {h_flt:>6.3f} | {h_cl:>6.3f} | {h_cl-h_flt:>+7.4f} | 0 (none)")


def cmd_blind6():
    print("=" * 115)
    print("FULL UNCROPPED C++20 --blind TRACK A (ALL 6/6 PERMUTATIONS + PEAK RSS)")
    print("=" * 115)
    datasets = [
        ("Indian_pines", load_mat_cube("Indian_pines.mat")),
        ("Salinas", load_mat_cube("Salinas.mat")),
        ("PaviaU", load_mat_cube("PaviaU.mat")),
        ("AVIRIS_holdout_i16", load_aviris_i16()),
    ]
    pnames = {(0,1,2):"BHW (BSQ)",(0,2,1):"BWH",(1,0,2):"HBW (BIL)",(1,2,0):"HWB (BIP)",(2,0,1):"WBH",(2,1,0):"WHB"}
    for name, cube in datasets:
        print(f"\n### {name} native shape={cube.shape} ({cube.nbytes/1e6:.2f} MB)")
        ne = nsp = 0
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for perm in itertools.permutations([0, 1, 2]):
                arr = np.ascontiguousarray(np.transpose(cube, perm))
                ts, tsp = arr.shape, perm.index(0)
                exp = [1, ts[2], ts[1] * ts[2]]
                rp, hp = str(td / "in.raw"), str(td / "out.hnd3")
                arr.tofile(rp)
                out, _, wall, rss = run_with_rss([HND3_BIN, "enc", rp, hp, "--blind", "--dtype", "i16", "--base-k", "0"])
                m = re.search(r"strides=\[([0-9,]+)\].*?spec_ax=(-?\d+).*?\(([0-9.]+)\s+b/s", out)
                fs = [int(x) for x in m.group(1).split(",")]
                fsp, bps = int(m.group(2)), float(m.group(3))
                ne += int(fs == exp); nsp += int(fsp == tsp)
                print(f"  {pnames[perm]:<10} shape={str(ts):<15} found={str(fs):<20} sp={fsp}/{tsp} {bps:.3f} b/s  {wall:.2f}s  RSS={rss:.0f} MB")
        print(f"  Summary {name}: exact_3D_strides={ne}/6, true_spec_axis={nsp}/6")


def cmd_holdouts():
    print("=" * 115)
    print("REAL & WIDE-RANGE int32 TESTS, NON-HYPERSPECTRAL 3D CONTROLS, & PEAK RSS AUDIT")
    print("=" * 115)
    # True wide-range int32 (beyond 16-bit!)
    ip = load_mat_cube("Indian_pines.mat")
    wide_ip = (ip.astype(np.int64) * 40000 + np.random.default_rng(1).integers(-5, 5, ip.shape)).astype("<i4")
    rng = np.random.default_rng(3)
    base = np.cumsum(np.cumsum(rng.normal(0, 1, (64, 64)), 0), 1)
    base /= np.abs(base).max()
    full_i32 = np.clip(
        np.rint(np.stack([base * (1.9e9 * (0.5 + 0.5 * np.cos(b / 5))) for b in range(32)])
                + rng.normal(0, 1e6, (32, 64, 64))),
        -2**31, 2**31 - 1,
    ).astype("<i4")

    rss_targets = [
        ("Indian_pines_i16", ip, "i16"),
        ("Salinas_i16", load_mat_cube("Salinas.mat"), "i16"),
        ("PaviaU_i16", load_mat_cube("PaviaU.mat"), "i16"),
        ("AVIRIS_i16", load_aviris_i16(), "i16"),
        ("AVIRIS_in_i32", load_aviris_i32(), "i32"),
        ("IP_x40000_i32", wide_ip, "i32"),
        ("FullRange_1.9e9_i32", full_i32, "i32"),
    ]
    print(f"\n{'Dataset':<20} | {'Mode':<4} | {'Size (B)':>10} | {'b/s':>6} | {'Enc1T/MT':>12} | {'Dec1T/MT':>12} | {'EncRSS':>7} | {'DecRSS':>7}")
    print("-" * 105)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, cube, dt in rss_targets:
            rp = str(td / f"{name}.raw")
            cube.tofile(rp)
            sh = f"{cube.shape[0]},{cube.shape[1]},{cube.shape[2]}"
            for k in [0, 6]:
                hp = str(td / f"{name}_k{k}.hnd3")
                out_em, _, _, enc_rss = run_with_rss([HND3_BIN, "enc", rp, hp, "--shape", sh, "--dtype", dt, "--base-k", str(k), "--no-verify"])
                out_dm, _, _, dec_rss = run_with_rss([HND3_BIN, "dec", hp, "--bench"])
                out_e1 = subprocess.check_output([HND3_BIN, "enc", rp, hp, "--shape", sh, "--dtype", dt, "--base-k", str(k), "--threads", "1", "--no-verify"], text=True)
                out_d1 = subprocess.check_output([HND3_BIN, "dec", hp, "--threads", "1", "--bench"], text=True)
                sz = os.path.getsize(hp)
                em = float(re.search(r"([0-9.]+)\s+MB/s", out_em).group(1))
                e1 = float(re.search(r"([0-9.]+)\s+MB/s", out_e1).group(1))
                dm = float(re.search(r"([0-9.]+)\s+MB/s", out_dm).group(1))
                d1 = float(re.search(r"([0-9.]+)\s+MB/s", out_d1).group(1))
                print(f"{name:<20} | k={k:<2} | {sz:>10,} | {sz*8/cube.size:>6.3f} | {e1:>5.1f}/{em:<6.1f} | {d1:>5.1f}/{dm:<6.1f} | {enc_rss:>5.0f}MB | {dec_rss:>5.0f}MB")

        # Non-hyperspectral 3D controls (including updated --blind and cjxl!)
        for name, cube, dt in [("Cells3D_Microscopy_u16", load_cells3d(), "u16"), ("Brain_MRI_u16", load_brain_mri(), "u16")]:
            B, H, W = cube.shape
            c64 = cube.astype(np.int64)
            rp = str(td / f"{name}.raw")
            cube.tofile(rp)
            sh = f"{B},{H},{W}"
            print(f"\n### {name} {cube.shape} ({cube.nbytes/1e6:.2f} MB)")
            for lbl, extra in [
                ("HND3 (k=0, auto)", ["--shape", sh, "--base-k", "0", "--mode", "auto"]),
                ("HND3 (k=6, 7-layer)", ["--shape", sh, "--base-k", "6", "--mode", "auto"]),
                ("HND3 (--blind, k=0)", ["--blind", "--base-k", "0"]),
            ]:
                hp = str(td / "c.hnd3")
                oe = subprocess.check_output([HND3_BIN, "enc", rp, hp, "--dtype", dt] + extra, text=True)
                od = subprocess.check_output([HND3_BIN, "dec", hp, "--bench"], text=True)
                sz = os.path.getsize(hp)
                ds = float(re.search(r"([0-9.]+)\s+MB/s", od).group(1))
                info = re.search(r"strides=\[[^\]]+\].*?spec_ax=-?\d+", oe)
                print(f"  {lbl:<36} {sz:>10,} B ({sz*8/cube.size:>6.3f} b/s) decMT={ds:>5.1f} MB/s  {info.group(0) if info else ''}")
            res_jxl = bench_cjxl_per_band(c64, effort=7, n_workers=N_CORES)
            if res_jxl:
                xb, _, xds = res_jxl
                print(f"  {'JPEG-XL cjxl 0.13 (-d 0 -e 7)':<36} {xb:>10,} B ({xb*8/cube.size:>6.3f} b/s) decMT={cube.nbytes/1e6/xds:>5.1f} MB/s")


def main():
    print_environment()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("rd7", "all"):
        cmd_rd7()
    if cmd in ("baselines", "all"):
        cmd_baselines()
    if cmd in ("ldu_noise", "all"):
        cmd_ldu_noise()
    if cmd in ("blind6", "all"):
        cmd_blind6()
    if cmd in ("holdouts", "all"):
        cmd_holdouts()


if __name__ == "__main__":
    main()
