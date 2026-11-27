#!/usr/bin/env python3
"""
Apples-to-Apples HND4 vs Official NTNU CCSDS-123.0-B-2 (Issue 2: Hybrid Entropy
Coder + Near-Lossless) Benchmark on Matched Centered Crops.

Requires the MIT-licensed WHISPERS 2024 CCSDS-123.0-B-2 verification model:
    git clone --depth 1 https://github.com/NTNU-SmallSat-Lab/ccsds123_0-b-2_high_level_model /tmp/ccsds123_b2

Usage:
    python3 bench/bench_ccsds123_b2.py [--crop 64] [--workers 12]
"""
import argparse
import concurrent.futures
import contextlib
import io
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CCSDS_B2_DIR = Path(os.environ.get("CCSDS123_B2_DIR", "/tmp/ccsds123_b2"))
sys.path.insert(0, str(CCSDS_B2_DIR))

from bench.bench_hnd3 import compute_sam_deg, load_aviris_i16, load_mat_cube


def run_ccsds_b2_one(args):
    name, cube, P, E = args
    from ccsds123_i2_hlm import constants as const
    from ccsds123_i2_hlm import header as hd
    from ccsds123_i2_hlm import hybrid_encoder as hyb_enc
    from ccsds123_i2_hlm import predictor as pred

    B, H, W = cube.shape
    h = hd.Header(f"{name}-s16le-{B}x{H}x{W}.raw")
    h.sample_type = hd.SampleType.SIGNED_INTEGER
    h.dynamic_range = 0  # D=16
    h.sample_encoding_order = hd.SampleEncodingOrder.BI
    h.sub_frame_interleaving_depth = 1  # BIL
    h.entropy_coder_type = hd.EntropyCoderType.HYBRID
    h.prediction_bands_num = P
    h.prediction_mode = hd.PredictionMode.FULL
    h.local_sum_type = hd.LocalSumType.WIDE_NEIGHBOR_ORIENTED
    h.register_size = 0  # R=64
    h.weight_component_resolution = 9    # Omega=13 (standard)
    h.weight_update_change_interval = 2  # t_inc=64 (2^6)
    h.weight_update_initial_parameter = 5  # nu_min=-1
    h.weight_update_final_parameter = 9    # nu_max=3

    if E == 0:
        h.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.LOSSLESS
        h.sample_representative_flag = hd.SampleRepresentativeFlag.NOT_INCLUDED
        h.fixed_damping_value = 0
        h.fixed_offset_value = 0
    else:
        h.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY
        h.absolute_error_limit_assignment_method = hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT
        h.absolute_error_limit_bit_depth = 5
        h.absolute_error_limit_value = E
        h.set_absolute_error_limit_table_array_to_default()
        h.sample_representative_flag = hd.SampleRepresentativeFlag.INCLUDED
        h.sample_representative_resolution = 3
        h.fixed_damping_value = 3  # standard phi=3/8
        h.fixed_offset_value = 3   # standard psi=3/8

    h.set_damping_table_array_to_default()
    h.set_damping_offset_table_array_to_default()
    h.check_legal_config()

    ic = const.ImageConstants(h)
    bip = cube.astype(np.int64).transpose(1, 2, 0)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        p = pred.Predictor(h, ic, bip)
        p.run_predictor()
        he = hyb_enc.HybridEncoder(h, ic, p.get_predictor_output())
        he.run_encoder()
    dt = time.perf_counter() - t0

    hdr_bits = h.get_header_bitstreams()[0]
    tot_bits = len(hdr_bits) + len(he.bitstream)
    tot_bytes = (tot_bits + 7) // 8
    rec_bsq = p.clipped_quantizer_bin_center.transpose(2, 0, 1).astype(np.int64).copy()
    orig64 = cube.astype(np.int64)
    # In CCSDS-123.0-B-2, sample t=0 (y=0, x=0) is always coded losslessly (NTNU leaves bin_center[0,0] unpopulated)
    rec_bsq[:, 0, 0] = orig64[:, 0, 0]
    diff = np.abs(orig64 - rec_bsq)
    linf = int(diff.max())
    assert linf <= E, f"CCSDS-123.0-B-2 L_inf violated: {linf} > {E}"
    dyn = float(orig64.max() - orig64.min())
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(dyn / math.sqrt(mse))
    sam_m, _ = (0.0, 0.0) if mse == 0.0 else compute_sam_deg(orig64, rec_bsq)
    return (name, P, E, tot_bytes, tot_bytes * 8.0 / cube.size, psnr, sam_m, linf, dt)


def run_hnd4_one(cube, E):
    B, H, W = cube.shape
    with tempfile.TemporaryDirectory() as td:
        rp = f"{td}/in.raw"
        hp = f"{td}/out.hnd4"
        dp = f"{td}/dec.raw"
        cube.tofile(rp)
        subprocess.run(
            [str(ROOT / "hnd3"), "enc", rp, hp, "--shape", f"{B},{H},{W}", "--dtype", "i16",
             "--base-k", "0", "--max-error", str(E), "--crc", "--threads", "4"],
            check=True, stdout=subprocess.DEVNULL,
        )
        subprocess.run([str(ROOT / "hnd3"), "dec", hp, dp, "--threads", "4"], check=True, stdout=subprocess.DEVNULL)
        sz = os.path.getsize(hp)
        rec = np.fromfile(dp, dtype="<i2").reshape(cube.shape).astype(np.int64)
        orig64 = cube.astype(np.int64)
        diff = np.abs(orig64 - rec)
        linf = int(diff.max())
        assert linf <= E, f"HND4 L_inf violated: {linf} > {E}"
        dyn = float(orig64.max() - orig64.min())
        mse = float(np.mean(diff.astype(np.float64) ** 2))
        psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(dyn / math.sqrt(mse))
        sam_m, _ = (0.0, 0.0) if mse == 0.0 else compute_sam_deg(orig64, rec)
        return sz, sz * 8.0 / cube.size, psnr, sam_m, linf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=64, help="Centered spatial crop size (default 64x64)")
    ap.add_argument("--full-indian-pines", action="store_true", help="Run full uncropped Indian Pines (200, 145, 145)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if not (CCSDS_B2_DIR / "ccsds123_i2_hlm").exists():
        print(f"ERROR: CCSDS-123.0-B-2 verification model not found at {CCSDS_B2_DIR}", file=sys.stderr)
        sys.exit(1)

    C = args.crop
    datasets = {}
    if args.full_indian_pines:
        datasets["Indian_pines_full"] = load_mat_cube("Indian_pines.mat")
    else:
        for name, full in [
            ("Indian_pines", load_mat_cube("Indian_pines.mat")),
            ("Salinas", load_mat_cube("Salinas.mat")),
            ("PaviaU", load_mat_cube("PaviaU.mat")),
            ("AVIRIS_holdout", load_aviris_i16()),
        ]:
            B, H, W = full.shape
            y0, x0 = (H - C) // 2, (W - C) // 2
            datasets[name] = np.ascontiguousarray(full[:, y0:y0 + C, x0:x0 + C], dtype="<i2")

    tasks = []
    for name, cube in datasets.items():
        for E in (0, 1, 2, 4):
            tasks.append((name, cube, 3, E))
        tasks.append((name, cube, 15, 0))

    t_all = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_ccsds_b2_one, tasks))
    scope = "full uncropped (200x145x145)" if args.full_indian_pines else f"centered ({C}x{C}) crops"
    print(f"Completed {len(results)} NTNU CCSDS-123.0-B-2 Hybrid runs on {scope} in {time.perf_counter() - t_all:.1f}s\n")

    res_map = {(r[0], r[1], r[2]): r for r in results}
    print("=" * 124)
    print(f"APPLES-TO-APPLES: HND4 vs OFFICIAL NTNU CCSDS-123.0-B-2 (HYBRID ENTROPY CODER) ON {scope.upper()}")
    print("=" * 124)
    for name, cube in datasets.items():
        print(f"\n### {name} crop {cube.shape} ({cube.nbytes / 1e6:.2f} MB)")
        print(f"{'E':>3} | {'CCSDS-123.0-B-2 (P=3, Hyb)':>28} | {'B2 PSNR / SAM':>18} | {'HND4 (k=0, CRC)':>20} | {'HND4 PSNR / SAM':>18} | {'Delta (HND4-B2)':>15}")
        print("-" * 116)
        for E in (0, 1, 2, 4):
            _, _, _, b2_B, b2_bps, b2_psnr, b2_sam, _, _ = res_map[(name, 3, E)]
            h4_B, h4_bps, h4_psnr, h4_sam, _ = run_hnd4_one(cube, E)
            p_b2 = "inf" if math.isinf(b2_psnr) else f"{b2_psnr:.2f}dB"
            p_h4 = "inf" if math.isinf(h4_psnr) else f"{h4_psnr:.2f}dB"
            print(f"E={E:<1} | {b2_B:>9,} B ({b2_bps:>5.3f} b/s) | {p_b2:>8} / {b2_sam:5.3f}° | {h4_B:>8,} B ({h4_bps:>5.3f} b/s) | {p_h4:>8} / {h4_sam:5.3f}° | {h4_bps - b2_bps:>+7.3f} b/s")
        _, _, _, b2_15_B, b2_15_bps, _, _, _, _ = res_map[(name, 15, 0)]
        print(f"  Note: CCSDS-123.0-B-2 (P=15, Hybrid, E=0 lossless) = {b2_15_B:,} B ({b2_15_bps:.3f} b/s)")


if __name__ == "__main__":
    main()
