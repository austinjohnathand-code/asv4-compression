"""End-to-End Benchmark & Ablation Harness for FTCC (Forest-Tree Shape-Adaptive Compression).

Runs:
  1. Synthetic structured datasets (32 MB each):
       - tabular_telemetry
       - volumetric_3d
       - distant_records
       - mixed_text_logs
       - mixed_binary_entropy
  2. Multi-offset slices (32 MB across 5 offsets + 64 MB scaling slice) from a
     1.05 GB (1,056 MB) on-disk multi-regime file (or any user-supplied --file).

Compares:
  - Baselines: zstd (L3), zstd+LDM (L3, window=128MB), lz4
  - Stage 1+2 Only + zstd finisher (with identity escape hatch)
  - Full Stage 1-4 Pipeline + zstd finisher
  - Full Stage 1-4 Pipeline + lz4 finisher

Measures net compressed size, compression ratio, descriptor/metadata overhead (bytes & %),
residual size after finisher, encode/decode throughput (MB/s), peak memory (MB), and
bit-exact SHA-256 verification on every run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import lz4.frame
import zstandard as zstd

from corpus import build_synthetic_suite, ensure_1gb_corpus, read_file_slice
from ftcc import CodecConfig, ClumpingConfig, decode_buffer, encode_buffer


MB = 1024 * 1024


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _measure(fn: Callable[[], Any]) -> Tuple[Any, float, float]:
    """Run fn() and return (result, elapsed_sec, peak_python_alloc_mb)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, elapsed, peak / MB


def run_baseline_zstd(raw: bytes, level: int = 3, ldm: bool = False) -> Dict[str, Any]:
    raw_sha = sha256_hex(raw)
    if ldm:
        params = zstd.ZstdCompressionParameters.from_level(level, enable_ldm=True, window_log=27)
        comp = zstd.ZstdCompressor(compression_params=params)
    else:
        comp = zstd.ZstdCompressor(level=level)
    decomp = zstd.ZstdDecompressor()

    encoded, enc_s, enc_mem = _measure(lambda: comp.compress(raw))
    decoded, dec_s, dec_mem = _measure(lambda: decomp.decompress(encoded))
    ok = sha256_hex(decoded) == raw_sha
    assert ok, "Baseline zstd roundtrip failed!"

    n = len(raw)
    return {
        "method": "zstd_ldm" if ldm else "zstd",
        "raw_bytes": n,
        "compressed_bytes": len(encoded),
        "ratio": n / len(encoded),
        "overhead_bytes": 0,
        "overhead_pct": 0.0,
        "residual_bytes": len(encoded),
        "enc_mb_s": (n / MB) / max(enc_s, 1e-6),
        "dec_mb_s": (n / MB) / max(dec_s, 1e-6),
        "enc_ms": enc_s * 1000.0,
        "dec_ms": dec_s * 1000.0,
        "peak_mem_mb": max(enc_mem, dec_mem),
        "bit_exact": ok,
    }


def run_baseline_lz4(raw: bytes) -> Dict[str, Any]:
    raw_sha = sha256_hex(raw)
    encoded, enc_s, enc_mem = _measure(lambda: lz4.frame.compress(raw))
    decoded, dec_s, dec_mem = _measure(lambda: lz4.frame.decompress(encoded))
    ok = sha256_hex(decoded) == raw_sha
    assert ok, "Baseline lz4 roundtrip failed!"

    n = len(raw)
    return {
        "method": "lz4",
        "raw_bytes": n,
        "compressed_bytes": len(encoded),
        "ratio": n / len(encoded),
        "overhead_bytes": 0,
        "overhead_pct": 0.0,
        "residual_bytes": len(encoded),
        "enc_mb_s": (n / MB) / max(enc_s, 1e-6),
        "dec_mb_s": (n / MB) / max(dec_s, 1e-6),
        "enc_ms": enc_s * 1000.0,
        "dec_ms": dec_s * 1000.0,
        "peak_mem_mb": max(enc_mem, dec_mem),
        "bit_exact": ok,
    }


def run_ftcc(raw: bytes, method_name: str, cfg: CodecConfig) -> Dict[str, Any]:
    raw_sha = sha256_hex(raw)
    (encoded, st), enc_s, enc_mem = _measure(lambda: encode_buffer(raw, cfg))
    decoded, dec_s, dec_mem = _measure(lambda: decode_buffer(encoded))
    ok = sha256_hex(decoded) == raw_sha
    assert ok, f"FTCC ({method_name}) bit-exact roundtrip FAILED!"

    n = len(raw)
    return {
        "method": method_name,
        "raw_bytes": n,
        "compressed_bytes": st.compressed_bytes,
        "ratio": st.ratio,
        "overhead_bytes": st.total_overhead_bytes,
        "overhead_pct": st.overhead_pct,
        "metadata_raw_bytes": st.metadata_raw_bytes,
        "metadata_comp_bytes": st.metadata_compressed_bytes,
        "template_raw_bytes": st.template_raw_bytes,
        "residual_bytes": st.payload_compressed_bytes,
        "stream_sizes": st.stream_compressed_sizes,
        "num_clumps": st.num_clumps,
        "clump_mean_kb": st.clump_mean_bytes / 1024.0,
        "mode_counts": st.mode_counts,
        "layout_counts": st.layout_counts,
        "escape_reverts": st.escape_hatch_reverts,
        "tree1_templates": st.tree1_templates_used,
        "tree2_edges": st.tree2_edges_used,
        "tree2_max_depth": st.tree2_max_depth,
        "tree3_l0": st.tree3_level0_clumps,
        "tree3_l1": st.tree3_level1_clumps,
        "stage_ms": {
            "s1": round(st.stage1_ms, 1),
            "s2": round(st.stage2_ms, 1),
            "s3": round(st.stage3_ms, 1),
            "s4_esc": round(st.stage4_escape_ms, 1),
            "s4_fin": round(st.stage4_finish_ms, 1),
        },
        "enc_mb_s": (n / MB) / max(enc_s, 1e-6),
        "dec_mb_s": (n / MB) / max(dec_s, 1e-6),
        "enc_ms": enc_s * 1000.0,
        "dec_ms": dec_s * 1000.0,
        "peak_mem_mb": max(enc_mem, dec_mem),
        "bit_exact": ok,
    }


def warmup_jit():
    """Pre-compile Numba JIT kernels on tiny buffer so benchmark timings exclude JIT compile."""
    from corpus import gen_tabular_telemetry
    tiny = gen_tabular_telemetry(256 * 1024, seed=0)
    cfg = CodecConfig(
        finisher="zstd",
        clumping=ClumpingConfig(min_clump_size=65536, max_clump_size=131072),
    )
    enc, _ = encode_buffer(tiny, cfg)
    dec = decode_buffer(enc)
    assert dec == tiny


def benchmark_dataset(name: str, raw: bytes) -> Dict[str, Any]:
    clump_cfg = ClumpingConfig(min_clump_size=262144, max_clump_size=1048576)

    cfg_s12_zstd = CodecConfig(
        finisher="zstd",
        zstd_level=3,
        enable_stage2=True,
        enable_stage3=False,
        enable_escape_hatch=True,
        clumping=clump_cfg,
    )
    cfg_full_zstd = CodecConfig(
        finisher="zstd",
        zstd_level=3,
        enable_stage2=True,
        enable_stage3=True,
        enable_escape_hatch=True,
        clumping=clump_cfg,
    )
    cfg_full_lz4 = CodecConfig(
        finisher="lz4",
        enable_stage2=True,
        enable_stage3=True,
        enable_escape_hatch=True,
        clumping=clump_cfg,
    )

    runs = [
        run_baseline_zstd(raw, level=3, ldm=False),
        run_baseline_zstd(raw, level=3, ldm=True),
        run_baseline_lz4(raw),
        run_ftcc(raw, "ftcc_s12_zstd", cfg_s12_zstd),
        run_ftcc(raw, "ftcc_full_zstd", cfg_full_zstd),
        run_ftcc(raw, "ftcc_full_lz4", cfg_full_lz4),
    ]

    z_base = runs[0]["compressed_bytes"]
    l_base = runs[2]["compressed_bytes"]
    for r in runs:
        if r["method"].endswith("lz4"):
            r["delta_vs_base_pct"] = 100.0 * (r["compressed_bytes"] - l_base) / l_base
        else:
            r["delta_vs_base_pct"] = 100.0 * (r["compressed_bytes"] - z_base) / z_base

    return {
        "dataset": name,
        "size_mb": len(raw) / MB,
        "runs": runs,
    }


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 124)
    print(
        f"{'Dataset':<24} {'Method':<15} {'Net MB':>8} {'Ratio':>7} {'vs Base':>8} "
        f"{'Ovhd B':>8} {'Ovhd%':>7} {'Enc MB/s':>9} {'Dec MB/s':>9} {'PeakMB':>7} {'OK':>3}"
    )
    print("=" * 124)
    for ds in results:
        for i, r in enumerate(ds["runs"]):
            ds_label = ds["dataset"] if i == 0 else ""
            delta_s = f"{r['delta_vs_base_pct']:+6.1f}%"
            print(
                f"{ds_label:<24} {r['method']:<15} {r['compressed_bytes']/MB:>8.2f} "
                f"{r['ratio']:>6.2f}x {delta_s:>8} {r['overhead_bytes']:>8d} "
                f"{r['overhead_pct']:>6.4f}% {r['enc_mb_s']:>9.1f} {r['dec_mb_s']:>9.1f} "
                f"{r['peak_mem_mb']:>7.1f} {'Y' if r['bit_exact'] else 'N':>3}"
            )
        print("-" * 124)


def main() -> None:
    parser = argparse.ArgumentParser(description="FTCC Benchmark & Ablation Harness")
    parser.add_argument("--file", type=str, default="", help="Optional path to external 1 GB+ file")
    parser.add_argument("--synth-mb", type=int, default=32, help="Synthetic dataset size in MB")
    parser.add_argument("--slice-mb", type=int, default=32, help="1 GB+ file slice size in MB")
    parser.add_argument("--out-json", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    print("Warming up Numba JIT kernels...")
    warmup_jit()

    all_results: List[Dict[str, Any]] = []

    # Part 1: Synthetic structured datasets (32 MB each)
    print(f"\nBuilding {args.synth_mb} MB synthetic structured datasets...")
    suite = build_synthetic_suite(size_mb=args.synth_mb)
    for name, raw in suite.items():
        print(f"Benchmarking synthetic dataset: {name} ({len(raw)/MB:.1f} MB)...")
        all_results.append(benchmark_dataset(name, raw))

    # Part 2: 1 GB+ file sampled across multiple offsets (32 MB slices + 64 MB slice)
    if args.file:
        corpus_path = Path(args.file)
    else:
        corpus_path = ensure_1gb_corpus(Path("bench_data/corpus_1gb.bin"), target_mb=1056)

    file_mb = corpus_path.stat().st_size // MB
    print(f"\nBenchmarking multi-offset slices from {corpus_path} ({file_mb} MB)...")
    offsets = [0, 256, 512, 768, max(0, file_mb - args.slice_mb)]
    for off in offsets:
        raw = read_file_slice(corpus_path, off, args.slice_mb)
        label = f"1gb_off{off:04d}m_{args.slice_mb}m"
        print(f"Benchmarking {label} ({len(raw)/MB:.1f} MB)...")
        all_results.append(benchmark_dataset(label, raw))

    # Plus one 64 MB slice at offset 256 MB to verify 64 MB throughput/memory scaling
    if file_mb >= 320:
        raw64 = read_file_slice(corpus_path, 256, 64)
        label64 = "1gb_off0256m_64m"
        print(f"Benchmarking {label64} ({len(raw64)/MB:.1f} MB)...")
        all_results.append(benchmark_dataset(label64, raw64))

    print_summary_table(all_results)

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved full JSON metrics to {out_path.resolve()}")


if __name__ == "__main__":
    main()
