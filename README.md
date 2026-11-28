# `asv4-compression` — Two C++20 Codecs: ASV4 (Streaming 1D) & HND3/HND4 (3D Hyperspectral)

This repository contains **two independent C++20 codecs** built under a staged falsification methodology:

1. **ASV4 (`v4.1`, [`v3/cpp/asv3.cpp`](v3/cpp/asv3.cpp) → `./asv4`)**: A single-pass, pipe-safe, schema-free lossless streaming compressor (ARM64 NEON + `libzstd 1.5.5`) for binary dumps, neural network weights (`.safetensors`), and mixed structured files.
2. **HND3 / HND4 (`phase4-v4.0`, modular library [`include/hnd3.h`](include/hnd3.h) + `libhnd3.a` → `./hnd3`)**: A zero-dependency C++20/C99 reversible & guaranteed-$L_\infty$ near-lossless 3D tensor codec for hyperspectral cubes (`i16`/`u16`/`i32`) combining complete-pivoted integer `LDU`-RKLT, arbitrary-shape 2D 5/3 integer lifting, 16-stripe band-parallel 32-bit rANS, 7-layer progressive bitplane coding, streaming spatial tiling (`HND4 --tile-size T`) with random-access single-tile decode (`--tile TY,TX`), per-header & per-layer Castagnoli CRC32C (`--crc`), and blind 3D stride/spectral-axis recovery (`--blind`).

> **Units, Hardware, & Honest Scope**: Benchmarks ran on an **Apple M3 Max (16 logical CPUs, 128 GiB RAM, macOS 27.2.1, Apple Clang 16.0.0; Homebrew LLVM 23.1.6 for `libFuzzer + ASan + UBSan`)**. In the **ASV4** section, file sizes and Peak RSS use **binary `MiB` (`2^20 B`)** (matching [`v3/REPORT_V4.md`](v3/REPORT_V4.md)) while throughput is in decimal `MB/s` (`10^6 B/s`). In the **HND3/HND4** section, all sizes, RSS, and throughputs use **decimal `MB` (`10^6 B`)** (matching [`PHASE3_REPORT.md`](PHASE3_REPORT.md) and [`PHASE4_REPORT.md`](PHASE4_REPORT.md)). GitHub Actions CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) builds, tests (`ubuntu-latest` `x86_64` & `macos-latest` `arm64`), and runs `libFuzzer + ASan + UBSan` on both codecs. Known limits: HND3/HND4 is 3D-only; untiled `HND3` holds full cubes in memory (`~17–22×` raw `i16`, solved via `HND4 --tile-size 128` at `0.9×–1.7×` Decode RSS); and in guaranteed-$L_\infty$ near-lossless mode (`E ≥ 1`), closed-loop **CCSDS-123.0-B-2 (Hybrid)** beats HND4's open-loop pre-quantizer by `0.14–1.07 b/s` ([`PHASE4_REPORT.md`](PHASE4_REPORT.md) §3).

---

## Part 1: ASV4 (`v4.1`) — Blind Streaming Compressor

ASV4 discovers binary structure online without schemas or backward seeks, applies fused SIMD bijective transforms (including nibble-rotated byte-plane transposition `T2_NIBROT`), coalesces transformed and raw spans into lazy persistent `zstd` streams, and falls back to raw `zstd --long=27` on unstructured or adversarial data.

- **Real public LLM weights & astronomy rasters** *(note: ZipNN optimizes for multi-threaded aligned tensor loading, whereas ASV4 is single-threaded and schema-free; single-run margins over ZipNN are `0.17%–0.39%`)*:
  - **GPT-2 FP32 (`real_gpt2_f32.safetensors`, `522.7 MiB`)**: `394.91 MiB` (`−10.38%` vs. `zstd -19 --long=27`) at `211 MB/s` enc / `1,252 MB/s` dec (1T) vs. ZipNN 0.5.4 `397.59 MiB`, OpenZL `le32` `398.43 MiB`, Blosc2 `401.61 MiB`.
  - **SmolLM2-135M BF16 (`real_smollm2_bf16.safetensors`, `256.6 MiB`)**: `170.44 MiB` (`−15.09%` vs. `zstd -3 --long=27`) at `208 MB/s` enc / `927 MB/s` dec (1T) vs. ZipNN 0.5.4 `171.10 MiB`, Blosc2 `177.25 MiB`, OpenZL `bf16Decon` `178.68 MiB`.
  - **Hubble WFPC2 FITS (`real_hubble_eagle_f32.fits`, `29.5 MiB`)**: `15.28 MiB` (`−16.86%` vs. `zstd -19 --long=27`; supervised OpenZL `15.14 MiB`) at `146 MB/s` enc / `680 MB/s` dec.
- **Anti-regression floor & bounds**: Worst-case overhead across all 5 negative/adversarial controls (`enwik9`, source `.tar`, Mach-O binaries, media, CSPRNG random) is `+0.0021%` (L3) / `+0.0028%` (L19). `L19 ≤ L3` on `16/16` datasets, `32/32` streaming SHA-256 PASS, `0.89×` avg / `2.06×` max Scoring Work Multiplier, `≤ 192 MiB` combined dual-stream zstd window ceiling (peak OS process RSS `≤ 286.8 MiB` decode / `≤ 312.9 MiB` encode), and **coverage-guided `libFuzzer + ASan + UBSan` verified** (`164,475` runs across all 7 transform IDs in [`fuzz/fuzz_asv4.cpp`](fuzz/fuzz_asv4.cpp), `0` crashes/errors/leaks).

---

## How It Works

```mermaid
flowchart LR
    IN["Input Stream (stdin/file)"] --> FG["16 MiB Flush Group + SHA-256"]
    FG --> SEG["Coupled Stage 1 & 2: NEON M(L) + Per-Plane zstd-1 + Viterbi DP + LDM/zstd-3 Gates"]
    SEG -->|"T0 (Fallback)"| RAW["S_RAW (Persistent zstd --long)"]
    SEG -->|"T1 / T2_XOR / T2_ELEM / T2_BIT / T2_NIBROT"| XF["Fused NEON Transform → S_XFORM (Persistent zstd)"]
    RAW --> OUT["Pipe-Safe Interleaved ASV3/v4 Stream + 32 B SHA-256 Trailer"]
    XF --> OUT
```

See **[`ARCHITECTURE.md`](ARCHITECTURE.md)** for full Mermaid diagrams of the two-tier scorer, `T2_NIBROT` bit-lane information theory ($I(E_{4..0}; M_{6..4}) \approx +0.0523\text{ bits/bf16}$), interleaved wire format, and bounds-checked streaming decoder.

---

## Benchmark Results (16 Datasets / 4.95 GiB, Single-Threaded ARM64)

### Head-to-Head on Real Public AI Weights & Astronomy Data (Sizes in Binary `MiB`)
*(All external baselines verified bit-exact lossless round-trip; aligned `.safetensors` baselines include exact header bytes. ZipNN optimizes for multi-threaded model-loading throughput on aligned tensors, whereas ASV4 is a general-purpose, single-threaded, schema-free streaming compressor requiring no `.safetensors` parser. OpenZL `le32` `398.43 MiB` reflects the verified chunk-32 fix (0-indexed final chunk of 33) documented in [`v3/AUDIT_V3.md`](v3/AUDIT_V3.md) and [`v3/REPORT_V4.md`](v3/REPORT_V4.md). On `real_gpt2_f32` and `real_smollm2_bf16`, L3 and L19 match because high-$H_0$ `T2_NIBROT` lanes cap `c_xform` to L3 to prevent `ZSTD_btultra2` noise-chasing regressions.)*

| Real Public Dataset | Orig (`MiB`) | `zstd -3` / `-19` (`libzstd 1.5.5` C API) | `Blosc2` (`zstd7`) | `OpenZL 0.3.0` *(Verified Round-Trip)* | `ZipNN 0.5.4` *(Aligned, Verified)* | **Blind `ASV4 v4.1` (L3 / L19)** |
|---|---:|---:|---:|---:|---:|---:|
| `real_gpt2_f32.safetensors` | 522.7 | 440.52 / 440.63 | 401.61 (`−8.86%`) | 398.43 (`le32`, `−9.58%`) / 402.50 (`f32Decon` aligned) / 404.12 (`tr4`) / 432.17 (`f32Decon` raw) | 397.59 (`float32`, `−9.77%`) | **394.91 (`−10.35%`) / 394.91 (`−10.38%`)** |
| `real_smollm2_bf16.safetensors` | 256.6 | 200.74 / 200.67 | 177.25 (`−11.67%`) | 178.68 (`bf16Decon`, `−10.96%`) / 183.00 (`le16`, `−8.81%`) | 171.10 (`bfloat16`, `−14.74%`) | **170.44 (`−15.09%`) / 170.44 (`−15.06%`)** |
| `real_hubble_eagle_f32.fits` | 29.5 | 18.40 / 18.38 | — | **15.14** (`le32`/`be32`, `−17.66%`) | — | **15.33 (`−16.66%`) / 15.28 (`−16.86%`)** |
| `real_mixed_public.bin` | 283.9 | 99.66 / 87.42 | 131.51 | 97.18 (`serial`) | — | **94.28 (`−5.40%`) / 82.25 (`−5.92%`)** |
| `real_nyc_taxi_struct104.bin` | 256.0 | 53.66 / **36.89** | 64.37 | 51.38 (`serial`) / 63.93 (`struct104+T`) | — | **53.66 (`+0.001%`) / 36.89 (`+0.002%`)** |
| `real_nyc_taxi_columnar.arrow` | 803.1 | 107.72 / **87.83** | 148.84 | 112.47 (`serial`) | — | **107.72 (`+0.001%`) / 87.83 (`+0.002%`)** |

### Full 16-Dataset Same-Binary Table ([`v3/final_refresh_results_v4.1.csv`](v3/final_refresh_results_v4.1.csv))
*(Sizes and Process RSS in binary `MiB`; speeds in decimal `MB/s`. For reference `B2_capi` `zstd -3` baseline speeds per dataset, see [`v3/REPORT_V4.md`](v3/REPORT_V4.md).)*

| Dataset | Orig `MiB` | L3 vs `B2_capi` | L19 vs `B3a_capi` | Enc L3 (MB/s) | Dec L3 / L19 (MB/s) | Enc / Dec L3 RSS (`MiB`) | Regs | Work Mult | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `real_gpt2_f32.safetensors` | 522.7 | **−10.354%** | **−10.375%** | 211.0 | 1251.5 / 1240.9 | 154.2 / 154.3 | 2 | 0.95× | PASS |
| `real_smollm2_bf16.safetensors` | 256.6 | **−15.094%** | **−15.063%** | 208.3 | 926.6 / 935.6 | 151.8 / 150.8 | 3 | 0.85× | PASS |
| `real_hubble_eagle_f32.fits` | 29.5 | **−16.658%** | **−16.860%** | 146.3 | 680.4 / 636.3 | 111.8 / 89.6 | 3 | 1.04× | PASS |
| `real_mixed_public.bin` | 283.9 | **−5.401%** | **−5.920%** | 115.8 | 848.9 / 942.3 | 241.0 / 225.1 | 9 | 0.93× | PASS |
| `real_nyc_taxi_struct104.bin` | 256.0 | +0.001% | +0.002% | 112.5 | 715.0 / 985.7 | 191.0 / 194.6 | 1 | 0.62× | PASS |
| `real_nyc_taxi_columnar.arrow` | 803.1 | +0.001% | +0.002% | 118.5 | 986.4 / 1231.3 | 192.8 / 193.7 | 1 | 1.18× | PASS |
| `target_numeric_f32.bin` | 256.0 | **−41.758%** | **−33.479%** | 157.8 | 847.1 / 732.5 | 202.4 / 202.2 | 1 | 1.29× | PASS |
| `target_struct64.bin` | 256.0 | **−68.167%** | **−54.072%** | 179.7 | 695.7 / 691.5 | 194.8 / 198.1 | 1 | 0.78× | PASS |
| `target_sqlite.db` | 311.3 | **−9.566%** | **−6.554%** | 89.8 | 737.3 / 632.8 | 282.6 / 286.0 | 2 | 1.00× | PASS |
| `target_mixed_structured.bin` | 256.0 | **−42.944%** | **−32.186%** | 132.6 | 825.1 / 718.9 | 312.9 / 286.8 | 8 | 1.08× | PASS |
| `target_snapshots.bin` | 256.0 | +0.001% | +0.001% | 343.0 | 1364.3 / 1414.0 | 215.0 / 207.0 | 1 | 0.25× | PASS |
| `neg_enwik9.xml` (1 GB) | 953.7 | **+0.001%** | **+0.001%** | 108.4 | 779.0 / 781.0 | 197.4 / 196.8 | 1 | 0.73× | PASS |
| `neg_source_tar.tar` | 273.2 | **+0.002%** | **+0.002%** | 111.3 | 1040.9 / 1051.1 | 191.6 / 193.6 | 1 | 0.97× | PASS |
| `adv_binaries.bin` | 160.0 | **+0.002%** | **+0.003%** | 81.8 | 1136.5 / 1123.4 | 191.0 / 193.9 | 1 | 2.06× | PASS |
| `adv_compressed_media.tar` | 126.0 | **+0.000%** | **+0.000%** | 299.1 | 1332.3 / 1281.9 | 213.0 / 211.0 | 1 | 0.33× | PASS |
| `adv_random.bin` | 64.0 | **+0.000%** | **+0.000%** | 374.4 | 1128.7 / 1112.0 | 150.5 / 142.8 | 1 | 0.25× | PASS |

---

## Part 2: HND3 / HND4 (`phase4-v4.0`) — Reversible & Near-Lossless 3D Hyperspectral Codec

**HND3 / HND4** ([`include/hnd3.h`](include/hnd3.h) + `libhnd3.a` → `./hnd3`, with `HND3 v3` wire-compatible with [`phase3_codec.py`](phase3_codec.py)) is a zero-dependency C++20/C99 3D tensor codec featuring complete-pivoted `LDU`-RKLT (ARM64 NEON `vmlal_s32` for `i16`/`u16`, overflow-hardened scalar `int64`/`__int128` + variable-width escape for full-range `i32`), arbitrary-shape 2D 5/3 integer lifting, 16-stripe parallel 3D-context 32-bit rANS, single-stream 7-layer progressive decoding (`base_k = 6`), **HND4 streaming spatial tiling (`--tile-size T`) with `O(B · T²)` bounded Peak RSS, `≤ 64 MiB` tile-row strip I/O, and random-access single-tile decoding (`--tile TY,TX`)**, **per-header & per-layer Castagnoli CRC32C (`--crc`, verifying even lossy previews)**, **auto-gated reversible monotonic histogram compaction (`--hist-lut auto|on|off`)**, **guaranteed-$L_\infty$ near-lossless coding (`--max-error E`)**, default-on `--verify`, windowed native C++20 blind 3D shape recovery (`--blind`), and `libFuzzer + ASan + UBSan` verification (`129k+` runs in [`fuzz/fuzz_hnd3.cpp`](fuzz/fuzz_hnd3.cpp), `0` errors).

### Real Public Hyperspectral Cubes (Uncropped, Official References, Decimal `MB = 10⁶ B`, Best-of-3)

| Uncropped Real Cube | Official `Empordà` CCSDS-123.0-B-1 (`P=3 / P=15` SA) | `LDU` + OpenJPEG J2K (`Dec 1T/MT`) | `LDU` + `cjxl 0.13` (`Dec MT`) | **HND3 `k=0` (`Dec 1T/MT`)** | **HND3 `k=6` (`Dec 1T/MT`)** | `HND4 tile=128` (`b/s` · `Dec` · **Dec RSS**) | `k=6` Preview (`SAM_mean`) | `--blind` (`6/6` Perms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Indian Pines** `(200, 145, 145)`, `8.41 MB` | `6.579 / 6.545 b/s` *(B2 Hyb: `6.577 / 6.544`)* | `6.739` (`22/167`) | `6.558` (`12`) | **`6.485`** (`70/205`) | `6.561` (`28/126`) | `6.734` · `109 MB/s` · **`97 MB`** | `16.4%` → `54.1 dB` (`0.32°`) | `6/6` exact |
| **Salinas** `(204, 512, 217)`, `45.33 MB` | `4.881 / 4.864 b/s` | `4.803` (`22/63`) | **`4.043`** (`18`) | `4.657` (`38/157`) | `4.628` (`21/106`) | `4.737` · `114 MB/s` · **`108 MB`** (`7.0×↓`) | `3.4%` → `60.1 dB` (`0.34°`) | `6/6` exact |
| **PaviaU** `(103, 610, 340)`, `42.72 MB` | `6.551 / 6.620 b/s` | `6.640` (`21/99`) | `6.558` (`33`) | `6.366` (`50/175`) | **`6.354`** (`26/125`) | `6.542` · `124 MB/s` · **`70 MB`** (`10.2×↓`) | `12.1%` → `52.9 dB` (`0.78°`) | `6/6` exact |
| **AVIRIS Holdout** `(224, 745, 398)`, `132.84 MB` | `3.014 / 3.007 b/s` | `3.068` (`29/60`) | **`2.868`** (`19`) | `3.068` (`41/157`) | `2.940` (`23/116`) | `3.119` (`k=6: 3.052`) · **`114 MB/s`** · **`144 MB`** (`15.1×↓`) | `0.6%` → `55.3 dB` (`0.50°`) | `6/6` exact |

*Reproduce all tables via `python3 bench/fetch_hnd3_data.py && make bench-hnd3` (and `python3 bench/bench_ccsds123_b2.py` for NTNU CCSDS-123.0-B-2). See [`PHASE4_REPORT.md`](PHASE4_REPORT.md) for HND4 tiled Peak RSS & `≤ 64 MiB` tile-row strip I/O (`15.1×` Decode RSS reduction on AVIRIS at `114.1 MB/s` full-scene disk decode, `22.5–63.7 ms` single-tile random access), auto-gated `FLAG_HIST_LUT` sparse-histogram compaction on non-hyperspectral 3D volumes (`Cells3D`: `9.976 → 8.938 b/s`, beating `cjxl 0.13`'s `9.618 b/s` at `2.5×` decode speed; `Brain_MRI`: `7.083 → 2.717–2.796 b/s` at `160.5 MB/s`, `0%` false triggers on dense hyperspectral cubes), guaranteed-$L_\infty$ near-lossless R-D vs. **NTNU CCSDS-123.0-B-2 (Hybrid)** (where HND4 beats Issue 2 lossless on full Indian Pines `6.485 vs 6.544 b/s`, while Issue 2's closed-loop quantizer beats HND4 by `0.14–1.07 b/s` at `E=1..4`), and `libFuzzer + ASan + UBSan` results. See [`PHASE3_REPORT.md`](PHASE3_REPORT.md) for 7-layer R-D curves, discrete-entropy `LDU` audit, and full-range `int32` (`±1.9e9`).*

---

## Quickstart

### Build & Test (C++20; Verified on Apple Silicon ARM64, Portable Scalar Paths Included)
Requires a C++20 compiler (`clang++` or `g++`) and `libzstd` (`>= 1.5` for `./asv4`; `./hnd2` and `./hnd3` have **zero external dependencies**). The top-level [`Makefile`](Makefile) auto-detects Homebrew, Conda, or system `libzstd` paths and builds `./asv4` (from [`v3/cpp/asv3.cpp`](v3/cpp/asv3.cpp)), `./hnd2`, and `./hnd3`.

```bash
# Build all C++20 targets (./asv4, v3/cpp/asv3, ./hnd2, libhnd3.a, ./hnd3)
make

# Run UBSan + portable SHA-256 + NEON/scalar parity + C99 ABI + 5k libc++ debug fuzz + Phase 1-4 suites
make test

# Build coverage-guided libFuzzer + ASan + UBSan harnesses (/tmp/fuzz_hnd3, /tmp/fuzz_asv4)
make fuzz
```

### Usage: ASV4 (Files or Unix Pipes)
```bash
# Compress & decompress files (automatic streaming SHA-256 verification on decode)
./asv4 c --level 3 model.safetensors model.asv4
./asv4 d model.asv4 model.restored

# Pipe-safe streaming (zero backward seeks)
cat model.safetensors | ./asv4 c --level 3 | ./asv4 d > /dev/null

# Archival preset (--level 19, guaranteed <= --level 3) with JSON stats & trace
./asv4 c --level 19 --stats stats.json --trace input.bin output.asv4
```

### Usage: HND3 / HND4 (3D Hyperspectral & Scientific Tensors)
```bash
# 1) Untiled 7-layer quality-embedded lossless HND3
./hnd3 enc cube.raw cube.hnd3 --shape 204,512,217 --dtype i16 --base-k 6
./hnd3 dec cube.hnd3 cube_lossless.raw              # Full bit-exact decode + SHA-256 check
./hnd3 dec cube.hnd3 preview.raw --stop-layer 6     # Instant 60.1 dB base preview (3.4% of stream)

# 2) HND4 streaming spatial tiling (bounded O(B*T^2) Peak RSS + per-layer CRC32C + random access)
./hnd3 enc aviris.raw aviris.hnd4 --shape 224,745,398 --dtype i16 --base-k 6 --tile-size 128
./hnd3 dec aviris.hnd4 aviris_rec.raw               # Streaming decode (144-151 MB RSS, ~85-114 MB/s, SHA256+CRC32C=PASS)
./hnd3 dec aviris.hnd4 tile11.raw --tile 1,1        # Random-access single-tile lossless decode (~64 ms)
./hnd3 dec aviris.hnd4 prev11.raw --tile 1,1 --stop-layer 6  # Verified CRC32C preview of one tile (~35 ms)

# 3) Guaranteed-L_inf near-lossless mode (|x - x_hat| <= 2 on every sample, verified on encode)
./hnd3 enc aviris.raw aviris_e2.hnd4 --shape 224,745,398 --dtype i16 --tile-size 256 --max-error 2

# 4) Pure C++20 blind mode (zero shape hints, automatic 6/6 stride & spectral-axis discovery)
./hnd3 enc headerless.raw blind.hnd3 --blind --dtype i16 --base-k 0
./hnd3 dec blind.hnd3 restored.raw
```

### Reproduce All 16 Benchmarks (4.95 GiB)
```bash
# 1. Build ./asv4, v3/cpp/asv3, and the exact in-process libzstd C API baseline harness
make bench

# 2. Generate the 10-file baseline corpus in v3/corpus/ (downloads enwik9)
python3 v3/build_corpus.py

# 3. Download & assemble the 6 real public dumps in v3/corpus_real/
#    (GPT-2 FP32, SmolLM2-135M BF16, NASA Hubble FITS, NYC TLC Taxi Struct/Arrow, Mixed Public)
python3 v3/download_real_corpus.py

# 4. Run the full 16-dataset same-binary L3 & L19 wall-clock / Peak RSS benchmark
python3 v3/bench_final_refresh.py
```

---

## Repository Map

| Path | Description |
|---|---|
| [`include/hnd3.h`](include/hnd3.h), [`cpp/hnd3_core.hpp`](cpp/hnd3_core.hpp), [`cpp/hnd3_rans.hpp`](cpp/hnd3_rans.hpp), [`cpp/hnd3_lib.cpp`](cpp/hnd3_lib.cpp), [`cpp/hnd3_cli.cpp`](cpp/hnd3_cli.cpp) | **Phase 4 modular zero-dependency C++20/C99 HND3/HND4 library (`libhnd3.a`) & CLI (`./hnd3`)** (streaming tiling, CRC32C, near-lossless, C ABI, native `--blind`) |
| [`fuzz/fuzz_hnd3.cpp`](fuzz/fuzz_hnd3.cpp), [`fuzz/fuzz_asv4.cpp`](fuzz/fuzz_asv4.cpp) | Coverage-guided `libFuzzer + ASan + UBSan` targets for HND3/HND4 and ASV4 (`make fuzz`) |
| [`bench/fetch_hnd3_data.py`](bench/fetch_hnd3_data.py), [`bench/bench_hnd3.py`](bench/bench_hnd3.py), [`bench/bench_ccsds123_b2.py`](bench/bench_ccsds123_b2.py) | Reproducible SHA-256 dataset fetcher, full benchmark suite (`make bench-hnd3`), & NTNU CCSDS-123.0-B-2 comparison |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Multi-platform GitHub Actions CI (`ubuntu-latest` x86_64, `macos-latest` arm64, + `libFuzzer + ASan + UBSan`) |
| [`Makefile`](Makefile) | Auto-detecting C++20 build (`make`, `make libhnd3`, `make bench`, `make bench-hnd3`, `make test`, `make fuzz`) |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System architecture & Mermaid diagrams for ASV4 v4.1 and HND3/HND4 (Phases 0–4) |
| [`cpp/hnd2.cpp`](cpp/hnd2.cpp) | Phase 2 C++20 NEON HND2 engine (7-layer quality-embedded lossless) |
| [`phase3_codec.py`](phase3_codec.py), [`phase2_codec.py`](phase2_codec.py), [`phase1_codec.py`](phase1_codec.py) | Wire-compatible Python+Numba reference codecs for Phases 1–3 |
| [`phase0_harness.py`](phase0_harness.py) | Phase 0 falsification harness v2.2 (Blue Book CCSDS-123, real Rice, Gates 0A–0D, Track B) |
| [`test_phase0.py`](test_phase0.py) .. [`test_phase4.py`](test_phase4.py) | Automated verification suites (transforms, CCSDS-123, blind strides, Py↔C++ interop, HND4 tiling/CRC/near-lossless/C ABI) |
| [`PHASE0_REPORT.md`](PHASE0_REPORT.md) .. [`PHASE4_REPORT.md`](PHASE4_REPORT.md) | Staged empirical falsification & real-cube benchmark reports (Phases 0–4) |
| [`v3/cpp/asv3.cpp`](v3/cpp/asv3.cpp) | **Production C++20 NEON ASV4 v4.1 streaming encoder & hardened decoder** |
| [`v3/cpp/asv3_v3.0.cpp`](v3/cpp/asv3_v3.0.cpp) | Frozen `v3.0` C++20 NEON reference snapshot |
| [`v3/cpp/zstd_capi_bench.cpp`](v3/cpp/zstd_capi_bench.cpp) | Exact in-process single-threaded `libzstd` C API baseline harness (`B2_capi` / `B3a_capi`) |
| [`v3/download_real_corpus.py`](v3/download_real_corpus.py) | Downloads & builds all 6 real public benchmark datasets in `v3/corpus_real/` |
| [`v3/build_corpus.py`](v3/build_corpus.py) | Builds the original 10-file pre-registered corpus in `v3/corpus/` |
| [`v3/REPORT_V4.md`](v3/REPORT_V4.md) | Complete **ASV4 v4.1** technical report (16 datasets, OpenZL/Blosc2/ZipNN comparisons, theoretical analysis) |
| [`v3/REPORT_V3.md`](v3/REPORT_V3.md) | Staged Phase 0 → 1A → 1B → 2 build report & Kill Gate 1B falsification record |
| [`v3/AUDIT_V3.md`](v3/AUDIT_V3.md) | Forensic audit artifacts (corpus provenance, C-API baselines, divergence traces, OpenZL chunk-32 fix (0-indexed final chunk of 33)) |
| [`v3/final_refresh_results_v4.1.csv`](v3/final_refresh_results_v4.1.csv) | Canonical 16-dataset `v4.1` L3 & L19 CSV (plus archived `v3.0` and `v4.0` CSVs) |
| [`v3/codec.py`](v3/codec.py), [`v3/transforms.py`](v3/transforms.py) | Phase 1A Python/Numba reference implementation (wire-compatible with `v3`) |
| [`v3/stage3_forest.py`](v3/stage3_forest.py) | Phase 1B Stage 3 delta-forest ablation (falsified & cut per Kill Gate 1B) |
| [`v3/bench_final_refresh.py`](v3/bench_final_refresh.py) | Same-binary 16-dataset L3 & L19 wall-clock / Peak RSS benchmark driver |
| [`ftcc/`](ftcc/) | Initial v1 forest-tree energy-clumping prototype (superseded by v3/v4) |
| [`LICENSE`](LICENSE) | MIT License |

---

## References & Credits

### Algorithms & Prior Art
- **P. Hao & Q. Shi (2001)**, *Matrix Factorizations for Reversible Integer Mapping*, IEEE Trans. Signal Processing (`PLUS` / SERM integer lifting).
- **I. Daubechies & W. Sweldens (1998)**, *Factoring Wavelet Transforms into Lifting Steps*, J. Fourier Anal. Appl. (3-shear diagonal lifting & lifting framework).
- **A. R. Calderbank, I. Daubechies, W. Sweldens, & B.-L. Yeo (1998)**, *Wavelet Transforms That Map Integers to Integers*, Appl. Comput. Harmon. Anal. (reversible 5/3 integer wavelet).
- **L. Galli & S. Salzo (2004)**, *Lossless Hyperspectral Compression Using KLT*, Proc. IEEE IGARSS, vol. 1, p. 316 (quasi-complete-pivoted reversible KLT).
- **I. Blanes & J. Serra-Sagristà (2009)**, *Clustered Reversible-KLT for Progressive Lossy-to-Lossless 3D Image Coding*, Proc. IEEE DCC 2009, pp. 233–242; and *Cost and Scalability Improvements to the Karhunen–Loève Transform for Remote-Sensing Image Coding*, IEEE Trans. Geosci. Remote Sens. 48(7), 2010 (multilevel clustered RKLT lineage behind `clust` mode).
- **I. Blanes & J. Serra-Sagristà (2011)**, *Pairwise Orthogonal Transform for Spectral Image Coding*, IEEE Trans. Geosci. Remote Sens. 49(3) (published online 2010; POT spectral decorrelation).
- **J. Duda (2009 / 2013)**, *Asymmetric Numeral Systems*, arXiv:0902.0271 / arXiv:1311.2540 (32-bit rANS entropy coding).
- **A. de Cheveigné & H. Kawahara (2002)**, *YIN, a Fundamental Frequency Estimator for Speech and Music*, J. Acoust. Soc. Am. (rectified fundamental stride selection).
- **CCSDS 123.0-B-1 / 123.0-B-2**, *Low-Complexity Lossless and Near-Lossless Multispectral and Hyperspectral Image Compression*, Consultative Committee for Space Data Systems.
- **External reference implementations benchmarked**: `libzstd` (Meta), `OpenZL`, `Blosc2`, `ZipNN`, UAB/ESA `Empordà` (CCSDS-123.0-B-1), NTNU SmallSat Lab CCSDS-123.0-B-2 High-Level Verification Model (Vorhaug, Boyle, & Orlandić, WHISPERS 2024), `OpenJPEG`, `libjxl` (`cjxl`/`djxl` 0.13.0), `CharLS`, and `ZFP`.

### Public Datasets
- **Hyperspectral cubes**: Purdue University / NASA AVIRIS *Indian Pines*, NASA/JPL AVIRIS *Salinas*, University of Pavia ROSIS *Pavia University*, and NASA AVIRIS holdout scene via HuggingFace `danjacobellis/aviris_1k_val`.
- **Non-hyperspectral 3D controls**: `scikit-image` public 3D fluorescence microscopy (`cells3d.tif`, Allen Institute for Cell Science) and 3D medical brain MRI (`brain.tiff`).
- **ASV4 streaming corpus**: OpenAI `gpt2` (`safetensors`), HuggingFaceTB `SmolLM2-135M` (`safetensors`), NASA/ESA Hubble WFPC2 Eagle Nebula (`FITS`), NYC TLC Trip Record Data, and Matt Mahoney's `enwik9`.

### Authorship & AI-Assisted Engineering Note
This codebase, staged falsification harnesses, C++20 NEON kernels, and empirical audit reports were developed collaboratively with AI-assisted implementation, code review, and independent adversarial verification.

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).
