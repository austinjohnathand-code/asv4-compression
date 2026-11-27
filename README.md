# ASV4 — Blind, Schema-Free Adaptive Stride-Transposition Streaming Compressor

**ASV4 (v4.1)** is a single-pass, pipe-safe, schema-free lossless compressor written in C++20 with ARM64 NEON SIMD intrinsics and native `libzstd 1.5.5`. Without knowing file formats, headers, or data types upfront, ASV4 discovers binary structure online, applies fused SIMD bijective transforms (including nibble-rotated byte-plane transposition `T2_NIBROT`), coalesces transformed and raw spans into lazy persistent `zstd` streams, and falls back transparently to raw `zstd --long=27` on unstructured or adversarial data.

- **Beats dedicated & supervised baselines blindly on real LLM weights**:
  - **OpenAI GPT-2 FP32 (`real_gpt2_f32.safetensors`, 522.7 MB)**: **`394.91 MB`** (**−10.38%** vs. `zstd -19 --long=27`, **−24.45%** vs. raw) at **211 MB/s** encode and **1,252 MB/s** decode on one ARM64 core — beating verified lossless **ZipNN 0.5.4 `float32` (`397.59 MB`)**, **OpenZL `le32` (`398.43 MB`)**, **Blosc2 (`401.61 MB`)**, **OpenZL `Float32Deconstruct` (`402.50 MB` aligned / `432.17 MB` raw)**, and **OpenZL `struct4+TransposeSplit` (`404.12 MB`)**.
  - **SmolLM2-135M BF16 (`real_smollm2_bf16.safetensors`, 256.6 MB)**: **`170.44 MB`** (**−15.09%** vs. `zstd -3 --long=27`, **−33.58%** vs. raw) at **208 MB/s** encode and **927–936 MB/s** decode — beating verified lossless **ZipNN 0.5.4 `bfloat16` (`171.10 MB`)**, **Blosc2 (`177.25 MB`)**, **OpenZL `BFloat16Deconstruct` (`178.68 MB`)**, and **OpenZL `le16` (`183.00 MB`)**.
- **Real scientific astronomy rasters (`real_hubble_eagle_f32.fits`, NASA/ESA Hubble WFPC2 `>f4`)**: **`15.28 MB`** (**−16.86%** vs. `zstd -19 --long=27`, within `0.14 MB` of supervised OpenZL `15.14 MB`) at **146 MB/s** encode and **680 MB/s** decode.
- **Zero-regression safety floor**: Worst-case overhead across all 5 negative & adversarial controls (1 GB `enwik9`, source `.tar`, 1,834 concatenated ARM64 Mach-O binaries, compressed media, CSPRNG random) is **`+0.0021%` (L3) / `+0.0028%` (L19)** — over 50× below the `0.15%` cap.
- **Strictly monotonic, bounded, and hardened**: `L19 ≤ L3` on **16/16** datasets, **`32/32` streaming SHA-256 PASS**, **`0.89×` avg / `2.06×` max** Scoring Work Multiplier (`≤ 3.0×` budget), **`680–1,414 MB/s`** single-core decode, **`O(1)`** bounded window memory (`≤ 192 MB` combined dual-stream ceiling), UBSan-verified bounds checks, and portable FIPS 180-4 SHA-256 fallback for Linux/x86-64.

---

## How It Works

```mermaid
flowchart LR
    IN["Input Stream (stdin/file)"] --> FG["16 MB Flush Group + SHA-256"]
    FG --> SEG["Coupled Stage 1 & 2: NEON M(L) + Per-Plane zstd-1 + Viterbi DP + LDM/zstd-3 Gates"]
    SEG -->|"T0 (Fallback)"| RAW["S_RAW (Persistent zstd --long)"]
    SEG -->|"T1 / T2_XOR / T2_ELEM / T2_BIT / T2_NIBROT"| XF["Fused NEON Transform → S_XFORM (Persistent zstd)"]
    RAW --> OUT["Pipe-Safe Interleaved ASV3/v4 Stream + 32 B SHA-256 Trailer"]
    XF --> OUT
```

See **[`ARCHITECTURE.md`](ARCHITECTURE.md)** for full Mermaid diagrams of the two-tier scorer, `T2_NIBROT` bit-lane information theory ($I(E_{4..0}; M_{6..4}) \approx +0.0523\text{ bits/bf16}$), interleaved wire format, and bounds-checked streaming decoder.

---

## Benchmark Results (16 Datasets / 4.95 GiB, Single-Threaded ARM64)

### Head-to-Head on Real Public AI Weights & Astronomy Data
*(All external baselines verified bit-exact lossless round-trip; aligned `.safetensors` baselines include exact header bytes. Note that ZipNN optimizes for multi-threaded model-loading throughput on aligned tensors, whereas ASV4 is a general-purpose, single-threaded, schema-free streaming compressor requiring no `.safetensors` parser.)*

| Real Public Dataset | Orig (MB) | `zstd -3` / `-19` (`libzstd 1.5.5` C API) | `Blosc2` (`zstd7`) | `OpenZL 0.3.0` *(Verified Round-Trip)* | `ZipNN 0.5.4` *(Aligned, Verified)* | **Blind `ASV4 v4.1` (L3 / L19)** |
|---|---:|---:|---:|---:|---:|---:|
| `real_gpt2_f32.safetensors` | 522.7 | 440.52 / 440.63 | 401.61 (`−8.86%`) | 398.43 (`le32`, `−9.58%`) / 402.50 (`f32Decon` aligned) / 404.12 (`tr4`) / 432.17 (`f32Decon` raw) | 397.59 (`float32`, `−9.77%`) | **394.91 (`−10.35%`) / 394.91 (`−10.38%`)** |
| `real_smollm2_bf16.safetensors` | 256.6 | 200.74 / 200.67 | 177.25 (`−11.67%`) | 178.68 (`bf16Decon`, `−10.96%`) / 183.00 (`le16`, `−8.81%`) | 171.10 (`bfloat16`, `−14.74%`) | **170.44 (`−15.09%`) / 170.44 (`−15.06%`)** |
| `real_hubble_eagle_f32.fits` | 29.5 | 18.40 / 18.38 | — | **15.14** (`le32`/`be32`, `−17.66%`) | — | **15.33 (`−16.66%`) / 15.28 (`−16.86%`)** |
| `real_mixed_public.bin` | 283.9 | 99.66 / 87.42 | 131.51 | 97.18 (`serial`) | — | **94.28 (`−5.40%`) / 82.25 (`−5.92%`)** |
| `real_nyc_taxi_struct104.bin` | 256.0 | 53.66 / **36.89** | 64.37 | 51.38 (`serial`) / 63.93 (`struct104+T`) | — | **53.66 (`+0.001%`) / 36.89 (`+0.002%`)** |
| `real_nyc_taxi_columnar.arrow` | 803.1 | 107.72 / **87.83** | 148.84 | 112.47 (`serial`) | — | **107.72 (`+0.001%`) / 87.83 (`+0.002%`)** |

### Full 16-Dataset Same-Binary Table ([`v3/final_refresh_results_v4.1.csv`](v3/final_refresh_results_v4.1.csv))

| Dataset | Orig MB | L3 vs `B2_capi` | L19 vs `B3a_capi` | Enc L3 (MB/s) | Dec L3 / L19 (MB/s) | Dec L3 RSS (MB) | Regs | Work Mult | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `real_gpt2_f32.safetensors` | 522.7 | **−10.354%** | **−10.375%** | 211.0 | 1251.5 / 1240.9 | 154.3 | 2 | 0.95× | PASS |
| `real_smollm2_bf16.safetensors` | 256.6 | **−15.094%** | **−15.063%** | 208.3 | 926.6 / 935.6 | 150.8 | 3 | 0.85× | PASS |
| `real_hubble_eagle_f32.fits` | 29.5 | **−16.658%** | **−16.860%** | 146.3 | 680.4 / 636.3 | 89.6 | 3 | 1.04× | PASS |
| `real_mixed_public.bin` | 283.9 | **−5.401%** | **−5.920%** | 115.8 | 848.9 / 942.3 | 225.1 | 9 | 0.93× | PASS |
| `real_nyc_taxi_struct104.bin` | 256.0 | +0.001% | +0.002% | 112.5 | 715.0 / 985.7 | 194.6 | 1 | 0.62× | PASS |
| `real_nyc_taxi_columnar.arrow` | 803.1 | +0.001% | +0.002% | 118.5 | 986.4 / 1231.3 | 193.7 | 1 | 1.18× | PASS |
| `target_numeric_f32.bin` | 256.0 | **−41.758%** | **−33.479%** | 157.8 | 847.1 / 732.5 | 202.2 | 1 | 1.29× | PASS |
| `target_struct64.bin` | 256.0 | **−68.167%** | **−54.072%** | 179.7 | 695.7 / 691.5 | 198.1 | 1 | 0.78× | PASS |
| `target_sqlite.db` | 311.3 | **−9.566%** | **−6.554%** | 89.8 | 737.3 / 632.8 | 286.0 | 2 | 1.00× | PASS |
| `target_mixed_structured.bin` | 256.0 | **−42.944%** | **−32.186%** | 132.6 | 825.1 / 718.9 | 286.8 | 8 | 1.08× | PASS |
| `target_snapshots.bin` | 256.0 | +0.001% | +0.001% | 343.0 | 1364.3 / 1414.0 | 207.0 | 1 | 0.25× | PASS |
| `neg_enwik9.xml` (1 GB) | 953.7 | **+0.001%** | **+0.001%** | 108.4 | 779.0 / 781.0 | 196.8 | 1 | 0.73× | PASS |
| `neg_source_tar.tar` | 273.2 | **+0.002%** | **+0.002%** | 111.3 | 1040.9 / 1051.1 | 193.6 | 1 | 0.97× | PASS |
| `adv_binaries.bin` | 160.0 | **+0.002%** | **+0.003%** | 81.8 | 1136.5 / 1123.4 | 193.9 | 1 | 2.06× | PASS |
| `adv_compressed_media.tar` | 126.0 | **+0.000%** | **+0.000%** | 299.1 | 1332.3 / 1281.9 | 211.0 | 1 | 0.33× | PASS |
| `adv_random.bin` | 64.0 | **+0.000%** | **+0.000%** | 374.4 | 1128.7 / 1112.0 | 142.8 | 1 | 0.25× | PASS |

---

## HND3 — Reversible N-D Hyperspectral & Scientific Tensor Engine (`phase3-v3.1`)

In addition to ASV4's 1D/record streaming engine, this repository includes **HND3** ([`cpp/hnd3.cpp`](cpp/hnd3.cpp) → `./hnd3`, wire-compatible with [`phase3_codec.py`](phase3_codec.py)), a **zero-dependency C++20 ARM64 NEON** reversible 3D/N-D tensor codec featuring complete-pivoted `LDU`-RKLT (`vmlal_s32` FMA with bit-exact `-DHND3_FORCE_SCALAR` parity), arbitrary-shape 2D 5/3 integer lifting, **16-stripe parallel 3D-context 32-bit rANS**, **single-stream 7-layer quality-embedded lossless coding**, built-in FIPS 180-4 SHA-256, and **windowed native C++20 blind stride-lattice recovery (`--blind`)**.

### Real Public Hyperspectral Cubes (Exact Uncropped Apples-to-Apples, Decimal `MB = 10⁶ B`, All `SHA-256 PASS`)

| Uncropped Real Cube | Custom CCSDS-123.0-B-1 (`P=3 / P=15`) | Official `OpenJPEG J2K + LDU-RKLT` | **`.hnd3` One-Shot (`k=0`)** | **`.hnd3` 7-Layer (`k=6`)** | `k=6` Preview (Same File!) | Full Uncropped `--blind` (`6/6` Perms) |
|---|---:|---:|---:|---:|---:|---:|
| **Indian Pines** `(200, 145, 145)`, `8.41 MB` | `6.798 / 6.772 b/s` | `6.739 b/s` (`22 MB/s` dec) | `6.485 b/s` (`196 MB/s` dec) | **`6.561 b/s` (`2.44×`)** (`122 MB/s`) | `16.4%` (`1.07 b/s`) → **`54.1 dB`** (`530 MB/s`) | **`6/6` exact** (`6.485 b/s`) |
| **Salinas** `(204, 512, 217)`, `45.33 MB` | `5.131 / 5.108 b/s` | `4.803 b/s` (`21 MB/s` dec) | `4.657 b/s` (`157 MB/s` dec) | **`4.628 b/s` (`3.46×`)** (`111 MB/s`) | **`3.4%` (`0.16 b/s`, `100×`)** → **`60.1 dB`** (`328 MB/s`) | **`6/6` exact** (`4.654–4.657 b/s`) |
| **PaviaU** `(103, 610, 340)`, `42.72 MB` | `6.846 / 6.907 b/s` | `6.640 b/s` (`21 MB/s` dec) | `6.367 b/s` (`186 MB/s` dec) | **`6.354 b/s` (`2.52×`)** (`122 MB/s`) | `12.1%` (`0.77 b/s`, `21×`) → **`52.9 dB`** (`407 MB/s`) | **`6/6` exact** (`6.349–6.367 b/s`) |
| **Unseen AVIRIS** `(224, 745, 398)`, `132.84 MB` | `3.112 / 3.078 b/s` | `3.068 b/s` (`27 MB/s` dec) | `3.068 b/s` (`159 MB/s` dec) | **`2.940 b/s` (`5.44×`)** (`117 MB/s`) | **`0.6%` (`0.02 b/s`, `888×`)** → **`55.3 dB`** (`318 MB/s`) | **`6/6` exact** (`3.068–3.113 b/s`) |

See **[`PHASE3_REPORT.md`](PHASE3_REPORT.md)** for full 7-layer R-D tables (`k=6..0` bytes, `PSNR`, $L_\infty$, decode ms), `CCSDS-123 + 3D-rANS` / `CharLS JPEG-LS` / `ZFP` / `zstd -19` baselines, `265.67 MB` real `int32` overflow-safe results, non-hyperspectral 3D microscopy & MRI controls (including honest domain boundaries), Peak RSS measurements, NEON↔scalar SHA-256 parity, and 20,200-iteration UBSan + `libc++` debug fuzzing.

```bash
# Build all C++20 binaries (./asv4, ./hnd2, ./hnd3) and run UBSan + cross-language verification
make
make test

# Encode & decode a 3D hyperspectral cube (7-layer quality-embedded lossless .hnd3)
./hnd3 enc cube.raw cube.hnd3 --shape 204,512,217 --dtype i16 --base-k 6
./hnd3 dec cube.hnd3 cube_lossless.raw              # Full bit-exact decode + SHA-256 check
./hnd3 dec cube.hnd3 preview.raw --stop-layer 6     # Instant 60 dB base preview (~348 MB/s)

# Pure C++20 blind mode (zero shape hints, automatic BSQ/BIL/BIP stride & spectral-axis discovery)
./hnd3 enc headerless.raw blind.hnd3 --blind --dtype u16 --base-k 0
./hnd3 dec blind.hnd3 restored.raw
```

---

## Quickstart

### Build (macOS Apple Silicon or Linux x86-64/ARM64)
Requires a C++20 compiler (`clang++` or `g++`) and `libzstd` (`>= 1.5`). The top-level [`Makefile`](Makefile) auto-detects Homebrew, Conda, or system `libzstd` paths and builds `./asv4` (from [`v3/cpp/asv3.cpp`](v3/cpp/asv3.cpp), which retains the `v3/` directory for historical continuity).

```bash
# Build ./asv4 (and v3/cpp/asv3)
make

# Run UBSan + portable FIPS 180-4 SHA-256 + Unix pipe smoke test
make test
```

### Usage (Files or Unix Pipes)
```bash
# Compress & decompress files (automatic streaming SHA-256 verification on decode)
./asv4 c --level 3 model.safetensors model.asv4
./asv4 d model.asv4 model.restored

# Pipe-safe streaming (zero backward seeks)
cat model.safetensors | ./asv4 c --level 3 | ./asv4 d > /dev/null

# Archival preset (--level 19, guaranteed <= --level 3) with JSON stats & trace
./asv4 c --level 19 --stats stats.json --trace input.bin output.asv4
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
| [`Makefile`](Makefile) | Auto-detecting C++20 build (`make` → `./asv4`, `./hnd2`, `./hnd3`; `make bench`; `make test`) |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System architecture & Mermaid diagrams for both ASV4 v4.1 and HND3 (Phases 0–3) |
| [`cpp/hnd3.cpp`](cpp/hnd3.cpp) | **Phase 3 zero-dependency C++20 NEON HND3 engine** (16-stripe rANS, `vmlal_s32` LDU, native `--blind`) |
| [`cpp/hnd2.cpp`](cpp/hnd2.cpp) | Phase 2 C++20 NEON HND2 engine (7-layer quality-embedded lossless) |
| [`phase3_codec.py`](phase3_codec.py), [`phase2_codec.py`](phase2_codec.py), [`phase1_codec.py`](phase1_codec.py) | Wire-compatible Python+Numba reference codecs for Phases 1–3 |
| [`phase0_harness.py`](phase0_harness.py) | Phase 0 falsification harness v2.2 (Blue Book CCSDS-123, real Rice, Gates 0A–0D, Track B) |
| [`test_phase0.py`](test_phase0.py), [`test_phase1.py`](test_phase1.py), [`test_phase2.py`](test_phase2.py), [`test_phase3.py`](test_phase3.py) | Automated verification suites (transforms, CCSDS-123, blind strides, Py↔C++ cross-interop) |
| [`PHASE0_REPORT.md`](PHASE0_REPORT.md), [`PHASE1_REPORT.md`](PHASE1_REPORT.md), [`PHASE2_REPORT.md`](PHASE2_REPORT.md), [`PHASE3_REPORT.md`](PHASE3_REPORT.md) | Staged empirical falsification & real-cube benchmark reports (Phases 0–3) |
| [`v3/cpp/asv3.cpp`](v3/cpp/asv3.cpp) | **Production C++20 NEON ASV4 v4.1 streaming encoder & hardened decoder** |
| [`v3/cpp/asv3_v3.0.cpp`](v3/cpp/asv3_v3.0.cpp) | Frozen `v3.0` C++20 NEON reference snapshot |
| [`v3/cpp/zstd_capi_bench.cpp`](v3/cpp/zstd_capi_bench.cpp) | Exact in-process single-threaded `libzstd` C API baseline harness (`B2_capi` / `B3a_capi`) |
| [`v3/download_real_corpus.py`](v3/download_real_corpus.py) | Downloads & builds all 6 real public benchmark datasets in `v3/corpus_real/` |
| [`v3/build_corpus.py`](v3/build_corpus.py) | Builds the original 10-file pre-registered corpus in `v3/corpus/` |
| [`v3/REPORT_V4.md`](v3/REPORT_V4.md) | Complete **ASV4 v4.1** technical report (16 datasets, OpenZL/Blosc2/ZipNN comparisons, theoretical analysis) |
| [`v3/REPORT_V3.md`](v3/REPORT_V3.md) | Staged Phase 0 → 1A → 1B → 2 build report & Kill Gate 1B falsification record |
| [`v3/AUDIT_V3.md`](v3/AUDIT_V3.md) | Forensic audit artifacts (corpus provenance, C-API baselines, divergence traces, OpenZL chunk-32 fix) |
| [`v3/final_refresh_results_v4.1.csv`](v3/final_refresh_results_v4.1.csv) | Canonical 16-dataset `v4.1` L3 & L19 CSV (plus archived `v3.0` and `v4.0` CSVs) |
| [`v3/codec.py`](v3/codec.py), [`v3/transforms.py`](v3/transforms.py) | Phase 1A Python/Numba reference implementation (wire-compatible with `v3`) |
| [`v3/stage3_forest.py`](v3/stage3_forest.py) | Phase 1B Stage 3 delta-forest ablation (falsified & cut per Kill Gate 1B) |
| [`v3/bench_final_refresh.py`](v3/bench_final_refresh.py) | Same-binary 16-dataset L3 & L19 wall-clock / Peak RSS benchmark driver |
| [`ftcc/`](ftcc/) | Initial v1 forest-tree energy-clumping prototype (superseded by v3/v4) |
