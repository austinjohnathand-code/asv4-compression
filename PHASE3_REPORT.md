# Phase 3 Technical Report (`phase3-v3.2`) — HND3 3D Hyperspectral Codec

HND3 ([`cpp/hnd3.cpp`](cpp/hnd3.cpp) → `./hnd3`, wire-compatible with [`phase3_codec.py`](phase3_codec.py)) is a zero-dependency C++20 reversible 3D tensor codec for hyperspectral cubes (`i16`, `u16`, `i32`). It combines complete-pivoted integer `LDU`-RKLT spectral decorrelation (ARM64 NEON `vmlal_s32` for 16-bit inputs, scalar `int64` for 32-bit inputs), arbitrary-shape 2D 5/3 integer lifting, 16-stripe band-parallel 32-bit rANS entropy coding, optional 7-layer progressive bitplane refinement (`base_k = 6`), built-in FIPS 180-4 SHA-256 verification, and native C++20 blind 3D stride/spectral-axis recovery (`--blind`).

---

## 1. Scope, Known Limitations, & Benchmark Environment

Putting limitations upfront:
- **3D only**: HND3 operates strictly on 3D tensors `(D0, D1, D2)`. Higher-dimensional volumes (such as 4D `cells3d.tif`) must be sliced or reshaped to 3D externally.
- **High working-set RAM (no tiling/streaming)**: Full-cube `int64` transforms and subband buffers require `~17–22×` raw `int16` size (`~1.0 GB` for a `45 MB` cube, `~2.9 GB` for a `133 MB` cube). Multi-gigabyte flight lines require spatial tiling (not yet implemented).
- **No guaranteed-$L_\infty$ near-lossless mode**: Progressive layers (`k = 6..1`) reduce MSE/SAM smoothly, but inverse `LDU`-RKLT amplification prevents tight $L_\infty$ guarantees at intermediate layers (unlike CCSDS-123.0-B-2).
- **No preview integrity checksum**: Built-in SHA-256 covers exact lossless reconstruction (`stop_layer = 0`) only; lossy previews (`stop_layer > 0`) print `SHA256=N/A (preview)`.
- **No CCSDS-123.0-B-2 reference run**: Official comparisons below use **UAB/ESA Empordà (CCSDS-123.0-B-1, Issue 1)**; CCSDS-123.0-B-2 (Issue 2 hybrid entropy coder & near-lossless) was not run.
- **`--blind` stride horizon**: Windowing the FFT ACF to `min(N, 4_194_304)` samples caps detectable strides at `≤ 1,048,576` (BSQ frames larger than `~1024×1024` exceed the window).
- **Platform & fuzzing scope**: Benchmarks, UBSan, `-DHND3_FORCE_SCALAR` parity, and mutation fuzzing ran on **Apple M3 Max (16 logical CPUs, 128 GiB RAM, macOS 27.2.1, Apple Clang 16.0.0)**. GitHub Actions CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) covers `ubuntu-latest` (`x86_64`) and `macos-latest` (`arm64`). Because Apple Clang 16's macOS ASan runtime crashes before `main()` on Darwin 27 (`sanitizer_malloc_mac.inc:189`), local fuzzing used UBSan + `libc++` debug hardening (`_LIBCPP_HARDENING_MODE_DEBUG`) rather than ASan or coverage-guided `libFuzzer`.
- **Units & reproducibility**: All HND3 sizes, Peak RSS (`/usr/bin/time -l`), and throughputs use **decimal `MB = 10^6 B`**. Every table below is reproducible via [`bench/fetch_hnd3_data.py`](bench/fetch_hnd3_data.py) (SHA-256 dataset manifest) and [`bench/bench_hnd3.py`](bench/bench_hnd3.py) (`make bench-hnd3`). All timings report best-of-3 runs (`--no-verify` for pure encode throughput; default CLI runs encode-then-verify before writing).

---

## 2. Uncropped Apples-to-Apples Baselines (4 Hyperspectral Cubes)

All codecs were evaluated on the exact same uncropped native cubes, in real encoded bytes, with verified bit-exact round-trip:
- **Official UAB/ESA Empordà (`emporda.jar`, OpenJDK 21)**: CCSDS-123.0-B-1 Issue 1 reference (`P=3` & `P=15` Sample-Adaptive, `P=3` Block-Adaptive, full prediction, neighbor-oriented local sum, BSQ).
- **Custom CCSDS-123.0-B-1 (Numba, [`phase0_harness.py`](phase0_harness.py))**: Included for transparency alongside Empordà; note Empordà achieves `0.07–0.29 b/s` better rates than our custom Numba implementation.
- **Official OpenJPEG (`imagecodecs 2023.1.23`, J2K Part 1 reversible)**: Per-band lossless raw, plus our **`LDU`-RKLT + per-band OpenJPEG J2K (Part 1)** (exact serialized `LDU` spectral header + per-band lossless `.jp2`; note this is *not* a Part 2 MCT codestream). Timed both **1T** and **16-thread MT** (`ThreadPoolExecutor` across bands) for fair speed comparison.
- **Official JPEG-XL (`cjxl`/`djxl` v0.13.0, `-d 0 -e 7`)**: Per-band lossless raw, plus **`LDU`-RKLT + per-band `cjxl`** (16-thread parallel across bands).
- **Official CharLS JPEG-LS** (1T & 16T MT), **Official ZFP Reversible 3D (`i32` mode)**, and **`zstd -19` (2-byte shuffled)**.

### 2.1 Indian Pines `(200, 145, 145)` — `8,410,000 B` (`8.41 MB`, `i16`)

| Codec / Method | Progressive? | Bytes | Rate (`b/s`) | Ratio | Enc `1T / MT` (MB/s) | Dec `1T / MT` (MB/s) |
|---|---|---:|---:|---:|---:|---:|
| `Custom CCSDS-123 (P=15) + 3D-rANS` | No | `3,356,782` | `6.386` | `2.51×` | (Numba) | — |
| **HND3 (`k=0`, 1-shot, `auto`)** | Spatial | `3,408,480` | `6.485` | `2.47×` | `51.0 / 115.2` | `69.5 / 205.1` |
| `Empordà CCSDS-123.0-B-1 (P=15, SA)` *(official)* | No | `3,440,472` | `6.545` | `2.44×` | `8.4 / —` | `6.9 / —` |
| `LDU-RKLT + per-band cjxl 0.13 (-d 0 -e 7)` | Per-band | `3,446,892` | `6.558` | `2.44×` | `— / 9.6` | `— / 12.4` |
| **HND3 (`k=6`, 7-layer, `auto`)** | **7-layer SNR** | `3,448,539` | `6.561` | `2.44×` | `21.2 / 80.1` | `28.2 / 125.5` |
| `Empordà CCSDS-123.0-B-1 (P=3, SA)` *(official)* | No | `3,457,844` | `6.579` | `2.43×` | `10.4 / —` | `7.7 / —` |
| `Empordà CCSDS-123.0-B-1 (P=3, BA)` *(official)* | No | `3,515,340` | `6.688` | `2.39×` | `9.6 / —` | `7.6 / —` |
| `LDU-RKLT + per-band OpenJPEG J2K (Part 1)` | Per-band | `3,542,254` | `6.739` | `2.37×` | `11.1 / 27.9` | `22.2 / 167.0` |
| `Custom CCSDS-123.0-B-1 (P=15, Rice)` | No | `3,559,591` | `6.772` | `2.36×` | (Numba) | — |
| `JPEG-XL cjxl 0.13` *(per-band raw)* | Per-band | `4,007,907` | `7.625` | `2.10×` | `— / 8.9` | `— / 8.5` |
| `CharLS JPEG-LS` *(per-band)* | No | `4,165,882` | `7.926` | `2.02×` | `89.7 / 753.5` | `89.2 / 698.1` |
| `OpenJPEG J2K Part 1` *(per-band raw)* | Per-band | `4,236,668` | `8.060` | `1.99×` | `14.5 / 163.8` | `19.7 / 222.5` |
| `zstd -19` *(byte-shuffled 2B)* | No | `4,485,477` | `8.534` | `1.87×` | `6.8 / —` | `1,434.9 / —` |
| `ZFP Reversible 3D` *(official, `i32`)* | No | `6,399,544` | `12.175` | `1.31×` | `229.6 / —` | `264.7 / —` |

### 2.2 Salinas `(204, 512, 217)` — `45,330,432 B` (`45.33 MB`, `i16`)

| Codec / Method | Progressive? | Bytes | Rate (`b/s`) | Ratio | Enc `1T / MT` (MB/s) | Dec `1T / MT` (MB/s) |
|---|---|---:|---:|---:|---:|---:|
| `LDU-RKLT + per-band cjxl 0.13 (-d 0 -e 7)` | Per-band | `11,455,684` | `4.043` | `3.96×` | `— / 18.2` | `— / 17.7` |
| **HND3 (`k=6`, 7-layer, `auto`)** | **7-layer SNR** | `13,111,158` | `4.628` | `3.46×` | `15.7 / 54.4` | `20.7 / 105.9` |
| **HND3 (`k=0`, 1-shot, `auto`)** | Spatial | `13,194,691` | `4.657` | `3.44×` | `27.6 / 68.0` | `37.9 / 157.4` |
| `Custom CCSDS-123 (P=15) + 3D-rANS` | No | `13,583,657` | `4.795` | `3.34×` | (Numba) | — |
| `LDU-RKLT + per-band OpenJPEG J2K (Part 1)` | Per-band | `13,608,877` | `4.803` | `3.33×` | `16.3 / 46.1` | `22.1 / 62.7` |
| `Empordà CCSDS-123.0-B-1 (P=15, SA)` *(official)* | No | `13,779,056` | `4.864` | `3.29×` | `11.3 / —` | `9.2 / —` |
| `Empordà CCSDS-123.0-B-1 (P=3, SA)` *(official)* | No | `13,827,872` | `4.881` | `3.28×` | `14.0 / —` | `11.4 / —` |
| `Empordà CCSDS-123.0-B-1 (P=3, BA)` *(official)* | No | `14,372,552` | `5.073` | `3.15×` | `13.9 / —` | `11.1 / —` |
| `Custom CCSDS-123.0-B-1 (P=15, Rice)` | No | `14,471,088` | `5.108` | `3.13×` | (Numba) | — |
| `JPEG-XL cjxl 0.13` *(per-band raw)* | Per-band | `14,654,924` | `5.173` | `3.09×` | `— / 22.2` | `— / 21.1` |
| `zstd -19` *(byte-shuffled 2B)* | No | `17,156,071` | `6.055` | `2.64×` | `5.1 / —` | `893.5 / —` |
| `CharLS JPEG-LS` *(per-band)* | No | `17,674,288` | `6.238` | `2.56×` | `101.1 / 1242.5` | `106.6 / 1306.3` |
| `OpenJPEG J2K Part 1` *(per-band raw)* | Per-band | `19,138,425` | `6.755` | `2.37×` | `16.3 / 212.1` | `23.5 / 293.4` |
| `ZFP Reversible 3D` *(official, `i32`)* | No | `30,156,576` | `10.644` | `1.50×` | `237.3 / —` | `294.9 / —` |

### 2.3 PaviaU `(103, 610, 340)` — `42,724,400 B` (`42.72 MB`, `i16`)

| Codec / Method | Progressive? | Bytes | Rate (`b/s`) | Ratio | Enc `1T / MT` (MB/s) | Dec `1T / MT` (MB/s) |
|---|---|---:|---:|---:|---:|---:|
| **HND3 (`k=6`, 7-layer, `auto`)** | **7-layer SNR** | `16,967,890` | `6.354` | `2.52×` | `19.8 / 84.1` | `25.6 / 125.3` |
| **HND3 (`k=0`, 1-shot, `auto`)** | Spatial | `17,000,291` | `6.366` | `2.51×` | `41.4 / 116.1` | `49.8 / 175.1` |
| `Empordà CCSDS-123.0-B-1 (P=3, SA)` *(official)* | No | `17,492,744` | `6.551` | `2.44×` | `13.6 / —` | `10.4 / —` |
| `LDU-RKLT + per-band cjxl 0.13 (-d 0 -e 7)` | Per-band | `17,512,133` | `6.558` | `2.44×` | `— / 30.3` | `— / 32.8` |
| `Custom CCSDS-123 (P=15) + 3D-rANS` | No | `17,525,157` | `6.563` | `2.44×` | (Numba) | — |
| `Empordà CCSDS-123.0-B-1 (P=15, SA)` *(official)* | No | `17,677,344` | `6.620` | `2.42×` | `11.4 / —` | `9.0 / —` |
| `LDU-RKLT + per-band OpenJPEG J2K (Part 1)` | Per-band | `17,729,879` | `6.640` | `2.41×` | `14.8 / 68.6` | `21.4 / 98.8` |
| `Empordà CCSDS-123.0-B-1 (P=3, BA)` *(official)* | No | `18,007,080` | `6.744` | `2.37×` | `13.4 / —` | `10.7 / —` |
| `Custom CCSDS-123.0-B-1 (P=15, Rice)` | No | `18,442,758` | `6.907` | `2.32×` | (Numba) | — |
| `zstd -19` *(byte-shuffled 2B)* | No | `24,068,251` | `9.013` | `1.78×` | `3.8 / —` | `2,387.8 / —` |
| `JPEG-XL cjxl 0.13` *(per-band raw)* | Per-band | `24,438,125` | `9.152` | `1.75×` | `— / 26.2` | `— / 26.8` |
| `CharLS JPEG-LS` *(per-band)* | No | `24,829,636` | `9.299` | `1.72×` | `85.8 / 1030.7` | `82.4 / 999.7` |
| `OpenJPEG J2K Part 1` *(per-band raw)* | Per-band | `25,892,643` | `9.697` | `1.65×` | `12.8 / 160.3` | `18.0 / 227.4` |
| `ZFP Reversible 3D` *(official, `i32`)* | No | `33,026,800` | `12.368` | `1.29×` | `232.0 / —` | `273.7 / —` |

### 2.4 Out-of-Sample Holdout: NASA AVIRIS `(224, 745, 398)` — `132,836,480 B` (`132.84 MB`, `i16`)

*Source: HuggingFace `danjacobellis/aviris_1k_val` (`validation-00003-of-00066.parquet`, row 0), prime spatial factors `745 = 5×149`, `398 = 2×199`. Provenance caveat: the parquet stored raw big-endian NASA AVIRIS `>i2` bytes decoded as `<i2`; we detected this after the first run because `zstd -19` compressed the unswapped bytes to `3.57 b/s` while all spectral codecs stalled above `10.5 b/s`, and byte-swapped to restore authentic radiance in `[-50, 2033]` (`bench/fetch_hnd3_data.py` reproduces this deterministically with SHA-256 verification).*

| Codec / Method | Progressive? | Bytes | Rate (`b/s`) | Ratio | Enc `1T / MT` (MB/s) | Dec `1T / MT` (MB/s) |
|---|---|---:|---:|---:|---:|---:|
| `LDU-RKLT + per-band cjxl 0.13 (-d 0 -e 7)` | Per-band | `23,808,693` | `2.868` | `5.58×` | `— / 20.0` | `— / 18.9` |
| `Custom CCSDS-123 (P=15) + 3D-rANS` | No | `23,989,094` | `2.889` | `5.54×` | (Numba) | — |
| **HND3 (`k=6`, 7-layer, `auto`)** | **7-layer SNR** | `24,408,930` | `2.940` | `5.44×` | `16.6 / 73.1` | `22.5 / 116.0` |
| `Empordà CCSDS-123.0-B-1 (P=15, SA)` *(official)* | No | `24,965,668` | `3.007` | `5.32×` | `15.1 / —` | `12.2 / —` |
| `Empordà CCSDS-123.0-B-1 (P=3, SA)` *(official)* | No | `25,021,220` | `3.014` | `5.31×` | `20.1 / —` | `15.3 / —` |
| **HND3 (`k=0`, 1-shot, `auto`)** | Spatial | `25,468,981` | `3.068` | `5.22×` | `33.2 / 93.7` | `40.7 / 157.4` |
| `LDU-RKLT + per-band OpenJPEG J2K (Part 1)` | Per-band | `25,473,453` | `3.068` | `5.21×` | `22.7 / 52.8` | `28.6 / 59.8` |
| `Custom CCSDS-123.0-B-1 (P=15, Rice)` | No | `25,554,044` | `3.078` | `5.20×` | (Numba) | — |
| `JPEG-XL cjxl 0.13` *(per-band raw)* | Per-band | `25,579,812` | `3.081` | `5.19×` | `— / 22.8` | `— / 21.9` |
| `Empordà CCSDS-123.0-B-1 (P=3, BA)` *(official)* | No | `26,580,196` | `3.202` | `5.00×` | `18.9 / —` | `16.3 / —` |
| `OpenJPEG J2K Part 1` *(per-band raw)* | Per-band | `27,620,094` | `3.327` | `4.81×` | `31.9 / 416.0` | `44.7 / 562.9` |
| `CharLS JPEG-LS` *(per-band)* | No | `27,911,424` | `3.362` | `4.76×` | `144.4 / 1595.1` | `158.5 / 1687.5` |
| `zstd -19` *(byte-shuffled 2B)* | No | `29,650,031` | `3.571` | `4.48×` | `6.1 / —` | `1,520.3 / —` |
| `ZFP Reversible 3D` *(official, `i32`)* | No | `70,300,032` | `8.468` | `1.89×` | `236.4 / —` | `351.6 / —` |

### Summary of Baseline Trade-offs
- **vs. Official UAB/ESA Empordà (CCSDS-123.0-B-1)**: HND3 (`k=6`) improves compression over official Empordà on Salinas (`-0.236 b/s`, `4.9%`), PaviaU (`-0.197 b/s`, `3.0%`), and AVIRIS (`-0.067 b/s`, `2.2%`), while trailing Empordà `P=15` by `+0.016 b/s` on Indian Pines (`k=0` beats Empordà by `-0.060 b/s`), with 7-layer progressive decoding and higher throughput (`21–28 MB/s` 1T / `106–126 MB/s` MT vs. Empordà Java `7–15 MB/s` 1T).
- **vs. `LDU`-RKLT + per-band OpenJPEG**: HND3 (`k=6`) beats `LDU + OpenJPEG` in bitrate on all 4 cubes (`-0.13` to `-0.29 b/s`). At 1T, `k=6` decode (`21–28 MB/s`) is roughly on par with `LDU + OpenJPEG` (`21–29 MB/s`), while `k=0` (`38–70 MB/s`) is `1.4×–3.1×` faster. At 16T MT, HND3 `k=6` (`106–126 MB/s`) is `1.3×–1.9×` faster on large cubes (`Salinas`, `PaviaU`, `AVIRIS`), while per-band OpenJPEG without `LDU` reaches `222–563 MB/s` MT at `1.5–3.3 b/s` higher bitrate.
- **vs. `LDU`-RKLT + per-band JPEG-XL (`cjxl 0.13 -d 0 -e 7`)**: `LDU + cjxl` achieves the smallest files on Salinas (`4.043` vs. `4.628 b/s`) and AVIRIS (`2.868` vs. `2.940 b/s`), ties `k=6` on Indian Pines (`6.558` vs. `6.561 b/s`), and trails HND3 on PaviaU (`6.558` vs. `6.354 b/s`), at `3.8×–10×` slower MT decode (`12–33 MB/s`) and without single-stream 3D progressive layers.

---

## 3. Complete-Pivoted `LDU`-RKLT Rounding-Noise Audit

Measuring exact same-matrix integer lifting error $e = Y_{\text{LDU}} - U_{\text{KLT}}(X - \mu)$ against `float64` KLT (with exact empirical discrete zero-order histogram entropy $\Delta H_{\text{disc}} = H_{\text{disc}}(Y_{\text{LDU}}) - H_{\text{disc}}(\mathrm{round}(Y_{\text{flt}}))$, verified `0` Givens fallbacks — `complete_pivoted_ldu` has no Givens fallback branch):

| Cube / Stress Test | `auto` Mode | `max\|L\|` | `max\|U\|` | `max\|c_k\|` | `max\|1/c_k\|` | `RMS(e)` | `Var(e)` | `H_flt` (`b/s`) | `H_ldu` (`b/s`) | `ΔH_disc` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Indian Pines** (`200×200`) | `clustered`* | `1.0000` | `1.0000` | `1.533` | `1.420` | `1.054` | `1.111` | `6.379` | `6.392` | `+0.0130` |
| **Salinas** (`204×204`) | `full` | `1.0000` | `1.0000` | `1.474` | `1.614` | `1.082` | `1.171` | `4.616` | `4.671` | `+0.0547` |
| **PaviaU** (`103×103`) | `full` | `1.0000` | `1.0000` | `2.037` | `2.317` | `1.345` | `1.809` | `6.313` | `6.347` | `+0.0343` |
| **AVIRIS Holdout** (`224×224`) | `full` | `1.0000` | `1.0000` | `1.671` | `1.604` | `0.970` | `0.941` | `2.513` | `2.854` | `+0.3405` |
| **Synth 60 dB** (`204×204`, `σ_n=1`) | `clustered`* | `1.0000` | `1.0000` | `3.502` | `2.798` | `2.054` | `4.217` | `2.490` | `3.389` | `+0.8985` |

*\*Rows measure full `B×B` `LDU` vs. full `float64` KLT. In `auto` mode, Indian Pines selects `clustered` (`M ≤ 8` groups, keeping `|L|, |U| ≤ 1`), whose pre-wavelet zero-order entropy is `9.273 b/s` because cluster heads retain spatial low-frequency energy subsequently removed by the 2D 5/3 wavelet. Complete pivoting eliminates Phase 0's `> 10^4` Schur-complement blowup, while leaving an `O(1)`-LSB integer-lifting rounding floor (`RMS ≈ 0.97–2.05`) that increases discrete entropy by `+0.34`–`+0.90 b/s` when tail-PC noise is `≤ 1` count.*

---

## 4. 7-Layer Progressive Rate–Distortion Curves (`k = 6 → 0`, with SAM)

Exact cumulative bytes per layer, `PSNR_rng = 20·log10((max − min) / RMSE)`, mean and 99th-percentile **Spectral Angle Mapper (`SAM` in degrees)**, max absolute error $L_\infty$, and best-of-3 decode latency (16T MT & 1T):

### 4.1 Salinas `(204, 512, 217)` (`45.33 MB`, `dyn = 9,218`, `LDU_hdr = 87,766 B`)

| Layer | Cum. Bytes | `%` Stream | `b/s` | Ratio | `PSNR_rng` | `SAM_mean` | `SAM_p99` | $L_\infty$ | MT `ms` (`MB/s`) | 1T `MB/s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k=6` (Base) | `451,494` | `3.4%` | `0.159` | `100.40×` | `60.11 dB` | `0.339°` | `0.694°` | `178` | `137.0` (`330.4`) | `61.7` |
| `k=5` | `835,924` | `6.4%` | `0.295` | `54.23×` | `62.61 dB` | `0.258°` | `0.475°` | `72` | `147.0` (`309.4`) | `49.0` |
| `k=4` | `1,863,641` | `14.2%` | `0.658` | `24.32×` | `65.56 dB` | `0.184°` | `0.327°` | `37` | `174.0` (`260.3`) | `42.2` |
| `k=3` | `4,009,579` | `30.6%` | `1.415` | `11.31×` | `69.10 dB` | `0.123°` | `0.221°` | `20` | `199.0` (`227.9`) | `34.9` |
| `k=2` | `7,205,748` | `55.0%` | `2.543` | `6.29×` | `72.84 dB` | `0.080°` | `0.145°` | `13` | `229.0` (`197.6`) | `29.4` |
| `k=1` | `10,486,368` | `80.0%` | `3.701` | `4.32×` | `75.48 dB` | `0.059°` | `0.109°` | `13` | `270.0` (`167.6`) | `24.5` |
| `k=0` (Lossless) | `13,111,158` | `100.0%` | `4.628` | `3.46×` | `∞` | `0.000°` | `0.000°` | `0` | `415.0` (`109.1`) | `21.5` |

### 4.2 PaviaU `(103, 610, 340)` (`42.72 MB`, `dyn = 8,000`, `LDU_hdr = 23,504 B`)

| Layer | Cum. Bytes | `%` Stream | `b/s` | Ratio | `PSNR_rng` | `SAM_mean` | `SAM_p99` | $L_\infty$ | MT `ms` (`MB/s`) | 1T `MB/s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k=6` (Base) | `2,047,876` | `12.1%` | `0.767` | `20.86×` | `52.89 dB` | `0.784°` | `3.398°` | `228` | `92.0` (`461.9`) | `85.2` |
| `k=5` | `3,729,715` | `22.0%` | `1.397` | `11.46×` | `57.50 dB` | `0.463°` | `2.020°` | `99` | `115.0` (`372.0`) | `61.8` |
| `k=4` | `6,026,857` | `35.5%` | `2.257` | `7.09×` | `62.73 dB` | `0.255°` | `1.138°` | `37` | `140.0` (`305.2`) | `47.1` |
| `k=3` | `8,805,731` | `51.9%` | `3.298` | `4.85×` | `67.47 dB` | `0.148°` | `0.665°` | `20` | `162.0` (`263.6`) | `37.0` |
| `k=2` | `11,836,251` | `69.8%` | `4.433` | `3.61×` | `70.87 dB` | `0.100°` | `0.449°` | `17` | `210.0` (`203.2`) | `31.4` |
| `k=1` | `14,664,950` | `86.4%` | `5.492` | `2.91×` | `72.68 dB` | `0.081°` | `0.365°` | `16` | `215.0` (`198.6`) | `28.1` |
| `k=0` (Lossless) | `16,967,890` | `100.0%` | `6.354` | `2.52×` | `∞` | `0.000°` | `0.000°` | `0` | `348.0` (`122.9`) | `24.6` |

### 4.3 Indian Pines `(200, 145, 145)` (`8.41 MB`, `dyn = 8,649`, `LDU_hdr = 12,294 B`)

| Layer | Cum. Bytes | `%` Stream | `b/s` | Ratio | `PSNR_rng` | `SAM_mean` | `SAM_p99` | $L_\infty$ | MT `ms` (`MB/s`) | 1T `MB/s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k=6` (Base) | `564,412` | `16.4%` | `1.074` | `14.90×` | `54.07 dB` | `0.316°` | `0.415°` | `146` | `14.0` (`605.2`) | `132.9` |
| `k=5` | `863,402` | `25.0%` | `1.643` | `9.74×` | `59.04 dB` | `0.178°` | `0.228°` | `66` | `24.0` (`355.2`) | `81.9` |
| `k=4` | `1,258,515` | `36.5%` | `2.394` | `6.68×` | `63.84 dB` | `0.103°` | `0.130°` | `34` | `27.0` (`313.2`) | `59.8` |
| `k=3` | `1,804,938` | `52.3%` | `3.434` | `4.66×` | `68.61 dB` | `0.059°` | `0.076°` | `17` | `36.0` (`231.3`) | `43.9` |
| `k=2` | `2,423,203` | `70.3%` | `4.610` | `3.47×` | `73.17 dB` | `0.035°` | `0.044°` | `11` | `39.0` (`213.0`) | `36.8` |
| `k=1` | `2,991,355` | `86.7%` | `5.691` | `2.81×` | `76.11 dB` | `0.025°` | `0.032°` | `8` | `44.0` (`190.9`) | `32.3` |
| `k=0` (Lossless) | `3,448,539` | `100.0%` | `6.561` | `2.44×` | `∞` | `0.000°` | `0.000°` | `0` | `78.0` (`107.5`) | `27.7` |

### 4.4 AVIRIS Holdout `(224, 745, 398)` (`132.84 MB`, `dyn = 2,083`, `LDU_hdr = 105,330 B`)

| Layer | Cum. Bytes | `%` Stream | `b/s` | Ratio | `PSNR_rng` | `SAM_mean` | `SAM_p99` | $L_\infty$ | MT `ms` (`MB/s`) | 1T `MB/s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k=6` (Base) | `149,607` | `0.6%` | `0.018` | `887.90×` | `55.27 dB` | `0.497°` | `1.578°` | `95` | `397.0` (`334.3`) | `54.6` |
| `k=5` | `301,394` | `1.2%` | `0.036` | `440.74×` | `56.90 dB` | `0.423°` | `1.497°` | `59` | `453.0` (`293.1`) | `52.1` |
| `k=4` | `741,323` | `3.0%` | `0.089` | `179.19×` | `58.97 dB` | `0.350°` | `1.496°` | `43` | `504.0` (`263.4`) | `45.3` |
| `k=3` | `2,160,341` | `8.9%` | `0.260` | `61.49×` | `59.67 dB` | `0.335°` | `1.623°` | `20` | `539.0` (`246.2`) | `39.3` |
| `k=2` | `6,819,310` | `27.9%` | `0.821` | `19.48×` | `61.39 dB` | `0.262°` | `1.357°` | `14` | `616.0` (`215.8`) | `33.1` |
| `k=1` | `16,009,012` | `65.6%` | `1.928` | `8.30×` | `62.65 dB` | `0.216°` | `0.282°` | `13` | `707.0` (`188.0`) | `26.7` |
| `k=0` (Lossless) | `24,408,930` | `100.0%` | `2.940` | `5.44×` | `∞` | `0.000°` | `0.000°` | `0` | `1,162.0` (`114.3`) | `22.6` |

*Note: Truncating an `.hnd3` file at any layer boundary (verified in `test_phase3.py`) decodes identically to `--stop-layer k` on the full file. At `k=2 → k=1`, $L_\infty$ plateaus (`13→13`, `17→16`, `14→13`) because inverse `LDU`-RKLT distributes 1-bit wavelet rounding across bands.*

---

## 5. Full Uncropped Native `--blind` Recovery (`k=0`, All 6/6 Permutations)

| Uncropped Cube (`Size`) | Perms (`BHW/BWH/HBW/HWB/WBH/WHB`) | Exact 3D Strides `[1, s1, s2]` | True Spectral Axis | Blind Rate (`b/s`) | Wall Time | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| **Indian Pines** `(200, 145, 145)` (`8.41 MB`) | `6/6` | `6/6` | `6/6` | `6.485–6.486` | `2.59–2.76 s` | `571–575 MB` |
| **Salinas** `(204, 512, 217)` (`45.33 MB`) | `6/6` | `6/6` | `6/6` | `4.654–4.657` | `3.16–3.46 s` | `1,384–1,387 MB` |
| **PaviaU** `(103, 610, 340)` (`42.72 MB`) | `6/6` | `6/6` | `6/6` | `6.349–6.367` | `2.90–3.17 s` | `1,344–1,348 MB` |
| **AVIRIS Holdout** `(224, 745, 398)` (`132.84 MB`) | `6/6` | `6/6` | `6/6` | `3.068–3.113` | `3.46–4.42 s` | `3,119–3,129 MB` |
| **Incompressible Noise** `(16, 32, 32)` | 1D raw | `[1]` raw fallback | n/a | `< +0.2%` tax | `< 0.01 s` | `< 20 MB` |

---

## 6. True Wide-Range `int32`, Peak RSS, & Non-Hyperspectral 3D Controls

### 6.1 `int16` & True Wide-Range `int32` + Peak RSS Audit

After upgrading the entropy coder's escape path (`tok 30` for `[2^15, 2^16)`, `tok 31` 6-bit variable width up to `m = 62`, handling `|v| < 2^62`), both 16-bit-in-`i32` AND true full-range `int32` (`±1.9×10^9`, whose post-transform coefficients exceed `2^31`) round-trip bit-exact across C++ and Python:

| Dataset (`Raw Size`, `dtype`) | Mode | Size (B) | Rate (`b/s`) | Enc `1T / MT` (MB/s) | Dec `1T / MT` (MB/s) | Enc RSS | Dec RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| **Indian Pines** (`8.41 MB`, `i16`) | `k=0` / `k=6` | `3,408,480` / `3,448,539` | `6.485` / `6.561` | `50.6/115.1` / `20.8/78.2` | `68.2/194.0` / `27.6/124.4` | `203` / `205 MB` | `159` / `161 MB` |
| **Salinas** (`45.33 MB`, `i16`) | `k=0` / `k=6` | `13,194,691` / `13,111,158` | `4.657` / `4.628` | `29.1/67.8` / `15.6/55.3` | `40.7/155.4` / `21.8/112.3` | `1,040` / `1,034 MB` | `807` / `806 MB` |
| **PaviaU** (`42.72 MB`, `i16`) | `k=0` / `k=6` | `17,000,291` / `16,967,890` | `6.366` / `6.354` | `43.2/117.8` / `19.5/84.3` | `53.1/181.7` / `25.5/120.3` | `1,000` / `1,018 MB` | `755` / `755 MB` |
| **AVIRIS** (`132.84 MB`, `i16` NEON) | `k=0` / `k=6` | `25,468,981` / `24,408,930` | `3.068` / `2.940` | `33.3/93.3` / `16.8/72.7` | `41.1/155.5` / `22.5/117.1` | `2,959` / `2,924 MB` | `2,307` / `2,307 MB` |
| **AVIRIS** (`265.67 MB`, `i16`-in-`i32` scalar) | `k=0` / `k=6` | `25,468,981` / `24,408,930` | `3.068` / `2.940` | `17.7/74.0` / `14.1/70.0` | `18.8/93.7` / `15.9/83.6` | `3,088` / `3,063 MB` | `2,722` / `2,724 MB` |
| **Indian Pines × 40,000** (`16.82 MB`, `i32`, `|x|≤3.8e8`) | `k=0` / `k=6` | `13,190,422` / `11,955,456` | `25.095` / `22.745` | `86.3/166.0` / `50.1/134.0` | `103.7/244.3` / `61.5/200.2` | `240` / `239 MB` | `189` / `195 MB` |
| **Smooth Full-Range `i32`** (`0.52 MB`, `[-1.26e9, +1.90e9]`) | `k=0` / `k=6` | `431,439` / `401,581` | `26.333` / `24.511` | `88.9/92.0` / `48.2/53.0` | `107.9/158.7` / `54.5/75.9` | `9` / `11 MB` | `9` / `10 MB` |

### 6.2 Non-Hyperspectral 3D Controls

| 3D Control Volume | HND3 `k=0` | HND3 `k=6` | HND3 `--blind` (`k=0`) | `cjxl 0.13 (-d 0 -e 7)` | `OpenJPEG J2K` | `CharLS JPEG-LS` | `zstd -19` (2B shuf) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Cells3D Microscopy** `(60, 256, 256)` `u16` (`7.86 MB`) | `9.976 b/s` (`191 MB/s` MT) | `10.073 b/s` (`131 MB/s`) | `9.976 b/s` *(exact `(60,256,256)`, `sp=0`)* | **`9.618 b/s`** (`60 MB/s`) | `10.278 b/s` | `10.019 b/s` | `10.654 b/s` |
| **Brain MRI** `(10, 256, 256)` `u16` (`1.31 MB`) | `7.083 b/s` (`164 MB/s` MT) | `7.376 b/s` (`88 MB/s`) | `7.083 b/s` *(exact `(10,256,256)`, `sp=0`)* | **`1.883 b/s`** (`43 MB/s`) | `6.749 b/s` | `6.185 b/s` | `4.226 b/s` |

*Both non-hyperspectral controls are won in bitrate by JPEG-XL (`cjxl`), and sparse-background Brain MRI is also won by `zstd -19`, `JPEG-LS`, and `OpenJPEG`. Evaluating all candidate axes on an identical centered 3D subcube (plus a `SPEC_NONE` candidate) eliminated the earlier `+15%` blind-mode penalty on Brain MRI (`8.158 → 7.083 b/s`).*

---

## 7. Verification, Hardening, & Bug-Fix Record (`v3.2`)

- **Default-on encode verification**: `./hnd3 enc` decodes and checks exact byte equality against the input in memory before writing (disable with `--no-verify` for pure encode timing).
- **Hardening & regression tests (`test_phase3.py`, 15 tests across `test_phase1..3`)**:
  - Full-range `int32` (`±1.9×10^9`, `k=0` & `k=6`) C++ and Python wire-exact round-trip.
  - High-ratio constant all-zero `(64, 256, 256)` cube (`636×` compression) — resolved by replacing the `256×` ratio cap with spectral-header pre-validation, lazy per-level subband allocation (`LL` decoded first), and exact final rANS state verification (`state == RANS_L && wpos == -1`).
  - Strict CLI validation (`--dtype` required, exact file-size == `shape × itemsize`, `--spec-axis ∈ [0,2]`, `--base-k ∈ [0,15]`, `--levels ∈ [0,8]`, production binary excludes `fuzz`).
  - Degenerate shapes `(1,32,32)`, `(16,1,64)`, `(16,3,3)`, `(2,5,7)`, truncated layer-boundary decode, and 1T == 16T bit-exact encode/decode determinism.
- **NEON ↔ scalar parity (ARM64)**: `-DHND3_FORCE_SCALAR` produces bit-identical `.hnd3` output and cross-decodes identically to NEON on Apple M3 Max (`k=0` and `k=6`). Note: `enc` uses floating-point Jacobi eigenvectors, so exact cross-architecture encoder bitstream identity between ARM64 and x86-64 is not claimed (lossless cross-decode is pure integer math and covered by CI).
- **Mutation fuzzing**: `make test` builds a dedicated `-DHND3_ENABLE_FUZZ -fsanitize=undefined -fno-sanitize-recover=all -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG` test binary and runs 5,000 mutation iterations (`4,931` rejected, `69` accepted, `0` crashes/UBSan/hardened-bounds errors; accepted mutations flip unused padding bits in extra-bit byte tails or unused header bitmask bits without altering any decoded pixel, so SHA-256 passes). Note that `libc++` hardening checks `std::vector::operator[]` bounds, while raw-pointer loops rely on explicit bounds checks; full coverage-guided `libFuzzer` + ASan on Linux x86-64 remains future work.
