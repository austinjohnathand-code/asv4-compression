# Phase 2 Report — Unified Single-Stream Embedded `.hnd2` & C++20 NEON Production Engine

Phase 2 delivers two major milestones on top of Phase 1 (`phase1-v1.0`):

1. **Unified Single-Stream Quality-Embedded Format (`.hnd2`)**: Combines Phase 1's one-shot tightest lossless mode (`base_k = 0`) and a **7-layer $L_2$-aligned dyadic bitplane-embedded mode (`base_k = 6`)** inside one clean binary wire specification. Truncating a single lossless `.hnd2` file at any layer boundary `k ∈ {6..0}` immediately decodes full-resolution previews from **`52.9–60.1 dB`** (`15×–91×` prefix) all the way to **bit-exact lossless (`SHA-256 PASS`)**—costing only **`+0.47%` (`+0.022 b/s`)** on Salinas!
2. **Zero-Dependency C++20 ARM64 NEON + Multi-Threaded Engine ([cpp/hnd2.cpp](file:///Users/destinatus/Documents/compression/cpp/hnd2.cpp) → `./hnd2`)**: Built-in FIPS 180-4 SHA-256, Cyclic Jacobi symmetric eigensolver, cache-blocked NEON `LDU`-RKLT, multi-threaded arbitrary-shape 2D 5/3 integer lifting, and parallel-level 3D-context 32-bit rANS + 360°-context bitplane refinement. **100% wire-compatible bit-for-bit with [phase2_codec.py](file:///Users/destinatus/Documents/compression/phase2_codec.py)**, zero compiler warnings (`-Wall -Wextra`), and **100% UBSan clean** (`-fsanitize=undefined -fno-sanitize-recover=all`).

---

## 1. Unified `.hnd2` Wire Format Architecture

- **Header (`62` bytes, little-endian)**: Magic `"HND2"`, version `2`, flags (`FLAG_BLIND`, `FLAG_LOSSY`, `FLAG_RAW`), dtype (`i16`/`u16`/`i32`), spectral mode (`none`/`full_ldu`/`clustered_ldu`), DWT levels `n_lv`, `spec_axis`, `base_k`, `q_base`, `orig_shape (D0, D1, D2)`, `sha256[32]`, and `spec_len`.
- **Spectral Stage (`spec_bytes`)**: Exact Q15 `int16` complete-pivoted `LDU`-RKLT (`|L_ij|, |U_ij| ≤ 1`) + greedy log-balanced Daubechies-Sweldens 3-rounding diagonal ladder (`0` Givens fallbacks up to full `204×204`).
- **Coarse-to-Fine Base Layer (`k = base_k`)**: Deterministic $L_2$-aligned bitplane shifts (`offsets = [2, 2, 2, 1, 1, 1, 0, 0, 0, 0]` for `L = 3`) coded with 16-context × 32-symbol 12-bit 32-bit rANS + packed extra bits. When `base_k = 0`, the stream ends here (one-shot fastest/tightest lossless).
- **Refinement Layers (`k = base_k - 1 .. 0`, when `base_k > 0`)**: For each subband whose shift drops by 1 bit, codes a unified 16-context × 3-symbol rANS pass:
  - **Significance (`prev == 0`, ctx `0..11`)**: Uses **360° spatial neighborhood** (causal `W/N/NW/NE` in current layer + full `E/S/SW/SE` in already-decoded previous layer) + coarser-scale parent + bidirectional spectral neighbors (`b-1` current / `b+1` previous).
  - **Magnitude Refinement (`prev != 0`, ctx `12..15`)**: Binned by `|prev| ∈ {1, 2, 3..5, ≥6}` to capture exact Laplace tail bias.
  - **Midpoint Reconstruction**: Truncating at any layer `k > 0` reconstructs `sign * ((mag << p) + (1 << (p-1)))`, while `k = 0` (`p = 0`) is bit-exact lossless (`SHA-256 PASS`).

---

## 2. Single-Stream 7-Layer Quality-Embedded Curves (`base_k = 6`, Native Uncropped Cubes)

Every row below is decoded from a **truncated prefix of the exact same lossless `.hnd2` file**:

| Layer (`q_eff`) | Indian Pines `(200, 145, 145)` | Salinas `(204, 512, 217)` | PaviaU `(103, 610, 340)` |
|---|---:|---:|---:|
| **`k = 6` (`~64`, base)** | `571 KB` (`1.087 b/s`, `14.7×`) → **`54.07 dB`** | `498 KB` (`0.176 b/s`, **`91.0×`**) → **`60.11 dB`** | `2.34 MB` (`0.875 b/s`, `18.3×`) → **`52.89 dB`** |
| `k = 5` (`~32`) | `870 KB` (`1.655 b/s`, `9.67×`) → `59.04 dB` | `909 KB` (`0.321 b/s`, `49.8×`) → `62.61 dB` | `4.11 MB` (`1.540 b/s`, `10.4×`) → `57.50 dB` |
| `k = 4` (`~16`) | `1.26 MB` (`2.402 b/s`, `6.66×`) → `63.84 dB` | `2.02 MB` (`0.712 b/s`, `22.5×`) → `65.57 dB` | `6.49 MB` (`2.429 b/s`, `6.59×`) → `62.72 dB` |
| `k = 3` (`~8`) | `1.81 MB` (`3.435 b/s`, `4.66×`) → `68.62 dB` | `4.25 MB` (`1.499 b/s`, `10.7×`) → `69.10 dB` | `9.32 MB` (`3.489 b/s`, `4.59×`) → `67.47 dB` |
| `k = 2` (`~4`) | `2.42 MB` (`4.605 b/s`, `3.47×`) → `73.18 dB` | `7.49 MB` (`2.643 b/s`, `6.05×`) → `72.84 dB` | `12.37 MB` (`4.633 b/s`, `3.45×`) → `70.87 dB` |
| `k = 1` (`~2`) | `2.98 MB` (`5.678 b/s`, `2.82×`) → `76.11 dB` | `10.79 MB` (`3.808 b/s`, `4.20×`) → `75.48 dB` | `15.21 MB` (`5.695 b/s`, `2.81×`) → `72.68 dB` |
| **`k = 0` (Full Lossless)** | **`3.44 MB` (`6.544 b/s`, `2.44×`) → ∞ (SHA-256)** | **`13.42 MB` (`4.736 b/s`, `3.38×`) → ∞ (SHA-256)** | **`17.51 MB` (`6.558 b/s`, `2.44×`) → ∞ (SHA-256)** |
| *vs. Std CCSDS-123 (`Rice`)* | *Beats `6.734` by **`-0.190 b/s` (`+2.8%`)*** | *Beats `5.094` by **`-0.358 b/s` (`+7.0%`)*** | *Beats `6.839` by **`-0.281 b/s` (`+4.1%`)*** |

---

## 3. C++20 NEON `./hnd2` Production Benchmark (Apple Silicon, 16-Core MT & 1T)

All lossless runs verified with built-in FIPS 180-4 **SHA-256 PASS** and exact byte-for-byte output comparison:

| Native Real Cube | Mode (`base_k`, `spec`) | Compressed Size | Rate (`Ratio`) | Enc (MT) | Dec (MT) | Dec (1T) | Preview `k=6` (MT) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Indian Pines** `(200, 145, 145)`, `8.41 MB` | `k=0`, `auto` (`clust`) | `3,378,159 B` | **`6.427 b/s` (`2.49×`)** | **`96.9 MB/s`** | **`127.9 MB/s`** | `56.0 MB/s` | — |
| | `k=6`, `auto` (`clust`) | `3,439,806 B` | `6.544 b/s` (`2.44×`) | `47.6 MB/s` | `64.4 MB/s` | `26.4 MB/s` | **`340.5 MB/s`** (`25 ms`) |
| **Salinas** `(204, 512, 217)`, `45.33 MB` | `k=0`, `auto` (`full`) | `13,355,675 B` | **`4.714 b/s` (`3.39×`)** | `54.0 MB/s` | `92.5 MB/s` | `23.0 MB/s` | — |
| | `k=0`, `clustered` | `13,365,866 B` | **`4.718 b/s` (`3.39×`)** | **`108.3 MB/s`** | **`134.8 MB/s`** | **`59.5 MB/s`** | — |
| | `k=6`, `auto` (`full`) | `13,416,816 B` | **`4.736 b/s` (`3.38×`)** | `33.0 MB/s` | `51.2 MB/s` | `15.7 MB/s` | **`168.5 MB/s`** (`269 ms`) |
| | `k=6`, `clustered` | `13,420,900 B` | `4.737 b/s` (`3.38×`) | `48.2 MB/s` | `66.5 MB/s` | `26.5 MB/s` | — |
| **PaviaU** `(103, 610, 340)`, `42.72 MB` | `k=0`, `auto` (`full`) | `17,198,193 B` | **`6.441 b/s` (`2.48×`)** | `89.1 MB/s` | `108.1 MB/s` | `33.5 MB/s` | — |
| | `k=0`, `clustered` | `17,542,809 B` | `6.570 b/s` (`2.44×`) | **`112.6 MB/s`** | **`130.6 MB/s`** | `56.7 MB/s` | — |
| | `k=6`, `auto` (`full`) | `17,512,080 B` | `6.558 b/s` (`2.44×`) | `46.0 MB/s` | `60.2 MB/s` | `20.4 MB/s` | **`234.9 MB/s`** (`182 ms`) |
| | `k=6`, `clustered` | `17,854,131 B` | `6.686 b/s` (`2.39×`) | `52.0 MB/s` | `66.8 MB/s` | `27.1 MB/s` | — |

---

## 4. Verification Summary & Phase 3 Candidates

- **Compiler & UBSan**: `make hnd2` builds cleanly with zero warnings under `-O3 -std=c++20 -mcpu=native -Wall -Wextra`; UBSan (`-fsanitize=undefined -fno-sanitize-recover=all`) passed on full native `(200, 145, 145)` Indian Pines (`base_k = 0` & `base_k = 6`).
- **Cross-Language Suite**: [test_phase2.py](file:///Users/destinatus/Documents/compression/test_phase2.py) (**4/4 PASS in 1.3s**):
  - `Python encode → C++ decode` and `C++ encode → Python decode` across `i16`/`u16`/`i32`, odd shapes, `spec_axis ∈ {0, 1, 2}`, `base_k ∈ {0, 6}`, and byte-identical truncated `stop_layer = 3` reconstruction.
  - Blind Track A 1D (`BSQ`/`BIP` + noise raw fallback) encoded by Python and decoded by C++ `./hnd2`.
  - Full native odd `(200, 145, 145)` `Indian_pines.mat` C++ ↔ Python interop.
- **Phase 3 Candidates**:
  1. **Band-parallel / 4-way interleaved rANS** inside each subband (`B` bands decoded in parallel across all 16 cores instead of only 3 orientations) — projected **`400–700+ MB/s`** lossless decode on Apple Silicon.
  2. **Pure C++20 Blind Track A Stride Detector** (`./hnd2 enc --blind`) using a lightweight radix-2 FFT Pearson ACF so `./hnd2` handles headerless 1D streams natively without Python.
  3. **ASV4 Unified Dispatch (`T5_HND2`)**: Auto-route raw 3D scientific tensors through `HND2` while keeping `ASV4` `T1..T4` for 1D/float/LLM streams.
