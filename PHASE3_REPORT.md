# Phase 3 Report (`phase3-v3.1`) — Band-Stripe Parallel `.hnd3`, True NEON `vmlal_s32` FMA, Native C++20 Blind Track A, & Full Audit

Phase 3 ([cpp/hnd3.cpp](file:///Users/destinatus/Documents/compression/cpp/hnd3.cpp) → `./hnd3`, wire-compatible with [phase3_codec.py](file:///Users/destinatus/Documents/compression/phase3_codec.py), verified by [test_phase3.py](file:///Users/destinatus/Documents/compression/test_phase3.py)) upgrades Phase 2 (`phase2-v2.0`) with **16-stripe band-parallel 32-bit rANS**, **ARM64 NEON 32×32→64 widening FMA (`vmlal_s32`)**, **64-bit extra-bit buffering**, **windowed native C++20 `--blind` Track A**, and **hardened decoder bounds/rANS checks**.

> **Reporting Conventions**: All sizes and throughput rates use **decimal megabytes (`1 MB = 10^6 bytes`)** (e.g., `8,410,000 B = 8.41 MB`, `45,330,432 B = 45.33 MB`, `132,836,480 B = 132.84 MB`). Every external baseline and every `.hnd3` run below is measured on the **exact same uncropped native cube** in **real encoded bytes** with **bit-exact round-trip verification** (built-in FIPS 180-4 SHA-256 for `.hnd3` / `ASV4`; array-equality check for all external codecs). All benchmarks were measured on Apple Silicon (16-core ARM64, macOS, `Apple clang 16.0.0`, `-O3 -std=c++20 -mcpu=native -Wall -Wextra`).

---

## 1. Phase 3 Architecture Upgrades (`.hnd2` → `.hnd3`)

1. **Band-Stripe Parallel 32-bit rANS (`n_stripes = min(B, 16)`)**:
   - Partitions each subband's spectral axis into up to `16` independent contiguous band stripes, each with its own compact 12-bit rANS frequency table.
   - **Systems win**: Unlocks all 16 CPU cores across every subband and every bitplane refinement layer.
   - **Information-theoretic win**: Because `LDU`-RKLT orders spectral components by descending energy, per-stripe CDF tables adapt cleanly to each energy tier (`PC1..PC12` vs. noise floor), saving `0.06–0.20 b/s` on large cubes—causing **7-layer embedded `base_k = 6` to beat one-shot `base_k = 0`** on Salinas, PaviaU, and the unseen 132.84 MB AVIRIS holdout!
2. **True ARM64 NEON 32×32→64 Widening FMA (`vmlal_s32` / `vmlal_high_s32`) + Portable Scalar Parity**:
   - Processes 8 pixels/iter in `ldu_row_madd_i32_neon` for `int16`/`uint16` cubes, with automatic overflow-safe scalar `int64` dispatch for `int32` cubes (and `-DHND3_FORCE_SCALAR` for cross-architecture parity verification).
3. **64-Bit Extra-Bit Buffer (`FastBitReader` / `FastBitWriter`)**: Replaces per-bit inner loops with 2-instruction 64-bit shifts.
4. **Windowed Native C++20 Blind Track A (`./hnd3 enc --blind`)**:
   - Windows the Cooley-Tukey radix-2 FFT ACF to `min(N, 4_194_304)` samples (covering strides up to `1,048,576` in bounded time/RAM), rejects non-dividing spurious 4th+ candidate strides (`N_full % s_next != 0`), picks the true spectral axis via multi-level clustered `LDU` + 2D 5/3 entropy scoring, and falls back cleanly to raw framing on incompressible noise—with **zero Python dependency**.
5. **Hardened Decoder**: Pre-allocation header/spectral-count sanity checks, anti-memory-bomb sample-to-stream cap, rANS frequency/CDF sum validation, and truncated-state detection (`state < RANS_L`) across base and refinement stripes.

---

## 2. Exact Uncropped Apples-to-Apples Baselines (All 4 Hyperspectral Cubes)

Every codec below ran on the **exact same uncropped files** (no `//8` crops).
- **CCSDS-123 provenance**: `CCSDS-123.0-B-1` is our verified bit-exact Blue Book Issue 1 implementation in [phase0_harness.py](file:///Users/destinatus/Documents/compression/phase0_harness.py) (Numba-compiled, Sections 4.4–4.9 local-sum sign-LMS predictor + sample-adaptive Golomb-Rice coder, verified bit-exact round-trip). Both **`P=3` and `P=15`** are reported on every cube (on PaviaU, `P=3` beats `P=15` because 103 noisy ROSIS bands converge slower with 15 weights). We also report **`CCSDS-123 (P=15) + 3D-rANS`** (packing the exact CCSDS-123 prediction residuals with our 3D-context 32-bit rANS coder—a strong non-progressive proxy for Issue 2's hybrid entropy coder).
- **Official external progressive/lossless binaries** (via `imagecodecs 2023.1.23` + `zstandard`):
  - **OpenJPEG JPEG-2000 Part 1** (`jpeg2k_encode(..., level=0, reversible=True)`, per-band lossless `.jp2`)
  - **OpenJPEG JPEG-2000 Part 2 + `LDU`-RKLT** (exact serialized `LDU`-RKLT spectral header bytes + per-band lossless OpenJPEG `.jp2`, verified bit-exact round-trip)
  - **CharLS JPEG-LS** (`jpegls_encode(..., level=0)`, per-band lossless)
  - **ZFP Reversible 3D** (`zfp_encode(..., mode='R')`, official 3D reversible `int32` mode)
  - **`zstd -19` (byte-shuffled 2B)** and **`ASV4 v4.1` (L19, blind byte-stream)**

### 2.1 Indian Pines `(200, 145, 145)` — `8,410,000 B` (`8.41 MB`, `int16`)

| Codec / Method | Progressive? | Encoded Bytes | Rate (`b/s`) | Ratio | Enc (MB/s) | Dec (MB/s) |
|---|---|---:|---:|---:|---:|---:|
| `CCSDS-123 (P=15) + 3D-rANS` *(custom)* | No | `3,356,782` | `6.386` | `2.51×` | `14.5` | — |
| **Phase 3 `.hnd3` (`k=0`, 1-shot, MT)** | Spatial only | `3,408,480` | `6.485` | `2.47×` | **`110.9`** | **`189.2`** |
| **Phase 3 `.hnd3` (`k=6`, 7-layer, MT)** | **7-layer SNR** | **`3,448,539`** | **`6.561`** | **`2.44×`** | `74.0` | `115.5` |
| `OpenJPEG J2K Part 2 + LDU-RKLT` *(official)* | Spatial/SNR | `3,542,254` | `6.739` | `2.37×` | `15.7` | `22.4` |
| `CCSDS-123.0-B-1 (P=15, Rice)` *(custom)* | No | `3,559,591` | `6.772` | `2.36×` | `18.4` | — |
| `CCSDS-123.0-B-1 (P=3, Rice)` *(custom)* | No | `3,573,440` | `6.798` | `2.35×` | `27.7` | — |
| `CharLS JPEG-LS` *(official, per-band)* | No | `4,165,882` | `7.926` | `2.02×` | `81.1` | `92.6` |
| `OpenJPEG J2K Part 1` *(official, per-band)* | Spatial/SNR | `4,236,668` | `8.060` | `1.99×` | `14.2` | `20.0` |
| `zstd -19` *(byte-shuffled 2B)* | No | `4,485,477` | `8.534` | `1.87×` | `7.2` | `1,419.5` |
| `ASV4 v4.1` *(L19, blind byte-stream)* | No | `4,673,175` | `8.891` | `1.80×` | `6.2` | `318.4` |
| `ZFP Reversible 3D` *(official, `i32` mode)* | No | `6,399,544` | `12.175` | `1.31×` | `198.3` | `264.4` |

### 2.2 Salinas `(204, 512, 217)` — `45,330,432 B` (`45.33 MB`, `int16`)

| Codec / Method | Progressive? | Encoded Bytes | Rate (`b/s`) | Ratio | Enc (MB/s) | Dec (MB/s) |
|---|---|---:|---:|---:|---:|---:|
| **Phase 3 `.hnd3` (`k=6`, 7-layer, MT)** | **7-layer SNR** | **`13,111,158`** | **`4.628`** | **`3.46×`** | `54.2` | `106.0` |
| **Phase 3 `.hnd3` (`k=0`, 1-shot, MT)** | Spatial only | `13,194,920` | `4.657` | `3.44×` | **`67.0`** | **`153.8`** |
| `CCSDS-123 (P=15) + 3D-rANS` *(custom)* | No | `13,583,657` | `4.795` | `3.34×` | `14.6` | — |
| `OpenJPEG J2K Part 2 + LDU-RKLT` *(official)* | Spatial/SNR | `13,608,877` | `4.803` | `3.33×` | `15.2` | `20.8` |
| `CCSDS-123.0-B-1 (P=15, Rice)` *(custom)* | No | `14,471,088` | `5.108` | `3.13×` | `17.8` | — |
| `CCSDS-123.0-B-1 (P=3, Rice)` *(custom)* | No | `14,536,922` | `5.131` | `3.12×` | `26.8` | — |
| `ASV4 v4.1` *(L19, blind byte-stream)* | No | `16,256,429` | `5.738` | `2.79×` | `4.3` | `527.7` |
| `zstd -19` *(byte-shuffled 2B)* | No | `17,156,071` | `6.055` | `2.64×` | `5.0` | `1,001.7` |
| `CharLS JPEG-LS` *(official, per-band)* | No | `17,674,288` | `6.238` | `2.56×` | `99.6` | `111.6` |
| `OpenJPEG J2K Part 1` *(official, per-band)* | Spatial/SNR | `19,138,425` | `6.755` | `2.37×` | `16.3` | `23.1` |
| `ZFP Reversible 3D` *(official, `i32` mode)* | No | `30,156,576` | `10.644` | `1.50×` | `237.0` | `292.2` |

### 2.3 PaviaU `(103, 610, 340)` — `42,724,400 B` (`42.72 MB`, `int16`)

| Codec / Method | Progressive? | Encoded Bytes | Rate (`b/s`) | Ratio | Enc (MB/s) | Dec (MB/s) |
|---|---|---:|---:|---:|---:|---:|
| **Phase 3 `.hnd3` (`k=6`, 7-layer, MT)** | **7-layer SNR** | **`16,967,890`** | **`6.354`** | **`2.52×`** | `78.0` | `116.4` |
| **Phase 3 `.hnd3` (`k=0`, 1-shot, MT)** | Spatial only | `17,001,192` | `6.367` | `2.51×` | **`118.8`** | **`173.0`** |
| `CCSDS-123 (P=15) + 3D-rANS` *(custom)* | No | `17,525,157` | `6.563` | `2.44×` | `14.1` | — |
| `OpenJPEG J2K Part 2 + LDU-RKLT` *(official)* | Spatial/SNR | `17,729,880` | `6.640` | `2.41×` | `14.4` | `21.0` |
| `CCSDS-123.0-B-1 (P=3, Rice)` *(custom)* | No | `18,280,016` | `6.846` | `2.34×` | `25.4` | — |
| `CCSDS-123.0-B-1 (P=15, Rice)` *(custom)* | No | `18,442,758` | `6.907` | `2.32×` | `17.8` | — |
| `zstd -19` *(byte-shuffled 2B)* | No | `24,068,251` | `9.013` | `1.78×` | `3.3` | `2,351.1` |
| `ASV4 v4.1` *(L19, blind byte-stream)* | No | `24,519,769` | `9.182` | `1.74×` | `3.3` | `611.1` |
| `CharLS JPEG-LS` *(official, per-band)* | No | `24,829,636` | `9.299` | `1.72×` | `82.4` | `84.4` |
| `OpenJPEG J2K Part 1` *(official, per-band)* | Spatial/SNR | `25,892,643` | `9.697` | `1.65×` | `12.6` | `17.2` |
| `ZFP Reversible 3D` *(official, `i32` mode)* | No | `33,026,800` | `12.368` | `1.29×` | `230.2` | `267.8` |

### 2.4 Frozen Unseen Holdout: NASA AVIRIS `(224, 745, 398)` — `132,836,480 B` (`132.84 MB`, `int16`)

*Source: HuggingFace `danjacobellis/aviris_1k_val` (`validation-00003-of-00066.parquet`, row 0), never seen during Phase 0–3 development, with ugly prime spatial factors `745 = 5×149`, `398 = 2×199`. Data-integrity note: the parquet stored raw big-endian NASA AVIRIS `>i2` radiance bytes interpreted as `<i2` (immediately caught because `zstd -19` beat all spectral codecs at `3.57 b/s` and `val & 0xFF` had only 9 unique values); byte-swapping recovers authentic NASA AVIRIS radiance (`range=[-50, 2033]`).*

| Codec / Method | Progressive? | Encoded Bytes | Rate (`b/s`) | Ratio | Enc (MB/s) | Dec (MB/s) |
|---|---|---:|---:|---:|---:|---:|
| `CCSDS-123 (P=15) + 3D-rANS` *(custom)* | No | `23,989,094` | `2.889` | `5.54×` | `15.6` | — |
| **Phase 3 `.hnd3` (`k=6`, 7-layer, MT)** | **7-layer SNR** | **`24,408,930`** | **`2.940`** | **`5.44×`** | `71.7` | `112.5` |
| **Phase 3 `.hnd3` (`k=0`, 1-shot, MT)** | Spatial only | `25,468,981` | `3.068` | `5.22×` | **`91.4`** | **`154.2`** |
| `OpenJPEG J2K Part 2 + LDU-RKLT` *(official)* | Spatial/SNR | `25,473,453` | `3.068` | `5.21×` | `21.6` | `27.3` |
| `CCSDS-123.0-B-1 (P=15, Rice)` *(custom)* | No | `25,554,044` | `3.078` | `5.20×` | `18.0` | — |
| `CCSDS-123.0-B-1 (P=3, Rice)` *(custom)* | No | `25,839,324` | `3.112` | `5.14×` | `26.7` | — |
| `OpenJPEG J2K Part 1` *(official, per-band)* | Spatial/SNR | `27,620,094` | `3.327` | `4.81×` | `31.1` | `42.7` |
| `CharLS JPEG-LS` *(official, per-band)* | No | `27,911,424` | `3.362` | `4.76×` | `135.6` | `158.0` |
| `zstd -19` *(byte-shuffled 2B)* | No | `29,650,031` | `3.571` | `4.48×` | `6.5` | `1,472.9` |
| `ZFP Reversible 3D` *(official, `i32` mode)* | No | `70,300,032` | `8.468` | `1.89×` | `237.1` | `329.8` |

> **Apples-to-Apples Takeaway**: Across **all 4 uncropped hyperspectral cubes** (including the frozen 132.84 MB AVIRIS holdout), **7-layer progressive `.hnd3` (`k=6`) beats OpenJPEG JPEG-2000 Part 2 + `LDU`-RKLT by `0.13–0.29 b/s` (`4.1×–5.5×` faster decode) AND beats CCSDS-123.0-B-1 (`P=3` & `P=15`) by `0.14–0.49 b/s`**. On Salinas and PaviaU it is **#1 overall** even against non-progressive `CCSDS-123 (P=15) + 3D-rANS` (by `0.17–0.21 b/s`), while on Indian Pines and AVIRIS non-progressive `CCSDS-123 + 3D-rANS` edges ahead by `0.05–0.18 b/s` at `7–8×` slower sequential speed and with no progressive previews.

### 2.5 Empirical Rounding-Noise Variance Audit (`204×204` / `224×224` Complete-Pivoted `LDU`-RKLT)

Phase 0's row-pivoted `PLUS`/SERM factorization suffered Schur-complement multiplier blowup (`> 10^4`) beyond `M = 8`. Complete-pivoted `LDU` (`P·diag(s0)·U_KLT·Q = L·diag(sd)·D_ladder·U`) guarantees `|L_ij|, |U_ij| ≤ 1.0000` and pairs diagonal factors via a greedy log-balanced ladder (`0` Givens fallbacks up to `224×224`). Measuring exact same-matrix integer lifting error `e = Y_ldu − U_KLT·(X − μ)` against ideal non-reversible `float64` KLT (all verified bit-exact inverse round-trip):

| Cube / Stress Test | Matrix `B×B` | `max|L|` | `max|U|` | `max|c_k|` | `max|1/c_k|` | `RMS(e)` | `Var(e)` | Tail-20% `RMS` | `max|e|` | `ΔH` vs `float64` KLT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Indian Pines** | `200×200` | `1.0000` | `1.0000` | `1.533` | `1.420` | `1.054` | `1.111` | `0.972` | `8.65` | `+0.0140 b/s` |
| **Salinas** | `204×204` | `1.0000` | `1.0000` | `1.474` | `1.614` | `1.082` | `1.171` | `0.908` | `7.88` | `+0.0481 b/s` |
| **PaviaU** | `103×103` | `1.0000` | `1.0000` | `2.037` | `2.317` | `1.345` | `1.809` | `1.440` | `12.53` | `+0.0213 b/s` |
| **Unseen AVIRIS** | `224×224` | `1.0000` | `1.0000` | `1.671` | `1.604` | `0.970` | `0.941` | `0.701` | `9.66` | `+0.3967 b/s` |
| **High-SNR Synth (`60 dB`, `σ_n=1`)** | `204×204` | `1.0000` | `1.0000` | `3.502` | `2.798` | `2.054` | `4.217` | `1.958` | `18.24` | `+0.9515 b/s` |

- **Key mechanism & honest floor**: Complete pivoting eliminates catastrophic blowup completely (`Var(e) = 0.94–1.81` across all 4 real cubes, zero Givens fallbacks). On standard sensor noise floors (Indian Pines, Salinas, PaviaU), the rounding overhead is negligible (`+0.014–0.048 b/s`). On ultra-low-noise/high-SNR cubes whose tail PCs fall near or below `1` count (`AVIRIS` tail `σ < 1`, `60 dB` synthetic `σ_n = 1`), the irreducible $O(1)$-LSB integer-lifting rounding floor contributes `+0.40–0.95 b/s` relative to unquantized `float64` KLT—explaining cleanly why non-progressive causal `CCSDS-123 + 3D-rANS` edges `.hnd3` by `0.051 b/s` on `AVIRIS`.

---

## 3. Full 7-Layer Progressive Rate–Distortion Curves (`k = 6 → 0`)

Exact cumulative byte offsets walked directly from each `.hnd3` (`base_k = 6`, `mode = auto`) container, with `PSNR_rng = 20·log10((max − min) / RMSE)`, `PSNR_16b = 20·log10(65535 / RMSE)`, max absolute error $L_\infty = \max|x - \hat{x}|$, and 16-core C++ `./hnd3 dec --stop-layer k --bench` decode latency.

### 3.1 Salinas `(204, 512, 217)` (`45.33 MB`, `dyn = 9,218`, `LDU_hdr = 87,766 B`)

| Layer | Cum. Bytes | `%` Stream | Cum. `b/s` | Ratio | `PSNR_rng` | `PSNR_16b` | $L_\infty$ | Dec (ms) | Dec (MB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`k=6` (Base)** | **`451,494`** | **`3.4%`** | **`0.159`** | **`100.40×`** | **`60.11 dB`** | `77.15 dB` | `178` | **`138.0`** | **`327.6`** |
| `k=5` | `835,924` | `6.4%` | `0.295` | `54.23×` | `62.61 dB` | `79.64 dB` | `72` | `178.0` | `254.5` |
| `k=4` | `1,863,641` | `14.2%` | `0.658` | `24.32×` | `65.56 dB` | `82.60 dB` | `37` | `190.0` | `239.1` |
| `k=3` | `4,009,579` | `30.6%` | `1.415` | `11.31×` | `69.10 dB` | `86.14 dB` | `20` | `212.0` | `214.0` |
| `k=2` | `7,205,748` | `55.0%` | `2.543` | `6.29×` | `72.84 dB` | `89.88 dB` | `13` | `234.0` | `193.3` |
| `k=1` | `10,486,368` | `80.0%` | `3.701` | `4.32×` | `75.48 dB` | `92.52 dB` | `13` | `268.0` | `169.0` |
| **`k=0` (Lossless)** | **`13,111,158`** | **`100.0%`** | **`4.628`** | **`3.46×`** | **`∞`** | **`∞`** | **`0`** | `426.0` | `106.4` |

### 3.2 PaviaU `(103, 610, 340)` (`42.72 MB`, `dyn = 8,000`, `LDU_hdr = 23,504 B`)

| Layer | Cum. Bytes | `%` Stream | Cum. `b/s` | Ratio | `PSNR_rng` | `PSNR_16b` | $L_\infty$ | Dec (ms) | Dec (MB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`k=6` (Base)** | **`2,047,876`** | **`12.1%`** | **`0.767`** | **`20.86×`** | **`52.89 dB`** | `71.16 dB` | `228` | **`105.0`** | **`406.7`** |
| `k=5` | `3,729,715` | `22.0%` | `1.397` | `11.46×` | `57.50 dB` | `75.77 dB` | `99` | `129.0` | `331.7` |
| `k=4` | `6,026,857` | `35.5%` | `2.257` | `7.09×` | `62.73 dB` | `80.99 dB` | `37` | `143.0` | `298.7` |
| `k=3` | `8,805,731` | `51.9%` | `3.298` | `4.85×` | `67.47 dB` | `85.74 dB` | `20` | `175.0` | `244.5` |
| `k=2` | `11,836,251` | `69.8%` | `4.433` | `3.61×` | `70.87 dB` | `89.13 dB` | `17` | `205.0` | `208.5` |
| `k=1` | `14,664,950` | `86.4%` | `5.492` | `2.91×` | `72.68 dB` | `90.95 dB` | `16` | `228.0` | `187.5` |
| **`k=0` (Lossless)** | **`16,967,890`** | **`100.0%`** | **`6.354`** | **`2.52×`** | **`∞`** | **`∞`** | **`0`** | `378.0` | `113.2` |

### 3.3 Indian Pines `(200, 145, 145)` (`8.41 MB`, `dyn = 8,649`, `LDU_hdr = 12,294 B`)

| Layer | Cum. Bytes | `%` Stream | Cum. `b/s` | Ratio | `PSNR_rng` | `PSNR_16b` | $L_\infty$ | Dec (ms) | Dec (MB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`k=6` (Base)** | **`564,412`** | **`16.4%`** | **`1.074`** | **`14.90×`** | **`54.07 dB`** | `71.66 dB` | `146` | **`16.0`** | **`529.7`** |
| `k=5` | `863,402` | `25.0%` | `1.643` | `9.74×` | `59.04 dB` | `76.63 dB` | `66` | `23.0` | `362.6` |
| `k=4` | `1,258,515` | `36.5%` | `2.394` | `6.68×` | `63.84 dB` | `81.43 dB` | `34` | `28.0` | `305.3` |
| `k=3` | `1,804,938` | `52.3%` | `3.434` | `4.66×` | `68.61 dB` | `86.20 dB` | `17` | `32.0` | `261.4` |
| `k=2` | `2,423,203` | `70.3%` | `4.610` | `3.47×` | `73.17 dB` | `90.76 dB` | `11` | `40.0` | `212.5` |
| `k=1` | `2,991,355` | `86.7%` | `5.691` | `2.81×` | `76.11 dB` | `93.70 dB` | `8` | `45.0` | `186.5` |
| **`k=0` (Lossless)** | **`3,448,539`** | **`100.0%`** | **`6.561`** | **`2.44×`** | **`∞`** | **`∞`** | **`0`** | `73.0` | `115.2` |

### 3.4 Unseen NASA AVIRIS Holdout `(224, 745, 398)` (`132.84 MB`, `dyn = 2,083`, `LDU_hdr = 105,330 B`)

| Layer | Cum. Bytes | `%` Stream | Cum. `b/s` | Ratio | `PSNR_rng` | `PSNR_16b` | $L_\infty$ | Dec (ms) | Dec (MB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`k=6` (Base)** | **`149,607`** | **`0.6%`** | **`0.018`** | **`887.90×`** | **`55.27 dB`** | `85.22 dB` | `95` | **`417.0`** | **`318.3`** |
| `k=5` | `301,394` | `1.2%` | `0.036` | `440.74×` | `56.90 dB` | `86.85 dB` | `59` | `454.0` | `292.7` |
| `k=4` | `741,323` | `3.0%` | `0.089` | `179.19×` | `58.97 dB` | `88.92 dB` | `43` | `509.0` | `261.1` |
| `k=3` | `2,160,341` | `8.9%` | `0.260` | `61.49×` | `59.67 dB` | `89.63 dB` | `20` | `573.0` | `231.9` |
| `k=2` | `6,819,310` | `27.9%` | `0.821` | `19.48×` | `61.39 dB` | `91.34 dB` | `14` | `666.0` | `199.5` |
| `k=1` | `16,009,012` | `65.6%` | `1.928` | `8.30×` | `62.65 dB` | `92.61 dB` | `13` | `741.0` | `179.2` |
| **`k=0` (Lossless)** | **`24,408,930`** | **`100.0%`** | **`2.940`** | **`5.44×`** | **`∞`** | **`∞`** | **`0`** | `1,199.0` | `110.8` |

> **Progressive Curve Verification**: `k=6` is genuinely ultra-compact—only **`0.6%`–`16.4%`** of the lossless stream (`15×`–`888×` compression!) delivers a **`52.9–60.1 dB`** preview, ramping smoothly across all 7 bitplane layers (`Trunks → Branches → Leaves → Moss`) down to $L_\infty = 0$ exact reconstruction.

---

## 4. Full Uncropped Native C++20 `--blind` Track A (All 6/6 Permutations × 4 Cubes = 24/24 Exact)

Windowing the radix-2 FFT ACF to `min(N, 4_194_304)` samples and rejecting non-dividing 4th+ candidate strides (`N_full % s_next != 0`) enables full uncropped blind recovery across ugly prime dimensions (`217 = 7×31`, `103`, `745 = 5×149`, `398 = 2×199`) and fixes PaviaU's previous spurious 4th-stride edge case. Every single run passed full FIPS 180-4 SHA-256 verification:

| Uncropped Cube (`Size`) | Tested Perms (`BHW/BWH/HBW/HWB/WBH/WHB`) | Exact 3D Strides `[1, s1, s2]` | True Spectral Axis | Blind Rate (`b/s`, `k=0`) | Wall-Clock Time | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| **Indian Pines** `(200, 145, 145)` (`8.41 MB`) | All `6/6` | **`6/6` exact** | **`6/6` exact** | `6.485–6.486` | `2.59–2.76 s` | `571–575 MB` |
| **Salinas** `(204, 512, 217)` (`45.33 MB`) | All `6/6` | **`6/6` exact** | **`6/6` exact** | `4.654–4.657` | `3.16–3.46 s` | `1,384–1,387 MB` |
| **PaviaU** `(103, 610, 340)` (`42.72 MB`) | All `6/6` | **`6/6` exact** | **`6/6` exact** | `6.349–6.367` | `2.90–3.17 s` | `1,344–1,348 MB` |
| **Unseen AVIRIS** `(224, 745, 398)` (`132.84 MB`) | All `6/6` | **`6/6` exact** | **`6/6` exact** | `3.068–3.113` | `3.46–4.42 s` | `3,119–3,129 MB` |
| **Incompressible Noise** `(16, 32, 32)` | Raw 1D | `[1]` raw fallback | n/a | `< +0.2%` tax | `< 0.01 s` | `< 20 MB` |

---

## 5. Non-Hyperspectral 3D Controls & Real `int32` Overflow-Safe Path

### 5.1 Real `int32` (`265.67 MB`) & Peak Encode/Decode RSS Audit (`/usr/bin/time -l`)

| Dataset (`Raw Size`, `dtype`) | Mode | Encoded Size | `b/s` (`Ratio`) | Enc (MT) | Dec (MT) | Dec (1T) | Enc RSS | Dec RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Indian Pines** (`8.41 MB`, `i16`) | `k=0` / `k=6` | `3.41` / `3.45 MB` | `6.485` / `6.561` | `113.8` / `77.7 MB/s` | `196.4` / `122.4 MB/s` | `64.6` / `27.5 MB/s` | `202` / `199 MB` | `160` / `163 MB` |
| **Salinas** (`45.33 MB`, `i16`) | `k=0` / `k=6` | `13.19` / `13.11 MB` | `4.657` / `4.628` | `67.8` / `55.0 MB/s` | `156.8` / `110.5 MB/s` | `39.4` / `21.6 MB/s` | `1,040` / `1,034 MB` | `807` / `802 MB` |
| **PaviaU** (`42.72 MB`, `i16`) | `k=0` / `k=6` | `17.00` / `16.97 MB` | `6.367` / `6.354` | `118.8` / `84.0 MB/s` | `186.0` / `121.9 MB/s` | `50.2` / `25.1 MB/s` | `999` / `1,018 MB` | `755` / `755 MB` |
| **AVIRIS Holdout** (`132.84 MB`, `i16` NEON) | `k=0` / `k=6` | `25.47` / `24.41 MB` | `3.068` / `2.940` (`5.44×`) | `96.0` / `73.1 MB/s` | `159.2` / `116.8 MB/s` | `38.3` / `21.8 MB/s` | `2,961` / `2,934 MB` | `2,308` / `2,304 MB` |
| **AVIRIS Holdout** (`265.67 MB`, `i32` scalar `i64`) | `k=0` / `k=6` | `25.47` / `24.41 MB` | `3.068` / `2.940` (`10.88×`) | `80.6` / `67.7 MB/s` | `96.8` / `80.9 MB/s` | `18.3` / `15.4 MB/s` | `3,096` / `3,063 MB` | `2,719` / `2,723 MB` |

- **Honest Speed & Memory Trade-offs**:
  1. **Single-thread progressive tax**: `1T` full lossless `k=6` decode (`21.6–27.5 MB/s`) is roughly **`1.8×–2.3×` slower** than one-shot `k=0` (`38.3–64.6 MB/s`)—the direct cost of executing 6 sequential bitplane refinement passes (`k=5..0`) in addition to the base layer. Under 16-core MT, band-stripe parallelism closes that gap to `1.4×–1.6×` (`110–122 MB/s` vs. `157–196 MB/s`).
  2. **Working-set RSS**: Full-cube `int64` transform buffers plus subband/parent arrays require **`~17–22×` raw `int16`** (`~10–12×` raw `int32`) peak RAM (`~1.0 GB` for a `45 MB` cube, `~2.9 GB` for a `133 MB` cube). Tiled spatial streaming (`256×256` tiles) is the natural Phase 4 path to drop RSS below `2×` raw for multi-gigabyte scenes.
  3. **Real `int32` proof**: Encoding the exact same 224-band AVIRIS scene as native `int32` (`265.67 MB`) automatically dispatches to the overflow-safe `int64` lifting path (`use_i32_neon = false`), producing the **exact same `24,408,930 B` bitstream (`10.88×`, `SHA-256 PASS`)** while cleanly illustrating the NEON speedup (`38.3` vs. `18.3 MB/s` `1T` `k=0`, `2.09×`).

### 5.2 Non-Hyperspectral 3D Controls (Fluorescence Microscopy & Medical Brain MRI)

| Real 3D Control Volume | `.hnd3 k=0` (`auto`) | `.hnd3 k=6` (`7-layer`) | `.hnd3 --blind` (`k=0`) | `OpenJPEG J2K` | `CharLS JPEG-LS` | `ZFP Rev 3D` | `zstd -19` (2B shuf) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Cells3D Microscopy** `(60, 256, 256)` `u16` (`7.86 MB`, `[0, 65535]`) | **`9.977 b/s`** (`172.9 MB/s` dec) | `10.073 b/s` (`122.1 MB/s`) | **`9.977 b/s`** *(exact `(60,256,256)`, `sp=0`)* | `10.278 b/s` (`16.5 MB/s`) | `10.019 b/s` (`77.4 MB/s`) | `15.173 b/s` (`229.8 MB/s`) | `10.654 b/s` (`1,557 MB/s`) |
| **Brain MRI** `(10, 256, 256)` `u16` (`1.31 MB`, sparse black bg) | `7.099 b/s` (`152.9 MB/s` dec) | `7.376 b/s` (`72.3 MB/s`) | `8.158 b/s` *(exact strides, `sp=2`)* | `6.749 b/s` (`21.7 MB/s`) | `6.185 b/s` (`115.6 MB/s`) | `11.790 b/s` (`263.4 MB/s`) | **`4.226 b/s`** (`1,139 MB/s`) |

> **Honest Domain Boundary**: On dense, full-dynamic-range 3D fluorescence microscopy (`Cells3D`), `.hnd3` **beats OpenJPEG J2K, CharLS JPEG-LS, ZFP, and `zstd -19`** (at `2.2×–10.5×` faster decode than JPEG-LS/J2K) and recovers the exact 3D volume blindly. Conversely, on a shallow `B=10` medical MRI slice stack dominated by exact-zero background (`Brain MRI`), `zstd -19` (`4.226 b/s`) and `JPEG-LS` (`6.185 b/s`) decisively outperform wavelet/KLT coding (`7.099 b/s`) because LZ77 dictionary/run-mode coding collapses zero-background runs far more efficiently than 2D 5/3 subbands.

---

## 6. Hardening, Cross-Architecture Parity, & Fuzz Verification

- **Zero compiler warnings**: `c++ -O3 -std=c++20 -mcpu=native -Wall -Wextra -pthread cpp/hnd3.cpp -o hnd3` → **0 warnings**.
- **ARM64 NEON ↔ Portable Scalar (`-DHND3_FORCE_SCALAR`) Bit-Exact Parity**: Verified on full native `Indian_pines` `(200, 145, 145)` for both `base_k = 0` (`3,408,480 B`, identical SHA-256 `390a0fee38c52a02...`) and `base_k = 6` (`3,448,539 B`, identical SHA-256 `fcf108d9afdfcd58...`), plus cross-direction NEON→Scalar and Scalar→NEON lossless decode parity (`SHA-256 = c7673a8224773dbb...`).
- **20,200-Iteration UBSan + `libc++` Debug Bounds-Checked Fuzz Pass**: Compiled with `-fsanitize=undefined -fno-sanitize-recover=all -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG` (runtime bounds checks on every `std::vector::operator[]`; note Apple Clang 16's macOS ASan runtime hits a known Darwin-27 startup `sanitizer_malloc_mac.inc:189` bug prior to `main()`). Fuzzing actually caught and drove fixes for two real crafted-input DoS paths (forged-dimension memory bomb before spectral-header validation, and missing `state < RANS_L` check on truncated rANS buffers). After hardening:
  - `10,000` iterations on `k=0` seed: `9,892` cleanly rejected, `108` accepted, **0 crashes / 0 UBSan / 0 bounds violations**.
  - `10,000` iterations on `k=6` seed: `9,850` cleanly rejected, `150` accepted, **0 crashes / 0 UBSan / 0 bounds violations**.
  - `200` iterations on full `3.45 MB` Indian Pines `k=6` seed: `197` cleanly rejected, `3` accepted, **0 crashes / 0 UBSan / 0 bounds violations**.
- **`make test`**: Permanently integrates UBSan + `_LIBCPP_HARDENING_MODE_DEBUG` 2,000-iteration fuzzing, `-DHND3_FORCE_SCALAR` bit-exact parity, and the full `test_phase2.py` + `test_phase3.py` Python↔C++ wire-compatibility suites (**100% PASS**).
