# Phase 4 Systems & Capability Report (`phase4-v4.0` — `libhnd3` + `HND4` + `libFuzzer/ASan`)

Phase 4 addresses every architectural and verification gap identified in the end-of-Phase-3 audit while preserving 100% bit-for-bit wire compatibility with **HND3 v3** ([`phase3_codec.py`](phase3_codec.py)):

1. **Modular C++20 Library & C99 ABI (`libhnd3.a` + [`include/hnd3.h`](include/hnd3.h))**: Split the monolithic 2,165-line `cpp/hnd3.cpp` into [`cpp/hnd3_core.hpp`](cpp/hnd3_core.hpp), [`cpp/hnd3_rans.hpp`](cpp/hnd3_rans.hpp), [`cpp/hnd3_lib.cpp`](cpp/hnd3_lib.cpp), and a thin CLI driver [`cpp/hnd3_cli.cpp`](cpp/hnd3_cli.cpp) (`./hnd3`), exposing both C++20 (`hnd3::encode`/`decode`/`encode_file`/`decode_file`/`inspect`/`crc32c`) and `extern "C"` (`hnd3_encode_buffer`/`hnd3_decode_buffer`/`hnd3_free`) APIs verified by a standalone C99 test harness.
2. **HND4 Streaming Spatial Tiling (`--tile-size T`), Tile-Row Strip Write Buffering, & Random-Access Tile Decode (`dec --tile TY,TX`)**: Bounds encode/decode RAM to `O(B · T²)` via strided file sampling (`≤ 16,384` pixels) for one shared global `LDU`-RKLT model followed by tile-by-tile streaming I/O, plus a bounded `≤ 64 MiB` multi-column tile-row strip write buffer for BSQ/BIL/BIP full-cube decodes. Cuts Peak Decode RSS on the `132.84 MB` AVIRIS holdout by **15.1×** (`2,169 MB → 144 MB`) while keeping full-scene disk decode at **`114.1 MB/s` (`91.6%` of untiled)** at `+0.051 b/s`.
3. **Per-Header & Per-Layer Castagnoli CRC32C (`--crc`, always on in HND4)**: Protects the 84 B header, optional `FLAG_HIST_LUT` prefix, shared `LDU` model, Tile Index Table, and every individual quality layer (`k = base_k..0`) of every tile—so **lossy progressive previews (`--stop-layer K`) and single-tile random-access decodes verify integrity (`CRC32C=PASS`)**.
4. **Guaranteed-$L_\infty$ Near-Lossless Coding (`--max-error E`) & CCSDS-123.0-B-2 Benchmark**: Uniform centered sample-domain quantizer guaranteeing `|x - x_hat| ≤ E` for all `i16`/`u16`/`i32` samples (including dtype min/max extremes), benchmarked head-to-head against the WHISPERS 2024 **NTNU CCSDS-123.0-B-2** verification model (Issue 2: Hybrid Entropy Coder + closed-loop near-lossless).
5. **Auto-Gated Reversible Monotonic Histogram Compaction LUT (`FLAG_HIST_LUT`, `--hist-lut auto|on|off`)**: Order-preserving 16-bit lossless palette compaction (`fmt=1` sparse ascending list for `K ≤ 4,096`; `fmt=2` 8 KB presence bitmap for `K > 4,096`) with a `> 240×`-separated sparsity auto-gate (`0%` false triggers on dense hyperspectral cubes). Drops `Cells3D` confocal microscopy `9.976 → 8.938 b/s` (**beating `cjxl 0.13` `9.618 b/s` by `−0.680 b/s` at `2.5×` faster decode**) and `Brain_MRI` `7.083 → 2.717 b/s` (`k=6`) / `2.796 b/s` (`k=0`, `160.5 MB/s`, closing `84%` of the gap to `cjxl`).
6. **Coverage-Guided `libFuzzer + ASan + UBSan` on Both `HND3`/`HND4` and `ASV4`**: Built and executed real LLVM 23.1.6 `libFuzzer + ASan + UBSan` harnesses ([`fuzz/fuzz_hnd3.cpp`](fuzz/fuzz_hnd3.cpp), [`fuzz/fuzz_asv4.cpp`](fuzz/fuzz_asv4.cpp)), caught and fixed one real signed-integer-overflow UB in `ldu_apply_stage`, proactively hardened all lifting/rANS/dequant/LUT arithmetic, and wired a `libFuzzer + ASan + UBSan` job into [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

> **Environment & Units**: Apple M3 Max (`16` logical CPUs, `128 GiB` RAM, macOS 27.2.1; Apple Clang 16.0.0 for release binaries, Homebrew LLVM 23.1.6 `clang++` for `libFuzzer + ASan + UBSan`). All sizes, RSS (from `/usr/bin/time -l`), and throughputs in this report use **decimal `MB = 10⁶ B`**. Reproduce via `python3 bench/bench_hnd3.py hnd4_tiled`, `python3 bench/bench_hnd3.py near_lossless`, and `python3 bench/bench_ccsds123_b2.py`.

---

## 1. HND4 Streaming Spatial Tiling: Peak RSS, Bitrate Delta, & Single-Tile Random Access

In `HND3 v3`, whole-cube `int64` working buffers drove Peak Encode/Decode RSS to `~17–22×` raw `i16` size (`2,979 / 2,169 MB` on the `132.84 MB` AVIRIS holdout). In `HND4 v4`, `encode_file` and `decode_file` stream one spatial tile `(B, th, tw)` at a time while sharing ONE global `LDU`-RKLT model (sampled via strided reads using `< 16 MB` RAM) and storing a compact `140 B/tile` Tile Index Table (`offset`, `tile_bytes`, `layer_bytes[16]`, `layer_crc32c[16]`).

All rows below were verified bit-exact lossless against the original uncropped raw files (`SHA256=PASS`, `CRC32C=PASS`).

### AVIRIS Holdout `(224, 745, 398)` — `132.84 MB` (`i16`)

| Format / Tile Grid | `k` | Bytes | `b/s` | Delta vs Untiled | Enc 16T | Dec 16T | **Enc RSS** | **Dec RSS** | 1-Tile Random Access (`lossless / k=6`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HND3` untiled | `0` | `25,468,981` | `3.068` | base | `78.9 MB/s` | `124.6 MB/s` | `2,981 MB` | `2,169 MB` | — |
| **`HND4 tile=128` (`6×4`)** | `0` | `25,894,683` | `3.119` | **`+0.051`** | `75.6 MB/s` | **`114.1 MB/s`** | **`225 MB` (`13.2×↓`)** | **`144 MB` (`15.1×↓`)** | **`45.2 ms` / —** |
| **`HND4 tile=256` (`3×2`)** | `0` | `25,629,782` | `3.087` | **`+0.019`** | `82.0 MB/s` | **`119.3 MB/s`** | **`698 MB` (`4.3×↓`)** | **`427 MB` (`5.1×↓`)** | `93.9 ms` / — |
| `HND4 tile=512` (`2×1`) | `0` | `25,574,961` | `3.080` | `+0.013` | `79.2 MB/s` | `118.3 MB/s` | `2,809 MB` | `1,629 MB` | `215.4 ms` / — |
| `HND3` untiled | `6` | `24,408,930` | `2.940` | base | `61.7 MB/s` | `99.0 MB/s` | `2,931 MB` | `2,168 MB` | — |
| **`HND4 tile=128` (`6×4`)** | `6` | `25,336,106` | `3.052` | **`+0.112`** | `57.5 MB/s` | **`85.3 MB/s`** | **`222 MB` (`13.2×↓`)** | **`151 MB` (`14.4×↓`)** | **`63.7 ms` / `35.3 ms`** |
| **`HND4 tile=256` (`3×2`)** | `6` | `24,656,971` | `2.970` | **`+0.030`** | `66.1 MB/s` | **`96.7 MB/s`** | **`698 MB` (`4.2×↓`)** | **`437 MB` (`5.0×↓`)** | `127.1 ms` / `72.8 ms` |
| `HND4 tile=512` (`2×1`) | `6` | `24,501,560` | `2.951` | `+0.011` | `66.9 MB/s` | `98.5 MB/s` | `2,789 MB` | `1,628 MB` | `290.8 ms` / `163.5 ms` |

### Salinas `(204, 512, 217)` (`45.33 MB`), PaviaU `(103, 610, 340)` (`42.72 MB`), & Indian Pines `(200, 145, 145)` (`8.41 MB`)

| Dataset | Format / Tile Grid | `k` | `b/s` | Delta | Enc / Dec 16T | **Enc RSS** | **Dec RSS** | 1-Tile Random Access (`lossless / k=6`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Salinas** | `HND3` untiled | `0 / 6` | `4.657 / 4.628` | base | `58.5 / 121.3` (`k=0`) | `1,049 / 1,044 MB` | `755 MB` | — |
| **Salinas** | `HND4 tile=128` (`4×2`) | `0` | `4.737` | `+0.080` | `70.3 / 113.7 MB/s` | **`183 MB` (`5.7×↓`)** | **`108 MB` (`7.0×↓`)** | `33.9 ms` / — |
| **Salinas** | `HND4 tile=128` (`4×2`) | `6` | `4.754` | `+0.126` | `56.1 / 85.7 MB/s` | **`206 MB` (`5.1×↓`)** | **`119 MB` (`6.3×↓`)** | `50.6 ms` / `25.7 ms` |
| **Salinas** | `HND4 tile=256` (`2×1`) | `0 / 6` | `4.671 / 4.646` | `+0.014 / +0.019` | `71.9 / 120.5` (`k=0`) | `502 MB` | `292 MB` | `127.5 ms` (`k=0`) |
| **PaviaU** | `HND3` untiled | `0 / 6` | `6.366 / 6.354` | base | `90.2 / 136.0` (`k=0`) | `1,016 / 995 MB` | `711 MB` | — |
| **PaviaU** | `HND4 tile=128` (`5×3`) | `0` | `6.542` | `+0.176` | `88.2 / 123.7 MB/s` | **`121 MB` (`8.4×↓`)** | **`70 MB` (`10.2×↓`)** | `22.5 ms` / — |
| **PaviaU** | `HND4 tile=128` (`5×3`) | `6` | `6.626` | `+0.272` | `62.3 / 77.2 MB/s` | **`117 MB` (`8.5×↓`)** | **`81 MB` (`8.8×↓`)** | `36.2 ms` / `17.6 ms` |
| **PaviaU** | `HND4 tile=256` (`3×2`) | `0 / 6` | `6.416 / 6.438` | `+0.049 / +0.084` | `89.2 / 119.0` (`k=0`) | `493 MB` | `296 MB` | `28.5 ms` (`k=0`) |
| **Indian Pines** | `HND3` untiled | `0 / 6` | `6.485 / 6.561` | base | `90.4 / 143.6` (`k=0`) | `204 / 209 MB` | `141 / 150 MB` | — |
| **Indian Pines** | `HND4 tile=64` (`3×3`) | `0` | `7.162` | `+0.677` | `59.7 / 88.9 MB/s` | **`71 MB`** | **`33 MB`** | `15.0 ms` / — |
| **Indian Pines** | `HND4 tile=128` (`2×2`) | `0 / 6` | `6.734 / 6.921` | `+0.249 / +0.360` | `63.1 / 108.7` (`k=0`) | `173 MB` | `97 MB` | `7.3 ms` (`k=0`) |

**Engineering trade-off (and v4.0.2 tile-row strip I/O upgrade)**: Initially, multi-column BSQ tiled decode issued `O(B · H · n_tx)` small row writes (`688,128` 256-byte syscalls on AVIRIS `tile=128`), capping disk decode at `51.0 MB/s` (`59.9 MB/s` after `v4.0.1`). Adding a bounded `≤ 64 MiB` multi-column tile-row strip buffer in `decode_file()` (`22.8 MB` on AVIRIS `tile=128`, parallel scatter across bands/rows, flushed as `B` contiguous band chunks for BSQ or 1 contiguous chunk per tile-row for BIL/BIP, with automatic fallback for ultra-wide swaths `> 64 MiB`) lifted end-to-end AVIRIS `tile=128` disk decode from `59.9` to **`114.1 MB/s`** (`1.90×` vs. `v4.0.1`, `2.24×` vs. initial `v4.0`'s `51.0 MB/s`; `91.6%` of untiled `124.6 MB/s`) while keeping Peak Decode RSS at **`144 MB` (`15.1×` below untiled)** — and lifted Salinas, PaviaU, and Indian Pines `tile=128` disk decode by `+75%` to `+81%` (`91–94%` of untiled). Because Peak RSS scales with `O(B · T² + min(64 MiB, B · T · W))`, memory stays strictly bounded on multi-gigabyte swaths. Avoid `T=64` on small `145×145` cubes where 16-stripe rANS tables cost `+0.68 b/s`.

---

## 2. Guaranteed-$L_\infty$ Near-Lossless Coding (`--max-error E`) on Full Uncropped Scenes

HND4 implements a uniform centered sample-domain quantizer (`step = 2E + 1`, `q = sign(x) · floor((|x| + E) / step)`, `x_hat = clamp(q · step, dtype_min, dtype_max)`) before `LDU`-RKLT and 2D 5/3 lifting. Every sample across `i16`, `u16`, and `i32` (including `INT16_MIN/MAX`, `UINT16_MAX`, `INT32_MIN/MAX` tested in [`test_phase4.py`](test_phase4.py)) strictly satisfies `|x - x_hat| ≤ E`, verified by default-on `--verify` (`VERIFY=PASS(Linf<=E)`).

| Scene (`tile_size`) | `E=0` (Lossless) | `E=1` (`b/s` · `PSNR` · `SAM`) | `E=2` (`b/s` · `PSNR` · `SAM`) | `E=4` (`b/s` · `PSNR` · `SAM`) | `E=8` (`b/s` · `PSNR` · `SAM`) | Dec 16T (`E=1..8`) |
|---|---:|---:|---:|---:|---:|---:|
| **Indian Pines** (`untiled`) | `6.485 b/s` (`2.47×`) | `5.013` · `80.50 dB` · `0.015°` | `4.406` · `75.77 dB` · `0.026°` | `3.793` · `70.49 dB` · `0.048°` | `3.249` · `64.94 dB` · `0.091°` | `107–113 MB/s` |
| **Salinas** (`tile=256`) | `4.671 b/s` (`3.43×`) | `3.467` · `81.05 dB` · `0.031°` | `3.047` · `76.29 dB` · `0.054°` | `2.733` · `71.04 dB` · `0.099°` | `2.601` · `65.49 dB` · `0.187°` | `110–114 MB/s` |
| **PaviaU** (`tile=256`) | `6.416 b/s` (`2.49×`) | `5.021` · `79.82 dB` · `0.036°` | `4.482` · `75.05 dB` · `0.062°` | `3.933` · `69.82 dB` · `0.114°` | `3.499` · `64.26 dB` · `0.215°` | `104–109 MB/s` |
| **AVIRIS Holdout** (`tile=256`) | `3.087 b/s` (`5.18×`) | `2.478` · `68.26 dB` · `0.104°` | `2.087` · `63.86 dB` · `0.178°` | `1.651` · `57.97 dB` · `0.337°` | **`0.806`** (`19.85×`) · `53.73 dB` · `0.569°` | `94–116 MB/s` |

*(Every entry strictly verified `MaxErr == E` on decode with `CRC32C=PASS`.)*

---

## 3. Apples-to-Apples vs. Official CCSDS-123.0-B-2 (Issue 2: Hybrid Entropy Coder)

Benchmarked against the MIT-licensed WHISPERS 2024 **NTNU-SmallSat-Lab CCSDS-123.0-B-2 High-Level Verification Model** (verified bit-exact against official CCSDS `Test1-20190201` test vectors and the CNES reference encoder; standard `FULL` prediction, `WIDE_NEIGHBOR_ORIENTED` local sum, `BIL`, `HYBRID` entropy coder, `Ω=13`, `t_inc=64`, `ν_min=-1`, `ν_max=3`, near-lossless `φ=3/8, ψ=3/8`; reproducible via [`bench/bench_ccsds123_b2.py`](bench/bench_ccsds123_b2.py)). Because the NTNU verification model is pure Python (`~10k samples/s`), we ran **full uncropped Indian Pines `(200, 145, 145)`** plus **exact matched centered `(B, 64, 64)` crops** across all four scenes.

### 3.1 Full Uncropped `Indian Pines` `(200, 145, 145)` (`8.41 MB`)

| `E` ($L_\infty$) | CCSDS-123.0-B-2 (`P=3`, Hybrid) | B2 `PSNR` / `SAM` | **HND4 (`k=0`, CRC32C)** | HND4 `PSNR` / `SAM` | Delta (`HND4 − B2`) |
|---:|---:|---:|---:|---:|---:|
| **`E=0` (lossless)** | `3,456,794 B` (`6.577 b/s`) *(P=15: `6.544 b/s`)* | `∞` / `0.000°` | **`3,408,591 B` (`6.485 b/s`)** | `∞` / `0.000°` | **`−0.092 b/s`** (`−0.059` vs `P=15`) |
| `E=1` | **`2,563,682 B` (`4.877 b/s`)** | `80.50 dB` / `0.015°` | `2,634,985 B` (`5.013 b/s`) | `80.50 dB` / `0.015°` | `+0.136 b/s` |
| `E=2` | **`2,184,976 B` (`4.157 b/s`)** | `75.73 dB` / `0.026°` | `2,315,886 B` (`4.406 b/s`) | `75.77 dB` / `0.026°` | `+0.249 b/s` |
| `E=4` | **`1,762,323 B` (`3.353 b/s`)** | `70.50 dB` / `0.048°` | `1,993,930 B` (`3.793 b/s`) | `70.49 dB` / `0.048°` | `+0.441 b/s` |

### 3.2 All Four Scenes on Matched Centered `(B, 64, 64)` Crops

| Scene Crop | `E=0` B2 (`P=3 / P=15`) vs **HND4** | `E=1` B2 (`P=3`) vs **HND4** | `E=2` B2 (`P=3`) vs **HND4** | `E=4` B2 (`P=3`) vs **HND4** |
|---|---:|---:|---:|---:|
| **Indian Pines** `(200, 64, 64)` | `6.509 / 6.505` vs `6.816` (`+0.308`) | **`4.787`** vs `5.287` (`+0.500`) | **`4.066`** vs `4.636` (`+0.569`) | **`3.260`** vs `4.006` (`+0.746`) |
| **Salinas** `(204, 64, 64)` | `5.015 / 5.044` vs `5.153` (`+0.138`) | **`3.340`** vs `3.765` (`+0.425`) | **`2.649`** vs `3.273` (`+0.624`) | **`1.911`** vs `2.831` (`+0.920`) |
| **PaviaU** `(103, 64, 64)` | `6.783 / 6.849` vs `7.473` (`+0.690`) | **`5.372`** vs `5.875` (`+0.502`) | **`4.643`** vs `5.198` (`+0.555`) | **`3.774`** vs `4.465` (`+0.690`) |
| **AVIRIS Holdout** `(224, 64, 64)` | `3.052 / 3.038` vs `3.288` (`+0.236`) | **`1.510`** vs `2.340` (`+0.830`) | **`0.942`** vs `1.817` (`+0.875`) | **`0.473`** vs `1.538` (`+1.065`) |

### Honest Architectural Analysis (Where CCSDS-123.0-B-2 Wins vs. Where HND4 Wins)
- **Why CCSDS-123.0-B-2 beats HND4 cleanly in near-lossless (`+0.14` to `+1.07 b/s`)**: CCSDS-123.0-B-2 places the uniform scalar quantizer **inside the causal DPCM prediction loop** (`q_z(t) = quant(s_z(t) - s_hat_z(t))`) with **zero post-quantization lifting rounding noise**, and feeds the resulting zero-dominated stream into its 16-table **Hybrid Entropy Coder** (specifically engineered for sub-1-bit/sample coding, reaching `0.473 b/s` on AVIRIS at `E=4`). By contrast, HND4 applies open-loop sample-domain quantization *before* integer `LDU`-RKLT + 2D 5/3 lifting—so the integer lifting cascade injects an irreducible `~0.5–1.0`-unit rounding-noise floor in the *quantized* domain, which becomes a visible floor as `E` grows (`+0.44 b/s` on full Indian Pines at `E=4`, `+1.07 b/s` on low-entropy AVIRIS crop).
- **Why HND4 trailed on `64×64` lossless crops despite beating Issue 2 (`6.485 vs 6.544 b/s`) on full Indian Pines**: On a `64×64` crop (`4,096` pixels), HND4's `LDU` + 16-stripe rANS table headers cost `~0.3–0.6 b/s`, whereas on full scenes (`21k–296k` pixels) that header amortizes to `< 0.03 b/s`.
- **Where HND4 has clear systems advantages**: `94–116 MB/s` multi-core parallel decode (vs. strictly serial DPCM causal loops), single-stream 7-layer progressive previews (`0.6%–16%` of stream → `53–60 dB` with verified per-layer CRC32C), `22.5–63.7 ms` random-access single-tile decoding, and bounded `70–151 MB` streaming Decode RSS.

---

## 4. Auto-Gated Reversible Monotonic Histogram Compaction LUT (`FLAG_HIST_LUT`)

Scientific 16-bit 3D volumes outside hyperspectral imaging (confocal microscopy, medical MRI, non-linearly calibrated detectors) frequently occupy only a sparse subset of the 16-bit code space (e.g., `Brain_MRI` uses `226` squared intensities $k^2 \in [0, 51076]$ with `adj_ratio = 0.004`; `Cells3D` uses `10,749` non-adjacent intensities with `adj_ratio = 0.000`). Previously, those large empty gaps inflated 5/3 wavelet detail coefficients by `4–200×`, costing `7.083 b/s` on `Brain_MRI` and `9.976 b/s` on `Cells3D`.

`HND4` adds a zero-overhead, strictly order-preserving 16-bit histogram compaction LUT (`FLAG_HIST_LUT = 1 << 6`, `--hist-lut auto|on|off`) stored as a compact prefix inside `spec_bytes` (covered by `HeaderCRC32C`):
- **`fmt=1` (`K ≤ 4,096`)**: `1 + 2 + 2K` B strictly ascending value table (`455 B = 0.006 b/s` on `Brain_MRI`).
- **`fmt=2` (`K > 4,096`)**: `1 + 8,192` B raw 65,536-bit presence bitmap (`8,193 B = 0.017 b/s` on `Cells3D`).
- **Auto-gate (`--hist-lut auto`, default)**: Fires *only* on 16-bit lossless cubes when `K ≥ 2`, `K / span < 0.50`, `adj_pairs / (K - 1) < 0.50`, and `wire_bytes × 8 / N < 0.15 b/s`. Because all four dense hyperspectral cubes sit at `adj_ratio = 0.974–0.999` vs. `0.000–0.004` on sparse volumes (`> 240×` separation), dense hyperspectral cubes have a **0% trigger rate** and remain **byte-for-byte identical `HND3 v3`** (`0.000 b/s` delta). Because the map is strictly monotonic, progressive previews (`--stop-layer K`) map cleanly back onto the exact original palette via `inv[clamp(v, 0, K - 1)]`.

| Non-Hyperspectral 3D Volume | `HND3` (LUT off) | **HND4 `k=0` (`auto` & `--blind`)** | **HND4 `k=6` (7-layer)** | **HND4 `k=0` (`mode=none`)** | `cjxl 0.13 (-d 0 -e 7)` | Delta vs `cjxl` |
|---|---:|---:|---:|---:|---:|---:|
| **`Cells3D` Microscopy** `(60, 256, 256) u16` (`7.86 MB`) | `9.976 b/s` | **`8.938 b/s`** (`139.8 MB/s` dec; `--blind`: `136.4 MB/s`) | `9.024 b/s` (`102.0 MB/s`) | **`8.892 b/s`** (`143.4 MB/s`) | `9.618 b/s` (`55.7 MB/s`) | **`−0.680 b/s` (`2.5×` faster dec)** |
| **`Brain_MRI`** `(10, 256, 256) u16` (`1.31 MB`) | `7.083 b/s` | `2.796 b/s` (`160.5 MB/s` dec; `--blind`: `153.2 MB/s`) | **`2.717 b/s`** (`94.1 MB/s`) | `2.667 b/s` (`163.1 MB/s`) | **`1.883 b/s`** (`42.3 MB/s`) | `+0.834 b/s` (`84%` of gap closed; `3.8×` faster dec) |
| **Dense Hyperspectral (`Indian Pines` / `Salinas` / `PaviaU` / `AVIRIS`)** | `6.485 / 4.657 / 6.366 / 3.068` | **Identical (`0.000 b/s` delta, `HND3 v3`, `lut=0`)** | Identical | — | — | 0% false triggers |

*(Honest note: on `Brain_MRI`, `FLAG_HIST_LUT` closes `84%` of the gap to `cjxl` and beats `zstd -19` `4.226`, `CharLS` `6.185`, and `OpenJPEG` `6.749` easily; `cjxl`'s remaining `0.83 b/s` edge comes from its 2D MA-tree zero-block predictor on large flat zero-background corners.)*

---

## 5. Real Coverage-Guided `libFuzzer + ASan + UBSan` Results (LLVM 23.1.6)

Both codecs now have dedicated coverage-guided `libFuzzer` harnesses built with Homebrew LLVM 23.1.6 (`-O2 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all`, `make fuzz`) and wired into GitHub Actions `ubuntu-latest` CI:

| Target | Harness | Seeds / Scope | Executed Units | Rate | New Coverage Units | Peak RSS | Findings & Fixes |
|---|---|---|---:|---:|---:|---:|---|
| **`HND3` v3 + `HND4` v4** | [`fuzz/fuzz_hnd3.cpp`](fuzz/fuzz_hnd3.cpp) | `v3 k=0/k=6/clust/blind` + `v4 tiled/NL/u16/i32` + **`fmt=1` & `fmt=2` `FLAG_HIST_LUT` seeds**; exercises `inspect`, full decode, progressive preview, random-access tile, **and `verify_crc=false` checksum-bypassed deep parsing** | **81,785** (`46s`) + **47,764** (`26s` LUT pass) | `1,777–1,837/s` | `841` + `1,235` (`1,307` edges / `4,304` features) | `458 MB` | **Caught 1 real UB** in initial pass at exec #24,975 (`__ubsan_handle_mul_overflow_abort` in `ldu_apply_stage` on crafted `cq × tok31`). Fixed via `__int128` + unsigned modular helpers across `LDU`, 5/3 lifting, rANS activity/shift, `dequant_mid`, and `nl_dequant`. **0 crashes/ASan/UBSan/leaks post-fix & across LUT pass.** |
| **`ASV3` / `ASV4`** | [`fuzz/fuzz_asv4.cpp`](fuzz/fuzz_asv4.cpp) | Seeded across **all 7 transform IDs** (`T0`, `T1`, `T2_XOR`, `T2_ELEM`, `T2_BIT`, `T2_NIBROT`, `T3_2D`) with exact 24-byte `MetaRecord` payloads via `fmemopen()` into `asv3::run_decode` | **164,475** (`31s`) + **120,629** (`26s` exact-record pass) | `4,639–5,305/s` | `535` + `567` | `117 MB` | **0 crashes, 0 ASan errors, 0 UBSan errors, 0 leaks.** |

---

## 6. Verification Summary

- `make all` compiles `./asv4`, `./hnd2`, `libhnd3.a`, and `./hnd3` with **0 warnings** under `-Wall -Wextra`.
- `make test` passes **23/23 unit & integration tests** across `test_phase1.py`, `test_phase2.py`, `test_phase3.py`, and [`test_phase4.py`](test_phase4.py) (tiled BSQ/BIL/BIP ragged edge tiles & tile-row strip decode across `i16`/`u16`/`i32`, `FLAG_HIST_LUT` `fmt=1` sparse `u16`/`i16` & `fmt=2` bitmap `u16` untiled/tiled/progressive-palette/`--blind` + real `Brain_MRI`/`Cells3D` round-trips, dense full-range `±1.9e9` `int32` multi-band UBSan round-trip, random-access single-tile & preview decode, per-header/tile-index/layer CRC32C tamper & truncation checks, guaranteed-$L_\infty$ near-lossless including dtype min/max extremes, blind `HND4` `FLAG_BLIND` preservation, header-only `hnd3 info` CRC32C verification, strict integer/shape CLI validation on both `hnd3` and `asv4`, and standalone C99 `cc -std=c99` linkage against `libhnd3.a`), plus UBSan + portable FIPS 180-4 SHA-256 + ARM64 NEON↔scalar bit-exact parity + 5,000-iteration `libc++`/`libstdc++`-hardened debug fuzz.
- **Post-release audit & v4.0.2 upgrades**:
  - Fixed a latent signed `int64_t` overflow in `pack_rans_stripe` (`cpp/hnd3_rans.hpp`) on dense full-range `±1.9e9` `int32` cubes by factoring ONE shared `unsigned __int128` activity/EMA helper used identically by encoder and decoder.
  - Hardened `complete_pivoted_ldu` against degenerate/singular pivots and `int32_t` quantization overflows, added `std::endian::little` static check, added `layer_bytes` sum validation and full `ifstream` short-read/64-bit-overflow checks in `encode_file`/`decode_file`.
  - Optimized `HND4` streaming file I/O (`extract_tile_from_file` contiguous `≤ 4 MB` BSQ band-strip reads + `decode_file` `≤ 64 MiB` multi-column tile-row strip write buffer across BSQ/BIL/BIP), lifting AVIRIS `tile=128` (`k=0`) encode from `53.6 → 75.6–84.5 MB/s` and full-scene disk decode from `51.0 → 59.9 → 114.1 MB/s` (`2.24×` end-to-end across `v4.0.1` + `v4.0.2`; `91.6%` of untiled) at `144 MB` Decode RSS (`15.1×` below untiled).
  - Added auto-gated reversible monotonic histogram compaction LUT (`FLAG_HIST_LUT`, `--hist-lut auto|on|off`), lifting `Cells3D` past `cjxl 0.13` (`8.938 vs 9.618 b/s` at `2.5×` decode speed) and cutting `Brain_MRI` from `7.083` to `2.717–2.796 b/s` with `0%` false triggers on dense hyperspectral cubes.
  - Added header-only `hnd3::inspect_file`, strict `strtol`/`strtoull` CLI validation on both `./hnd3` and `./asv4` (closing an infinite-loop bug on `./asv4 c --block-mb 0`), and accurate `SHA256=` labels.
