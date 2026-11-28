# Phase 4 Systems & Capability Report (`phase4-v4.0` — `libhnd3` + `HND4` + `libFuzzer/ASan`)

Phase 4 addresses every architectural and verification gap identified in the end-of-Phase-3 audit while preserving 100% bit-for-bit wire compatibility with **HND3 v3** ([`phase3_codec.py`](phase3_codec.py)):

1. **Modular C++20 Library & C99 ABI (`libhnd3.a` + [`include/hnd3.h`](include/hnd3.h))**: Split the monolithic 2,165-line `cpp/hnd3.cpp` into [`cpp/hnd3_core.hpp`](cpp/hnd3_core.hpp), [`cpp/hnd3_rans.hpp`](cpp/hnd3_rans.hpp), [`cpp/hnd3_lib.cpp`](cpp/hnd3_lib.cpp), and a thin CLI driver [`cpp/hnd3_cli.cpp`](cpp/hnd3_cli.cpp) (`./hnd3`), exposing both C++20 (`hnd3::encode`/`decode`/`encode_file`/`decode_file`/`inspect`/`crc32c`) and `extern "C"` (`hnd3_encode_buffer`/`hnd3_decode_buffer`/`hnd3_free`) APIs verified by a standalone C99 test harness.
2. **HND4 Streaming Spatial Tiling (`--tile-size T`) & Random-Access Tile Decode (`dec --tile TY,TX`)**: Bounds encode/decode RAM to `O(B · T²)` via strided file sampling (`≤ 16,384` pixels) for one shared global `LDU`-RKLT model followed by tile-by-tile streaming I/O. Cuts Peak Decode RSS on the `132.84 MB` AVIRIS holdout by **18.2×** (`2,169 MB → 119 MB`) at `+0.051 b/s`.
3. **Per-Header & Per-Layer Castagnoli CRC32C (`--crc`, always on in HND4)**: Protects the 84 B header, shared `LDU` model, Tile Index Table, and every individual quality layer (`k = base_k..0`) of every tile—so **lossy progressive previews (`--stop-layer K`) and single-tile random-access decodes verify integrity (`CRC32C=PASS`)**.
4. **Guaranteed-$L_\infty$ Near-Lossless Coding (`--max-error E`) & CCSDS-123.0-B-2 Benchmark**: Uniform centered sample-domain quantizer guaranteeing `|x - x_hat| ≤ E` for all `i16`/`u16`/`i32` samples (including dtype min/max extremes), benchmarked head-to-head against the WHISPERS 2024 **NTNU CCSDS-123.0-B-2** verification model (Issue 2: Hybrid Entropy Coder + closed-loop near-lossless).
5. **Coverage-Guided `libFuzzer + ASan + UBSan` on Both `HND3`/`HND4` and `ASV4`**: Built and executed real LLVM 23.1.6 `libFuzzer + ASan + UBSan` harnesses ([`fuzz/fuzz_hnd3.cpp`](fuzz/fuzz_hnd3.cpp), [`fuzz/fuzz_asv4.cpp`](fuzz/fuzz_asv4.cpp)), caught and fixed one real signed-integer-overflow UB in `ldu_apply_stage`, proactively hardened all lifting/rANS/dequant arithmetic, and wired a `libFuzzer + ASan + UBSan` job into [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

> **Environment & Units**: Apple M3 Max (`16` logical CPUs, `128 GiB` RAM, macOS 27.2.1; Apple Clang 16.0.0 for release binaries, Homebrew LLVM 23.1.6 `clang++` for `libFuzzer + ASan + UBSan`). All sizes, RSS (from `/usr/bin/time -l`), and throughputs in this report use **decimal `MB = 10⁶ B`**. Reproduce via `python3 bench/bench_hnd3.py hnd4_tiled`, `python3 bench/bench_hnd3.py near_lossless`, and `python3 bench/bench_ccsds123_b2.py`.

---

## 1. HND4 Streaming Spatial Tiling: Peak RSS, Bitrate Delta, & Single-Tile Random Access

In `HND3 v3`, whole-cube `int64` working buffers drove Peak Encode/Decode RSS to `~17–22×` raw `i16` size (`2,979 / 2,169 MB` on the `132.84 MB` AVIRIS holdout). In `HND4 v4`, `encode_file` and `decode_file` stream one spatial tile `(B, th, tw)` at a time while sharing ONE global `LDU`-RKLT model (sampled via strided reads using `< 16 MB` RAM) and storing a compact `140 B/tile` Tile Index Table (`offset`, `tile_bytes`, `layer_bytes[16]`, `layer_crc32c[16]`).

All rows below were verified bit-exact lossless against the original uncropped raw files (`SHA256=PASS`, `CRC32C=PASS`).

### AVIRIS Holdout `(224, 745, 398)` — `132.84 MB` (`i16`)

| Format / Tile Grid | `k` | Bytes | `b/s` | Delta vs Untiled | Enc 16T | Dec 16T | **Enc RSS** | **Dec RSS** | 1-Tile Random Access (`lossless / k=6`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HND3` untiled | `0` | `25,468,981` | `3.068` | base | `82.9 MB/s` | `118.7 MB/s` | `2,979 MB` | `2,169 MB` | — |
| **`HND4 tile=128` (`6×4`)** | `0` | `25,894,683` | `3.119` | **`+0.051`** | `53.6 MB/s` | `51.0 MB/s` | **`229 MB` (`13.0×↓`)** | **`119 MB` (`18.2×↓`)** | **`90.2 ms` / —** |
| **`HND4 tile=256` (`3×2`)** | `0` | `25,629,782` | `3.087` | **`+0.019`** | `72.8 MB/s` | `79.8 MB/s` | **`695 MB` (`4.3×↓`)** | **`382 MB` (`5.7×↓`)** | `179.1 ms` / — |
| `HND4 tile=512` (`2×1`) | `0` | `25,574,961` | `3.080` | `+0.013` | `77.0 MB/s` | `94.4 MB/s` | `2,805 MB` | `1,630 MB` | `304.1 ms` / — |
| `HND3` untiled | `6` | `24,408,930` | `2.940` | base | `69.7 MB/s` | `98.5 MB/s` | `2,929 MB` | `2,168 MB` | — |
| **`HND4 tile=128` (`6×4`)** | `6` | `25,336,106` | `3.052` | **`+0.112`** | `47.1 MB/s` | `51.8 MB/s` | **`222 MB` (`13.2×↓`)** | **`126 MB` (`17.2×↓`)** | **`105.9 ms` / `74.8 ms`** |
| **`HND4 tile=256` (`3×2`)** | `6` | `24,656,971` | `2.970` | **`+0.030`** | `58.6 MB/s` | `69.0 MB/s` | **`685 MB` (`4.3×↓`)** | **`389 MB` (`5.6×↓`)** | `209.5 ms` / `150.5 ms` |
| `HND4 tile=512` (`2×1`) | `6` | `24,501,560` | `2.951` | `+0.011` | `65.9 MB/s` | `80.7 MB/s` | `2,792 MB` | `1,630 MB` | `383.0 ms` / `250.4 ms` |

### Salinas `(204, 512, 217)` (`45.33 MB`), PaviaU `(103, 610, 340)` (`42.72 MB`), & Indian Pines `(200, 145, 145)` (`8.41 MB`)

| Dataset | Format / Tile Grid | `k` | `b/s` | Delta | Enc / Dec 16T | **Enc RSS** | **Dec RSS** | 1-Tile Random Access (`lossless / k=6`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Salinas** | `HND3` untiled | `0 / 6` | `4.657 / 4.628` | base | `61.5 / 117.2` (`k=0`) | `1,050 MB` | `755 MB` | — |
| **Salinas** | `HND4 tile=128` (`4×2`) | `0` | `4.737` | `+0.080` | `52.4 / 60.6 MB/s` | **`201 MB` (`5.2×↓`)** | **`104 MB` (`7.3×↓`)** | `71.2 ms` / — |
| **Salinas** | `HND4 tile=128` (`4×2`) | `6` | `4.754` | `+0.126` | `42.9 / 50.8 MB/s` | **`205 MB` (`5.1×↓`)** | **`107 MB` (`7.1×↓`)** | `87.5 ms` / `62.1 ms` |
| **Salinas** | `HND4 tile=256` (`2×1`) | `0 / 6` | `4.671 / 4.646` | `+0.014 / +0.019` | `63.4 / 81.8` (`k=0`) | `514 MB` | `291 MB` | `203.7 ms` (`k=0`) |
| **PaviaU** | `HND3` untiled | `0 / 6` | `6.366 / 6.354` | base | `102.6 / 138.3` (`k=0`) | `1,018 MB` | `711 MB` | — |
| **PaviaU** | `HND4 tile=128` (`5×3`) | `0` | `6.542` | `+0.176` | `64.9 / 66.6 MB/s` | **`122 MB` (`8.3×↓`)** | **`62 MB` (`11.5×↓`)** | `42.9 ms` / — |
| **PaviaU** | `HND4 tile=128` (`5×3`) | `6` | `6.626` | `+0.272` | `43.9 / 53.6 MB/s` | **`122 MB` (`8.2×↓`)** | **`68 MB` (`10.5×↓`)** | `56.6 ms` / `38.9 ms` |
| **PaviaU** | `HND4 tile=256` (`3×2`) | `0 / 6` | `6.416 / 6.438` | `+0.049 / +0.084` | `73.1 / 77.9` (`k=0`) | `496 MB` | `276 MB` | `66.7 ms` (`k=0`) |
| **Indian Pines** | `HND3` untiled | `0 / 6` | `6.485 / 6.561` | base | `89.8 / 141.5` (`k=0`) | `208 MB` | `142 MB` | — |
| **Indian Pines** | `HND4 tile=64` (`3×3`) | `0` | `7.162` | `+0.677` | `34.2 / 37.0 MB/s` | **`82 MB`** | **`36 MB`** | `32.6 ms` / — |
| **Indian Pines** | `HND4 tile=128` (`2×2`) | `0 / 6` | `6.734 / 6.921` | `+0.249 / +0.360` | `45.0 / 48.4` (`k=0`) | `171 MB` | `86 MB` | `12.1 ms` (`k=0`) |

**Engineering trade-off**: Because Peak RSS scales with `O(B · T²)` rather than total scene area, `tile=128` keeps decode RAM at `62–126 MB` even on `224`-band scenes (and stays constant on multi-gigabyte swaths). The trade-off is per-tile BSQ row-seek I/O overhead (`~1.5–2.0×` slower full-scene decode than untiled `HND3`) and `~0.02–0.18 b/s` extra table/boundary cost at `T=128–256` (avoid `T=64` on small `145×145` cubes where 16-stripe rANS tables cost `+0.68 b/s`).

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
- **Where HND4 has clear systems advantages**: `94–116 MB/s` multi-core parallel decode (vs. strictly serial DPCM causal loops), single-stream 7-layer progressive previews (`0.6%–16%` of stream → `53–60 dB` with verified per-layer CRC32C), `43–106 ms` random-access single-tile decoding, and bounded `62–126 MB` streaming Decode RSS.

---

## 4. Real Coverage-Guided `libFuzzer + ASan + UBSan` Results (LLVM 23.1.6)

Both codecs now have dedicated coverage-guided `libFuzzer` harnesses built with Homebrew LLVM 23.1.6 (`-O2 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all`, `make fuzz`) and wired into GitHub Actions `ubuntu-latest` CI:

| Target | Harness | Seeds / Scope | Executed Units | Rate | New Coverage Units | Peak RSS | Findings & Fixes |
|---|---|---|---:|---:|---:|---:|---|
| **`HND3` v3 + `HND4` v4** | [`fuzz/fuzz_hnd3.cpp`](fuzz/fuzz_hnd3.cpp) | `v3 k=0/k=6/clust/blind` + `v4 tiled/NL/u16/i32`; exercises `inspect`, full decode, progressive preview, random-access tile, **and `verify_crc=false` checksum-bypassed deep parsing** | **81,785** (`46s` post-fix) | `1,777/s` | `841` | `454 MB` | **Caught 1 real UB** at exec #24,975 (`__ubsan_handle_mul_overflow_abort` in `ldu_apply_stage` on crafted `cq × tok31`). Fixed via `__int128` + unsigned modular helpers across `LDU`, 5/3 lifting, rANS activity/shift, `dequant_mid`, and `nl_dequant`. **0 crashes/ASan/UBSan/leaks post-fix.** |
| **`ASV3` / `ASV4`** | [`fuzz/fuzz_asv4.cpp`](fuzz/fuzz_asv4.cpp) | Seeded across **all 7 transform IDs** (`T0`, `T1`, `T2_XOR`, `T2_ELEM`, `T2_BIT`, `T2_NIBROT`, `T3_2D`) via `fmemopen()` into `asv3::run_decode` | **164,475** (`31s`) | `5,305/s` | `535` | `117 MB` | **0 crashes, 0 ASan errors, 0 UBSan errors, 0 leaks.** |

---

## 5. Verification Summary

- `make all` compiles `./asv4`, `./hnd2`, `libhnd3.a`, and `./hnd3` with **0 warnings** under `-Wall -Wextra`.
- `make test` passes **22/22 unit & integration tests** across `test_phase1.py`, `test_phase2.py`, `test_phase3.py`, and [`test_phase4.py`](test_phase4.py) (tiled BSQ/BIL/BIP ragged edge tiles across `i16`/`u16`/`i32`, dense full-range `±1.9e9` `int32` multi-band UBSan round-trip, random-access single-tile & preview decode, per-header/tile-index/layer CRC32C tamper & truncation checks, guaranteed-$L_\infty$ near-lossless including dtype min/max extremes, blind `HND4` `FLAG_BLIND` preservation, header-only `hnd3 info` CRC32C verification, strict integer/shape CLI validation on both `hnd3` and `asv4`, and standalone C99 `cc -std=c99` linkage against `libhnd3.a`), plus UBSan + portable FIPS 180-4 SHA-256 + ARM64 NEON↔scalar bit-exact parity + 5,000-iteration `libc++`/`libstdc++`-hardened debug fuzz.
- **Post-release full-stack audit fixes (`v4.0.1`)**:
  - Fixed a latent signed `int64_t` overflow in `pack_rans_stripe` (`cpp/hnd3_rans.hpp`) on dense full-range `±1.9e9` `int32` cubes by factoring ONE shared `unsigned __int128` activity/EMA helper used identically by encoder and decoder (preventing encoder/decoder context divergence under extreme `int32` ranges).
  - Hardened `complete_pivoted_ldu` against degenerate/singular pivots and `int32_t` quantization overflows, added `std::endian::little` static check, added `layer_bytes` sum validation and full `ifstream` short-read/64-bit-overflow checks in `encode_file`/`decode_file`.
  - Optimized `HND4` BSQ file I/O (`extract_tile_from_file` contiguous `≤ 4 MB` band-strip reads + full-width `scatter_tile_to_file` writes), lifting AVIRIS `tile=128` (`k=0`) encode throughput from `53.6` to **`84.5 MB/s`** (`+58%`) and disk decode from `51.0` to **`59.9 MB/s`** (`200.0 MB/s` in-memory `--bench`).
  - Added header-only `hnd3::inspect_file` (verifying `HeaderCRC32C` without loading multi-GB payloads), strict `strtol`/`strtoull` CLI validation on both `./hnd3` and `./asv4` (closing an infinite-loop bug on `./asv4 c --block-mb 0`), accurate `SHA256=` labels (`N/A (near-lossless)` / `N/A (single-tile)` / `N/A (no-out; CRC32C)` + in-memory SHA-256 on single-tile `--bench`), and verified `fuzz_asv4` (`120,629` runs, `0` errors) with exact 24-byte `MetaRecord` seeds covering all 7 transform IDs.
