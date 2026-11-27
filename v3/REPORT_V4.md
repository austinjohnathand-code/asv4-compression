# ASV4 Architecture & Benchmark Report (v4.1): Blind Schema-Free Compression Surpassing OpenZL (`le32`, `Float32Deconstruct`, `BFloat16Deconstruct`), Blosc2, and ZipNN on Real FP32 & BF16 LLM Weights

**Binary / Source**: `v3/cpp/asv3.cpp` (`VERSION = 4`, backward-compatible with `v3` streams; frozen `v3.0` archived at `v3/cpp/asv3_v3.0.cpp`)  
**Hardware / Toolchain**: Apple Silicon ARM64 (`arm64` NEON), 128 GB RAM, `clang++` C++20 (`-O3 -std=c++20 -mcpu=native -Wall -Wextra`), `libzstd 1.5.5`, `openzl 0.3.0`, `blosc2 2.7.1`, `zipnn 0.5.4`, Python 3.11 + `numpy 1.26.4`.  
**Canonical Data**: `v3/final_refresh_results_v4.1.csv` (32/32 live encode + streaming SHA-256 decode runs across 16 datasets / 4.95 GiB from one binary) | Frozen baselines: `v3/final_refresh_results_v3.0.csv`, `v3/final_refresh_results_v4.0.csv` | Full audit trail: `v3/AUDIT_V3.md`.

---

## 1. Executive Summary

**Research Thesis**: Can a blind, schema-free, single-pass, pipe-safe streaming compressor match or surpass schema-guided format engines (Meta's OpenZL `le32`/`le16`/`Float32Deconstruct`/`BFloat16Deconstruct`, Blosc2, and dedicated AI weight compressor ZipNN) on real-world production numeric/mixed files while keeping worst-case overhead on unstructured and adversarial inputs under `+0.15%`?

**ASV4 v4.1 Answer**: **Yes.** Building on the hardened, UBSan-clean `v3.0` C++20 NEON architecture, `v4.1` combines four focused improvements—fused ARM64 NEON **`T2_NIBROT`** across `w ∈ {2, 4, 8}` (`BF16`/`FP16`, `FP32`, `FP64`), **per-plane `zstd -1`/`-3` probe scoring**, a **combined-gain stationary gate**, and **high-entropy `T2_NIBROT` L19 level-capping** (with explicit retirement of `T3_2D` from encoder selection)—achieving:

- **Surpasses OpenZL (`le32`, `Float32Deconstruct`, `struct4+TransposeSplit`), Blosc2, and ZipNN on real GPT-2 FP32 weights (`real_gpt2_f32.safetensors`, OpenAI `gpt2`, 522.7 MB)**:
  - **ASV4 v4.1 (L3 & L19 monotonic)**: **`394.91 MB` (`−10.354%` vs `B2_capi` / `−10.375%` vs `B3a_capi`, `−24.45%` vs raw)** at **`211.0 MB/s`** single-threaded encode, **`1,251.5 MB/s`** L3 / **`1,240.9 MB/s`** L19 decode, **`154.3 MB`** Decode RSS, **`Regs = 2`**, and **`0.95×`** Work Mult.
  - Beats verified lossless **ZipNN 0.5.4 `float32` (`397.59 MB`, aligned)** by **2.68 MB**, **OpenZL `le32` (`398.43 MB`)** by **3.52 MB**, **Blosc2 `bitshuffle4+zstd7` (`401.61 MB`)** by **6.70 MB**, **OpenZL `Float32Deconstruct` (`402.50 MB` aligned / `432.17 MB` unaligned)** by **7.59–37.26 MB**, **OpenZL `struct4+TransposeSplit` (`404.12 MB`)** by **9.21 MB**, and **ASV3 `v3.0` (`410.42 MB`)** by **15.51 MB**—with **zero format knowledge**.
- **Surpasses OpenZL (`BFloat16Deconstruct`, `le16`), Blosc2, and ZipNN on real production BF16 LLM weights (`real_smollm2_bf16.safetensors`, HuggingFace `SmolLM2-135M`, 256.6 MB)**:
  - **ASV4 v4.1 (L3 & L19 monotonic)**: **`170.44 MB` (`−15.094%` vs `B2_capi` / `−15.063%` vs `B3a_capi`, `−33.58%` vs raw)** at **`208.3 MB/s`** encode, **`926.6 MB/s`** L3 / **`935.6 MB/s`** L19 decode, **`150.8 MB`** Decode RSS, **`Regs = 3`**, and **`0.85×`** Work Mult.
  - Beats verified lossless **ZipNN 0.5.4 `bfloat16` (`171.10 MB`, aligned)** by **0.66 MB**, **Blosc2 `bitshuffle2+zstd7` (`177.25 MB`)** by **6.81 MB**, **OpenZL `BFloat16Deconstruct` (`178.68 MB`)** by **8.24 MB**, and **OpenZL `le16` (`183.00 MB`)** by **12.56 MB**.
- **Within 0.14 MB of supervised OpenZL on real NASA/ESA Hubble astronomy frames (`real_hubble_eagle_f32.fits`, 29.5 MB big-endian `>f4`)**:
  - **ASV4 v4.1**: **`15.33 MB` (`−16.658%`)** L3 / **`15.28 MB` (`−16.860%`)** L19 (`146.3 MB/s` enc, `680.4 MB/s` dec, `Regs = 3`) vs verified supervised **OpenZL `le32`/`be32` (`15.14 MB`, `−17.66%`)**.
- **Monotonic `--level 19 ≤ --level 3` across all 16/16 datasets**:
  - Eliminated the `v4.0` GPT-2 L19 inversion (`401.29 MB → 394.91 MB`, L19 decode `886 → 1,241 MB/s`) and further improved `real_mixed_public.bin` L19 (`82.88 MB → 82.25 MB`, **`−5.920%`** vs `B3a_capi` at `942.3 MB/s` decode).
- **Zero-regression safety floor & flat `Work Mult`**:
  - Worst-case tax across all 5 negative & adversarial controls is **`+0.0021%` (L3) / `+0.0028%` (L19)**—over 50× below the `+0.15%` cap—with `adv_binaries.bin` locked at `Regs = 1`.
  - Retiring dead `T3_2D` from encoder selection unified and lowered Scoring Work Multiplier to **`0.89×` avg / `2.06×` max** for **both** L3 and L19 (every file well inside `≤ 3.0×`), with decode throughput **`680.4–1,364.3 MB/s` (L3)** and **`632.8–1,414.0 MB/s` (L19)** across every single dataset.

---

## 2. Head-to-Head vs. Supervised OpenZL, Blosc2, ZipNN, and Frozen `v3.0`

All external baselines below were verified lossless round-trip (`ozl.DCtx().decompress()` / `zipnn.decompress()` compared byte-for-byte against original input buffers; ZipNN & aligned OpenZL float graphs include the exact `.safetensors` header byte count so file-level totals are apples-to-apples). Note that ZipNN is specifically designed and optimized for multi-threaded model-loading throughput (reaching >1.1 GB/s single-threaded on aligned tensors here), whereas ASV4 targets general-purpose, single-threaded, schema-free streaming ratio and throughput without any `.safetensors` parser.

| Real Public Dataset | Orig (MB) | `B2_capi` (`z3`) | `B3a_capi` (`z19`) | `B6 Blosc2` (`zstd7`) | `B5 OpenZL` *(Round-Trip Verified)* | `ZipNN 0.5.4` *(Aligned, Verified)* | Frozen `ASV3 v3.0` (L3 / L19) | **Blind `ASV4 v4.1` (L3 / L19)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `real_gpt2_f32.safetensors` | 522.7 | 440.52 | 440.63 | 401.61 (`−8.86%`) | 398.43 (`le32`, `−9.58%`) / 402.50 (`f32Decon` aligned) / 404.12 (`tr4`) / 432.17 (`f32Decon` raw) | 397.59 (`float32`, `−9.77%`) | 410.42 (`−6.83%`) / 409.57 (`−7.05%`) | **394.91 (`−10.35%`) / 394.91 (`−10.38%`)** |
| `real_smollm2_bf16.safetensors` | 256.6 | 200.74 | 200.67 | 177.25 (`−11.67%`) | 178.68 (`bf16Decon`, `−10.96%`) / 183.00 (`le16`, `−8.81%`) | 171.10 (`bfloat16`, `−14.74%`) | *(added in v4.1)* | **170.44 (`−15.09%`) / 170.44 (`−15.06%`)** |
| `real_hubble_eagle_f32.fits` | 29.5 | 18.40 | 18.38 | — | **15.14** (`le32`/`be32`, `−17.66%`) | — | 15.78 (`−14.23%`) / 15.54 (`−15.45%`) | **15.33 (`−16.66%`) / 15.28 (`−16.86%`)** |
| `real_mixed_public.bin` | 283.9 | 99.66 | 87.42 | 131.51 | 97.18 (`serial`) | — | 96.18 (`−3.50%`) / 84.04 (`−3.86%`) | **94.28 (`−5.40%`) / 82.25 (`−5.92%`)** |
| `real_nyc_taxi_struct104.bin` | 256.0 | 53.66 | **36.89** | 64.37 | 51.38 (`serial`) / 63.93 (`struct104+T`) | — | 53.66 (`+0.001%`) / 36.89 (`+0.002%`) | **53.66 (`+0.001%`) / 36.89 (`+0.002%`)** |
| `real_nyc_taxi_columnar.arrow` | 803.1 | 107.72 | **87.83** | 148.84 | 112.47 (`serial`) | — | 107.72 (`+0.001%`) / 87.83 (`+0.002%`) | **107.72 (`+0.001%`) / 87.83 (`+0.002%`)** |

> **Audit Correction on `OpenZL 0.3.0` (`real_gpt2_f32.safetensors`)**: During v4 Phase 0 profiling, full chunk-by-chunk `ozl.DCtx().decompress()` verification caught a Python buffer-lifetime bug in our earlier `v3` OpenZL benchmark loop (`np.frombuffer(ch[:n], ...)` passed inline): on the trailing 33rd chunk (`10.71 MB`, where `len % 4 == 3` created a temporary slice), `openzl 0.3.0` dropped `9.005 MB`, producing the earlier phantom `389.42 MB` (`le32`) and `394.98 MB` (`struct4+T`) numbers. Keeping `ch_n` alive and verifying all 33 chunks via `DCtx` establishes the true lossless OpenZL sizes: **`398.43 MB`** (`le32`), **`402.50 MB`** (`Float32Deconstruct` aligned), **`404.12 MB`** (`struct4+TransposeSplit`), and **`432.17 MB`** (`Float32Deconstruct` on the raw unaligned `.safetensors` file).

---

## 3. Architecture of the `v4.1` Upgrades

ASV4 retains the exact single-pass, pipe-safe `ASV3` container (`Magic = "ASV3"`, `VERSION = 4`, zero backward seeks, lazy persistent `libzstd` contexts; `run_decode` transparently decodes both `VERSION = 3` and `VERSION = 4`) and adds **zero extra zstd streams** or framing overhead.

### 3.1 Bijective Nibble-Rotated Byte-Plane Transpose (`T2_NIBROT`, `tid = 6`)
For any element width $w \ge 2$ and direction bit $h \in \{0, 1\}$ (`h = 0` for little-endian `+1`, `h = 1` for big-endian `−1`; for `w == 2`, both directions are identical so only `h = 0` is evaluated):
- **Forward**:
  $$\text{rot}[r][c] = \big(\text{src}[r][c] \ll 4\big) \;\big|\; \big(\text{src}[r][(c \mp 1) \bmod w] \gg 4\big)$$
  followed by column-major transposition (`dst[c * rows + r] = rot[r][c]`).
- **Inverse** (exact bijection):
  $$\text{dst}[r][c] = \big(\text{rot}[r][c] \gg 4\big) \;\big|\; \big(\text{rot}[r][(c \pm 1) \bmod w] \ll 4\big)$$
- **Why it beats `T1`, `T2_BIT`, OpenZL, and ZipNN on FP32 & BF16 neural weights**:
  1. **Lower theoretical byte-plane entropy (capturing $I(E_{4..0}; M_{\text{top}})$) + eliminating greedy LZ77 false matches**: In Gaussian neural weights ($\mathcal{N}(0, 0.02)$), the steep density decay across each exponent octave creates real mutual information between the low exponent bits and top mantissa bits ($I(E_{4..0}; M_{6..4}) \approx +0.0523\text{ bits/bf16}$), while the top 3 exponent bits are nearly constant ($H(E_{7..5}) \approx 0$). Isolating `[E7..E0]` (`2.545 b`) and `[M6..M0, S]` (`7.972 b`) into separate byte lanes (as ZipNN / `BFloat16Deconstruct` do) discards that cross-lane coupling (`10.517 bits/bf16` theoretical byte-plane limit). In contrast, `T2_NIBROT(2, h=0)` packs `rot[0] = [M3..M0 | S, E7..E5]` (`5.000 b`) and `rot[1] = [E4..E0 | M6..M4]` (`5.465 b`), dropping total byte-plane entropy to **`10.465 bits/bf16`**—beating ZipNN's byte-plane limit by `0.052 bits/bf16`—while simultaneously raising minimum plane entropy above `5.0 bits/byte` so `zstd -3` stops chasing false 4-byte LZ77 matches and stays in pure Huffman mode at **99.95% of Shannon entropy**. The identical mechanism holds on `float32`: `T2_NIBROT(4, h=0)` drops total per-plane $H_0$ from `26.646 b/f32` (`T1(4)` lanes `[8.0, 8.0, 7.97, 2.68]`) to **`26.464 b/f32`** (`[5.0, 8.0, 8.0, 5.47]`, capturing `0.182 b/f32` of joint `(E4..E0, M22..M20)` coupling).
  2. **Alignment-free across phases**: `T2_NIBROT` operates without alignment search across all byte offsets. For `w == 4` (`FP32`), **3 of 4 phases (`tensor_start % 4 ∈ {0, 2, 3}`—including GPT-2's `14,291 % 4 = 3`)** preserve full intra-float `(E4..E0, M22..M20)` coupling (`26.464 b/f32`), while phase `1` retains the `5.0 / 5.65` anti-false-match Huffman benefit (`26.646 b/f32`). For `w == 2` (`BF16`), even offsets (`tensor_start % 2 == 0`, including SmolLM2's `30,536 % 2 = 0`) capture the full `0.052 b/bf16` intra-float coupling (`10.465 b/bf16`), while odd offsets (`tensor_start % 2 == 1`) fall back to `10.648 b/bf16` while preserving the anti-false-match Huffman effect (whereas `OpenZL Float32Deconstruct` degrades by `+29.67 MB` on unaligned `.safetensors`).
  3. **Memory-bandwidth-limited ARM64 NEON**: For `w == 2` (`vld2q_u8` / `vsliq_n_u8` / `vst2q_u8`) and `w == 4` (`vld4q_u8` / `vsliq_n_u8` / `vst4q_u8`), the nibble rotate fuses directly into the transpose with zero scratch buffer—delivering **`1,251.5 MB/s`** decode on GPT-2 FP32 and **`926.6–935.6 MB/s`** decode on SmolLM2 BF16.

### 3.2 Per-Plane `zstd -1` / `zstd -3` Probe Scoring (`w ∈ [2, 8]`)
In `v3.0`, scoring a 16 KB transposed probe in one `zstd -1` call forced a single Huffman table averaged across skewed exponent planes and uniform mantissa planes, blinding the selector to `T1` and `T2_NIBROT`. In `v4.1`, `TwoTierScorer::score_spec_l1` and `score_spec_l3` score each plane independently whenever `2 ≤ w ≤ 8` and `n / w ≥ 1024` (subtracting 12 B frame header per extra plane), matching `4 MB` streaming blocks (`≥ 512 KB/plane` = independent 128 KB zstd blocks) at zero extra `Work Mult`. Fundamental word widths `{2, 4, 8}` are pinned in `eval_w` (keeping Tier 1 at `≤ 4–5` candidates) so `BF16`, `FP32`, and `FP64` are never evicted by Miller-Madow $H_0$ refinement bias.

### 3.3 Combined-Gain Stationary Gate & Monotonic L19 Level Cap
- **Combined-Gain Gate**: Replaces rigid per-sample relative `0.94` rejection (which rejected 5 × 16 MB GPT-2 groups whenever one 256 KB sample hit a near-zero 4 MB causal attention mask where `+180 B` looked like `+119%` relative) with combined `(zk_a + zk_b) <= 0.94 * (z0_a + z0_b)` plus a strict per-sample absolute guard (`zk_i <= z0_i + 2048 B`).
- **Monotonic L19 Cap on High-Entropy `T2_NIBROT`**: When `c_xform` initializes on a `T2_NIBROT` flush group at `cfg.level > 3` whose first 256 KB compresses at `zstd -3` to `> 60%` of raw (uncorrelated neural weights like GPT-2 at `85%` or SmolLM2 BF16 at `69%`), `c_xform.cap_level_before_init(3)` prevents `ZSTD_btultra2` from emitting counterproductive cross-window LZ77 sequence headers (`+6.38 MB` in `v4.0`), while correlated rasters like Hubble FITS (`52% < 60%`) keep full L19 (`15.28 MB`).
- **Explicit Retirement of `T3_2D` from Encoder Selection**: `T3_2D` never fired on real Hubble FITS and stopped firing on `target_numeric_f32` under per-plane scoring in `v4.0`. Removing it from `TwoTierScorer` (while retaining `t3_2d_paeth_inverse` in `run_decode` for `v3` backward compatibility) dropped L19 `Work Mult` from `1.04× / 2.31×` to **`0.89× / 2.06×`** and tripled L19 decode speed on `target_numeric_f32` (`248.7 → 732.5 MB/s`).

---

## 4. Full 16-Dataset Same-Binary Release Table (`v3/final_refresh_results_v4.1.csv`)

Every L3 and L19 number below was measured live from the exact same compiled `v3/cpp/asv3` binary against exact in-process single-threaded `libzstd 1.5.5` C API (`B2_capi` = `zstd -3 --long=27`, `B3a_capi` = `zstd -19 --long=27`), with wall-clock throughput, OS Peak RSS (`/usr/bin/time -l`), and bit-exact streaming SHA-256 verification on all 32 runs (5,309,949,619 B / 4.95 GiB total corpus).

| Dataset | Orig MB | L3 vs `B2_capi` | L19 vs `B3a_capi` | Enc L3 (MB/s) | Dec L3 / L19 (MB/s) | Enc / Dec L3 RSS (MB) | Regs (L3/L19) | Work (L3/L19) | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `real_gpt2_f32.safetensors` | 522.7 | **−10.354%** *(394.91M)* | **−10.375%** *(394.91M)* | **211.0** | **1251.5 / 1240.9** | **154.2 / 154.3** | 2 / 2 | 0.95× / 0.95× | PASS |
| `real_smollm2_bf16.safetensors` | 256.6 | **−15.094%** *(170.44M)* | **−15.063%** *(170.44M)* | **208.3** | **926.6 / 935.6** | **151.8 / 150.8** | 3 / 3 | 0.85× / 0.85× | PASS |
| `real_hubble_eagle_f32.fits` | 29.5 | **−16.658%** *(15.33M)* | **−16.860%** *(15.28M)* | **146.3** | **680.4** / 636.3 | **111.8 / 89.6** | 3 / 3 | 1.04× / 1.04× | PASS |
| `real_mixed_public.bin` | 283.9 | **−5.401%** *(94.28M)* | **−5.920%** *(82.25M)* | 115.8 | **848.9 / 942.3** | 241.0 / 225.1 | 9 / 9 | 0.93× / 0.93× | PASS |
| `real_nyc_taxi_struct104.bin` | 256.0 | +0.001% | +0.002% | 112.5 | 715.0 / 985.7 | 191.0 / 194.6 | 1 / 1 | 0.62× / 0.62× | PASS |
| `real_nyc_taxi_columnar.arrow` | 803.1 | +0.001% | +0.002% | 118.5 | 986.4 / 1231.3 | 192.8 / 193.7 | 1 / 1 | 1.18× / 1.18× | PASS |
| `target_numeric_f32.bin` | 256.0 | **−41.758%** | **−33.479%** | 157.8 | 847.1 / **732.5** | 202.4 / 202.2 | 1 / 1 | 1.29× / 1.29× | PASS |
| `target_struct64.bin` | 256.0 | **−68.167%** | **−54.072%** | 179.7 | 695.7 / 691.5 | 194.8 / 198.1 | 1 / 1 | 0.78× / 0.78× | PASS |
| `target_sqlite.db` | 311.3 | **−9.566%** | **−6.554%** | 89.8 | 737.3 / 632.8 | 282.6 / 286.0 | 2 / 2 | 1.00× / 1.00× | PASS |
| `target_mixed_structured.bin` | 256.0 | **−42.944%** | **−32.186%** | 132.6 | 825.1 / **718.9** | 312.9 / 286.8 | 8 / 8 | 1.08× / 1.08× | PASS |
| `target_snapshots.bin` | 256.0 | +0.001% | +0.001% | 343.0 | 1364.3 / 1414.0 | 215.0 / 207.0 | 1 / 1 | 0.25× / 0.25× | PASS |
| `neg_enwik9.xml` (1 GB) | 953.7 | **+0.001%** | **+0.001%** | 108.4 | 779.0 / 781.0 | 197.4 / 196.8 | 1 / 1 | 0.73× / 0.73× | PASS |
| `neg_source_tar.tar` | 273.2 | **+0.002%** | **+0.002%** | 111.3 | 1040.9 / 1051.1 | 191.6 / 193.6 | 1 / 1 | 0.97× / 0.97× | PASS |
| `adv_binaries.bin` | 160.0 | **+0.002%** | **+0.003%** | 81.8 | 1136.5 / 1123.4 | 191.0 / 193.9 | 1 / 1 | 2.06× / 2.06× | PASS |
| `adv_compressed_media.tar` | 126.0 | **+0.000%** | **+0.000%** | 299.1 | 1332.3 / 1281.9 | 213.0 / 211.0 | 1 / 1 | 0.33× / 0.33× | PASS |
| `adv_random.bin` | 64.0 | **+0.000%** | **+0.000%** | 374.4 | 1128.7 / 1112.0 | 150.5 / 142.8 | 1 / 1 | 0.25× / 0.25× | PASS |

**v4.1 Release Totals (32/32 Live Runs, 16/16 Monotonic `L19 ≤ L3`)**:
`L3 & L19 Work Mult = 0.89× avg / 2.06× max` (`≤ 3.0×`) | `worst neg/adv tax = +0.0021% (L3) / +0.0028% (L19)` (`≤ +0.15%`) | `L3 decode = 680.4–1,364.3 MB/s` | `L19 decode = 632.8–1,414.0 MB/s` | `32/32 SHA-256 PASS`.

---

## 5. Security, Memory Bounds, and Honest Non-Wins

- **Decoder Memory Safety (UBSan Verified)**: Explicit bounds checks guard every `MetaRecord` read (`fhdr.wlog ∈ [10, 27]`, `delta_op == 0`, `rec.w >= 1`, `tail_len <= orig_len`, `r_pos + orig_len <= raw_q.size()`, `x_pos + xlen <= xform_q.size()`, `r_pos + tail_len <= raw_q.size()`, EOF trailer `read_exact`, and rejection of non-zero `S_RESID`). Verified 10/10 clean rejections on crafted corrupt/OOB inputs under `-fsanitize=undefined`, plus Unix `stdin`/`stdout` pipe tests on `real_smollm2_bf16.safetensors`, `real_hubble_eagle_f32.fits`, and `real_gpt2_f32.safetensors` under UBSan + portable FIPS 180-4 SHA-256 (`-DASV3_FORCE_PORTABLE_SHA256`).
- **Bounded $O(1)$ Working Memory**: Single-stream `S_RAW` files use `wlog = 27` (`128 MB` window; `143–211 MB` Decode RSS up through 1 GB `enwik9`). Dual-stream files cap uninitialized contexts to `wlog = 26` (`64 MB`): simultaneous wake (`real_gpt2_f32`, `real_smollm2_bf16`, `real_mixed_public`) uses `64 + 64 = 128 MB` combined window (`151–225 MB` RSS), while sequential wake (`target_sqlite`, `target_mixed_structured`) is bounded at `128 + 64 = 192 MB` combined window (`286–287 MB` RSS).
- **Honest Accounting of Non-Wins**:
  1. **`real_nyc_taxi_struct104.bin` & `real_nyc_taxi_columnar.arrow`**: Intra-row categorical correlations in NYC Taxi 104 B structs make raw LZ77 beat stride transpose by `+15.8%`, and 64K-row Apache Arrow RecordBatches interleave ~512 KB columns below the cross-stream savings floor. `ASV4` backs off cleanly to `T0` (`+0.001% / +0.002%`, `Regs = 1`).
  2. **`real_mixed_public.bin`**: 100% of the `−5.401%` L3 / `−5.920%` L19 savings come from cleanly isolating the 64.1 MB GPT-2 `float32` slice while leaving the surrounding tar, taxi struct, `enwik9`, and Arrow segments untouched at `T0`.
  3. **`target_numeric_f32.bin` L19 (`−33.48%` vs `v3.0`'s `−36.13%`)**: Explicit trade-off from retiring `T3_2D` from encoder selection in favor of `T2_XOR(4)` (3× faster L19 decode: `732.5` vs `248.7 MB/s`; lower global L19 `Work Mult`).

---

## 6. Reproducibility Commands

```bash
# Build release C++20 ARM64 NEON binary (zero warnings)
clang++ -O3 -std=c++20 -mcpu=native -Wall -Wextra \
  -I/opt/anaconda3/include -L/opt/anaconda3/lib -Wl,-rpath,/opt/anaconda3/lib \
  v3/cpp/asv3.cpp -lzstd -o v3/cpp/asv3

# Verify portable Linux/non-Apple SHA-256 + UBSan build
clang++ -O2 -std=c++20 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all \
  -DASV3_FORCE_PORTABLE_SHA256 \
  -I/opt/anaconda3/include -L/opt/anaconda3/lib -Wl,-rpath,/opt/anaconda3/lib \
  v3/cpp/asv3.cpp -lzstd -o /tmp/asv41_ubsan

# Pipe-safe single-pass encode + streaming SHA-256 decode on BF16 LLM weights
cat v3/corpus_real/real_smollm2_bf16.safetensors | ./v3/cpp/asv3 c --level 3 | ./v3/cpp/asv3 d > /dev/null

# Reproduce full 16-dataset L3 & L19 table (v3/final_refresh_results.csv)
python3 v3/bench_final_refresh.py
```
