# Phase 1 Report — End-to-End `.hnd1` Reversible Hyperspectral & N-D Codec

This report documents the falsification experiments, architecture breakthroughs, and full native-resolution **real-binary `.hnd1`** benchmark results of **Phase 1** ([`phase1_codec.py`](phase1_codec.py), verified by [`test_phase1.py`](test_phase1.py)).

---

## 1. Three Phase 1 Empirical Discoveries That Shaped the Codec

### 1.1 Gate 1A: Decomposing Phase 0's `R6` Oracle Gain (`+0.35` to `+1.77 b/s`)
Phase 0 found massive oracle block-KLT gains (`R6`, `4×4×4, K=16`) on all three real cubes. Before building a spatially switched spectral transform, we decomposed `R6` into **(a) fixed global basis `Q1` with per-cluster diagonal variances** vs. **(b) per-cluster basis rotation**:

| Real Cube (`(B, H, W)`) | `R6` Total Net Gain | **Variance-Only** (Fixed Global Basis) | **Basis Rotation** (Local Eigenvectors) | Tile `64×64` / `32×32` SERM vs. Global |
|---|---:|---:|---:|---|
| **Indian Pines** `(200, 144, 144)` | `+1.773 b/s` | **`+1.679 b/s` (94.7%)** | `+0.093 b/s` (5.3%) | Worse by `+0.12` / `+0.33 b/s` |
| **Salinas** `(204, 512, 216)` | `+0.833 b/s` | **`+0.808 b/s` (97.0%)** | `+0.026 b/s` (3.0%) | Worse by `+0.08` / `+0.26 b/s` |
| **PaviaU** `(103, 608, 336)` | `+0.353 b/s` | **`+0.320 b/s` (90.7%)** | `+0.033 b/s` (9.3%) | Worse by `+0.19` / `+0.50 b/s` |

> **Verdict (Gate 1A)**: **91%–97% of `R6`'s gain is local variance heteroscedasticity (entropy-coder context modeling), NOT basis rotation.** Tiled spatial SERM and causal `LL`-clustered `K`-SERM fail to amortize boundary discontinuities and headers. Phase 1 therefore drops spatial basis switching and puts its complexity budget into **(1) zero-fallback full/multi-PC spectral decorrelation** and **(2) 3D causal context modeling + real 32-bit rANS**.

### 1.2 Complete-Pivoted `LDU`-RKLT (Eliminating Givens Fallbacks at Arbitrary $M$)
In Phase 0, single-row-`S` PLUS factorization fell back to Givens on $M \ge 16$ (`fb = 6–12`, rounding noise `1.1–3.9 LSB²`), leaving a `0.16–0.35 b/s` gap to `float KLT`. Replacing it with **complete-pivoted `LDU`-RKLT** ($P\,\mathrm{diag}(s_0)\,A\,Q = L\,\mathrm{diag}(s_d)\,D_{\text{ladder}}\,U$):
- Enforces $|L_{ij}| \le 1$, $|U_{ij}| \le 1$ everywhere (stored losslessly in **Q15 `int16`**!).
- Factors $D$ via a greedy log-balanced ordering (`c_max = 1.44–2.08` even for full **$204 \times 204$** KLT!) and the Daubechies-Sweldens 3-rounding diagonal ladder (`y -= c x; x += (1/c - 1) y; y += x; x += (c - 1) y`).
- Achieves **zero Givens fallbacks for every $M$ tested (`M = 2..204`)** and cache-tiled parallel Numba execution (`22–37 MB/s` encode, `29–46 MB/s` decode).
- `"auto"` mode selects **Full $B \times B$ `LDU`-RKLT** whenever its header amortizes `< 0.04 b/s` (`Salinas`, `PaviaU`), and **Multi-PC Clustered `LDU`-RKLT (`gs=16, keep_pcs=3`)** on smaller cubes (`Indian Pines`).

### 1.3 Gate 1B: Exact $L_2$-Weighted 2D 5/3 Quantization Kills `R4` Everywhere
Computing the exact biorthogonal $L_2$ synthesis norms of 3-level 2D 5/3 lifting (`LL = 5.3747`, `L3-LH/HL = 2.9199`, `L3-HH = 1.5861`, `L2-LH/HL = 1.5923`, `L2-HH = 0.9219`, `L1-LH/HL = 1.0383`, `L1-HH = 0.7188`) and setting $q_s = \max(1, \mathrm{round}(q_{\text{base}} / g_s))$ boosts `2D 5/3` PSNR by **`+1.7` to `+3.1 dB`** at equivalent rates and beats `R4` (folded 4×4 orthonormal RKLT) across the entire rate–distortion curve on all three real cubes.

---

## 2. Real-Binary `.hnd1` Lossless Results vs. Blue Book CCSDS-123

All `.hnd1` numbers below are **actual serialized binary container bytes** (magic `HND1`, SHA-256 digest, `LDU` lifting headers, 3D-context rANS frequency tables, rANS words, and packed extra bits) verified with **100% bit-exact decode & SHA-256 PASS** on **full native uncropped cubes** (including odd spatial dimensions `145×145` and `512×217`!).

### 2.1 Apples-to-Apples Comparison on Cropped `//8` Cubes (Matching Phase 0 CCSDS-123)

| Dataset | Standard CCSDS-123 Blue Book (`Rice`) | CCSDS-123 Blue Book + **Our `3D-rANS`** | **Phase 1 `LDU-RKLT + 2D 5/3 + 3D-rANS`** | Gain vs. Std CCSDS-123 | Gain vs. CCSDS-123 + `3D-rANS` |
|---|---:|---:|---:|---:|---:|
| **Indian Pines** `(200, 144, 144)` | `6.734 b/s` (`P=15`) | `6.386 b/s` (`P=15`) | **`6.423 b/s`** (`LDU16`, `2.49×`) | **`-0.311 b/s` (`+4.6%`)** | `-0.037 b/s` |
| **Salinas** `(204, 512, 216)` | `5.094 b/s` (`P=15`) | `4.794 b/s` (`P=15`) | **`4.715 b/s`** (`LDU16`/`Full`, `3.39×`) | **`-0.379 b/s` (`+7.4%`)** | **`-0.079 b/s` (`+1.65%`)** |
| **PaviaU** `(103, 608, 336)` | `6.839 b/s` (`P=3`) | `6.516 b/s` (`P=3`) | **`6.432 b/s`** (`Full LDU`, `2.49×`) | **`-0.407 b/s` (`+6.0%`)** | **`-0.084 b/s` (`+1.29%`)** |

> Even when CCSDS-123 Blue Book is upgraded with our own 3D-context rANS coder, **Phase 1 `LDU-RKLT + 2D 5/3 + 3D-rANS` beats it on both large real cubes (`Salinas` and `PaviaU`)**, beats standard Blue Book CCSDS-123 (Rice) by **`+4.6%` to `+7.4%`** on all three cubes, and even beats Phase 0's non-reversible `float KLT` Laplace estimate (`6.467 b/s`) on `PaviaU` with real verified binary bytes (`6.432 b/s`)!

### 2.2 Full Native Uncropped `.hnd1` Container Performance & Throughput

| Dataset (Native Shape, Dtype) | Raw Size | `.hnd1` `auto` Size | Rate (`Ratio`) | Encode / Decode (Python+Numba) | Integrity |
|---|---:|---:|---:|---:|---|
| **Indian Pines** `(200, 145, 145)`, `uint16` | `8,410,000 B` | **`3,378,095 B`** | **`6.427 b/s` (`2.49×`)** | `24.0–36.4` / **`45.0 MB/s`** | **SHA-256 PASS** |
| **Salinas** `(204, 512, 217)`, `int16` | `45,330,432 B` | **`13,354,729 B`** | **`4.714 b/s` (`3.39×`)** | `21.8–36.9` / **`28.5–45.8 MB/s`** | **SHA-256 PASS** |
| **PaviaU** `(103, 610, 340)`, `uint16` | `42,724,400 B` | **`17,199,328 B`** | **`6.441 b/s` (`2.48×`)** | `29.0–37.3` / **`36.4–44.8 MB/s`** | **SHA-256 PASS** |

---

## 3. Progressive Prefix Decode, $L_2$-Weighted Lossy `.hnd1`, & Blind Track A

### 3.1 Single-Stream Prefix Progressive Decode (Coarse-to-Fine Framing)
Stopping decode early on the **single lossless `.hnd1` bitstream** (`max_level = 0, 1, 2, 3`) reconstructs progressively sharper full-cube previews:

| Prefix Read | Indian Pines (`b/s`, `%`, `PSNR`) | Salinas (`b/s`, `%`, `PSNR`) | PaviaU (`b/s`, `%`, `PSNR`) |
|---|---:|---:|---:|
| `level <= 0` (`LL` only) | `0.124` (`1.9%`), `31.14 dB` | `0.093` (`2.0%`), `35.23 dB` | `0.093` (`1.4%`), `25.30 dB` |
| `level <= 1` (`LL+L3`) | `0.425` (`6.6%`), `33.62 dB` | `0.291` (`6.2%`), `38.37 dB` | `0.360` (`5.6%`), `28.29 dB` |
| `level <= 2` (`LL+L3+L2`) | `1.653` (`25.7%`), `37.31 dB` | `1.163` (`24.7%`), `43.10 dB` | `1.537` (`23.9%`), `32.97 dB` |
| `level <= 3` (**Full Lossless**) | **`6.427` (`100%`), ∞ (SHA-256)** | **`4.714` (`100%`), ∞ (SHA-256)** | **`6.441` (`100%`), ∞ (SHA-256)** |

### 3.2 Real-Binary $L_2$-Weighted Lossy `.hnd1` Sweep (`q_base = 2..64`)

| `q_base` | Indian Pines (`b/s`, `Ratio`, `PSNR`) | Salinas (`b/s`, `Ratio`, `PSNR`) | PaviaU (`b/s`, `Ratio`, `PSNR`) |
|---:|---:|---:|---:|
| `2` | `5.410` (`3.0×`), `75.99 dB` | `3.623` (`4.4×`), `75.37 dB` | `5.425` (`2.9×`), `72.63 dB` |
| `4` | `4.251` (`3.8×`), `72.53 dB` | `2.377` (`6.7×`), `72.32 dB` | `4.286` (`3.7×`), `70.45 dB` |
| `8` | `3.193` (`5.0×`), `68.06 dB` | `1.372` (`11.7×`), `68.73 dB` | `3.267` (`4.9×`), `67.02 dB` |
| `16` | `2.300` (`7.0×`), `63.52 dB` | `0.670` (`23.9×`), `65.30 dB` | `2.332` (`6.9×`), `62.34 dB` |
| `32` | `1.572` (`10.2×`), `58.48 dB` | `0.293` (`54.6×`), `62.32 dB` | `1.432` (`11.2×`), `56.89 dB` |
| `64` | **`1.011` (`15.8×`), `53.52 dB`** | **`0.161` (`99.4×`), `59.84 dB`** | **`0.784` (`20.4×`), `52.35 dB`** |

### 3.3 Blind Track A 1D-Stream Round-Trip (`encode_blind_1d` → `decode_blind_1d`)
With **zero shape or axis-order hints** (`BSQ`, `BIL`, `BIP` flat 1D arrays):
- **Indian Pines native odd `(200, 145, 145)`**: `3/3` exact strides (`[1, 145, 21025]`, `[1, 145, 29000]`, `[1, 200, 29000]`), `3/3` true spectral axis `B`, **`6.427 b/s` (`+0.00%` vs. schema-aware)**, **SHA-256 PASS**.
- **Salinas `(64, 256, 216)`**: `3/3` exact strides, `3/3` true `B`, **`5.943 b/s`**, **SHA-256 PASS**.
- **PaviaU `(64, 256, 216)`**: `3/3` exact primary 3D strides + `3/3` true `B`, **`6.567 b/s`**, **SHA-256 PASS** (incompressible noise falls back cleanly to raw framing with `< 0.2%` overhead).

---

## 4. Verification & Phase 2 Roadmap

- **Unit & Integration Tests**: `python3 -m unittest -v test_phase1.py` — **6/6 PASS in 3.4s** (complete-pivoted `LDU` round-trip & binary serialization for $M \in \{2..64\}$, sparse/escape 3D-context rANS, odd shapes + `int16`/`uint16`/`int32`, prefix progressive & monotonic $L_2$-lossy curves, blind 1D permutations + noise control, native odd `(200, 145, 145)` Indian Pines).
- **Phase 2 Candidates**:
  1. **Bitplane-embedded 3D-rANS** inside each subband so a *single* lossless `.hnd1` prefix delivers `52–76 dB` quality-embedded lossy previews (matching the `q=2..64` curve without re-encoding).
  2. **C++20 NEON / AVX2 port** of `LDU`-RKLT + 2D 5/3 + 4-way interleaved 32-bit rANS (targeting `300–600+ MB/s` decode, matching ASV4's production systems bar).
  3. **ASV4 integration**: Wire Track A (`recover_strides`) + `.hnd1` as a dedicated N-D tensor transform (`T5_HND1`) alongside ASV4's existing `T1..T4` 1D/float transforms.
