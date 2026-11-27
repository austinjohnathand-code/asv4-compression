# ASV4 Architecture (`ARCHITECTURE.md`)

**ASV4 (v4.1)** is a single-pass, pipe-safe, schema-free streaming compressor written in C++20 with ARM64 NEON SIMD intrinsics and persistent `libzstd 1.5.5` streaming contexts ([`v3/cpp/asv3.cpp`](v3/cpp/asv3.cpp)). It discovers hidden binary structure online—without upfront schemas, file extensions, or backward seeks—rearranges structured spans using fast bijective transforms, coalesces transformed and raw spans into separate persistent `zstd` streams, and falls back to raw `zstd --long=27` (`T0`) with `≤ +0.0028%` overhead whenever transforming does not win.

---

## 1. End-to-End Single-Pass Streaming Pipeline

Input is processed in `16 MB` flush groups (`flush_group_bytes`) with zero backward seeks. Any partial record trailing a transformed region at a flush-group boundary is carried forward in a tiny `carry` buffer (at most `unit_stride - 1` bytes) so adjacent flush groups remain anchored to the record grid without phase drift.

```mermaid
flowchart TD
    IN["Input Byte Stream (stdin / file)"] --> FG["Read 16 MiB Flush Group + Update Streaming SHA-256"]
    FG --> S12["Coupled Stage 1 & 2: Online Probing, Viterbi DP & Verification"]

    S12 --> REG{"Verified Region Transform"}
    REG -->|"T0 (Identity / Fallback)"| RAW_EMIT["Emit 4 MB Raw Blocks → raw_grp + Index Gear Anchors"]
    REG -->|"T1 / T2_XOR / T2_ELEM / T2_BIT / T2_NIBROT"| XF_EMIT["Region-Anchored 4 MB Blocks → Forward NEON Kernel → xform_grp"]

    RAW_EMIT --> META_Q["Append 24 B MetaRecord(s) → meta_raw"]
    XF_EMIT --> CARRY["Tail (lt unit_stride): Carry Across Group or Emit to raw_grp at Region End"]
    XF_EMIT --> META_Q
    CARRY --> META_Q

    META_Q --> Z_META["LazyZstdEncoder c_meta (L3, wlog=20)"]
    RAW_EMIT --> Z_RAW["LazyZstdEncoder c_raw (wlog=27 single / 26 dual)"]
    XF_EMIT --> Z_XF["LazyZstdEncoder c_xform (wlog=26 dual; cap L3 on high-H0 NIBROT)"]

    Z_META --> WRITE["Write TAG_DATA GroupHeader + Interleaved Payloads"]
    Z_RAW --> WRITE
    Z_XF --> WRITE

    WRITE --> LOOP{"More Input?"}
    LOOP -->|"Yes"| FG
    LOOP -->|"No (EOF)"| FIN["Flush carry + ZSTD_e_end Streams + Write TAG_EOF + 32 B SHA-256"]
```

---

## 2. Coupled Stage 1 & 2: Selector, Fast-Confirm & Safety Gates

Segmentation and transform selection are tightly coupled inside `segment_buffer_dp` to eliminate winner's-curse flapping, phase drift, and adversarial binary regressions (`adv_binaries.bin` stays locked at `Regs = 1, +0.002%`).

```mermaid
flowchart LR
    BUF["16 MiB Flush Group (256 x 64 KiB Slots)"] --> FC{"Stationary Fast-Confirm?"}

    FC -->|"Non-T0: consec ≥ 2, D_JS lt 0.12, i mod 8 ≠ 0"| FAST2["Fast-Confirm: Score {T0, active} (0.50x work)"]
    FC -->|"T0: consec ≥ 3, D_JS lt 0.04, i mod 4 ≠ 0"| FAST1["Fast-Confirm: Score {T0} only (0.25x work)"]
    FC -->|"Full Sweep / D_JS Spike / Mismatch"| FULL["Full Two-Tier Probe (16 KB Base + 64 KB Wide)"]

    FAST2 -->|"Active Still Wins"| DP["1D Viterbi DP (p_switch=48 KB, JS-Modulated)"]
    FAST2 -->|"Regime Changed"| FULL
    FAST1 --> DP

    subgraph Probe["Two-Tier Scorer (Per-Plane zstd -1 for w in 2..8)"]
        FULL --> H0_OUT{"Incompressible? (H0 gt 7.95 & z1 ≥ 99.5%)"}
        H0_OUT -->|"Yes"| T0_FAST["Return T0 Immediately"]
        H0_OUT -->|"No"| LAG["NEON Exact-Match Autocorrelation M(L) → Peaks (L1, L2)"]
        LAG --> MM["Miller-Madow H0 Pre-Filter + Pin {2, 4, 8} + L1"]
        MM --> TIER1["Tier 1: T1(w) Per-Plane zstd -1 + 2% Harmonic Tie-Break"]
        TIER1 --> T2_GATE{"Best T1 ≤ 0.98 * T0?"}
        T2_GATE -->|"No"| PICK["Select Best (Gain Gate ≤ 0.96 * T0)"]
        T2_GATE -->|"Yes"| TIER2["Tier 2 on Top-2 Widths: T2_XOR, T2_ELEM, T2_NIBROT (w=2,4,8), T2_BIT (w=4,8)"]
        TIER2 --> PICK
    end

    T0_FAST --> DP
    PICK --> DP

    DP --> GATES["Region Verification Gates"]

    subgraph Safety["Three-Layer Anti-Regression Guard"]
        GATES --> LDM["Gear Rolling-Hash LDM Guard vs S_RAW History (reject if overlap ≥ 15%)"]
        LDM --> Z3["2 x 256 KB Per-Plane zstd -3 Check (combined ≤ 0.94 & each ≤ z0 + 2048 B)"]
        Z3 --> FLOOR["Net Savings Floor: ≥ 256 KB (or ≥ 128 KB warm-numeric if S_XFORM active)"]
    end

    FLOOR -->|"Pass"| ACC["Accept Transformed Region"]
    FLOOR -->|"Reject"| REJ["Demote Region to T0 (S_RAW)"]
```

### Key Guardrails
- **Miller-Madow $H_0$ Pre-Filter + Pinned Word Widths `{2, 4, 8}`**: Prevents finite-sample entropy bias where wider candidate multiples (`16, 32, 64`) artificially appear lower-entropy than true fundamental widths (`BF16/FP16`, `FP32/INT32`, `FP64/INT64`).
- **Per-Plane `zstd -1` / `zstd -3` Scoring (`w ∈ [2, 8]`)**: Scores each byte plane independently (subtracting 12 B frame header per extra plane) so a 16 KB probe models the independent 128 KB zstd Huffman blocks of a 4 MB streaming block instead of averaging skewed exponent lanes with uniform mantissa lanes.
- **Gear Rolling-Hash LDM Duplication Guard (`TrailingRawIndex`)**: Maintains a 32,768-anchor shift-invariant ring buffer (~256 KB RAM) indexed **only over bytes actually routed to `S_RAW`**. Prevents splitting duplicated Mach-O/ELF tables or VM snapshot pages out of `S_RAW` where `zstd --long=27` already deduplicates them.
- **Stationary Fast-Confirm Path**: Holds average Scoring Work Multiplier to **`0.89×`** (max **`2.06×`**, well inside the `≤ 3.0×` budget) across all 16 datasets.

---

## 3. `T2_NIBROT`: Why a 4-Bit Nibble Rotation Beats OpenZL & ZipNN on Neural Weights

`T2_NIBROT` (`tid = 6`, `h = 0` LE `+1`, `h = 1` BE `−1`) applies a bijective cyclic 4-bit nibble shift across each `w`-byte word fused directly into ARM64 NEON `vld2q_u8`/`vld4q_u8` + `vsliq_n_u8` + `vst1q_u8` (zero scratch buffer, >5 GB/s).

```mermaid
flowchart LR
    subgraph BF16["BF16 Word (w=2, 16 bits)"]
        B1["Byte 1: [S, E7, E6, E5 | E4, E3, E2, E1]"]
        B0["Byte 0: [E0, M6, M5, M4 | M3, M2, M1, M0]"]
    end

    subgraph NIB["Fused NEON T2_NIBROT(2, h=0)"]
        R1["Plane 1 = (B1 shl 4) | (B0 shr 4): [E4..E0 | M6..M4] (H0 = 5.465 b)"]
        R0["Plane 0 = (B0 shl 4) | (B1 shr 4): [M3..M0 | S, E7..E5] (H0 = 5.000 b)"]
    end

    B1 --> R1
    B0 --> R1
    B0 --> R0
    B1 --> R0

    R1 --> Z["zstd -3 Pure Huffman (10.465 b/bf16 vs ZipNN 10.517 b/bf16)"]
    R0 --> Z
```

### Information-Theoretic & Systems Mechanism
1. **Captures Exponent-Mantissa Mutual Information ($I(E_{4..0}; M_{\text{top}})$)**: In Gaussian neural weights ($\mathcal{N}(0, 0.02)$), the steep density decay across each exponent octave creates real coupling ($I(E_{4..0}; M_{6..4}) \approx +0.0523\text{ bits/bf16}$), while top exponent bits are virtually constant ($H(E_{7..5}) \approx 0$). Isolating `[E7..E0]` and `[M6..M0, S]` into separate lanes (as ZipNN and OpenZL `BFloat16Deconstruct` do) discards this coupling (`10.517 bits/bf16` theoretical byte-plane limit). `T2_NIBROT(2, h=0)` packs `[E4..E0 | M6..M4]` into a single byte's Huffman table (`5.465 b`) and `[M3..M0 | S, E7..E5]` into the other (`5.000 b`), reaching **`10.465 bits/bf16`**—beating ZipNN's theoretical byte-plane limit. On FP32 (`w = 4`), `T2_NIBROT(4, h=0)` similarly drops total byte-plane $H_0$ from `26.646` (`T1(4)`) to **`26.464 b/f32`** (`−0.182 b/f32`).
2. **Eliminates Greedy `zstd -3` LZ77 False Matches**: Plain `T1` leaves a skewed ~10-symbol exponent lane (`H0 ≈ 2.68`) where `zstd -3` wastes bits emitting accidental 4-byte matches. Raising minimum plane entropy above `5.0 bits/byte` (~160 symbols) eliminates false matches so `zstd -3` stays in pure Huffman mode at **99.95% of Shannon entropy** and **`927–1,252 MB/s`** single-core decode.
3. **Monotonic L19 Level Cap**: When `c_xform` initializes on a high-entropy `T2_NIBROT` group (`zstd -3` ratio `> 60%` of raw on the first 256 KB), `c_xform.cap_level_before_init(3)` prevents `ZSTD_btultra2` from chasing noisy long-range matches (`+6.38 MB` regression in `v4.0`), guaranteeing `L19 ≤ L3` across all 16 datasets.

---

## 4. Interleaved Flush-Group Wire Format (`ASV3` Container, `VERSION = 4`)

Zero backward seeks during encode and a single forward cursor during decode enable arbitrary Unix piping (`cat in | asv3 c | asv3 d`). `run_decode` transparently accepts both `VERSION = 3` and `VERSION = 4`.

```mermaid
flowchart LR
    FH["FileHeader (8 B): magic='ASV3', ver=4, wlog=27, flags=0"] --> G1

    subgraph G1["Flush Group k (Repeats for every 16 MB)"]
        GH["GroupHeader (21 B): tag=0x01, grp_uncomp, lm, lr, lx, ld=0"]
        PM["S_META (lm B): zstd flush block of 24 B MetaRecords"]
        PR["S_RAW (lr B): zstd flush block of T0 + tail bytes"]
        PX["S_XFORM (lx B): zstd flush block of transformed blocks"]
        GH --> PM --> PR --> PX
    end

    G1 --> EOF

    subgraph EOF["FileTrailer"]
        TH["TrailerHeader (17 B): tag=0xFF, final_lens[4] (ld=0)"]
        FD["Final ZSTD_e_end Frame Epilogues (meta, raw, xform)"]
        TT["TrailerTail (40 B): uint64 total_orig_size + uint8 sha256[32]"]
        TH --> FD --> TT
    end
```

### 24-Byte `MetaRecord` Layout (Packed Little-Endian)
| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 4 B | `orig_len` | Uncompressed bytes represented by this block (`xlen + tail_len`) |
| 4 | 1 B | `transform_id` | `0=T0`, `1=T1`, `2=T2_ELEM`, `3=T2_XOR`, `4=T2_BIT`, `5=T3_2D` (v3 decode compat), `6=T2_NIBROT` |
| 5 | 1 B | `delta_op` | Must be `0` (`DELTA_NONE`; Stage 3 permanently cut per Kill Gate 1B) |
| 6 | 2 B | `w` | Element / record stride width (`≥ 1`) |
| 8 | 2 B | `h` | Secondary parameter (`T2_NIBROT`: `0=LE +1`, `1=BE −1`; `T3_2D`: row width) |
| 10 | 2 B | `phase` | Grid phase offset (`0` under Region-Anchored emission) |
| 12 | 4 B | `tail_len` | Trailing raw bytes (`< unit_stride`) stored in `S_RAW` at region end |
| 16 | 8 B | `parent_rel_offset` | Reserved (`0`) |

---

## 5. Pipe-Safe Streaming Handshake & Hardened Decode (`run_decode`)

Every `MetaRecord` and stream length is bounds-checked before dereferencing (verified 10/10 clean rejections on crafted OOB/corrupt streams under `-fsanitize=undefined` on ARM64), and dual-stream zstd window memory is strictly $O(1)$ bounded (`128 MiB` single-stream `S_RAW`; `128 MiB` simultaneous dual-stream `64+64 MiB`; `192 MiB` ceiling for sequential dual-stream `128+64 MiB`). Total OS process RSS—which additionally includes 16 MiB I/O flush-group staging, decompression frame buffers, and allocator overhead—peaks at `286.8 MiB` decode / `312.9 MiB` encode on `target_mixed_structured.bin`.

```mermaid
sequenceDiagram
    participant Enc as Encoder (run_encode)
    participant Pipe as OS Pipe / File Stream
    participant Dec as Hardened Decoder (run_decode)

    Enc->>Pipe: Write FileHeader (8 B: ASV3, ver=4, wlog=27)
    Pipe->>Dec: Read & validate magic, version in {3,4}, wlog in [10,27]

    loop Every 16 MiB Flush Group
        Enc->>Enc: Segment (DP + Gates) & NEON Transform
        Note over Enc: Cap uninitialized streams to wlog=26 if dual-stream active
        Enc->>Pipe: Write GroupHeader (ld=0) + S_META + S_RAW + S_XFORM
        Pipe->>Dec: Read GroupHeader (reject ld ≠ 0 or payload gt cap)
        Dec->>Dec: LazyZstdDecoder decompress_append (linear 4 MB growth)
        loop For each 24 B MetaRecord
            Dec->>Dec: Bounds check delta_op == 0, w ≥ 1, tail ≤ orig, r_pos & x_pos in range
            Dec->>Dec: Inverse NEON Transform (if non-T0) + Update SHA-256
            Dec->>Dec: Write reconstructed bytes to stdout/file
        end
        Dec->>Dec: Verify grp_dec == grp_uncomp & reclaim excess queue capacity
    end

    Enc->>Pipe: Write TrailerHeader (0xFF) + ZSTD_e_end epilogues + TrailerTail (size + SHA-256)
    Pipe->>Dec: Drain epilogues (reject trailing bytes) & compare size + SHA-256
    Note over Dec: Exit 0 iff byte count and 256-bit SHA-256 match bit-for-bit
```

---

## 6. HND3: Reversible 3D Hyperspectral Tensor Architecture (Phases 0–3)

Alongside ASV4's 1D/record streaming engine, **HND3 (`phase3-v3.2`, [`cpp/hnd3.cpp`](cpp/hnd3.cpp))** is a zero-dependency C++20 reversible 3D tensor codec (wire-compatible with [`phase3_codec.py`](phase3_codec.py), with bit-exact ARM64 NEON ↔ `-DHND3_FORCE_SCALAR` parity and variable-width escape coding supporting full-range `int32` inputs up to `|v| < 2^62`). It was developed across `Indian Pines`, `Salinas`, and `Pavia University`, and validated out-of-sample on a `132.84 MB` 224-band NASA AVIRIS scene plus non-hyperspectral 3D microscopy and MRI controls.

### 6.1 End-to-End HND3 Pipeline (Explicit & Native C++20 Blind Track A)

```mermaid
flowchart TD
    IN["Input Raw Tensor or Headerless 1D Stream (i16 / u16 / i32)"] --> MODE{"Mode?"}

    MODE -->|"--blind (Zero Shape Hints)"| FFT["Iterative Radix-2 FFT Centered Pearson ACF + Axis-Peeling"]
    FFT --> STR{"Valid 3D Lattice Strides [1, s1, s2]?"}
    STR -->|"No (e.g. Noise)"| RAWF["FLAG_RAW Fallback (62 B Header + Raw Bytes + SHA-256)"]
    STR -->|"Yes"| ROLE["Fast Center-Crop Multi-Level LDU + 2D 5/3 Role Picker → spec_axis"]
    ROLE --> PERM["Move spec_axis to Axis 0: (B, H, W)"]
    MODE -->|"Explicit --shape D0,D1,D2"| PERM

    PERM --> LDU["Complete-Pivoted LDU-RKLT (Q15 int16 L/U + NEON vmlal_s32 FMA)"]
    LDU --> DWT["Multi-Threaded Arbitrary-Shape 2D 5/3 Integer Lifting (L=3)"]
    DWT --> SUBS["Coarse-to-Fine Subband Schedule: LL → L3 → L2 → L1"]

    SUBS --> BASE["Base Layer (k = base_k): 16-Stripe Parallel 3D-Context 32-bit rANS + 64-bit Extra-Bits"]
    BASE --> REF{"base_k gt 0?"}
    REF -->|"No (k=0 One-Shot)"| OUT["62 B HND3 Header + SHA-256 + LDU Stages + Framed Stripes"]
    REF -->|"Yes (k=6 Embedded)"| LAY["Refinement Layers k = 5..0: 16-Stripe 360-deg Context Significance + Magnitude rANS"]
    LAY --> OUT
```

### 6.2 Complete-Pivoted `LDU`-RKLT, Rounding-Noise Audit, & 16-Stripe 7-Layer Embedded Layout

Complete-pivoted LU ($P\,\mathrm{diag}(s_0)\,U_{\text{KLT}}\,Q = L\,\mathrm{diag}(s_d)\,D_{\text{ladder}}\,U$) bounds all off-diagonal triangular entries to $|L_{ij}|, |U_{ij}| \le 1$ (stored losslessly in Q15 `int16`) and factors $D$ via a greedy log-balanced Daubechies-Sweldens 3-rounding diagonal ladder (no Givens fallback branch is required up to `224×224`). Unlike Phase 0's row-pivoted SERM (whose Schur-complement multipliers exploded $> 10^4$ beyond $M = 8$), complete pivoting keeps cumulative diagonal multipliers bounded to $\max|c_k|, \max|1/c_k| \in [1.42, 2.32]$ across all four real cubes—holding empirical integer-lifting rounding noise against the exact same `float64` KLT matrix to `RMS = 0.97–1.35` (`Var = 0.94–1.81`) and exact discrete zero-order entropy delta $\Delta H_{\text{disc}}$ to `+0.013`–`+0.055 b/s` on Indian Pines/Salinas/PaviaU (`+0.341 b/s` on AVIRIS and `+0.899 b/s` on a `60 dB` synthetic stress test where sub-integer/unit tail-PC noise exposes the irreducible $O(1)$-LSB lifting floor; full table in [`PHASE3_REPORT.md`](PHASE3_REPORT.md) §3). Partitioning each subband into `n_stripes = min(B, 16)` independent band stripes unlocks multi-core parallelism while letting each energy tier (`PC1..PC12` vs. noise floor `PC190..PC204`) train its own compact 12-bit rANS frequency table.

```mermaid
flowchart LR
    subgraph Header["HND3 Header & Spectral Model"]
        H["FileHeader (62 B): magic='HND3', ver=3, base_k, n_stripes, shape, SHA-256"]
        SP["LDU Stages: uint16 perms + bit-packed signs + Q15 int16 strict L/U + int32 ladder"]
        H --> SP
    end

    subgraph Base["Base Layer (k = base_k, e.g. 0.6%-16.4% of stream, 15x-888x Preview)"]
        B0["Subband LL: Stripes 0..S-1 (16-ctx x 32-sym 12-bit rANS + 64-bit EB)"]
        B1["Subbands L3, L2, L1: Stripes 0..S-1 (Spatial + Parent + Cross-Band EMA Context)"]
        B0 --> B1
    end

    subgraph Refine["Refinement Layers (k = 5 down to 0, L2-Aligned Offsets [2,2,2,1,1,1,0,0,0,0])"]
        R5["Layer k=5 (q ~ 32) → k=4 (16) → k=3 (8) → k=2 (4) → k=1 (2)"]
        R0["Layer k=0 (p = 0): Bit-Exact Lossless Reconstruction + Built-In SHA-256 Verify"]
        R5 --> R0
    end

    Header --> Base --> Refine
```

### 6.3 Native C++20 Blind Track A Stride-Lattice Discovery (`recover_strides_cpp`)

```mermaid
flowchart TD
    X["Flat 1D Signal x (Zero Shape / Divisor Hints)"] --> CEN["Global Mean-Centering + Variance Floor Guard (const → [1])"]
    CEN --> ACF0["Radix-2 FFT Exact Pearson ACF rho_raw[0..N/4]"]
    ACF0 --> LOOP["Current Stride s_prev (init = 1)"]

    LOOP --> P1["Cumulative Peel cur_peeled = cur[s:] - cur[:-s] → rho_peel"]
    LOOP --> P2["Direct xpeel = x[s:] - x[:-s] (if var ratio gt 0.002) → rho_xpeel"]
    P1 --> PK["Rectified Peak Finding (min_prom=0.10, min_val=0.20)"]
    P2 --> PK

    PK --> GRD["Guards: k=2 MA(1) Rebound + Non-Periodic 2*k Harmonic Check"]
    GRD --> YIN["Rectified YIN Fundamental Selection"]
    YIN --> DIV["Harmonic-Verified Divisor Refinement (2*d or 3*d for half-period)"]
    DIV --> NEXT["Append s_next = best_k * s_prev; Update cur if var drops"]
    NEXT -->|"len lt max_axes and peaks remain"| LOOP
    NEXT -->|"Done"| OUT["Recovered Stride Chain [1, s1, s2, ...]"]
```

---

## 7. Evolutionary History (Falsification Record)

| Version / Tag | Milestone | Outcome |
|---|---|---|
| **v1 (`ftcc/`)** | Energy clumping + forest of trees Python/Numba prototype | Superseded by v3 strict streaming/worst-case spec |
| **v3 Phase 0 & 1A** | Coupled Stage 1 & 2 (`codec.py`) + `ASV3` container | **PASS** — 100% SHA-256, pipe-safe, `≤ +0.002%` worst-case tax |
| **v3 Phase 1B** | Stage 3 bounded delta forest (`stage3_forest.py`) | **CUT PERMANENTLY** — regressed `+11.80%` on snapshots, `+3.03%` on binaries |
| **v3.0 (`asv3_v3.0.cpp`)** | C++20 NEON port, UBSan bounds checks, fast-confirm, tiled `T2_BIT`, portable SHA-256 | **RELEASED** — `−7.05%` GPT-2, `−15.45%` Hubble FITS, `≤ +0.0028%` tax, `0.83×` Work Mult |
| **v4.0** | `T2_NIBROT` (`w=4,8`) NEON, per-plane `zstd -1/-3` scoring, combined-gain gate | Surpassed OpenZL `le32` on GPT-2 (`394.91 MB`, `1,285 MB/s` dec), `−16.86%` Hubble FITS |
| **`v4.1` (`asv3.cpp`)** | `T2_NIBROT(2)` NEON for BF16, monotonic L19 cap, `T3_2D` encoder retirement | **RELEASED** — Beats OpenZL, Blosc2 & ZipNN on GPT-2 FP32 (`394.91 MB`) & SmolLM2 BF16 (`170.44 MB`), `16/16` monotonic `L19 ≤ L3`, `0.89×` Work Mult |
| **`phase0-v2.2`** | Hyperspectral N-D falsification harness, exact Blue Book CCSDS-123 (`P=3/15`), real Rice coder, 14-test suite | **PASS** — Killed folded spatial KLT (`0A`), passed `R6` (`0B`), passed 48/48 blind stride recovery (`0D`) |
| **`phase1-v1.0`** | Gate 1A (`R6` = 91–97% variance heteroscedasticity), complete-pivoted `LDU`-RKLT, 3D-context 32-bit rANS, $L_2$-weighted Track B (`.hnd1`) | **PASS** — Beats Blue Book CCSDS-123 on real cubes (`6.427 / 4.714 / 6.441 b/s`), `99.4×` @ `59.8 dB` on Salinas |
| **`phase2-v2.0`** | Unified 7-layer quality-embedded `.hnd2` + zero-dependency C++20 NEON engine (`./hnd2`) | **PASS** — `53–60 dB` base preview from single lossless file, `92–135 MB/s` dec, UBSan + Py↔C++ verified |
| **`phase3-v3.0`** | 16-stripe parallel rANS + true NEON `vmlal_s32` FMA + 64-bit EB + native C++20 `--blind` FFT (`./hnd3`) | **CURRENT** — 7-layer embedded beats one-shot & float KLT (`4.628 b/s` Salinas `+9.1%` vs CCSDS-123, `6.354 b/s` PaviaU `+7.1%`), up to `220 MB/s` lossless / `565 MB/s` preview dec, UBSan + Py↔C++ verified |
