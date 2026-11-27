# ASV4 Architecture (`ARCHITECTURE.md`)

**ASV4 (v4.1)** is a single-pass, pipe-safe, schema-free streaming compressor written in C++20 with ARM64 NEON SIMD intrinsics and persistent `libzstd 1.5.5` streaming contexts ([`v3/cpp/asv3.cpp`](v3/cpp/asv3.cpp)). It discovers hidden binary structure online—without upfront schemas, file extensions, or backward seeks—rearranges structured spans using fast bijective transforms, coalesces transformed and raw spans into separate persistent `zstd` streams, and falls back to raw `zstd --long=27` (`T0`) with `≤ +0.0028%` overhead whenever transforming does not win.

---

## 1. End-to-End Single-Pass Streaming Pipeline

Input is processed in `16 MB` flush groups (`flush_group_bytes`) with zero backward seeks. Any partial record trailing a transformed region at a flush-group boundary is carried forward in a tiny `carry` buffer (at most `unit_stride - 1` bytes) so adjacent flush groups remain anchored to the record grid without phase drift.

```mermaid
flowchart TD
    IN["Input Byte Stream (stdin / file)"] --> FG["Read 16 MB Flush Group + Update Streaming SHA-256"]
    FG --> S12["Coupled Stage 1 & 2: Online Probing, Viterbi DP & Verification"]

    S12 --> REG{"Verified Region Transform"}
    REG -->|"T0 (Identity / Fallback)"| RAW_EMIT["Emit 4 MB Raw Blocks -> raw_grp + Index Gear Anchors"]
    REG -->|"T1 / T2_XOR / T2_ELEM / T2_BIT / T2_NIBROT"| XF_EMIT["Region-Anchored 4 MB Blocks -> Forward NEON Kernel -> xform_grp"]

    RAW_EMIT --> META_Q["Append 24 B MetaRecord(s) -> meta_raw"]
    XF_EMIT --> CARRY["Tail (< unit_stride): Carry Across Group or Emit to raw_grp at Region End"]
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
    BUF["16 MB Flush Group (256 x 64 KB Slots)"] --> FC{"Stationary Fast-Confirm?"}

    FC -->|"Non-T0: consec>=2, D_JS<0.12, i%8!=0"| FAST2["Fast-Confirm: Score {T0, active} (0.50x work)"]
    FC -->|"T0: consec>=3, D_JS<0.04, i%4!=0"| FAST1["Fast-Confirm: Score {T0} only (0.25x work)"]
    FC -->|"Full Sweep / D_JS Spike / Mismatch"| FULL["Full Two-Tier Probe (16 KB Base + 64 KB Wide)"]

    FAST2 -->|"Active Still Wins"| DP["1D Viterbi DP (p_switch=48 KB, JS-Modulated)"]
    FAST2 -->|"Regime Changed"| FULL
    FAST1 --> DP

    subgraph Probe["Two-Tier Scorer (Per-Plane zstd -1 for w in [2..8])"]
        FULL --> H0_OUT{"Incompressible? (H0>7.95 & z1>=99.5%)"}
        H0_OUT -->|"Yes"| T0_FAST["Return T0 Immediately"]
        H0_OUT -->|"No"| LAG["NEON Exact-Match Autocorrelation M(L) -> Peaks (L1, L2)"]
        LAG --> MM["Miller-Madow H0 Pre-Filter + Pin {2, 4, 8} + L1"]
        MM --> TIER1["Tier 1: T1(w) Per-Plane zstd -1 + 2% Harmonic Tie-Break"]
        TIER1 --> T2_GATE{"Best T1 <= 0.98 * T0?"}
        T2_GATE -->|"No"| PICK["Select Best (Gain Gate <= 0.96 * T0)"]
        T2_GATE -->|"Yes"| TIER2["Tier 2 on Top-2 Widths: T2_XOR, T2_ELEM, T2_NIBROT (w=2,4,8), T2_BIT (w=4,8)"]
        TIER2 --> PICK
    end

    T0_FAST --> DP
    PICK --> DP

    DP --> GATES["Region Verification Gates"]

    subgraph Safety["Three-Layer Anti-Regression Guard"]
        GATES --> LDM["Gear Rolling-Hash LDM Guard vs S_RAW History (reject if overlap >= 15%)"]
        LDM --> Z3["2 x 256 KB Per-Plane zstd -3 Check (combined <= 0.94 & each <= z0 + 2048 B)"]
        Z3 --> FLOOR["Net Savings Floor: >= 256 KB (or >= 128 KB warm-numeric if S_XFORM active)"]
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
        R1["Plane 1 = (B1<<4)|(B0>>4): [E4..E0 | M6..M4] (H0 = 5.465 b)"]
        R0["Plane 0 = (B0<<4)|(B1>>4): [M3..M0 | S, E7..E5] (H0 = 5.000 b)"]
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

Every `MetaRecord` and stream length is bounds-checked before dereferencing (verified 10/10 clean rejections on crafted OOB/corrupt streams under `-fsanitize=undefined`), and dual-stream memory is strictly $O(1)$ bounded (`128 MB` single-stream `S_RAW`; `128 MB` simultaneous dual-stream `64+64 MB`; `192 MB` ceiling for sequential dual-stream `128+64 MB`).

```mermaid
sequenceDiagram
    participant Enc as Encoder (run_encode)
    participant Pipe as OS Pipe / File Stream
    participant Dec as Hardened Decoder (run_decode)

    Enc->>Pipe: Write FileHeader (8 B: ASV3, ver=4, wlog=27)
    Pipe->>Dec: Read & validate magic, version in {3,4}, wlog in [10,27]

    loop Every 16 MB Flush Group
        Enc->>Enc: Segment (DP + Gates) & NEON Transform
        Note over Enc: Cap uninitialized streams to wlog=26 if dual-stream active
        Enc->>Pipe: Write GroupHeader (ld=0) + S_META + S_RAW + S_XFORM
        Pipe->>Dec: Read GroupHeader (reject ld != 0 or payload > cap)
        Dec->>Dec: LazyZstdDecoder decompress_append (linear 4 MB growth)
        loop For each 24 B MetaRecord
            Dec->>Dec: Bounds check delta_op==0, w>=1, tail<=orig, r_pos/x_pos in range
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

## 6. Evolutionary History (Falsification Record)

| Version | Milestone | Outcome |
|---|---|---|
| **v1 (`ftcc/`)** | Energy clumping + forest of trees Python/Numba prototype | Superseded by v3 strict streaming/worst-case spec |
| **v3 Phase 0 & 1A** | Coupled Stage 1 & 2 (`codec.py`) + `ASV3` container | **PASS** — 100% SHA-256, pipe-safe, `≤ +0.002%` worst-case tax |
| **v3 Phase 1B** | Stage 3 bounded delta forest (`stage3_forest.py`) | **CUT PERMANENTLY** — regressed `+11.80%` on snapshots, `+3.03%` on binaries |
| **v3.0 (`asv3_v3.0.cpp`)** | C++20 NEON port, UBSan bounds checks, fast-confirm, tiled `T2_BIT`, portable SHA-256 | **RELEASED** — `−7.05%` GPT-2, `−15.45%` Hubble FITS, `≤ +0.0028%` tax, `0.83×` Work Mult |
| **v4.0** | `T2_NIBROT` (`w=4,8`) NEON, per-plane `zstd -1/-3` scoring, combined-gain gate | Surpassed OpenZL `le32` on GPT-2 (`394.91 MB`, `1,285 MB/s` dec), `−16.86%` Hubble FITS |
| **v4.1 (`asv3.cpp`)** | `T2_NIBROT(2)` NEON for BF16, monotonic L19 cap, `T3_2D` encoder retirement | **CURRENT** — Beats OpenZL, Blosc2 & ZipNN on both GPT-2 FP32 (`394.91 MB`) and SmolLM2 BF16 (`170.44 MB`), `16/16` monotonic `L19 ≤ L3`, `0.89×` Work Mult |
