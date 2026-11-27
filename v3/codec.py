"""ASV3 Single-Pass Streaming Codec (Phase 1A: Coupled Stage 1 & 2 + Stage 4 Container).

Architecture & Wire Format (v3 Spec):
  - True single-pass streaming (zero backward seeks on encode, single forward cursor
    on decode). Pipe-safe: `cat in | python3 codec.py c | python3 codec.py d`.
  - Coupled Stage 1 & 2:
      * 64 KB probe slots with 16 KB base probe + periodic 64 KB wide probe
      * Exact-match autocorrelation M(L) + Two-Tier zstd -1 scorer + H0 pre-filter
      * 1D Dynamic Programming (Viterbi) segmentation with regime-switch penalty P_switch
        and histogram Jensen-Shannon divergence check
      * Region-Anchored grid alignment (phase=0 at Region start; ragged tail emitted
        only at true Region end; decoupled processing block size 1/4/16 MB)
  - Stage 4 Interleaved Flush-Group Streaming Container ("ASV3"):
      * 4 persistent zstd --long=27 streaming contexts (S_META, S_RAW, S_XFORM, S_RESID)
      * Flushed via COMPRESSOBJ_FLUSH_BLOCK (ZSTD_e_flush) every flush_group_bytes (16 MB)
      * End-to-end streaming SHA-256 written in FileTrailer and verified on decode
  - Stage 3 (Bounded Delta Forest) is stubbed cleanly for Phase 1B ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import numpy as np
from numba import njit
import zstandard as zstd

from transforms import (
    T0_IDENTITY,
    T1_TRANSPOSE,
    T2_BITSHUFFLE,
    T2_ELEM_DELTA,
    T2_XOR_TRANSPOSE,
    T3_2D_PAETH,
    TRANSFORM_NAMES,
    ProbeResult,
    TransformSpec,
    TwoTierScorer,
    apply_transform_forward,
    apply_transform_inverse,
    js_divergence,
)
from stage3_forest import (
    DELTA_NONE,
    DELTA_SUB,
    DELTA_XOR,
    RingBuffer128M,
    Stage3DeltaForest,
    delta_add_u8,
    delta_xor_u8,
    fastcdc_boundaries,
)

MAGIC = b"ASV3"
VERSION = 3
WINDOW_LOG = 27  # 128 MB sliding window (matches W_win)

TAG_DATA = 0x01
TAG_EOF = 0xFF

# FileHeader: Magic(4s), Version(B), WindowLog(B), Flags(H) = 8 bytes
FILE_HDR_FMT = "<4sBBH"
FILE_HDR_SIZE = struct.calcsize(FILE_HDR_FMT)

# FlushGroup header: Tag(B), GroupUncompLen(I), LenMeta(I), LenRaw(I), LenXform(I), LenResid(I) = 21 bytes
GROUP_HDR_FMT = "<BIIIII"
GROUP_HDR_SIZE = struct.calcsize(GROUP_HDR_FMT)

# FileTrailer header: Tag(B), FinalLens(4I) = 17 bytes, followed by epilogue bytes, then TotalOrigSize(Q) + SHA256(32s)
TRAILER_HDR_FMT = "<BIIII"
TRAILER_HDR_SIZE = struct.calcsize(TRAILER_HDR_FMT)
TRAILER_TAIL_FMT = "<Q32s"
TRAILER_TAIL_SIZE = struct.calcsize(TRAILER_TAIL_FMT)

# Manifest record (24 bytes):
# orig_len(I), transform_id(B), delta_op(B), w(H), h(H), phase(H), tail_len(I), parent_rel_offset(Q)
META_REC_FMT = "<IBBHHHIQ"
META_REC_SIZE = struct.calcsize(META_REC_FMT)

MB = 1024 * 1024


@dataclass
class EncoderConfig:
    zstd_level: int = 19                 # Configurable -3 or -19
    window_log: int = WINDOW_LOG         # 27 (128 MB)
    flush_group_bytes: int = 16 * MB     # 8-16 MB Flush Groups
    block_size_bytes: int = 4 * MB       # Decoupled processing block size (1 / 4 / 16 MB)
    probe_step_bytes: int = 65536        # 64 KB probe step (matches min Region size)
    base_probe_bytes: int = 16384        # 16 KB base probe
    gain_gate: float = 0.96              # Require >= 4% zstd -1 gain over T0
    p_switch_bytes: int = 49152          # 48 KB DP regime-switch penalty (context-break tax)
    min_region_bytes: int = 65536        # Minimum Region size: 64 KB
    js_merge_threshold: float = 0.15     # Histogram JS divergence threshold
    ldm_dup_threshold: float = 0.15      # If >=15% sampled 16B hashes exist in trailing 128MB S_RAW, keep T0
    region_z3_gate: float = 0.94         # Require both start & mid 256KB zstd-3 <= 0.94 * T0 (>=6% real gain)
    min_region_saved_bytes: int = 262144 # Require >= 256 KB extrapolated savings to justify S_XFORM context split
    enable_stage3: bool = False          # Stubbed off in Phase 1A


@dataclass
class CodecStats:
    orig_bytes: int = 0
    comp_bytes: int = 0
    s_meta_bytes: int = 0
    s_raw_bytes: int = 0
    s_xform_bytes: int = 0
    s_resid_bytes: int = 0
    framing_bytes: int = 0
    regions_count: int = 0
    flush_groups: int = 0
    bytes_scored_zstd1: int = 0
    bytes_by_transform: Dict[str, int] = field(default_factory=dict)
    sha256: str = ""
    elapsed_sec: float = 0.0

    @property
    def scoring_work_multiplier(self) -> float:
        return (self.bytes_scored_zstd1 / self.orig_bytes) if self.orig_bytes > 0 else 0.0

    @property
    def meta_pct_of_out(self) -> float:
        return (100.0 * self.s_meta_bytes / self.comp_bytes) if self.comp_bytes > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "orig_bytes": self.orig_bytes,
            "comp_bytes": self.comp_bytes,
            "ratio": round(self.orig_bytes / max(1, self.comp_bytes), 4),
            "s_meta_bytes": self.s_meta_bytes,
            "s_meta_pct": round(self.meta_pct_of_out, 4),
            "s_raw_bytes": self.s_raw_bytes,
            "s_xform_bytes": self.s_xform_bytes,
            "s_resid_bytes": self.s_resid_bytes,
            "framing_bytes": self.framing_bytes,
            "regions_count": self.regions_count,
            "flush_groups": self.flush_groups,
            "bytes_scored_zstd1": self.bytes_scored_zstd1,
            "scoring_work_multiplier": round(self.scoring_work_multiplier, 3),
            "bytes_by_transform": self.bytes_by_transform,
            "sha256": self.sha256,
            "elapsed_sec": round(self.elapsed_sec, 3),
        }


def _make_zstd_stream_compressor(level: int, window_log: int) -> zstd.ZstdCompressionObj:
    params = zstd.ZstdCompressionParameters.from_level(
        level,
        enable_ldm=True,
        window_log=window_log,
    )
    return zstd.ZstdCompressor(compression_params=params).compressobj()


def _make_zstd_stream_decompressor(window_log: int) -> zstd.ZstdDecompressionObj:
    return zstd.ZstdDecompressor(max_window_size=1 << window_log).decompressobj()


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    """Read exactly n bytes from a pipe/stream, or fewer only on EOF."""
    if n == 0:
        return b""
    chunks = []
    rem = n
    while rem > 0:
        b = stream.read(rem)
        if not b:
            break
        chunks.append(b)
        rem -= len(b)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Coupled Stage 1 & 2: 1D DP Segmentation over Probe Slots
# ---------------------------------------------------------------------------

# Deterministic 256-entry 64-bit Gear table for O(1)/byte shift-invariant content-defined anchors
_GEAR_RNG = np.random.default_rng(0x41535633)
GEAR_TABLE = _GEAR_RNG.integers(0, 1 << 63, size=256, dtype=np.uint64)


@njit(cache=True)
def extract_cd_anchors(
    arr: np.ndarray,
    gear_table: np.ndarray,
    mask: np.uint64 = np.uint64(0x1FFF),
    max_anchors: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Shift-invariant Gear rolling-hash content-defined anchor extractor (avg 1 anchor / 8 KB).

    Returns (hashes[max_anchors], positions[max_anchors], count).
    Immune to arbitrary 1-byte alignment shifts across concatenated files!
    """
    n = arr.shape[0]
    hashes = np.zeros(max_anchors, dtype=np.uint64)
    positions = np.zeros(max_anchors, dtype=np.int64)
    if n < 64:
        return hashes, positions, 0

    h = np.uint64(0)
    cnt = 0
    # Warm up rolling hash for 32 bytes
    for i in range(32):
        h = (h << np.uint64(1)) + gear_table[arr[i]]

    i = 32
    lim = n - 32
    while i < lim:
        h = (h << np.uint64(1)) + gear_table[arr[i]]
        if (h & mask) == np.uint64(0):
            # Skip constant runs (e.g. zero padding)
            b0 = arr[i]
            if not (arr[i + 1] == b0 and arr[i + 4] == b0 and arr[i + 12] == b0 and arr[i + 24] == b0):
                # Compute 64-bit FNV-1a over 32 bytes at anchor i
                fh = np.uint64(14695981039346656037)
                for j in range(32):
                    fh = (fh ^ np.uint64(arr[i + j])) * np.uint64(1099511628211)
                hashes[cnt] = fh
                positions[cnt] = i
                cnt += 1
                if cnt >= max_anchors:
                    break
                # Skip ahead 1024 bytes after an anchor to prevent dense clusters
                i += 1024
                continue
        i += 1
    return hashes, positions, cnt


class TrailingRawIndex:
    """Bounded (~256 KB RAM) shift-invariant content-defined anchor index of trailing 128 MB history.

    Stores up to 32,768 content-defined 32-byte block hashes (~1 per 4-8 KB -> 128 MB+ coverage).
    Detects when a candidate non-T0 Region is a cross-file near-duplicate of earlier raw data
    (e.g. repeated Mach-O/ELF binaries) so ASV3 keeps it in S_RAW instead of breaking LDM matches.
    """

    def __init__(self, max_entries: int = 32768):
        self.max_entries = max_entries
        self._ring: List[int] = []
        self._pos = 0
        self._counts: Dict[int, int] = {}

    def overlap_fraction(self, arr: np.ndarray) -> float:
        if not self._counts or arr.shape[0] < 4096:
            return 0.0
        # Use finer mask (1/2048 B) when probing a single Region so even 256 KB regions yield ~100 anchors
        hs, _, cnt = extract_cd_anchors(arr, GEAR_TABLE, np.uint64(0x7FF), 512)
        if cnt < 4:
            return 0.0
        hits = 0
        for i in range(cnt):
            if int(hs[i]) in self._counts:
                hits += 1
        return hits / cnt

    def add_raw(self, arr: np.ndarray) -> None:
        if arr.shape[0] < 4096:
            return
        # Index anchors at 1/2048 B density? On 16 MB that's ~4000 anchors (capped at 4096 -> 8 groups = 32K = 128 MB!)
        hs, _, cnt = extract_cd_anchors(arr, GEAR_TABLE, np.uint64(0x7FF), 4096)
        for i in range(cnt):
            h = int(hs[i])
            if len(self._ring) < self.max_entries:
                self._ring.append(h)
                self._counts[h] = self._counts.get(h, 0) + 1
            else:
                old = self._ring[self._pos]
                cnt_old = self._counts.get(old, 1) - 1
                if cnt_old <= 0:
                    self._counts.pop(old, None)
                else:
                    self._counts[old] = cnt_old
                self._ring[self._pos] = h
                self._counts[h] = self._counts.get(h, 0) + 1
                self._pos = (self._pos + 1) % self.max_entries


def segment_buffer_dp(
    arr: np.ndarray,
    scorer: TwoTierScorer,
    cfg: EncoderConfig,
    prev_active_spec: TransformSpec,
    raw_index: Optional[TrailingRawIndex] = None,
) -> Tuple[List[Tuple[int, int, TransformSpec]], int]:
    """Run coupled Stage 1+2 probe scan + 1D DP (Viterbi) segmentation on a chunk.

    Returns (list of (start, end, TransformSpec) contiguous Regions, total bytes_scored_zstd1).
    """
    n = int(arr.shape[0])
    t0 = TransformSpec(T0_IDENTITY, 1, 0)
    if n < cfg.base_probe_bytes:
        return [(0, n, t0)], 0

    step = cfg.probe_step_bytes
    n_slots = max(1, (n + step - 1) // step)
    slot_bounds: List[Tuple[int, int]] = []
    for i in range(n_slots):
        s = i * step
        e = min(n, s + step)
        if e - s < cfg.base_probe_bytes and slot_bounds:
            # Absorb tiny trailing sliver into previous slot
            ps, _ = slot_bounds[-1]
            slot_bounds[-1] = (ps, e)
        else:
            slot_bounds.append((s, e))
    n_slots = len(slot_bounds)

    # 1. Score every 64 KB probe slot (16 KB base probe; 64 KB wide probe every 4th slot)
    probes: List[ProbeResult] = []
    bytes_scored = 0
    candidate_set = {t0}
    if prev_active_spec.tid != T0_IDENTITY:
        candidate_set.add(prev_active_spec)

    for i, (s, e) in enumerate(slot_bounds):
        slot_view = arr[s:e]
        p16 = slot_view[: min(cfg.base_probe_bytes, slot_view.shape[0])]
        # Wide 64 KB probe every 4th slot (Section 2.1)
        p64 = slot_view if (i % 4 == 0 and slot_view.shape[0] >= 32768) else None
        pr = scorer.score_probe(p16, p64)
        probes.append(pr)
        bytes_scored += pr.bytes_scored_zstd1
        if pr.best_spec.tid != T0_IDENTITY:
            candidate_set.add(pr.best_spec)

    # Fast path: if NO probe in the entire chunk beat T0 (common for enwik9, tar, media, random!)
    if len(candidate_set) == 1 and prev_active_spec.tid == T0_IDENTITY:
        return [(0, n, t0)], bytes_scored

    # Build ordered state list (keep T0 at index 0, plus top winning states)
    # Cap candidate states to at most 8 most frequent winners so DP is O(K * 8)
    win_counts: Dict[TransformSpec, int] = {}
    for pr in probes:
        if pr.best_spec.tid != T0_IDENTITY:
            win_counts[pr.best_spec] = win_counts.get(pr.best_spec, 0) + 1
    top_non_t0 = [sp for sp, _ in sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)[:7]]
    if prev_active_spec.tid != T0_IDENTITY and prev_active_spec not in top_non_t0:
        top_non_t0.append(prev_active_spec)
    states: List[TransformSpec] = [t0] + top_non_t0
    m = len(states)

    # 2. Compute per-slot cost matrix cost[i, st] (extrapolated compressed bytes for slot i)
    cost = np.empty((n_slots, m), dtype=np.float64)
    for i, (s, e) in enumerate(slot_bounds):
        slot_len = e - s
        p16 = arr[s: min(e, s + cfg.base_probe_bytes)]
        scale = slot_len / max(1, p16.shape[0])
        pr = probes[i]
        t0_c = pr.t0_bytes * scale
        cost[i, 0] = t0_c
        for j in range(1, m):
            sp = states[j]
            if sp in pr.scores:
                sz = pr.scores[sp]
            else:
                # Lazily evaluate state sp on this slot's 16 KB probe if sp is active nearby
                # OR if sp is one of the top-2 winners in the chunk (lets wide-probe T3_2D compete fairly!)
                has_nearby = (
                    j <= 2
                    or (i > 0 and probes[i - 1].best_spec == sp)
                    or (i + 1 < n_slots and probes[i + 1].best_spec == sp)
                    or (i == 0 and prev_active_spec == sp)
                )
                if has_nearby:
                    sz = scorer._zstd1_size(apply_transform_forward(p16, sp))
                    bytes_scored += p16.shape[0]
                    pr.scores[sp] = sz
                else:
                    # Conservative penalty (at least T0 cost)
                    sz = pr.t0_bytes + 512
            # Apply gain gate: if transform does not beat T0 by margin on this slot,
            # give it no bonus below T0 unless DP continuity justifies keeping the regime.
            cost[i, j] = sz * scale

    # 3. 1D Dynamic Programming (Viterbi) with explicit regime-switch penalty P_switch
    dp = np.full((n_slots, m), 1e18, dtype=np.float64)
    back = np.zeros((n_slots, m), dtype=np.int32)
    p_sw = float(cfg.p_switch_bytes)

    # Initial slot 0: switching away from prev_active_spec incurs P_switch
    prev_idx = states.index(prev_active_spec) if prev_active_spec in states else 0
    for j in range(m):
        sw_pen = 0.0 if j == prev_idx else p_sw
        dp[0, j] = cost[0, j] + sw_pen

    for i in range(1, n_slots):
        # Histogram Jensen-Shannon divergence between adjacent probes
        js = float(js_divergence(probes[i].hist, probes[i - 1].hist))
        # When adjacent histograms are very close (D_JS < 0.15), slightly reinforce staying in same state
        eff_sw = p_sw * (1.15 if js < cfg.js_merge_threshold else 0.85)
        for j in range(m):
            best_v = 1e18
            best_k = 0
            for k in range(m):
                v = dp[i - 1, k] + (0.0 if k == j else eff_sw)
                if v < best_v:
                    best_v = v
                    best_k = k
            dp[i, j] = best_v + cost[i, j]
            back[i, j] = best_k

    # Backtrack optimal state sequence
    path = [0] * n_slots
    best_last = int(np.argmin(dp[n_slots - 1]))
    path[n_slots - 1] = best_last
    for i in range(n_slots - 2, -1, -1):
        path[i] = int(back[i + 1, path[i + 1]])

    # 4. Merge contiguous slots with identical state into Regions
    raw_regions: List[Tuple[int, int, TransformSpec]] = []
    cur_s = slot_bounds[0][0]
    cur_sp = states[path[0]]
    for i in range(1, n_slots):
        sp = states[path[i]]
        if sp != cur_sp:
            raw_regions.append((cur_s, slot_bounds[i - 1][1], cur_sp))
            cur_s = slot_bounds[i][0]
            cur_sp = sp
    raw_regions.append((cur_s, slot_bounds[-1][1], cur_sp))

    # 5. Enforce minimum Region size + Stationary zstd -3 Region Verification + LDM Duplication Guard
    # (Catches Mach-O/ELF binaries where 16 KB zstd -1 overestimates gain or where pulling a
    # cross-file near-duplicate region out of S_RAW destroys zstd --long=27 LDM matches!)
    c3 = zstd.ZstdCompressor(level=3)
    verified: List[Tuple[int, int, TransformSpec]] = []
    for rs, re, sp in raw_regions:
        rlen = re - rs
        if sp.tid != T0_IDENTITY:
            if rlen < cfg.min_region_bytes:
                sp = t0
            else:
                sub = arr[rs:re]
                # Guard 1: Cross-file LDM duplication check against trailing 128 MB S_RAW history
                if raw_index is not None and raw_index.overlap_fraction(sub) >= cfg.ldm_dup_threshold:
                    sp = t0
                else:
                    # Guard 2: Stationary 2x256KB zstd -3 check (both start & mid must genuinely win!)
                    vlen = min(rlen, 262144)
                    s_a = sub[:vlen]
                    z0_a = len(c3.compress(memoryview(s_a)))
                    zk_a = len(c3.compress(memoryview(apply_transform_forward(s_a, sp))))
                    if rlen >= 2 * vlen:
                        mid = (rlen - vlen) // 2
                        s_b = sub[mid: mid + vlen]
                        z0_b = len(c3.compress(memoryview(s_b)))
                        zk_b = len(c3.compress(memoryview(apply_transform_forward(s_b, sp))))
                    else:
                        z0_b, zk_b = z0_a, zk_a

                    # Both slices must beat T0 by >= (1 - region_z3_gate) (default 6%)
                    # AND total extrapolated savings must beat min_region_saved_bytes (256 KB) context-split risk
                    est_saved = ((z0_a + z0_b) - (zk_a + zk_b)) * (rlen / max(1, 2 * vlen))
                    if (
                        zk_a > z0_a * cfg.region_z3_gate
                        or zk_b > z0_b * cfg.region_z3_gate
                        or est_saved < float(cfg.min_region_saved_bytes)
                    ):
                        sp = t0

        if verified and verified[-1][2] == sp:
            ps, _, _ = verified[-1]
            verified[-1] = (ps, re, sp)
        else:
            verified.append((rs, re, sp))

    return verified, bytes_scored


# ---------------------------------------------------------------------------
# Stage 4: Interleaved Flush-Group Streaming Encoder & Single-Cursor Decoder
# ---------------------------------------------------------------------------

class ASV3Encoder:
    """True single-pass streaming encoder (zero backward seeks, bounded memory)."""

    def __init__(self, out_stream: BinaryIO, cfg: EncoderConfig | None = None):
        self.out = out_stream
        self.cfg = cfg if cfg is not None else EncoderConfig()
        self.scorer = TwoTierScorer(gain_gate=self.cfg.gain_gate)
        self.stats = CodecStats()

        # 4 persistent zstd --long=27 streaming contexts (never reset across groups!)
        # S_META uses level 3 (tiny manifest, instant); payload streams use cfg.zstd_level
        self._c_meta = _make_zstd_stream_compressor(3, 20)
        self._c_raw = _make_zstd_stream_compressor(self.cfg.zstd_level, self.cfg.window_log)
        self._c_xform = _make_zstd_stream_compressor(self.cfg.zstd_level, self.cfg.window_log)
        self._c_resid = _make_zstd_stream_compressor(self.cfg.zstd_level, self.cfg.window_log)

        # Track whether each stream has received any bytes (so empty streams emit 0 bytes!)
        self._used = [False, False, False, False]

        # Running SHA-256 of uncompressed input
        self._sha = hashlib.sha256()

        # Region-anchored cross-FlushGroup state (at most S-1 < 4096 bytes carried!)
        self._active_spec = TransformSpec(T0_IDENTITY, 1, 0)
        self._carry = bytearray()
        self._header_written = False
        self._raw_index = TrailingRawIndex()
        self._forest: Optional[Stage3DeltaForest] = (
            Stage3DeltaForest(
                gear_table=GEAR_TABLE,
                extract_anchors_fn=extract_cd_anchors,
                window_bytes=1 << self.cfg.window_log,
                gain_gate=self.cfg.gain_gate,
                manifest_entry_bytes=META_REC_SIZE,
            )
            if self.cfg.enable_stage3
            else None
        )

    def _write_header(self) -> None:
        if self._header_written:
            return
        flags = (1 if self.cfg.enable_stage3 else 0) << 1
        hdr = struct.pack(FILE_HDR_FMT, MAGIC, VERSION, self.cfg.window_log, flags)
        self.out.write(hdr)
        self.stats.framing_bytes += len(hdr)
        self.stats.comp_bytes += len(hdr)
        self._header_written = True

    def _flush_group(
        self,
        group_uncomp_len: int,
        meta_buf: bytearray,
        raw_parts: List[bytes],
        xform_parts: List[bytes],
        resid_parts: List[bytes],
    ) -> None:
        if group_uncomp_len == 0:
            return
        self._write_header()

        m_in = bytes(meta_buf)
        r_in = b"".join(raw_parts)
        x_in = b"".join(xform_parts)
        d_in = b"".join(resid_parts)

        def _push(idx: int, cobj: zstd.ZstdCompressionObj, data: bytes) -> bytes:
            if not data:
                return b""
            self._used[idx] = True
            return cobj.compress(data) + cobj.flush(zstd.COMPRESSOBJ_FLUSH_BLOCK)

        cm = _push(0, self._c_meta, m_in)
        cr = _push(1, self._c_raw, r_in)
        cx = _push(2, self._c_xform, x_in)
        cd = _push(3, self._c_resid, d_in)

        ghdr = struct.pack(GROUP_HDR_FMT, TAG_DATA, group_uncomp_len, len(cm), len(cr), len(cx), len(cd))
        self.out.write(ghdr)
        if cm:
            self.out.write(cm)
        if cr:
            self.out.write(cr)
        if cx:
            self.out.write(cx)
        if cd:
            self.out.write(cd)

        self.stats.flush_groups += 1
        self.stats.framing_bytes += len(ghdr)
        self.stats.s_meta_bytes += len(cm)
        self.stats.s_raw_bytes += len(cr)
        self.stats.s_xform_bytes += len(cx)
        self.stats.s_resid_bytes += len(cd)
        self.stats.comp_bytes += len(ghdr) + len(cm) + len(cr) + len(cx) + len(cd)

    def _emit_region_blocks(
        self,
        data: np.ndarray,
        spec: TransformSpec,
        is_region_end: bool,
        meta_buf: bytearray,
        raw_parts: List[bytes],
        xform_parts: List[bytes],
        resid_parts: List[bytes],
    ) -> int:
        """Emit transformed blocks snapped to exact multiples of unit stride S.

        When Stage 3 (Phase 1B) is enabled, partitions regions via FastCDC (128-512 KB)
        and routes prefix-gated delta residuals to S_RESID while maintaining the
        symmetric 128 MB post-transform ring buffer.
        """
        n = int(data.shape[0])
        if n == 0:
            return 0

        tname = TRANSFORM_NAMES[spec.tid]
        if spec.tid == T0_IDENTITY:
            if self._forest is not None:
                cuts = fastcdc_boundaries(data, GEAR_TABLE, unit_stride=1)
                for ci in range(cuts.shape[0] - 1):
                    cs, ce = int(cuts[ci]), int(cuts[ci + 1])
                    chunk = data[cs:ce]
                    clen = ce - cs
                    dop, rel_off, payload = self._forest.process_chunk(chunk, spec)
                    meta_buf.extend(
                        struct.pack(META_REC_FMT, clen, T0_IDENTITY, dop, 1, 0, 0, 0, rel_off)
                    )
                    if dop == DELTA_NONE:
                        raw_parts.append(payload.tobytes())
                    else:
                        resid_parts.append(payload.tobytes())
                self.stats.bytes_by_transform[tname] = self.stats.bytes_by_transform.get(tname, 0) + n
                return n

            pos = 0
            while pos < n:
                end = min(n, pos + self.cfg.block_size_bytes)
                blen = end - pos
                meta_buf.extend(
                    struct.pack(META_REC_FMT, blen, T0_IDENTITY, DELTA_NONE, 1, 0, 0, 0, 0)
                )
                raw_parts.append(data[pos:end].tobytes())
                pos = end
            self.stats.bytes_by_transform[tname] = self.stats.bytes_by_transform.get(tname, 0) + n
            return n

        # Non-T0 Region: align every emitted block to exact multiple of unit stride S
        s_unit = spec.unit_stride()
        body = (n // s_unit) * s_unit
        rem = n - body

        emitted = 0
        if body > 0:
            if self._forest is not None:
                # Aligned FastCDC sub-blocks (256 KB avg) for Stage 3 delta matching
                cuts = fastcdc_boundaries(data[:body], GEAR_TABLE, unit_stride=s_unit)
                for ci in range(cuts.shape[0] - 1):
                    cs, ce = int(cuts[ci]), int(cuts[ci + 1])
                    blen = ce - cs
                    is_last = (ce == body)
                    tail = rem if (is_last and is_region_end) else 0
                    xf = apply_transform_forward(data[cs:ce], spec)
                    dop, rel_off, payload = self._forest.process_chunk(xf, spec)
                    if dop == DELTA_NONE:
                        xform_parts.append(payload.tobytes())
                    else:
                        resid_parts.append(payload.tobytes())
                    if tail > 0:
                        tail_arr = data[body:n]
                        self._forest.ring.append(tail_arr)
                        raw_parts.append(tail_arr.tobytes())
                    meta_buf.extend(
                        struct.pack(
                            META_REC_FMT,
                            blen + tail,
                            spec.tid,
                            dop,
                            spec.w,
                            spec.h,
                            0,
                            tail,
                            rel_off,
                        )
                    )
                    emitted += blen + tail
            else:
                blk_target = max(s_unit, (self.cfg.block_size_bytes // s_unit) * s_unit)
                pos = 0
                while pos < body:
                    end = min(body, pos + blk_target)
                    blen = end - pos
                    is_last_block = (end == body)
                    tail = rem if (is_last_block and is_region_end) else 0
                    xf = apply_transform_forward(data[pos:end], spec)
                    xform_parts.append(xf.tobytes())
                    if tail > 0:
                        raw_parts.append(data[body:n].tobytes())
                    meta_buf.extend(
                        struct.pack(
                            META_REC_FMT,
                            blen + tail,
                            spec.tid,
                            DELTA_NONE,
                            spec.w,
                            spec.h,
                            0,      # phase = 0 (strictly Region-anchored!)
                            tail,
                            0,
                        )
                    )
                    emitted += blen + tail
                    pos = end
            self.stats.bytes_by_transform[tname] = self.stats.bytes_by_transform.get(tname, 0) + body
            if rem > 0 and is_region_end:
                self.stats.bytes_by_transform["T0"] = self.stats.bytes_by_transform.get("T0", 0) + rem
        elif rem > 0 and is_region_end:
            tail_arr = data[:rem]
            if self._forest is not None:
                self._forest.ring.append(tail_arr)
            meta_buf.extend(struct.pack(META_REC_FMT, rem, T0_IDENTITY, DELTA_NONE, 1, 0, 0, 0, 0))
            raw_parts.append(tail_arr.tobytes())
            self.stats.bytes_by_transform["T0"] = self.stats.bytes_by_transform.get("T0", 0) + rem
            emitted += rem

        if rem > 0 and not is_region_end:
            self._carry.extend(data[body:n].tobytes())

        return emitted

    def encode_stream(self, in_stream: BinaryIO) -> CodecStats:
        t_start = time.perf_counter()
        self._write_header()

        while True:
            chunk = _read_exact(in_stream, self.cfg.flush_group_bytes)
            if not chunk:
                break
            self._sha.update(chunk)
            self.stats.orig_bytes += len(chunk)

            # Segment the newly arrived chunk (16 MB lookahead)
            arr = np.frombuffer(chunk, dtype=np.uint8)
            regions, scored = segment_buffer_dp(
                arr, self.scorer, self.cfg, self._active_spec, self._raw_index
            )
            self.stats.bytes_scored_zstd1 += scored
            # Update bounded trailing 128 MB raw fingerprint index ONLY with regions actually routed to S_RAW
            for rs, re, sp in regions:
                if sp.tid == T0_IDENTITY:
                    self._raw_index.add_raw(arr[rs:re])

            meta_buf = bytearray()
            raw_parts: List[bytes] = []
            xform_parts: List[bytes] = []
            resid_parts: List[bytes] = []
            group_emitted = 0

            for r_idx, (rs, re, sp) in enumerate(regions):
                is_first = (r_idx == 0)
                is_last = (r_idx == len(regions) - 1)

                # If active spec changed, flush any held carry from previous region as its ragged tail
                if self._carry and (not is_first or sp != self._active_spec):
                    carry_b = bytes(self._carry)
                    self._carry.clear()
                    if self._forest is not None:
                        self._forest.ring.append(carry_b)
                    meta_buf.extend(
                        struct.pack(META_REC_FMT, len(carry_b), T0_IDENTITY, DELTA_NONE, 1, 0, 0, 0, 0)
                    )
                    raw_parts.append(carry_b)
                    self.stats.bytes_by_transform["T0"] = self.stats.bytes_by_transform.get("T0", 0) + len(carry_b)
                    group_emitted += len(carry_b)

                if sp != self._active_spec or self.stats.regions_count == 0:
                    self.stats.regions_count += 1
                    self._active_spec = sp

                reg_view = arr[rs:re]
                if is_first and self._carry:
                    # Seamlessly prepend held < S carry bytes to keep exact Region grid phase=0!
                    reg_view = np.concatenate([np.frombuffer(bytes(self._carry), dtype=np.uint8), reg_view])
                    self._carry.clear()

                # Last region in chunk may continue into next FlushGroup -> hold < S carry!
                group_emitted += self._emit_region_blocks(
                    reg_view,
                    sp,
                    is_region_end=(not is_last),
                    meta_buf=meta_buf,
                    raw_parts=raw_parts,
                    xform_parts=xform_parts,
                    resid_parts=resid_parts,
                )

            self._flush_group(group_emitted, meta_buf, raw_parts, xform_parts, resid_parts)

        # EOF: flush any final carried ragged tail (< S bytes) as a tiny final group
        if self._carry:
            carry_b = bytes(self._carry)
            self._carry.clear()
            if self._forest is not None:
                self._forest.ring.append(carry_b)
            meta_buf = bytearray(
                struct.pack(META_REC_FMT, len(carry_b), T0_IDENTITY, DELTA_NONE, 1, 0, 0, 0, 0)
            )
            self.stats.bytes_by_transform["T0"] = self.stats.bytes_by_transform.get("T0", 0) + len(carry_b)
            self._flush_group(len(carry_b), meta_buf, [carry_b], [], [])

        # Finalize all used zstd contexts with ZSTD_e_end (COMPRESSOBJ_FLUSH_FINISH)
        epilogues = []
        for idx, cobj in enumerate((self._c_meta, self._c_raw, self._c_xform, self._c_resid)):
            ep = cobj.flush(zstd.COMPRESSOBJ_FLUSH_FINISH) if self._used[idx] else b""
            epilogues.append(ep)

        digest = self._sha.digest()
        thdr = struct.pack(
            TRAILER_HDR_FMT,
            TAG_EOF,
            len(epilogues[0]),
            len(epilogues[1]),
            len(epilogues[2]),
            len(epilogues[3]),
        )
        ttail = struct.pack(TRAILER_TAIL_FMT, self.stats.orig_bytes, digest)

        self.out.write(thdr)
        for ep in epilogues:
            if ep:
                self.out.write(ep)
        self.out.write(ttail)
        self.out.flush()

        self.stats.s_meta_bytes += len(epilogues[0])
        self.stats.s_raw_bytes += len(epilogues[1])
        self.stats.s_xform_bytes += len(epilogues[2])
        self.stats.s_resid_bytes += len(epilogues[3])
        self.stats.framing_bytes += len(thdr) + len(ttail)
        self.stats.comp_bytes += len(thdr) + sum(len(e) for e in epilogues) + len(ttail)
        self.stats.sha256 = digest.hex()
        self.stats.elapsed_sec = time.perf_counter() - t_start
        return self.stats


class ASV3Decoder:
    """Single-cursor streaming decoder (zero backward seeks, bounded memory, SHA-256 verified)."""

    def decode_stream(self, in_stream: BinaryIO, out_stream: BinaryIO) -> Tuple[int, str]:
        hdr = _read_exact(in_stream, FILE_HDR_SIZE)
        if len(hdr) != FILE_HDR_SIZE:
            raise ValueError("Truncated ASV3 FileHeader")
        magic, ver, wlog, flags = struct.unpack(FILE_HDR_FMT, hdr)
        if magic != MAGIC or ver != VERSION:
            raise ValueError(f"Invalid ASV3 header: magic={magic!r}, version={ver}")

        stage3_enabled = bool((flags >> 1) & 1)
        ring: Optional[RingBuffer128M] = RingBuffer128M(1 << wlog) if stage3_enabled else None

        d_meta = _make_zstd_stream_decompressor(20)
        d_raw = _make_zstd_stream_decompressor(wlog)
        d_xform = _make_zstd_stream_decompressor(wlog)
        d_resid = _make_zstd_stream_decompressor(wlog)

        sha = hashlib.sha256()
        total_out = 0

        # Per-stream pending byte buffers (carry any partial bytes cleanly if needed)
        meta_q = bytearray()
        raw_q = bytearray()
        xform_q = bytearray()
        resid_q = bytearray()

        while True:
            tag_b = _read_exact(in_stream, 1)
            if not tag_b:
                raise ValueError("Unexpected EOF before FileTrailer")
            tag = tag_b[0]

            if tag == TAG_DATA:
                rest = _read_exact(in_stream, GROUP_HDR_SIZE - 1)
                if len(rest) != GROUP_HDR_SIZE - 1:
                    raise ValueError("Truncated FlushGroup header")
                group_uncomp, lm, lr, lx, ld = struct.unpack("<IIIII", rest)

                if lm:
                    meta_q.extend(d_meta.decompress(_read_exact(in_stream, lm)))
                if lr:
                    raw_q.extend(d_raw.decompress(_read_exact(in_stream, lr)))
                if lx:
                    xform_q.extend(d_xform.decompress(_read_exact(in_stream, lx)))
                if ld:
                    resid_q.extend(d_resid.decompress(_read_exact(in_stream, ld)))

                # Step through newly decoded manifest entries for this FlushGroup
                m_pos = 0
                r_pos = 0
                x_pos = 0
                d_pos = 0
                group_dec = 0
                while m_pos + META_REC_SIZE <= len(meta_q):
                    orig_len, tid, delta_op, w, h, _phase, tail_len, parent_off = struct.unpack_from(
                        META_REC_FMT, meta_q, m_pos
                    )
                    m_pos += META_REC_SIZE

                    if tid == T0_IDENTITY:
                        if delta_op == DELTA_NONE:
                            out_b = bytes(raw_q[r_pos: r_pos + orig_len])
                            r_pos += orig_len
                            if ring is not None:
                                ring.append(out_b)
                        else:
                            assert ring is not None
                            res_arr = np.frombuffer(resid_q, dtype=np.uint8, count=orig_len, offset=d_pos)
                            d_pos += orig_len
                            parent_arr = ring.read_back(parent_off, orig_len)
                            rec_arr = delta_xor_u8(res_arr, parent_arr) if delta_op == DELTA_XOR else delta_add_u8(res_arr, parent_arr)
                            del res_arr
                            ring.append(rec_arr)
                            out_b = rec_arr.tobytes()
                    else:
                        xlen = orig_len - tail_len
                        if delta_op == DELTA_NONE:
                            xf_arr = np.frombuffer(xform_q, dtype=np.uint8, count=xlen, offset=x_pos)
                            x_pos += xlen
                            if ring is not None:
                                ring.append(xf_arr)
                            sp = TransformSpec(tid, w, h)
                            rec = apply_transform_inverse(xf_arr, sp).tobytes()
                            del xf_arr
                        else:
                            assert ring is not None
                            res_arr = np.frombuffer(resid_q, dtype=np.uint8, count=xlen, offset=d_pos)
                            d_pos += xlen
                            parent_arr = ring.read_back(parent_off, xlen)
                            xf_rec = delta_xor_u8(res_arr, parent_arr) if delta_op == DELTA_XOR else delta_add_u8(res_arr, parent_arr)
                            del res_arr
                            ring.append(xf_rec)
                            sp = TransformSpec(tid, w, h)
                            rec = apply_transform_inverse(xf_rec, sp).tobytes()

                        if tail_len > 0:
                            tail_b = bytes(raw_q[r_pos: r_pos + tail_len])
                            r_pos += tail_len
                            if ring is not None:
                                ring.append(tail_b)
                            out_b = rec + tail_b
                        else:
                            out_b = rec

                    out_stream.write(out_b)
                    sha.update(out_b)
                    total_out += len(out_b)
                    group_dec += len(out_b)

                if group_dec != group_uncomp:
                    raise ValueError(
                        f"FlushGroup length mismatch: decoded {group_dec} vs header {group_uncomp}"
                    )

                # Clear consumed stream slices (keeps decoder RAM strictly bounded!)
                del meta_q[:m_pos]
                del raw_q[:r_pos]
                del xform_q[:x_pos]
                del resid_q[:d_pos]

            elif tag == TAG_EOF:
                rest = _read_exact(in_stream, TRAILER_HDR_SIZE - 1)
                if len(rest) != TRAILER_HDR_SIZE - 1:
                    raise ValueError("Truncated FileTrailer header")
                fm, fr, fx, fd = struct.unpack("<IIII", rest)
                if fm:
                    d_meta.decompress(_read_exact(in_stream, fm))
                if fr:
                    d_raw.decompress(_read_exact(in_stream, fr))
                if fx:
                    d_xform.decompress(_read_exact(in_stream, fx))
                if fd:
                    d_resid.decompress(_read_exact(in_stream, fd))

                tail = _read_exact(in_stream, TRAILER_TAIL_SIZE)
                if len(tail) != TRAILER_TAIL_SIZE:
                    raise ValueError("Truncated FileTrailer SHA256")
                expected_size, expected_sha = struct.unpack(TRAILER_TAIL_FMT, tail)
                actual_sha = sha.digest()
                if total_out != expected_size:
                    raise ValueError(f"Size mismatch: decoded {total_out} vs trailer {expected_size}")
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"SHA-256 mismatch! expected={expected_sha.hex()}, got={actual_sha.hex()}"
                    )
                out_stream.flush()
                return total_out, actual_sha.hex()
            else:
                raise ValueError(f"Unknown FlushGroup tag: {tag:#x}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASV3 Single-Pass Streaming Codec (Phase 1A)")
    parser.add_argument("mode", choices=["c", "d"], help="c = compress, d = decompress")
    parser.add_argument("input", nargs="?", default="-", help="Input file or '-' for stdin")
    parser.add_argument("output", nargs="?", default="-", help="Output file or '-' for stdout")
    parser.add_argument("--level", type=int, default=19, help="zstd level (3 or 19)")
    parser.add_argument("--block-mb", type=int, default=4, help="Processing block size in MB (1/4/16)")
    parser.add_argument("--flush-mb", type=int, default=16, help="Flush group size in MB (8-16)")
    parser.add_argument("--p-switch", type=int, default=49152, help="DP switch penalty in bytes")
    parser.add_argument("--stage3", action="store_true", help="Enable Phase 1B Stage 3 Delta Forest (A3)")
    parser.add_argument("--stats-json", type=str, default="", help="Optional path to write stats JSON")
    args = parser.parse_args()

    in_f = sys.stdin.buffer if args.input == "-" else open(args.input, "rb")
    out_f = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")
    try:
        if args.mode == "c":
            cfg = EncoderConfig(
                zstd_level=args.level,
                block_size_bytes=args.block_mb * MB,
                flush_group_bytes=args.flush_mb * MB,
                p_switch_bytes=args.p_switch,
                enable_stage3=args.stage3,
            )
            enc = ASV3Encoder(out_f, cfg)
            stats = enc.encode_stream(in_f)
            if args.stats_json:
                Path(args.stats_json).write_text(json.dumps(stats.to_dict(), indent=2))
            elif args.output != "-":
                print(json.dumps(stats.to_dict(), indent=2), file=sys.stderr)
        else:
            dec = ASV3Decoder()
            total_out, sha_hex = dec.decode_stream(in_f, out_f)
            if args.output != "-":
                print(f"DECODE OK: {total_out} bytes, sha256={sha_hex}", file=sys.stderr)
    finally:
        if args.input != "-":
            in_f.close()
        if args.output != "-":
            out_f.close()


if __name__ == "__main__":
    main()
