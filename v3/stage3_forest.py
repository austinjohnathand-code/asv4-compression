"""Phase 1B: Stage 3 Bounded Delta Forest (A3 Ablation).

Implements Section 2 Stage 3 of the v3 Spec:
  - Fixed 128 MB (W_win = 1 << window_log) symmetric post-transform ring buffer
    shared in lockstep between encoder and decoder (O(1) parent read, no depth cap!).
  - O(N) FastCDC content-defined chunking (min 128 KB, avg 256 KB, max 512 KB)
    + Gear content-defined 32-byte anchor index scoped to trailing 128 MB window.
  - Honest Prefix-Aware Gain Gate:
      zstd1(R) + manifest_entry_bytes <= 0.96 * zstd1(C_i | prefix_dict = P_p)
    comparing OP_XOR and OP_SUB residuals against zstd -1 seeded with the parent
    as a raw prefix dictionary (DICT_TYPE_RAWCONTENT).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from numba import njit
import zstandard as zstd

from transforms import TransformSpec

DELTA_NONE = 0
DELTA_XOR = 1
DELTA_SUB = 2


@njit(cache=True)
def fastcdc_boundaries(
    arr: np.ndarray,
    gear_table: np.ndarray,
    unit_stride: int = 1,
    min_sz: int = 131072,
    avg_sz: int = 262144,
    max_sz: int = 524288,
) -> np.ndarray:
    """FastCDC content-defined chunk cut points snapped to unit_stride."""
    n = arr.shape[0]
    max_chunks = (n // max(1, min_sz)) + 4
    cuts = np.empty(max_chunks, dtype=np.int64)
    cuts[0] = 0
    n_cuts = 1

    # Mask for ~256 KB average: mask_s (before avg) stricter, mask_l (after avg) looser
    mask_s = np.uint64(0x7FFFF)  # ~1/524288
    mask_l = np.uint64(0x1FFFF)  # ~1/131072

    pos = 0
    while pos < n:
        rem = n - pos
        if rem <= min_sz:
            cuts[n_cuts] = n
            n_cuts += 1
            break
        lim = pos + max_sz
        if lim > n:
            lim = n
        mid = pos + avg_sz
        if mid > lim:
            mid = lim

        h = np.uint64(0)
        cut = lim
        i = pos + min_sz
        while i < mid:
            h = (h << np.uint64(1)) + gear_table[arr[i]]
            if (h & mask_s) == np.uint64(0):
                cut = i
                break
            i += 1
        if cut == lim:
            while i < lim:
                h = (h << np.uint64(1)) + gear_table[arr[i]]
                if (h & mask_l) == np.uint64(0):
                    cut = i
                    break
                i += 1

        # Snap cut to unit_stride alignment if needed
        if unit_stride > 1 and cut < n:
            span = cut - pos
            snapped = (span // unit_stride) * unit_stride
            if snapped < unit_stride:
                snapped = min(rem, unit_stride)
            cut = pos + snapped

        if cut <= pos:
            cut = lim
        cuts[n_cuts] = cut
        n_cuts += 1
        pos = cut

    return cuts[:n_cuts]


@njit(cache=True)
def delta_xor_u8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = a[i] ^ b[i]
    return out


@njit(cache=True)
def delta_sub_u8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = np.uint8((np.int64(a[i]) - np.int64(b[i])) & 0xFF)
    return out


@njit(cache=True)
def delta_add_u8(res: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = res.shape[0]
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = np.uint8((np.int64(res[i]) + np.int64(b[i])) & 0xFF)
    return out


class RingBuffer128M:
    """Strictly bounded W_win (default 128 MB) sliding ring buffer for post-transform segments.

    Provides O(1) append and O(1) read_back(rel_offset, length) across wrap-around.
    """

    def __init__(self, capacity: int = 128 * 1024 * 1024):
        self.capacity = capacity
        self._buf = bytearray(capacity)
        self._mv = memoryview(self._buf)
        self.total_written = 0  # Monotonic global byte cursor

    def append(self, data: bytes | bytearray | memoryview | np.ndarray) -> None:
        raw = memoryview(data).cast("B")
        n = len(raw)
        if n == 0:
            return
        if n >= self.capacity:
            # Keep only trailing capacity bytes
            tail = raw[n - self.capacity:]
            self._mv[:] = tail
            self.total_written += n
            return
        wpos = self.total_written % self.capacity
        first = min(n, self.capacity - wpos)
        self._mv[wpos: wpos + first] = raw[:first]
        if first < n:
            self._mv[0: n - first] = raw[first:]
        self.total_written += n

    def read_back(self, rel_offset: int, length: int) -> np.ndarray:
        """Read `length` bytes starting `rel_offset` bytes before current cursor."""
        if rel_offset < length or rel_offset > self.capacity or rel_offset > self.total_written:
            raise ValueError(
                f"Invalid ring read_back(rel_offset={rel_offset}, len={length}, total={self.total_written}, cap={self.capacity})"
            )
        start_abs = self.total_written - rel_offset
        rpos = start_abs % self.capacity
        if rpos + length <= self.capacity:
            # Contiguous copy
            return np.frombuffer(bytes(self._mv[rpos: rpos + length]), dtype=np.uint8)
        first = self.capacity - rpos
        out = bytes(self._mv[rpos: self.capacity]) + bytes(self._mv[0: length - first])
        return np.frombuffer(out, dtype=np.uint8)


class Stage3DeltaForest:
    """Bounded-memory O(N) delta forest with honest zstd-1 prefix-dictionary gain gate."""

    def __init__(
        self,
        gear_table: np.ndarray,
        extract_anchors_fn,
        window_bytes: int = 128 * 1024 * 1024,
        gain_gate: float = 0.96,
        manifest_entry_bytes: int = 24,
        max_candidates: int = 4,
    ):
        self.gear_table = gear_table
        self.extract_anchors = extract_anchors_fn
        self.window_bytes = window_bytes
        self.gain_gate = gain_gate
        self.manifest_entry_bytes = manifest_entry_bytes
        self.max_candidates = max_candidates

        self.ring = RingBuffer128M(window_bytes)
        # Anchor index: hash -> list of (abs_pos, spec_key)
        self._index: Dict[int, List[Tuple[int, Tuple[int, int, int]]]] = {}
        self._c1 = zstd.ZstdCompressor(level=1)

        # Telemetry
        self.chunks_tested = 0
        self.candidates_checked = 0
        self.deltas_accepted = 0
        self.bytes_accepted = 0
        self.bytes_scored_s3 = 0

    def process_chunk(
        self,
        post_xform: np.ndarray,
        spec: TransformSpec,
    ) -> Tuple[int, int, np.ndarray]:
        """Try finding a parent in the 128 MB ring buffer for post_xform chunk.

        Returns (delta_op, parent_rel_offset, payload_to_emit).
        Always appends post_xform (reconstructed state!) to the ring buffer and index.
        """
        n = int(post_xform.shape[0])
        cur_abs = self.ring.total_written
        spec_key = (spec.tid, spec.w, spec.h)
        self.chunks_tested += 1

        # Extract shift-invariant content-defined 32-byte anchors (~1 per 4 KB -> ~64 per 256 KB)
        hs, pos_arr, cnt = self.extract_anchors(post_xform, self.gear_table, np.uint64(0xFFF), 256)

        # Vote on candidate parent_abs_start = p_abs - c_off (sharing same spec_key, inside 128 MB ring)
        votes: Dict[int, int] = {}
        min_valid_abs = cur_abs - self.window_bytes
        for i in range(cnt):
            h = int(hs[i])
            c_off = int(pos_arr[i])
            bucket = self._index.get(h)
            if not bucket:
                continue
            # Lazily prune expired entries outside 128 MB ring
            if bucket[0][0] < min_valid_abs:
                bucket = [x for x in bucket if x[0] >= min_valid_abs]
                if bucket:
                    self._index[h] = bucket
                else:
                    del self._index[h]
                    continue
            for p_abs, p_key in bucket:
                if p_key != spec_key:
                    continue
                p_start = p_abs - c_off
                # Candidate parent slice [p_start .. p_start + n] must lie completely inside valid ring!
                if p_start >= min_valid_abs and (p_start + n) <= cur_abs:
                    votes[p_start] = votes.get(p_start, 0) + 1

        best_op = DELTA_NONE
        best_rel = 0
        best_payload = post_xform

        if votes:
            # Query up to K = 4 candidate parents with strongest anchor agreement (>= 2 anchors)
            top_cands = [
                p_start for p_start, v in sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
                if v >= 2
            ][: self.max_candidates]

            best_ratio = self.gain_gate
            child_bytes = post_xform.tobytes()

            for p_start in top_cands:
                self.candidates_checked += 1
                rel_off = cur_abs - p_start
                parent_arr = self.ring.read_back(rel_off, n)

                # Honest Prefix-Aware Gain Gate:
                # Compare against zstd -1 seeded with parent P_p as raw prefix dictionary!
                p_dict = zstd.ZstdCompressionDict(
                    parent_arr.tobytes(), dict_type=zstd.DICT_TYPE_RAWCONTENT
                )
                base_prefix_sz = len(zstd.ZstdCompressor(level=1, dict_data=p_dict).compress(child_bytes))
                self.bytes_scored_s3 += n

                r_xor = delta_xor_u8(post_xform, parent_arr)
                sz_xor = len(self._c1.compress(memoryview(r_xor)))
                self.bytes_scored_s3 += n

                r_sub = delta_sub_u8(post_xform, parent_arr)
                sz_sub = len(self._c1.compress(memoryview(r_sub)))
                self.bytes_scored_s3 += n

                if sz_sub <= sz_xor:
                    cand_op, cand_res, cand_sz = DELTA_SUB, r_sub, sz_sub
                else:
                    cand_op, cand_res, cand_sz = DELTA_XOR, r_xor, sz_xor

                ratio = (cand_sz + self.manifest_entry_bytes) / max(1, base_prefix_sz)
                if ratio <= best_ratio:
                    best_ratio = ratio
                    best_op = cand_op
                    best_rel = rel_off
                    best_payload = cand_res

        if best_op != DELTA_NONE:
            self.deltas_accepted += 1
            self.bytes_accepted += n

        # Append reconstructed post-transform chunk into ring buffer and register anchors
        self.ring.append(post_xform)
        for i in range(cnt):
            h = int(hs[i])
            abs_p = cur_abs + int(pos_arr[i])
            bucket = self._index.get(h)
            if bucket is None:
                self._index[h] = [(abs_p, spec_key)]
            else:
                if len(bucket) >= 8:
                    # Keep most recent entries
                    bucket = bucket[-7:]
                bucket.append((abs_p, spec_key))
                self._index[h] = bucket

        return best_op, best_rel, best_payload
