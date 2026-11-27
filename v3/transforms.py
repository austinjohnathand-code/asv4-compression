"""ASV3 Two-Tier Transform Catalog, Exact-Match Lag Discovery, and Entropy-Aware Scorer.

Implements Section 2.1 and 2.2 of the v3 Spec:
  - Fast exact-match autocorrelation M(L) = sum(I(b[i] == b[i+L])) on 16 KB base
    probes (L in [2, 256]) and 64 KB wide probes (strided/page lags up to 4096 B).
  - Tier 1: T0 (Identity), T1(w) (2D Byte-Plane Transpose)
  - Tier 2 (conditional if Tier 1 beats T0 by >= 2%):
      T2_ELEM(w) - Little-endian element-wise delta (carry preserved) + byte-plane transpose
      T2_XOR(w)  - Record-wise XOR + byte-plane transpose
      T2_BIT(w)  - Bit-plane shuffle (for w in {4, 8})
      T3_2D(w,h) - Byte-plane transpose + 2D Paeth spatial predictor along row stride h
  - Scorer: zstd -1 with optional per-plane H0 pre-filter, harmonic tie-breaking,
    and configurable gain gate (default <= 0.96 * zstd1(T0)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np
from numba import njit
import zstandard as zstd


# Transform IDs (compact uint8)
T0_IDENTITY = 0
T1_TRANSPOSE = 1
T2_ELEM_DELTA = 2
T2_XOR_TRANSPOSE = 3
T2_BITSHUFFLE = 4
T3_2D_PAETH = 5

TRANSFORM_NAMES = {
    T0_IDENTITY: "T0",
    T1_TRANSPOSE: "T1",
    T2_ELEM_DELTA: "T2_ELEM",
    T2_XOR_TRANSPOSE: "T2_XOR",
    T2_BITSHUFFLE: "T2_BIT",
    T3_2D_PAETH: "T3_2D",
}

FIXED_T1_WIDTHS = (2, 4, 8, 16, 32, 64)
WIDE_LAGS = (384, 512, 768, 1024, 2048, 4096)


@dataclass(frozen=True)
class TransformSpec:
    tid: int = T0_IDENTITY
    w: int = 1
    h: int = 0  # Only used by T3_2D (row stride in elements)

    def unit_stride(self) -> int:
        """Minimum alignment unit S for Region-Anchored grid alignment."""
        if self.tid == T0_IDENTITY:
            return 1
        if self.tid == T2_BITSHUFFLE:
            return 8 * self.w
        if self.tid == T3_2D_PAETH and self.h > 0:
            return self.w * self.h
        return max(1, self.w)

    def label(self) -> str:
        name = TRANSFORM_NAMES.get(self.tid, f"T{self.tid}")
        if self.tid == T0_IDENTITY:
            return name
        if self.tid == T3_2D_PAETH:
            return f"{name}({self.w},{self.h})"
        return f"{name}({self.w})"


# ---------------------------------------------------------------------------
# 2.1 Fast Lag Discovery: Exact-Match Autocorrelation M(L)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _exact_match_autocorr_base(buf: np.ndarray, max_lag: int = 256) -> np.ndarray:
    """Compute normalized exact-match rate M(L)/(N-L) for lags 2..max_lag."""
    n = buf.shape[0]
    scores = np.zeros(max_lag + 1, dtype=np.float32)
    if n <= max_lag + 64:
        return scores
    # Cap inner scan at 8192 comparisons per lag for speed while keeping 16 KB coverage
    # (stride-2 subsampling on 16 KB gives 8192 pairs across the entire 16 KB probe!)
    for lag in range(2, max_lag + 1):
        lim = n - lag
        step = 1 if lim <= 4096 else 2
        matches = 0
        cnt = 0
        for i in range(0, lim, step):
            if buf[i] == buf[i + lag]:
                matches += 1
            cnt += 1
        if cnt > 0:
            scores[lag] = matches / cnt
    return scores


@njit(cache=True)
def _exact_match_autocorr_wide(buf: np.ndarray) -> Tuple[int, float]:
    """Scan wide lags {384, 512, 768, 1024, 2048, 4096} + sparse stride-16 up to 4096."""
    n = buf.shape[0]
    if n < 8192:
        return 0, 0.0

    best_lag = 0
    best_score = 0.0
    # Evaluate explicit wide lags + sparse grid (272..4096 step 32)
    # Use 4096 sampled pairs across the 64 KB buffer
    for lag in (384, 512, 768, 1024, 2048, 4096):
        if n <= lag + 256:
            continue
        lim = n - lag
        step = max(1, lim // 4096)
        matches = 0
        cnt = 0
        for i in range(0, lim, step):
            if buf[i] == buf[i + lag]:
                matches += 1
            cnt += 1
        sc = matches / cnt if cnt > 0 else 0.0
        if sc > best_score:
            best_score = sc
            best_lag = lag

    # Sparse peaks up to 4096 (step 64)
    for lag in range(320, 4097, 64):
        if n <= lag + 256:
            continue
        lim = n - lag
        step = max(1, lim // 2048)
        matches = 0
        cnt = 0
        for i in range(0, lim, step):
            if buf[i] == buf[i + lag]:
                matches += 1
            cnt += 1
        sc = matches / cnt if cnt > 0 else 0.0
        if sc > best_score:
            best_score = sc
            best_lag = lag

    return best_lag, best_score


def discover_lag_peaks(
    probe_16k: np.ndarray,
    wide_probe_64k: np.ndarray | None = None,
) -> Tuple[int, int]:
    """Return top 2 non-harmonic exact-match lag peaks (L1, L2)."""
    scores = _exact_match_autocorr_base(probe_16k, 256)
    med = float(np.median(scores[2:])) if scores.shape[0] > 2 else 0.0
    thresh = med + 0.025  # Peak must clearly exceed background match rate

    # Find local peaks in [2, 256]
    peaks: List[Tuple[float, int]] = []
    for lag in range(2, 257):
        sc = float(scores[lag])
        if sc >= thresh:
            left = float(scores[lag - 1]) if lag > 2 else 0.0
            right = float(scores[lag + 1]) if lag < 256 else 0.0
            if sc >= left and sc >= right:
                peaks.append((sc, lag))

    if wide_probe_64k is not None and wide_probe_64k.shape[0] >= 8192:
        wlag, wsc = _exact_match_autocorr_wide(wide_probe_64k)
        if wlag > 0 and wsc >= thresh:
            peaks.append((wsc, wlag))

    peaks.sort(key=lambda x: x[0], reverse=True)
    chosen: List[int] = []
    for _sc, lag in peaks:
        is_harmonic = False
        for c in chosen:
            if lag % c == 0 or c % lag == 0:
                is_harmonic = True
                break
        if not is_harmonic:
            chosen.append(lag)
            if len(chosen) == 2:
                break

    l1 = chosen[0] if len(chosen) >= 1 else 0
    l2 = chosen[1] if len(chosen) >= 2 else 0
    return l1, l2


@njit(cache=True)
def _discover_row_stride_in_plane(plane: np.ndarray) -> int:
    """Detect secondary 2D row stride h inside a byte-plane (e.g. X=64..2048 of a 2D/3D grid)."""
    n = plane.shape[0]
    if n < 2048:
        return 0
    # Common 2D row widths + power-of-2 / common grid widths
    cands = np.array([32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048], dtype=np.int32)
    # Measure lag-1 diff as baseline; a 2D row stride h has low |p[i] - p[i-h]| or high match rate
    lim0 = min(n - 1, 4096)
    base_diff = 0
    for i in range(1, lim0):
        d = np.int64(plane[i]) - np.int64(plane[i - 1])
        if d < 0:
            d = -d
        base_diff += d
    base_avg = base_diff / max(1, lim0 - 1)

    # Compare candidate row-stride lag h against random lag 137 (do NOT cap at lag-1 diff,
    # because rectangular 2D grids like (256, 1024) can be smoother along X than along Y!)
    rand_diff = 0
    rlim = min(n - 137, 4096)
    if rlim > 64:
        for i in range(rlim):
            d = np.int64(plane[i + 137]) - np.int64(plane[i])
            if d < 0:
                d = -d
            rand_diff += d
        rand_avg = rand_diff / rlim
    else:
        rand_avg = 64.0

    best_h = 0
    best_avg = rand_avg * 0.55
    for ci in range(cands.shape[0]):
        h = int(cands[ci])
        if h * 3 >= n:
            continue
        lim = min(n - h, 4096)
        s = 0
        for i in range(lim):
            d = np.int64(plane[i + h]) - np.int64(plane[i])
            if d < 0:
                d = -d
            s += d
        avg = s / lim
        if avg < best_avg:
            best_avg = avg
            best_h = h

    return best_h


# ---------------------------------------------------------------------------
# 2.2 Bijective Transform Kernels (Numba-JIT, Zero Drift, Exact Inverse)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _t1_forward(data: np.ndarray, w: int) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    rows = n // w
    body = rows * w
    if rows == 0:
        out[:] = data[:]
        return out
    for c in range(w):
        base = c * rows
        idx = c
        for r in range(rows):
            out[base + r] = data[idx]
            idx += w
    for i in range(body, n):
        out[i] = data[i]
    return out


@njit(cache=True)
def _t1_inverse(data: np.ndarray, w: int) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    rows = n // w
    body = rows * w
    if rows == 0:
        out[:] = data[:]
        return out
    for c in range(w):
        base = c * rows
        idx = c
        for r in range(rows):
            out[idx] = data[base + r]
            idx += w
    for i in range(body, n):
        out[i] = data[i]
    return out


@njit(cache=True)
def _t2_xor_forward(data: np.ndarray, w: int) -> np.ndarray:
    """Record-wise XOR (rec[r] ^ rec[r-1]) fused with 2D byte-plane transpose."""
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    rows = n // w
    body = rows * w
    if rows == 0:
        out[:] = data[:]
        return out
    for c in range(w):
        base = c * rows
        idx = c
        prev = data[idx]
        out[base] = prev
        idx += w
        for r in range(1, rows):
            v = data[idx]
            out[base + r] = v ^ prev
            prev = v
            idx += w
    for i in range(body, n):
        out[i] = data[i]
    return out


@njit(cache=True)
def _t2_xor_inverse(data: np.ndarray, w: int) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    rows = n // w
    body = rows * w
    if rows == 0:
        out[:] = data[:]
        return out
    for c in range(w):
        base = c * rows
        idx = c
        acc = data[base]
        out[idx] = acc
        idx += w
        for r in range(1, rows):
            acc = acc ^ data[base + r]
            out[idx] = acc
            idx += w
    for i in range(body, n):
        out[i] = data[i]
    return out


def _t2_elem_forward(data: np.ndarray, w: int) -> np.ndarray:
    """Element-wise little-endian integer delta (carry preserved) + byte-plane transpose."""
    n = data.shape[0]
    rows = n // w
    body = rows * w
    if rows < 2 or w not in (2, 4, 8):
        return _t1_forward(data, w)
    dt = {2: np.dtype("<u2"), 4: np.dtype("<u4"), 8: np.dtype("<u8")}[w]
    elems = data[:body].view(dt)
    diffs = np.empty_like(elems)
    diffs[0] = elems[0]
    diffs[1:] = elems[1:] - elems[:-1]  # Exact modular unsigned subtraction with full carry
    trans = _t1_forward(diffs.view(np.uint8), w)
    if body == n:
        return trans
    out = np.empty(n, dtype=np.uint8)
    out[:body] = trans
    out[body:] = data[body:]
    return out


def _t2_elem_inverse(data: np.ndarray, w: int) -> np.ndarray:
    n = data.shape[0]
    rows = n // w
    body = rows * w
    if rows < 2 or w not in (2, 4, 8):
        return _t1_inverse(data, w)
    dt = {2: np.dtype("<u2"), 4: np.dtype("<u4"), 8: np.dtype("<u8")}[w]
    untrans = _t1_inverse(data[:body], w)
    elems = untrans.view(dt)
    rec = np.cumsum(elems, dtype=dt)
    rec_u8 = rec.view(np.uint8)
    if body == n:
        return rec_u8
    out = np.empty(n, dtype=np.uint8)
    out[:body] = rec_u8
    out[body:] = data[body:]
    return out


@njit(cache=True)
def _t2_bitshuffle_forward(data: np.ndarray, w: int) -> np.ndarray:
    """True bit-plane shuffle for w-byte elements (w in {4, 8}).

    Processes groups of 8 elements (unit = 8*w bytes). Within each group,
    transposes the 8x(8*w) bit matrix so bit plane b (0..8*w-1) stores the b-th
    bit of 8 consecutive elements packed into 1 byte, grouped contiguously by bit plane.
    """
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    unit = 8 * w
    groups = n // unit
    body = groups * unit
    if groups == 0:
        out[:] = data[:]
        return out

    # For each byte lane c in 0..w-1, and each bit k in 0..7:
    # bit plane index = c * 8 + k, occupying out[(c*8+k)*groups .. (c*8+k+1)*groups]
    for g in range(groups):
        g_base = g * unit
        for c in range(w):
            # Load 8 bytes (byte c of the 8 elements in group g)
            b0 = data[g_base + 0 * w + c]
            b1 = data[g_base + 1 * w + c]
            b2 = data[g_base + 2 * w + c]
            b3 = data[g_base + 3 * w + c]
            b4 = data[g_base + 4 * w + c]
            b5 = data[g_base + 5 * w + c]
            b6 = data[g_base + 6 * w + c]
            b7 = data[g_base + 7 * w + c]
            for bit in range(8):
                packed = (
                    (((b0 >> bit) & 1) << 0)
                    | (((b1 >> bit) & 1) << 1)
                    | (((b2 >> bit) & 1) << 2)
                    | (((b3 >> bit) & 1) << 3)
                    | (((b4 >> bit) & 1) << 4)
                    | (((b5 >> bit) & 1) << 5)
                    | (((b6 >> bit) & 1) << 6)
                    | (((b7 >> bit) & 1) << 7)
                )
                out[(c * 8 + bit) * groups + g] = np.uint8(packed)

    for i in range(body, n):
        out[i] = data[i]
    return out


@njit(cache=True)
def _t2_bitshuffle_inverse(data: np.ndarray, w: int) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    unit = 8 * w
    groups = n // unit
    body = groups * unit
    if groups == 0:
        out[:] = data[:]
        return out

    for g in range(groups):
        g_base = g * unit
        for c in range(w):
            p0 = data[(c * 8 + 0) * groups + g]
            p1 = data[(c * 8 + 1) * groups + g]
            p2 = data[(c * 8 + 2) * groups + g]
            p3 = data[(c * 8 + 3) * groups + g]
            p4 = data[(c * 8 + 4) * groups + g]
            p5 = data[(c * 8 + 5) * groups + g]
            p6 = data[(c * 8 + 6) * groups + g]
            p7 = data[(c * 8 + 7) * groups + g]
            for elem in range(8):
                val = (
                    (((p0 >> elem) & 1) << 0)
                    | (((p1 >> elem) & 1) << 1)
                    | (((p2 >> elem) & 1) << 2)
                    | (((p3 >> elem) & 1) << 3)
                    | (((p4 >> elem) & 1) << 4)
                    | (((p5 >> elem) & 1) << 5)
                    | (((p6 >> elem) & 1) << 6)
                    | (((p7 >> elem) & 1) << 7)
                )
                out[g_base + elem * w + c] = np.uint8(val)

    for i in range(body, n):
        out[i] = data[i]
    return out


@njit(cache=True)
def _paeth(a: np.int64, b: np.int64, c: np.int64) -> np.int64:
    p = a + b - c
    pa = p - a
    if pa < 0:
        pa = -pa
    pb = p - b
    if pb < 0:
        pb = -pb
    pc = p - c
    if pc < 0:
        pc = -pc
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


@njit(cache=True)
def _t3_2d_paeth_forward(data: np.ndarray, w: int, h: int) -> np.ndarray:
    """Byte-plane transpose (w) followed by 2D Paeth predictor along row stride h."""
    n = data.shape[0]
    trans = _t1_forward(data, w)
    rows = n // w
    if rows <= h or h <= 1:
        return trans
    out = np.empty(n, dtype=np.uint8)
    body = rows * w
    for c in range(w):
        base = c * rows
        # First row (0..h-1): left predictor
        out[base] = trans[base]
        for i in range(1, h):
            out[base + i] = np.uint8((np.int64(trans[base + i]) - np.int64(trans[base + i - 1])) & 0xFF)
        # Subsequent rows: Up on left edge (i % h == 0), Paeth(left, up, upleft) elsewhere
        for i in range(h, rows):
            b = np.int64(trans[base + i - h])
            if (i % h) == 0:
                pred = b
            else:
                a = np.int64(trans[base + i - 1])
                ul = np.int64(trans[base + i - h - 1])
                pred = _paeth(a, b, ul)
            out[base + i] = np.uint8((np.int64(trans[base + i]) - pred) & 0xFF)
    for i in range(body, n):
        out[i] = data[i]
    return out


@njit(cache=True)
def _t3_2d_paeth_inverse(data: np.ndarray, w: int, h: int) -> np.ndarray:
    n = data.shape[0]
    rows = n // w
    if rows <= h or h <= 1:
        return _t1_inverse(data, w)
    trans = np.empty(n, dtype=np.uint8)
    body = rows * w
    for c in range(w):
        base = c * rows
        acc = np.int64(data[base])
        trans[base] = np.uint8(acc)
        for i in range(1, h):
            acc = (acc + np.int64(data[base + i])) & 0xFF
            trans[base + i] = np.uint8(acc)
        for i in range(h, rows):
            b = np.int64(trans[base + i - h])
            if (i % h) == 0:
                pred = b
            else:
                a = np.int64(trans[base + i - 1])
                ul = np.int64(trans[base + i - h - 1])
                pred = _paeth(a, b, ul)
            trans[base + i] = np.uint8((np.int64(data[base + i]) + pred) & 0xFF)
    for i in range(body, n):
        trans[i] = data[i]
    return _t1_inverse(trans, w)


def apply_transform_forward(data: np.ndarray, spec: TransformSpec) -> np.ndarray:
    tid = spec.tid
    if tid == T0_IDENTITY:
        return data
    if tid == T1_TRANSPOSE:
        return _t1_forward(data, max(2, spec.w))
    if tid == T2_ELEM_DELTA:
        return _t2_elem_forward(data, spec.w)
    if tid == T2_XOR_TRANSPOSE:
        return _t2_xor_forward(data, max(2, spec.w))
    if tid == T2_BITSHUFFLE:
        return _t2_bitshuffle_forward(data, spec.w)
    if tid == T3_2D_PAETH:
        return _t3_2d_paeth_forward(data, max(1, spec.w), max(2, spec.h))
    raise ValueError(f"Unknown transform id={tid}")


def apply_transform_inverse(data: np.ndarray, spec: TransformSpec) -> np.ndarray:
    tid = spec.tid
    if tid == T0_IDENTITY:
        return data
    if tid == T1_TRANSPOSE:
        return _t1_inverse(data, max(2, spec.w))
    if tid == T2_ELEM_DELTA:
        return _t2_elem_inverse(data, spec.w)
    if tid == T2_XOR_TRANSPOSE:
        return _t2_xor_inverse(data, max(2, spec.w))
    if tid == T2_BITSHUFFLE:
        return _t2_bitshuffle_inverse(data, spec.w)
    if tid == T3_2D_PAETH:
        return _t3_2d_paeth_inverse(data, max(1, spec.w), max(2, spec.h))
    raise ValueError(f"Unknown transform id={tid}")


# ---------------------------------------------------------------------------
# Fast Per-Plane Order-0 Entropy Pre-Filter + zstd -1 Two-Tier Scorer
# ---------------------------------------------------------------------------

@njit(cache=True)
def _plane_h0_sum(data: np.ndarray, w: int) -> float:
    """Fast Miller-Madow bias-corrected per-plane order-0 Shannon entropy (bits/byte) for width w.

    Without Miller-Madow correction (+ (K_nonzero - 1) / (2 * rows * ln 2)), larger widths w
    have fewer rows per plane and artificially lower empirical H0, wrongly evicting true
    smaller strides (like w=4 or w=16).
    """
    n = data.shape[0]
    rows = n // w
    if rows < 8:
        return 8.0
    hist = np.empty(256, dtype=np.int32)
    inv_r = 1.0 / rows
    inv_ln2 = 1.4426950408889634
    total_h = 0.0
    for c in range(w):
        for b in range(256):
            hist[b] = 0
        idx = c
        for _ in range(rows):
            hist[data[idx]] += 1
            idx += w
        h = 0.0
        nonzero = 0
        for b in range(256):
            cnt = hist[b]
            if cnt > 0:
                nonzero += 1
                p = cnt * inv_r
                h -= p * (math.log(p) * inv_ln2)
        if nonzero > 1:
            h += ((nonzero - 1) * inv_ln2) / (2.0 * rows)
        total_h += h
    return total_h / w


@njit(cache=True)
def compute_histogram_256(data: np.ndarray) -> np.ndarray:
    """Normalized 256-bin byte histogram for Jensen-Shannon divergence."""
    n = data.shape[0]
    hist = np.zeros(256, dtype=np.float32)
    if n == 0:
        return hist
    for i in range(n):
        hist[data[i]] += 1.0
    inv_n = 1.0 / n
    for b in range(256):
        hist[b] *= inv_n
    return hist


@njit(cache=True)
def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence D_JS(p || q) in nats/bits (base-2, in [0, 1])."""
    d = 0.0
    inv_ln2 = 1.4426950408889634
    for i in range(256):
        pi = float(p[i])
        qi = float(q[i])
        m = 0.5 * (pi + qi)
        if pi > 1e-12:
            d += 0.5 * pi * math.log(pi / m) * inv_ln2
        if qi > 1e-12:
            d += 0.5 * qi * math.log(qi / m) * inv_ln2
    return d


@dataclass
class ProbeResult:
    best_spec: TransformSpec
    t0_bytes: int
    best_bytes: int
    scores: Dict[TransformSpec, int]
    bytes_scored_zstd1: int
    hist: np.ndarray


class TwoTierScorer:
    """Reusable zstd -1 scorer with H0 pre-filter, harmonic tie-breaking, and gain gate."""

    def __init__(
        self,
        gain_gate: float = 0.96,
        tier2_trigger: float = 0.98,
        harmonic_margin: float = 0.98,
        prefilter_top_k: int = 3,
    ):
        self.gain_gate = gain_gate
        self.tier2_trigger = tier2_trigger
        self.harmonic_margin = harmonic_margin
        self.prefilter_top_k = prefilter_top_k
        self._cctx = zstd.ZstdCompressor(level=1)

    def _zstd1_size(self, arr: np.ndarray) -> int:
        return len(self._cctx.compress(memoryview(arr)))

    def score_probe(
        self,
        probe_16k: np.ndarray,
        wide_probe_64k: np.ndarray | None = None,
    ) -> ProbeResult:
        n = probe_16k.shape[0]
        hist = compute_histogram_256(probe_16k)
        t0_spec = TransformSpec(T0_IDENTITY, 1, 0)
        t0_bytes = self._zstd1_size(probe_16k)
        scored_bytes = n
        scores: Dict[TransformSpec, int] = {t0_spec: t0_bytes}

        # Quick skip for clearly incompressible high-entropy probes (zstd-1 expands or barely shrinks)
        # Note: still check if wide lag wasn't missed; on pure random zstd-1 >= n
        h0_raw = _plane_h0_sum(probe_16k, 1)
        if h0_raw > 7.95 and t0_bytes >= int(n * 0.995):
            return ProbeResult(t0_spec, t0_bytes, t0_bytes, scores, scored_bytes, hist)

        # 2.1 Lag discovery
        l1, l2 = discover_lag_peaks(probe_16k, wide_probe_64k)
        cand_widths = list(FIXED_T1_WIDTHS)
        for lx in (l1, l2):
            if 2 <= lx <= 4096 and lx * 4 <= n and lx not in cand_widths:
                cand_widths.append(lx)

        # Pre-filter Tier 1 widths by sum of per-plane H0 to cap zstd-1 calls
        h0_ranked: List[Tuple[float, int]] = []
        for w in cand_widths:
            if w * 4 <= n:
                h0_ranked.append((_plane_h0_sum(probe_16k, w), w))
        h0_ranked.sort(key=lambda x: x[0])

        # Always keep fundamental word widths {4, 8} (so larger multiples 16/32/64 cannot
        # evict them via H0 refinement bias), plus top-2 remaining widths by H0, plus L1.
        eval_widths = [w for w in (4, 8) if w * 4 <= n]
        for _, w in h0_ranked:
            if w not in eval_widths:
                eval_widths.append(w)
                if len(eval_widths) >= 4:
                    break
        if l1 >= 2 and l1 * 4 <= n and l1 not in eval_widths and len(eval_widths) < 5:
            eval_widths.append(l1)

        # Tier 1 evaluation via zstd -1
        t1_results: List[Tuple[int, int]] = []  # (zstd1_bytes, w)
        for w in eval_widths:
            sp = TransformSpec(T1_TRANSPOSE, w, 0)
            sz = self._zstd1_size(_t1_forward(probe_16k, w))
            scored_bytes += n
            scores[sp] = sz
            t1_results.append((sz, w))

        # Harmonic tie-breaking among Tier 1 widths:
        # Prefer smaller width unless larger harmonic beats it by >= 2.0%
        t1_results.sort(key=lambda x: x[0])
        best_t1_sz = t0_bytes
        best_t1_w = 1
        if t1_results:
            # Start from smallest winning candidate and only upgrade to harmonic if >= 2% better
            by_w = sorted(t1_results, key=lambda x: x[1])
            best_t1_sz, best_t1_w = t1_results[0]
            for sz_small, w_small in by_w:
                if best_t1_w > w_small and best_t1_w % w_small == 0:
                    if best_t1_sz > sz_small * self.harmonic_margin:
                        best_t1_sz = sz_small
                        best_t1_w = w_small
                        break

        best_spec = t0_spec
        best_sz = t0_bytes
        if best_t1_w > 1 and best_t1_sz < best_sz:
            best_spec = TransformSpec(T1_TRANSPOSE, best_t1_w, 0)
            best_sz = best_t1_sz

        # Tier 2 (conditional: ONLY if best Tier 1 beats T0 by >= 2.0%)
        if best_t1_sz <= t0_bytes * self.tier2_trigger:
            top2_w = [best_t1_w]
            for _sz, w in t1_results:
                if w not in top2_w:
                    top2_w.append(w)
                    if len(top2_w) == 2:
                        break

            for w in top2_w:
                # T2_XOR(w)
                sp_xor = TransformSpec(T2_XOR_TRANSPOSE, w, 0)
                sz_xor = self._zstd1_size(_t2_xor_forward(probe_16k, w))
                scored_bytes += n
                scores[sp_xor] = sz_xor
                if sz_xor < best_sz:
                    best_sz = sz_xor
                    best_spec = sp_xor

                # T2_ELEM(w) for w in {2, 4, 8}
                if w in (2, 4, 8):
                    sp_el = TransformSpec(T2_ELEM_DELTA, w, 0)
                    sz_el = self._zstd1_size(_t2_elem_forward(probe_16k, w))
                    scored_bytes += n
                    scores[sp_el] = sz_el
                    if sz_el < best_sz:
                        best_sz = sz_el
                        best_spec = sp_el

                # T2_BIT(w) for w in {4, 8}
                if w in (4, 8) and n >= 8 * w * 4:
                    sp_bit = TransformSpec(T2_BITSHUFFLE, w, 0)
                    sz_bit = self._zstd1_size(_t2_bitshuffle_forward(probe_16k, w))
                    scored_bytes += n
                    scores[sp_bit] = sz_bit
                    if sz_bit < best_sz:
                        best_sz = sz_bit
                        best_spec = sp_bit

                # T3_2D(w, h) — only if secondary row stride h appears in transposed rows!
                # Pick highest-order non-constant byte lane (for float32 with constant MSB exponent,
                # lane w-1 is constant so inspect lane w-2 where top mantissa bits vary smoothly!).
                check_buf = wide_probe_64k if (wide_probe_64k is not None and wide_probe_64k.shape[0] >= 32768) else probe_16k
                if w <= 8:
                    rows_c = check_buf.shape[0] // w
                    mat = check_buf[: rows_c * w].reshape(rows_c, w)
                    chosen_plane = None
                    for lane in range(w - 1, -1, -1):
                        pl = mat[:, lane]
                        if pl.min() != pl.max():
                            chosen_plane = np.ascontiguousarray(pl)
                            break
                    h_peak = _discover_row_stride_in_plane(chosen_plane) if chosen_plane is not None else 0
                    if h_peak >= 32 and h_peak * w * 3 <= check_buf.shape[0]:
                        sp_2d = TransformSpec(T3_2D_PAETH, w, h_peak)
                        # Score on check_buf (normalize to 16 KB scale if wide_probe used)
                        t3_out = _t3_2d_paeth_forward(check_buf, w, h_peak)
                        if check_buf.shape[0] == n:
                            sz_2d = self._zstd1_size(t3_out)
                            scored_bytes += n
                        else:
                            # Score proportionally on check_buf and scale to n
                            sz_2d_wide = self._zstd1_size(t3_out)
                            sz_t0_wide = self._zstd1_size(check_buf)
                            scored_bytes += 2 * check_buf.shape[0]
                            sz_2d = int(t0_bytes * (sz_2d_wide / max(1, sz_t0_wide)))
                        scores[sp_2d] = sz_2d
                        if sz_2d < best_sz:
                            best_sz = sz_2d
                            best_spec = sp_2d

        # Harmonic tie-break final winner vs smaller-width same-family candidate
        if best_spec.tid != T0_IDENTITY and best_sz > t0_bytes * self.gain_gate:
            return ProbeResult(t0_spec, t0_bytes, t0_bytes, scores, scored_bytes, hist)

        return ProbeResult(best_spec, t0_bytes, best_sz, scores, scored_bytes, hist)
