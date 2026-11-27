"""Stage 2: Optimal N-Dimensional Shape Selection.

Evaluates candidate 1D, 2D (byte-plane transpose & stride-delta), and 3D cube/tensor
axis-permutation layouts on a sampled slice of each clump using a fast proxy metric
(Shannon entropy + delta entropy + run-break rate) rather than full compression passes.

Every layout transform is strictly bijective and lossless for arbitrary clump lengths
(preserving any trailing remainder bytes cleanly). Falls back to 1D identity when no
layout beats the baseline proxy by a safety margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
from numba import njit


LAYOUT_1D_IDENTITY = 0
LAYOUT_1D_DELTA = 1
LAYOUT_2D_TRANSPOSE = 2
LAYOUT_2D_TRANSPOSE_DELTA = 3
LAYOUT_3D_CUBE_PERMUTE = 4
LAYOUT_3D_CUBE_PERMUTE_DELTA = 5

LAYOUT_NAMES = {
    LAYOUT_1D_IDENTITY: "1D_IDENTITY",
    LAYOUT_1D_DELTA: "1D_DELTA",
    LAYOUT_2D_TRANSPOSE: "2D_TRANSPOSE",
    LAYOUT_2D_TRANSPOSE_DELTA: "2D_TRANSPOSE_DELTA",
    LAYOUT_3D_CUBE_PERMUTE: "3D_CUBE_PERM",
    LAYOUT_3D_CUBE_PERMUTE_DELTA: "3D_CUBE_PERM_DELTA",
}

# Candidate 2D strides (common record widths & word sizes)
DEFAULT_2D_STRIDES = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256)

# Candidate 3D cube geometries (W, H, perm_code) where tensor is (D, H, W):
#   perm_code 0: (W, D, H) — byte/channel plane first, row-contiguous
#   perm_code 1: (W, H, D) — byte/channel plane first, Z-slice contiguous (great for 3D grids!)
#   perm_code 2: (H, W, D) — Z-slice innermost
DEFAULT_3D_SHAPES = (
    # (W, H, perm_code)
    (2, 64, 1),
    (2, 128, 1),
    (2, 256, 1),
    (2, 1024, 1),
    (2, 4096, 1),
    (4, 64, 1),
    (4, 256, 1),
    (4, 1024, 1),
    (4, 4096, 1),
    (2, 64, 0),
    (4, 64, 0),
    (64, 64, 2),
    (128, 64, 2),
    (256, 64, 2),
)


@dataclass(frozen=True)
class LayoutDescriptor:
    layout_id: int = LAYOUT_1D_IDENTITY
    d0: int = 0   # 1D_DELTA: step; 2D: stride; 3D: W
    d1: int = 0   # 3D: H
    d2: int = 0   # 3D: perm_code

    def label(self) -> str:
        name = LAYOUT_NAMES.get(self.layout_id, f"L{self.layout_id}")
        if self.layout_id == LAYOUT_1D_IDENTITY:
            return name
        if self.layout_id in (LAYOUT_1D_DELTA, LAYOUT_2D_TRANSPOSE, LAYOUT_2D_TRANSPOSE_DELTA):
            return f"{name}(S={self.d0})"
        return f"{name}(W={self.d0},H={self.d1},P={self.d2})"


@njit(cache=True)
def fast_proxy_score(arr: np.ndarray) -> float:
    """Fast O(M) compression proxy in bits/byte (lower is better).

    Combines:
      - 0.50 * zero-order Shannon entropy H0(y)
      - 0.35 * first-order modular delta entropy H0(diff(y))
      - 0.15 * run-break rate scaled to [0, 8]
    """
    n = arr.shape[0]
    if n <= 1:
        return 0.0

    hist0 = np.zeros(256, dtype=np.int32)
    histd = np.zeros(256, dtype=np.int32)
    breaks = 0

    prev = np.int64(arr[0])
    hist0[prev] += 1
    for i in range(1, n):
        v = np.int64(arr[i])
        hist0[v] += 1
        d = (v - prev) & 0xFF
        histd[d] += 1
        if v != prev:
            breaks += 1
        prev = v

    inv_n = 1.0 / n
    inv_nm1 = 1.0 / (n - 1)
    h0 = 0.0
    hd = 0.0
    for b in range(256):
        c0 = hist0[b]
        if c0 > 0:
            p = c0 * inv_n
            h0 -= p * (math.log(p) * 1.4426950408889634)
        cd = histd[b]
        if cd > 0:
            p = cd * inv_nm1
            hd -= p * (math.log(p) * 1.4426950408889634)

    run_rate = 8.0 * (breaks * inv_nm1)
    return 0.50 * h0 + 0.35 * hd + 0.15 * run_rate


@njit(cache=True)
def _discover_dominant_strides(sample: np.ndarray, max_lag: int = 256) -> Tuple[int, int]:
    """Fast autocorrelation lag scan on first 4 KB to detect non-standard record strides."""
    m = min(sample.shape[0], 4096)
    if m < max_lag + 64:
        return 0, 0

    probe = m - max_lag
    if probe > 1024:
        probe = 1024

    # Baseline random-ish diff
    best1_lag = 0
    best1_val = 1e18
    best2_lag = 0
    best2_val = 1e18

    # Evaluate lags 3..max_lag (stride 1, 2 already in default set)
    for lag in range(3, max_lag + 1):
        s = 0
        for i in range(probe):
            d = np.int64(sample[i + lag]) - np.int64(sample[i])
            if d < 0:
                d = -d
            s += d
        val = float(s) / probe
        if val < best1_val:
            # Avoid picking harmonics too close if we already have a much better fundamental
            if best1_lag > 0 and (lag % best1_lag != 0):
                best2_val = best1_val
                best2_lag = best1_lag
            best1_val = val
            best1_lag = lag
        elif val < best2_val and (best1_lag == 0 or lag % best1_lag != 0):
            best2_val = val
            best2_lag = lag

    return best1_lag, best2_lag


@njit(cache=True)
def _delta_1d_forward(data: np.ndarray, step: int) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    if n == 0:
        return out
    lim = step if step < n else n
    for i in range(lim):
        out[i] = data[i]
    for i in range(lim, n):
        out[i] = (np.int64(data[i]) - np.int64(data[i - step])) & 0xFF
    return out


@njit(cache=True)
def _delta_1d_inverse(data: np.ndarray, step: int) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    if n == 0:
        return out
    lim = step if step < n else n
    for i in range(lim):
        out[i] = data[i]
    for i in range(lim, n):
        out[i] = (np.int64(data[i]) + np.int64(out[i - step])) & 0xFF
    return out


@njit(cache=True)
def _transpose_2d_forward(data: np.ndarray, stride: int, do_delta: bool) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    rows = n // stride
    body = rows * stride
    if rows == 0:
        for i in range(n):
            out[i] = data[i]
        return out

    if not do_delta:
        for c in range(stride):
            base = c * rows
            idx = c
            for r in range(rows):
                out[base + r] = data[idx]
                idx += stride
    else:
        for c in range(stride):
            base = c * rows
            idx = c
            prev = np.int64(data[idx])
            out[base] = np.uint8(prev)
            idx += stride
            for r in range(1, rows):
                v = np.int64(data[idx])
                out[base + r] = np.uint8((v - prev) & 0xFF)
                prev = v
                idx += stride

    for i in range(body, n):
        out[i] = data[i]
    return out


@njit(cache=True)
def _transpose_2d_inverse(data: np.ndarray, stride: int, do_delta: bool) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint8)
    rows = n // stride
    body = rows * stride
    if rows == 0:
        for i in range(n):
            out[i] = data[i]
        return out

    if not do_delta:
        for c in range(stride):
            base = c * rows
            idx = c
            for r in range(rows):
                out[idx] = data[base + r]
                idx += stride
    else:
        for c in range(stride):
            base = c * rows
            idx = c
            acc = np.int64(data[base])
            out[idx] = np.uint8(acc)
            idx += stride
            for r in range(1, rows):
                acc = (acc + np.int64(data[base + r])) & 0xFF
                out[idx] = np.uint8(acc)
                idx += stride

    for i in range(body, n):
        out[i] = data[i]
    return out


@njit(cache=True)
def _inner_delta_forward_inplace(buf: np.ndarray, n_slices: int, slice_len: int) -> None:
    """Apply modular delta along innermost dimension of shape (n_slices, slice_len)."""
    if slice_len <= 1:
        return
    for s in range(n_slices):
        base = s * slice_len
        prev = np.int64(buf[base])
        for k in range(1, slice_len):
            v = np.int64(buf[base + k])
            buf[base + k] = np.uint8((v - prev) & 0xFF)
            prev = v


@njit(cache=True)
def _inner_delta_inverse_inplace(buf: np.ndarray, n_slices: int, slice_len: int) -> None:
    if slice_len <= 1:
        return
    for s in range(n_slices):
        base = s * slice_len
        acc = np.int64(buf[base])
        for k in range(1, slice_len):
            acc = (acc + np.int64(buf[base + k])) & 0xFF
            buf[base + k] = np.uint8(acc)


def _cube_3d_forward(data: np.ndarray, w: int, h: int, perm_code: int, do_delta: bool) -> np.ndarray:
    n = data.shape[0]
    plane = w * h
    d = n // plane
    if d < 2:
        return data.copy()
    body = d * plane
    cube = data[:body].reshape((d, h, w))
    if perm_code == 0:
        perm = np.ascontiguousarray(cube.transpose((2, 0, 1)))  # (W, D, H)
        n_slices, slice_len = w * d, h
    elif perm_code == 1:
        perm = np.ascontiguousarray(cube.transpose((2, 1, 0)))  # (W, H, D)
        n_slices, slice_len = w * h, d
    else:
        perm = np.ascontiguousarray(cube.transpose((1, 2, 0)))  # (H, W, D)
        n_slices, slice_len = h * w, d

    flat = perm.reshape(-1)
    if do_delta:
        _inner_delta_forward_inplace(flat, n_slices, slice_len)

    if body == n:
        return flat
    out = np.empty(n, dtype=np.uint8)
    out[:body] = flat
    out[body:] = data[body:]
    return out


def _cube_3d_inverse(data: np.ndarray, w: int, h: int, perm_code: int, do_delta: bool) -> np.ndarray:
    n = data.shape[0]
    plane = w * h
    d = n // plane
    if d < 2:
        return data.copy()
    body = d * plane
    flat = data[:body].copy()

    if perm_code == 0:
        n_slices, slice_len = w * d, h
        if do_delta:
            _inner_delta_inverse_inplace(flat, n_slices, slice_len)
        cube = flat.reshape((w, d, h)).transpose((1, 2, 0))
    elif perm_code == 1:
        n_slices, slice_len = w * h, d
        if do_delta:
            _inner_delta_inverse_inplace(flat, n_slices, slice_len)
        cube = flat.reshape((w, h, d)).transpose((2, 1, 0))
    else:
        n_slices, slice_len = h * w, d
        if do_delta:
            _inner_delta_inverse_inplace(flat, n_slices, slice_len)
        cube = flat.reshape((h, w, d)).transpose((2, 0, 1))

    if body == n:
        return np.ascontiguousarray(cube).reshape(-1)
    out = np.empty(n, dtype=np.uint8)
    out[:body] = np.ascontiguousarray(cube).reshape(-1)
    out[body:] = data[body:]
    return out


def apply_layout_forward(data: np.ndarray, desc: LayoutDescriptor) -> np.ndarray:
    """Apply lossless reversible N-D layout transform."""
    lid = desc.layout_id
    if lid == LAYOUT_1D_IDENTITY:
        return data
    if lid == LAYOUT_1D_DELTA:
        return _delta_1d_forward(data, max(1, desc.d0))
    if lid == LAYOUT_2D_TRANSPOSE:
        return _transpose_2d_forward(data, max(2, desc.d0), False)
    if lid == LAYOUT_2D_TRANSPOSE_DELTA:
        return _transpose_2d_forward(data, max(2, desc.d0), True)
    if lid == LAYOUT_3D_CUBE_PERMUTE:
        return _cube_3d_forward(data, max(1, desc.d0), max(1, desc.d1), desc.d2, False)
    if lid == LAYOUT_3D_CUBE_PERMUTE_DELTA:
        return _cube_3d_forward(data, max(1, desc.d0), max(1, desc.d1), desc.d2, True)
    raise ValueError(f"Unknown layout_id={lid}")


def apply_layout_inverse(data: np.ndarray, desc: LayoutDescriptor) -> np.ndarray:
    """Exact inverse of apply_layout_forward."""
    lid = desc.layout_id
    if lid == LAYOUT_1D_IDENTITY:
        return data
    if lid == LAYOUT_1D_DELTA:
        return _delta_1d_inverse(data, max(1, desc.d0))
    if lid == LAYOUT_2D_TRANSPOSE:
        return _transpose_2d_inverse(data, max(2, desc.d0), False)
    if lid == LAYOUT_2D_TRANSPOSE_DELTA:
        return _transpose_2d_inverse(data, max(2, desc.d0), True)
    if lid == LAYOUT_3D_CUBE_PERMUTE:
        return _cube_3d_inverse(data, max(1, desc.d0), max(1, desc.d1), desc.d2, False)
    if lid == LAYOUT_3D_CUBE_PERMUTE_DELTA:
        return _cube_3d_inverse(data, max(1, desc.d0), max(1, desc.d1), desc.d2, True)
    raise ValueError(f"Unknown layout_id={lid}")


def select_optimal_layout(
    clump: np.ndarray,
    profile: np.ndarray | None = None,
    sample_bytes: int = 65536,
    improvement_margin: float = 0.94,
) -> Tuple[LayoutDescriptor, float, float]:
    """Select best N-D layout on a sampled slice using fast proxy scoring.

    Returns (best_descriptor, baseline_proxy_score, best_proxy_score).
    Automatically falls back to 1D_IDENTITY when no candidate beats baseline by
    the required margin or when the clump is high-entropy unstructured noise.
    """
    n = int(clump.shape[0])
    if n < 512:
        return LayoutDescriptor(LAYOUT_1D_IDENTITY), 0.0, 0.0

    # Fast early-out for high-entropy unstructured/random clumps
    if profile is not None:
        entropy_norm = float(profile[0])
        stride_coh = float(profile[4])
        if entropy_norm > 0.965 and stride_coh < 0.12:
            return LayoutDescriptor(LAYOUT_1D_IDENTITY), 8.0, 8.0

    # Take contiguous sample (at start of clump to preserve record/plane alignment!)
    m = min(n, sample_bytes)
    sample = clump[:m]

    base_score = float(fast_proxy_score(sample))
    best_desc = LayoutDescriptor(LAYOUT_1D_IDENTITY)
    best_score = base_score

    # Build candidate list
    candidates: List[LayoutDescriptor] = [
        LayoutDescriptor(LAYOUT_1D_DELTA, d0=1),
        LayoutDescriptor(LAYOUT_1D_DELTA, d0=2),
        LayoutDescriptor(LAYOUT_1D_DELTA, d0=4),
    ]

    # Discover custom strides via autocorrelation and combine with defaults
    s1, s2 = _discover_dominant_strides(sample, 256)
    stride_set = set(DEFAULT_2D_STRIDES)
    if s1 >= 3:
        stride_set.add(int(s1))
    if s2 >= 3:
        stride_set.add(int(s2))

    for s in sorted(stride_set):
        if s * 4 <= m:
            candidates.append(LayoutDescriptor(LAYOUT_2D_TRANSPOSE, d0=s))
            candidates.append(LayoutDescriptor(LAYOUT_2D_TRANSPOSE_DELTA, d0=s))

    # 3D cube/tensor candidates
    for w, h, p in DEFAULT_3D_SHAPES:
        if w * h * 2 <= m:
            candidates.append(LayoutDescriptor(LAYOUT_3D_CUBE_PERMUTE, d0=w, d1=h, d2=p))
            candidates.append(LayoutDescriptor(LAYOUT_3D_CUBE_PERMUTE_DELTA, d0=w, d1=h, d2=p))

    # Evaluate candidates with fast proxy metric
    for cand in candidates:
        transformed = apply_layout_forward(sample, cand)
        sc = float(fast_proxy_score(transformed))
        if sc < best_score:
            best_score = sc
            best_desc = cand

    # Require meaningful improvement margin over 1D identity baseline
    if best_desc.layout_id != LAYOUT_1D_IDENTITY and best_score > base_score * improvement_margin:
        return LayoutDescriptor(LAYOUT_1D_IDENTITY), base_score, base_score

    return best_desc, base_score, best_score
