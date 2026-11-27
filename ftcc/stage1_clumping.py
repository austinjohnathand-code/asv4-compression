"""Stage 1: Energy Clumping (Pre-Shaping / Partitioning).

Computes a fast multi-feature local energy profile across fixed-size windows:
  1. Normalized Shannon entropy H / 8
  2. Normalized mean absolute first-order byte delta
  3. Byte transition density (fraction of adjacent byte changes)
  4. Printable ASCII / whitespace ratio (separates text from binary regimes)
  5. Stride coherence (best periodic lag match across {4, 8, 16, 32, 64})

Detects sharp regime/energy shifts to place clump boundaries, then applies
practical guardrails (min_clump_size, max_clump_size, and adjacent profile merge)
to prevent descriptor bloat on 1 GB+ inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math
import numpy as np
from numba import njit


@dataclass
class ClumpingConfig:
    window_size: int = 4096
    min_clump_size: int = 262144       # 256 KB
    max_clump_size: int = 2097152      # 2 MB
    smooth_radius: int = 4             # 4 windows (16 KB) left/right smoothing
    split_threshold: float = 0.14      # Normalized L1 energy shift for split
    merge_threshold: float = 0.07      # Normalized L1 profile diff for neighbor merge


@dataclass
class ClumpSpec:
    clump_id: int
    start: int
    end: int
    profile: np.ndarray  # shape (5,) float32

    @property
    def length(self) -> int:
        return self.end - self.start


@njit(cache=True)
def _compute_window_features(data: np.ndarray, window_size: int) -> np.ndarray:
    n = data.shape[0]
    n_win = (n + window_size - 1) // window_size
    feats = np.empty((n_win, 5), dtype=np.float32)
    hist = np.empty(256, dtype=np.int32)
    lags = np.array([4, 8, 16, 32, 64], dtype=np.int32)

    for w in range(n_win):
        start = w * window_size
        end = start + window_size
        if end > n:
            end = n
        m = end - start
        if m <= 1:
            feats[w, 0] = 0.0
            feats[w, 1] = 0.0
            feats[w, 2] = 0.0
            feats[w, 3] = 0.0
            feats[w, 4] = 0.0
            continue

        for b in range(256):
            hist[b] = 0

        delta_sum = 0
        trans_cnt = 0
        ascii_cnt = 0

        prev = np.int64(data[start])
        hist[prev] += 1
        if (32 <= prev <= 126) or prev == 9 or prev == 10 or prev == 13:
            ascii_cnt += 1

        for i in range(start + 1, end):
            v = np.int64(data[i])
            hist[v] += 1
            d = v - prev
            if d < 0:
                d = -d
            delta_sum += d
            if v != prev:
                trans_cnt += 1
            if (32 <= v <= 126) or v == 9 or v == 10 or v == 13:
                ascii_cnt += 1
            prev = v

        # 1. Normalized Shannon entropy [0, 1]
        inv_m = 1.0 / m
        ent = 0.0
        for b in range(256):
            c = hist[b]
            if c > 0:
                p = c * inv_m
                ent -= p * (math.log(p) * 1.4426950408889634)
        feats[w, 0] = ent * 0.125

        # 2. Normalized mean abs byte delta [0, 1]
        mean_delta = delta_sum / ((m - 1) * 128.0)
        if mean_delta > 1.0:
            mean_delta = 1.0
        feats[w, 1] = mean_delta

        # 3. Transition density [0, 1]
        feats[w, 2] = trans_cnt / (m - 1)

        # 4. Printable ASCII ratio [0, 1]
        feats[w, 3] = ascii_cnt * inv_m

        # 5. Best stride coherence across {4, 8, 16, 32, 64} on up to 512 sampled pairs
        best_coh = 0.0
        probe_len = 512 if m > 576 else m - 64
        if probe_len > 16:
            for li in range(5):
                lag = lags[li]
                s_diff = 0
                for i in range(start, start + probe_len):
                    d = np.int64(data[i + lag]) - np.int64(data[i])
                    if d < 0:
                        d = -d
                    s_diff += d
                coh = 1.0 - (s_diff / (probe_len * 128.0))
                if coh > best_coh:
                    best_coh = coh
        if best_coh < 0.0:
            best_coh = 0.0
        feats[w, 4] = best_coh

    return feats


@njit(cache=True)
def _compute_energy_shifts(feats: np.ndarray, radius: int) -> np.ndarray:
    """Compute left-vs-right smoothed energy profile shift at every window boundary."""
    n_win = feats.shape[0]
    shifts = np.zeros(n_win, dtype=np.float32)
    if n_win < 2 * radius + 1:
        return shifts

    # Prefix sums for O(1) window-range means
    prefix = np.zeros((n_win + 1, 5), dtype=np.float64)
    for w in range(n_win):
        for f in range(5):
            prefix[w + 1, f] = prefix[w, f] + feats[w, f]

    inv_r = 1.0 / radius
    for w in range(radius, n_win - radius):
        l1 = 0.0
        for f in range(5):
            left_avg = (prefix[w, f] - prefix[w - radius, f]) * inv_r
            right_avg = (prefix[w + radius, f] - prefix[w, f]) * inv_r
            diff = left_avg - right_avg
            if diff < 0.0:
                diff = -diff
            l1 += diff
        shifts[w] = l1 * 0.2  # Normalize by 5 features -> [0, 1]
    return shifts


def partition_energy_clumps(
    data: np.ndarray,
    config: ClumpingConfig | None = None,
) -> List[ClumpSpec]:
    """Partition 1D uint8 array into energy-coherent clumps with min/max & merge guardrails."""
    if config is None:
        config = ClumpingConfig()

    n = int(data.shape[0])
    if n == 0:
        return []

    win = config.window_size
    if n <= config.min_clump_size:
        feats = _compute_window_features(data, min(win, max(64, n)))
        prof = feats.mean(axis=0).astype(np.float32)
        return [ClumpSpec(clump_id=0, start=0, end=n, profile=prof)]

    feats = _compute_window_features(data, win)
    n_win = feats.shape[0]
    shifts = _compute_energy_shifts(feats, config.smooth_radius)

    min_w = max(1, config.min_clump_size // win)
    max_w = max(min_w, config.max_clump_size // win)

    # Step 1: Place boundaries at sharp energy shifts (local maxima above threshold)
    # or at the best boundary inside the trailing quarter when hitting max_clump_size.
    boundaries_w = [0]
    cur_start = 0
    w = min_w
    while w < n_win:
        span = w - cur_start
        if span >= max_w:
            # Choose best natural split point in [cur_start + 3/4 max_w, w]
            search_lo = max(cur_start + min_w, w - max(1, max_w // 4))
            best_w = w
            best_s = -1.0
            for cand in range(search_lo, min(w + 1, n_win)):
                if shifts[cand] > best_s:
                    best_s = float(shifts[cand])
                    best_w = cand
            boundaries_w.append(best_w)
            cur_start = best_w
            w = cur_start + min_w
            continue

        s = float(shifts[w])
        if s >= config.split_threshold:
            # Check local peak within +/- 2 windows
            lo = max(cur_start + min_w, w - 2)
            hi = min(n_win - 1, w + 2)
            is_peak = True
            for k in range(lo, hi + 1):
                if shifts[k] > s:
                    is_peak = False
                    break
            if is_peak and (n_win - w) >= max(1, min_w // 2):
                boundaries_w.append(w)
                cur_start = w
                w = cur_start + min_w
                continue
        w += 1

    boundaries_w.append(n_win)

    # Build raw clumps with byte coordinates and mean energy profiles
    raw_clumps: List[list] = []
    for idx in range(len(boundaries_w) - 1):
        ws = boundaries_w[idx]
        we = boundaries_w[idx + 1]
        bs = ws * win
        be = min(n, we * win) if idx < len(boundaries_w) - 2 else n
        if be <= bs:
            continue
        prof = feats[ws:we].mean(axis=0).astype(np.float32)
        raw_clumps.append([bs, be, prof])

    # Step 2: Neighbor-merge pass (merge adjacent clumps with similar profiles up to max_clump_size)
    merged: List[list] = []
    for item in raw_clumps:
        if not merged:
            merged.append(item)
            continue
        prev = merged[-1]
        prev_len = prev[1] - prev[0]
        cur_len = item[1] - item[0]
        dist = float(np.abs(prev[2] - item[2]).mean())
        can_merge_size = (prev_len + cur_len) <= config.max_clump_size
        is_tiny_tail = cur_len < (config.min_clump_size // 2) and (prev_len + cur_len) <= int(config.max_clump_size * 1.5)
        if (dist <= config.merge_threshold and can_merge_size) or is_tiny_tail:
            total = prev_len + cur_len
            new_prof = ((prev[2] * prev_len + item[2] * cur_len) / total).astype(np.float32)
            prev[1] = item[1]
            prev[2] = new_prof
        else:
            merged.append(item)

    return [
        ClumpSpec(clump_id=i, start=c[0], end=c[1], profile=c[2])
        for i, c in enumerate(merged)
    ]
