"""Stage 3: Forest of Trees Across N-D Clumps & Coarse-to-Fine Refinement.

Builds three specialized trees across clumps with bounded depth and strict
marginal-gain pruning:
  - Tree 1 (Layout/Stride Hierarchy Template Tree):
      Groups clumps with compatible N-D shapes, builds a compact shared macro
      prototype (byte-wise modal template tiled across the group), and stores
      child deltas when total savings exceed template overhead.
  - Tree 2 (Similarity / Motif Cross-Clump Delta Tree):
      Indexes clumps via fast structural fingerprints (coarse sketch + sampled
      shingle hashes + histogram), links similar clumps across arbitrary
      distances (even 10s-100s MB apart) into a causal DAG/tree (parent < child,
      depth <= max_depth), and stores aligned SUB_MOD256 or XOR deltas.
  - Tree 3 (Coarse-to-Fine Multi-Level Refinement Tree):
      Decomposes shaped clumps into bounded macro-to-micro levels (Level 0:
      block-256 linear trend; Level 1: block-32 refinement) and prunes any
      refinement level whose marginal residual reduction fails to beat coarse
      band overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from numba import njit

from .stage1_clumping import ClumpSpec
from .stage2_layout import (
    LAYOUT_1D_IDENTITY,
    LAYOUT_2D_TRANSPOSE,
    LAYOUT_2D_TRANSPOSE_DELTA,
    LAYOUT_3D_CUBE_PERMUTE,
    LAYOUT_3D_CUBE_PERMUTE_DELTA,
    LayoutDescriptor,
    apply_layout_forward,
    fast_proxy_score,
)


# Clump encoding modes
MODE_IDENTITY = 0
MODE_SHAPED_ONLY = 1
MODE_TREE_TEMPLATE_DELTA = 2
MODE_TREE_SIMILARITY_DELTA = 3
MODE_TREE_COARSE_FINE = 4

MODE_NAMES = {
    MODE_IDENTITY: "IDENTITY",
    MODE_SHAPED_ONLY: "SHAPED_ONLY",
    MODE_TREE_TEMPLATE_DELTA: "TREE1_TEMPLATE",
    MODE_TREE_SIMILARITY_DELTA: "TREE2_SIMILARITY",
    MODE_TREE_COARSE_FINE: "TREE3_COARSE_FINE",
}

# Delta operations
OP_NONE = 0
OP_SUB_MOD256 = 1
OP_XOR = 2


@dataclass
class ForestConfig:
    max_depth: int = 3
    enable_tree1_template: bool = True
    enable_tree2_similarity: bool = True
    enable_tree3_coarse_fine: bool = True
    tree1_max_template_bytes: int = 4096
    tree2_min_improvement: float = 0.88   # Require >= 12% proxy improvement
    tree3_min_improvement: float = 0.94   # Require >= 6% net bits/byte improvement
    tree3_level1_improvement: float = 0.96


@dataclass
class ClumpPlan:
    clump_id: int
    start: int
    end: int
    mode: int = MODE_IDENTITY
    layout: LayoutDescriptor = field(default_factory=LayoutDescriptor)
    tree_parent: int = -1      # template_id for Tree 1; parent clump_id for Tree 2; -1 otherwise
    tree_op: int = OP_NONE     # SUB_MOD256 or XOR for Tree 1/2; levels (1 or 2) for Tree 3
    tree_depth: int = 0
    align_shift: int = 0       # Tree 2 byte alignment shift (child_pos - parent_pos)
    coarse_band: bytes = b""   # Tree 3 serialized coarse levels (Level 0 [+ Level 1])
    payload: np.ndarray | None = None  # Transformed residual bytes (uint8 1D)


@dataclass
class ForestResult:
    plans: List[ClumpPlan]
    templates: List[bytes]
    tree1_edges: int = 0
    tree2_edges: int = 0
    tree2_max_depth: int = 0
    tree3_level0_count: int = 0
    tree3_level1_count: int = 0


# ---------------------------------------------------------------------------
# Fast Numba Kernels for Delta, Tiling, Fingerprinting, and Coarse-to-Fine
# ---------------------------------------------------------------------------

@njit(cache=True)
def apply_op_forward_pair(
    child: np.ndarray,
    parent: np.ndarray,
    op: int,
    shift: int = 0,
) -> np.ndarray:
    """Compute child[k] (-) parent[k - shift] where 0 <= k - shift < len(parent)."""
    nc = child.shape[0]
    np_ = parent.shape[0]
    out = np.empty(nc, dtype=np.uint8)
    lo = shift if shift > 0 else 0
    hi = np_ + shift
    if hi > nc:
        hi = nc
    if lo > nc:
        lo = nc
    if hi < lo:
        hi = lo

    for i in range(0, lo):
        out[i] = child[i]
    if op == OP_SUB_MOD256:
        for i in range(lo, hi):
            out[i] = np.uint8((np.int64(child[i]) - np.int64(parent[i - shift])) & 0xFF)
    elif op == OP_XOR:
        for i in range(lo, hi):
            out[i] = child[i] ^ parent[i - shift]
    else:
        for i in range(lo, hi):
            out[i] = child[i]
    for i in range(hi, nc):
        out[i] = child[i]
    return out


@njit(cache=True)
def apply_op_inverse_pair(
    residual: np.ndarray,
    parent: np.ndarray,
    op: int,
    shift: int = 0,
) -> np.ndarray:
    nc = residual.shape[0]
    np_ = parent.shape[0]
    out = np.empty(nc, dtype=np.uint8)
    lo = shift if shift > 0 else 0
    hi = np_ + shift
    if hi > nc:
        hi = nc
    if lo > nc:
        lo = nc
    if hi < lo:
        hi = lo

    for i in range(0, lo):
        out[i] = residual[i]
    if op == OP_SUB_MOD256:
        for i in range(lo, hi):
            out[i] = np.uint8((np.int64(residual[i]) + np.int64(parent[i - shift])) & 0xFF)
    elif op == OP_XOR:
        for i in range(lo, hi):
            out[i] = residual[i] ^ parent[i - shift]
    else:
        for i in range(lo, hi):
            out[i] = residual[i]
    for i in range(hi, nc):
        out[i] = residual[i]
    return out


@njit(cache=True)
def apply_template_forward(data: np.ndarray, template: np.ndarray, op: int) -> np.ndarray:
    n = data.shape[0]
    tlen = template.shape[0]
    out = np.empty(n, dtype=np.uint8)
    if tlen == 0:
        for i in range(n):
            out[i] = data[i]
        return out
    if op == OP_SUB_MOD256:
        idx = 0
        for i in range(n):
            out[i] = np.uint8((np.int64(data[i]) - np.int64(template[idx])) & 0xFF)
            idx += 1
            if idx == tlen:
                idx = 0
    else:
        idx = 0
        for i in range(n):
            out[i] = data[i] ^ template[idx]
            idx += 1
            if idx == tlen:
                idx = 0
    return out


@njit(cache=True)
def apply_template_inverse(residual: np.ndarray, template: np.ndarray, op: int) -> np.ndarray:
    n = residual.shape[0]
    tlen = template.shape[0]
    out = np.empty(n, dtype=np.uint8)
    if tlen == 0:
        for i in range(n):
            out[i] = residual[i]
        return out
    if op == OP_SUB_MOD256:
        idx = 0
        for i in range(n):
            out[i] = np.uint8((np.int64(residual[i]) + np.int64(template[idx])) & 0xFF)
            idx += 1
            if idx == tlen:
                idx = 0
    else:
        idx = 0
        for i in range(n):
            out[i] = residual[i] ^ template[idx]
            idx += 1
            if idx == tlen:
                idx = 0
    return out


@njit(cache=True)
def _compute_fingerprint(clump: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fast shift-invariant structural fingerprint:
    1. sketch: 32-element mean byte profile across clump
    2. hist16: 16-bin coarse normalized byte histogram on first 16 KB
    3. content-defined anchor shingles (FNV-1a 8-byte hashes selected purely by content,
       so arbitrary byte shifts between recurring regions still match exact anchors!)
    """
    n = clump.shape[0]
    sketch = np.zeros(32, dtype=np.float32)
    hist16 = np.zeros(16, dtype=np.float32)
    max_sh = 160
    shingles = np.zeros(max_sh, dtype=np.uint64)
    sh_pos = np.zeros(max_sh, dtype=np.int64)
    n_found = 0

    if n < 64:
        return sketch, hist16, shingles, sh_pos, 0

    # 1. 32-point coarse sketch (sample 64 bytes around each anchor)
    step = n // 32
    for s in range(32):
        base = s * step
        lim = min(base + 64, n)
        acc = 0
        for i in range(base, lim):
            acc += np.int64(clump[i])
        sketch[s] = acc / ((lim - base) * 255.0)

    # 2. 16-bin histogram on first 16 KB
    hlim = min(n, 16384)
    for i in range(hlim):
        hist16[clump[i] >> 4] += 1.0
    inv_h = 1.0 / hlim
    for b in range(16):
        hist16[b] *= inv_h

    # 3. Content-defined 8-byte anchors sampled every 64 bytes (select where (h & 0x7F) == 0).
    # For a 1 MB clump: 16,384 probes -> ~128 content-defined anchors!
    # Also ensure non-trivial 8-byte blocks (not all identical bytes like zeros/spaces).
    span = n - 8
    probe_step = 64
    for pos in range(0, span, probe_step):
        b0 = clump[pos]
        # Skip constant 8-byte runs (e.g. 0x00 or 0x20 padding) to prevent false collisions
        if (
            clump[pos + 1] == b0
            and clump[pos + 2] == b0
            and clump[pos + 3] == b0
            and clump[pos + 4] == b0
        ):
            continue
        h = np.uint64(14695981039346656037)
        for j in range(8):
            h = (h ^ np.uint64(clump[pos + j])) * np.uint64(1099511628211)
        if (h & np.uint64(0x7F)) == np.uint64(0):
            shingles[n_found] = h
            sh_pos[n_found] = pos
            n_found += 1
            if n_found >= max_sh:
                break

    return sketch, hist16, shingles, sh_pos, n_found


@njit(cache=True)
def _coarse_band_extract(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Compute per-block coarse mean (uint8) for block_size."""
    n = arr.shape[0]
    m = (n + block_size - 1) // block_size
    coarse = np.empty(m, dtype=np.uint8)
    for b in range(m):
        s = b * block_size
        e = s + block_size
        if e > n:
            e = n
        acc = 0
        for i in range(s, e):
            acc += np.int64(arr[i])
        coarse[b] = np.uint8(acc // (e - s))
    return coarse


@njit(cache=True)
def _coarse_predict_linear_residual(arr: np.ndarray, coarse: np.ndarray, block_size: int) -> np.ndarray:
    """Deterministic integer linear interpolation between coarse block values -> mod-256 residual."""
    n = arr.shape[0]
    m = coarse.shape[0]
    res = np.empty(n, dtype=np.uint8)
    for b in range(m):
        s = b * block_size
        e = s + block_size
        if e > n:
            e = n
        c0 = np.int64(coarse[b])
        c1 = np.int64(coarse[b + 1]) if (b + 1) < m else c0
        span = e - s
        for off in range(span):
            pred = (c0 * (span - off) + c1 * off) // span
            res[s + off] = np.uint8((np.int64(arr[s + off]) - pred) & 0xFF)
    return res


@njit(cache=True)
def _coarse_reconstruct_linear(res: np.ndarray, coarse: np.ndarray, block_size: int) -> np.ndarray:
    """Exact bit-for-bit inverse of _coarse_predict_linear_residual."""
    n = res.shape[0]
    m = coarse.shape[0]
    out = np.empty(n, dtype=np.uint8)
    for b in range(m):
        s = b * block_size
        e = s + block_size
        if e > n:
            e = n
        c0 = np.int64(coarse[b])
        c1 = np.int64(coarse[b + 1]) if (b + 1) < m else c0
        span = e - s
        for off in range(span):
            pred = (c0 * (span - off) + c1 * off) // span
            out[s + off] = np.uint8((np.int64(res[s + off]) + pred) & 0xFF)
    return out


def coarse_fine_encode(
    shaped: np.ndarray,
    config: ForestConfig,
) -> Tuple[bool, int, bytes, np.ndarray]:
    """Try 2-level coarse-to-fine decomposition on a shaped clump.

    Level 0: block_size = 256 (macro structure)
    Level 1: block_size = 32  (mid-level refinement on Level-0 residual)

    Stops refining when marginal bit-rate reduction no longer beats coarse band overhead.
    Returns (accepted, num_levels, coarse_bytes, fine_residual).
    """
    n = shaped.shape[0]
    if n < 2048:
        return False, 0, b"", shaped

    base_bpb = float(fast_proxy_score(shaped))
    if base_bpb < 0.35:
        # Already near-zero entropy; coarse overhead won't pay off
        return False, 0, b"", shaped

    # Level 0: B0 = 256
    b0 = 256
    c0 = _coarse_band_extract(shaped, b0)
    r0 = _coarse_predict_linear_residual(shaped, c0, b0)

    c0_bpb = float(fast_proxy_score(c0))
    r0_bpb = float(fast_proxy_score(r0))
    est_base_bits = base_bpb * n
    # Include small framing overhead (16 bytes = 128 bits)
    est_l0_bits = r0_bpb * n + c0_bpb * c0.shape[0] + 128.0

    if est_l0_bits >= est_base_bits * config.tree3_min_improvement:
        return False, 0, b"", shaped

    # Level 1 refinement: B1 = 32 on r0
    b1 = 32
    # Center r0 around 128 before averaging so small signed residuals (-2..+2 -> 254..2)
    # don't average to ~128 across the 0/255 wrap boundary!
    r0_shifted = ((r0.astype(np.uint16) + 128) & 0xFF).astype(np.uint8)
    c1 = _coarse_band_extract(r0_shifted, b1)
    r1 = _coarse_predict_linear_residual(r0_shifted, c1, b1)

    c1_bpb = float(fast_proxy_score(c1))
    r1_bpb = float(fast_proxy_score(r1))
    est_l1_bits = r1_bpb * n + c0_bpb * c0.shape[0] + c1_bpb * c1.shape[0] + 192.0

    if est_l1_bits < est_l0_bits * config.tree3_level1_improvement:
        # Keep both Level 0 and Level 1
        coarse_bytes = c0.tobytes() + c1.tobytes()
        return True, 2, coarse_bytes, r1

    # Prune Level 1; keep Level 0 only
    return True, 1, c0.tobytes(), r0


def coarse_fine_decode(
    fine_res: np.ndarray,
    coarse_bytes: bytes,
    num_levels: int,
) -> np.ndarray:
    """Exact bit-for-bit inverse of coarse_fine_encode."""
    n = fine_res.shape[0]
    b0 = 256
    m0 = (n + b0 - 1) // b0
    coarse_arr = np.frombuffer(coarse_bytes, dtype=np.uint8)
    c0 = coarse_arr[:m0]

    if num_levels == 2:
        b1 = 32
        m1 = (n + b1 - 1) // b1
        c1 = coarse_arr[m0:m0 + m1]
        r0_shifted = _coarse_reconstruct_linear(fine_res, c1, b1)
        r0 = ((r0_shifted.astype(np.uint16) - 128) & 0xFF).astype(np.uint8)
        return _coarse_reconstruct_linear(r0, c0, b0)

    return _coarse_reconstruct_linear(fine_res, c0, b0)


# ---------------------------------------------------------------------------
# Stage 3 Orchestrator: Build Forest of 3 Trees with Pruning
# ---------------------------------------------------------------------------

def _build_group_template(
    raw_data: np.ndarray,
    clump_specs: List[ClumpSpec],
    group_ids: List[int],
    tile_len: int,
) -> np.ndarray:
    """Build a compact byte-wise median prototype of length tile_len across group clumps."""
    samples = []
    max_tiles = 64
    tiles_per_clump = max(1, max_tiles // max(1, len(group_ids)))
    for cid in group_ids:
        sp = clump_specs[cid]
        c_arr = raw_data[sp.start:sp.end]
        n_tiles = c_arr.shape[0] // tile_len
        if n_tiles == 0:
            continue
        take = min(n_tiles, tiles_per_clump)
        step = max(1, n_tiles // take)
        for t in range(0, n_tiles, step):
            if len(samples) >= max_tiles:
                break
            samples.append(c_arr[t * tile_len:(t + 1) * tile_len])
        if len(samples) >= max_tiles:
            break

    if len(samples) < 2:
        return np.empty(0, dtype=np.uint8)

    mat = np.stack(samples, axis=0)  # (K, tile_len)
    # Fast byte-wise median (uint8) captures the shared background/template across tiles
    return np.median(mat, axis=0).astype(np.uint8)


def build_forest_and_refine(
    raw_data: np.ndarray,
    clumps: List[ClumpSpec],
    layouts: List[LayoutDescriptor],
    config: ForestConfig | None = None,
) -> ForestResult:
    """Build Forest of 3 specialized trees across N-D clumps and produce refined plans."""
    if config is None:
        config = ForestConfig()

    n_clumps = len(clumps)
    plans: List[ClumpPlan] = []
    templates: List[bytes] = []

    # Step 0: Initialize baseline plans from Stage 2 (SHAPED_ONLY or IDENTITY)
    standalone_scores = np.zeros(n_clumps, dtype=np.float64)
    for i, sp in enumerate(clumps):
        raw_c = raw_data[sp.start:sp.end]
        lay = layouts[i]
        if lay.layout_id == LAYOUT_1D_IDENTITY:
            plan = ClumpPlan(
                clump_id=i,
                start=sp.start,
                end=sp.end,
                mode=MODE_IDENTITY,
                layout=lay,
                payload=raw_c,
            )
            standalone_scores[i] = float(fast_proxy_score(raw_c[:min(len(raw_c), 65536)]))
        else:
            shaped = apply_layout_forward(raw_c, lay)
            plan = ClumpPlan(
                clump_id=i,
                start=sp.start,
                end=sp.end,
                mode=MODE_SHAPED_ONLY,
                layout=lay,
                payload=shaped,
            )
            standalone_scores[i] = float(fast_proxy_score(shaped[:min(len(shaped), 65536)]))
        plans.append(plan)

    # -----------------------------------------------------------------------
    # Tree 2: Similarity / Motif Cross-Clump Delta Tree (Causal DAG, depth<=3)
    # Shift-invariant via 4KB-grid shingle offset voting!
    # -----------------------------------------------------------------------
    tree2_edges = 0
    tree2_max_depth = 0
    if config.enable_tree2_similarity and n_clumps >= 2:
        sketches = np.zeros((n_clumps, 32), dtype=np.float32)
        hists = np.zeros((n_clumps, 16), dtype=np.float32)
        shingle_index: Dict[int, List[Tuple[int, int]]] = {}
        depths = [0] * n_clumps

        for i, sp in enumerate(clumps):
            raw_c = raw_data[sp.start:sp.end]
            sk, hi, sh, sh_pos, n_sh = _compute_fingerprint(raw_c)
            sketches[i] = sk
            hists[i] = hi

            # Vote on (parent_id, align_shift) pairs via content-defined anchors
            pair_votes: Dict[Tuple[int, int], int] = {}
            for k_idx in range(n_sh):
                hv = int(sh[k_idx])
                c_pos = int(sh_pos[k_idx])
                prev_list = shingle_index.get(hv)
                if prev_list is not None:
                    for pid, p_pos in prev_list:
                        if depths[pid] < config.max_depth:
                            key = (pid, c_pos - p_pos)
                            pair_votes[key] = pair_votes.get(key, 0) + 1

            # Top 3 (parent_id, shift) candidates by shingle hits
            candidates: List[Tuple[int, int]] = []
            seen_pids = set()
            if pair_votes:
                top_pairs = sorted(pair_votes.items(), key=lambda kv: kv[1], reverse=True)
                for (pid, shift), hits in top_pairs:
                    if hits >= 2 and pid not in seen_pids:
                        candidates.append((pid, shift))
                        seen_pids.add(pid)
                        if len(candidates) >= 3:
                            break

            # Also add closest earlier clump by (sketch + hist) at shift=0 if not listed
            if i > 0 and len(candidates) < 3:
                best_p = -1
                best_d = 0.22
                for p in range(max(0, i - 256), i):
                    if depths[p] >= config.max_depth or p in seen_pids:
                        continue
                    len_p = clumps[p].length
                    len_c = sp.length
                    if len_p * 2 < len_c or len_c * 2 < len_p:
                        continue
                    d = float(np.abs(sketches[p] - sk).mean() + np.abs(hists[p] - hi).mean()) * 0.5
                    if d < best_d:
                        best_d = d
                        best_p = p
                if best_p >= 0:
                    candidates.append((best_p, 0))

            # Evaluate candidates on aligned overlap sample (up to 64 KB)
            best_score = standalone_scores[i] * config.tree2_min_improvement
            best_choice: Tuple[int, int, int, LayoutDescriptor] | None = None

            for pid, shift in candidates:
                p_sp = clumps[pid]
                p_raw = raw_data[p_sp.start:p_sp.end]
                # Require substantial overlap (>= 35% of child length)
                ov_lo = max(0, shift)
                ov_hi = min(sp.length, p_sp.length + shift)
                overlap = ov_hi - ov_lo
                if overlap < sp.length // 3:
                    continue

                s_len = min(overlap, 65536)
                c_samp = raw_c[ov_lo:ov_lo + s_len]
                p_samp = p_raw[ov_lo - shift:ov_lo - shift + s_len]

                for op in (OP_SUB_MOD256, OP_XOR):
                    diff_s = apply_op_forward_pair(c_samp, p_samp, op, 0)
                    sc_raw = float(fast_proxy_score(diff_s))
                    # Adjust score for any uncovered edge bytes (which remain at standalone_scores[i])
                    cov = overlap / sp.length
                    eff_raw = cov * sc_raw + (1.0 - cov) * standalone_scores[i]
                    if eff_raw < best_score:
                        best_score = eff_raw
                        best_choice = (pid, shift, op, LayoutDescriptor(LAYOUT_1D_IDENTITY))
                    if layouts[i].layout_id != LAYOUT_1D_IDENTITY:
                        diff_shaped = apply_layout_forward(diff_s, layouts[i])
                        sc_sh = float(fast_proxy_score(diff_shaped))
                        eff_sh = cov * sc_sh + (1.0 - cov) * standalone_scores[i]
                        if eff_sh < best_score:
                            best_score = eff_sh
                            best_choice = (pid, shift, op, layouts[i])

            if best_choice is not None:
                pid, shift, op, chosen_lay = best_choice
                p_sp = clumps[pid]
                p_raw = raw_data[p_sp.start:p_sp.end]
                diff_full = apply_op_forward_pair(raw_c, p_raw, op, shift)
                if chosen_lay.layout_id != LAYOUT_1D_IDENTITY:
                    payload = apply_layout_forward(diff_full, chosen_lay)
                else:
                    payload = diff_full
                depths[i] = depths[pid] + 1
                plans[i].mode = MODE_TREE_SIMILARITY_DELTA
                plans[i].layout = chosen_lay
                plans[i].tree_parent = pid
                plans[i].tree_op = op
                plans[i].tree_depth = depths[i]
                plans[i].align_shift = shift
                plans[i].payload = payload
                standalone_scores[i] = best_score
                tree2_edges += 1
                if depths[i] > tree2_max_depth:
                    tree2_max_depth = depths[i]

            # Register content-defined anchors of clump i into index
            for k_idx in range(n_sh):
                hv = int(sh[k_idx])
                c_pos = int(sh_pos[k_idx])
                lst = shingle_index.get(hv)
                if lst is None:
                    shingle_index[hv] = [(i, c_pos)]
                elif len(lst) < 12:
                    lst.append((i, c_pos))

    # -----------------------------------------------------------------------
    # Tree 1: Layout / Stride Hierarchy Template Tree
    # Groups remaining shaped clumps sharing the same layout/stride and builds
    # a shared macro prototype template if net savings beat template size.
    # -----------------------------------------------------------------------
    tree1_edges = 0
    if config.enable_tree1_template and n_clumps >= 2:
        layout_groups: Dict[LayoutDescriptor, List[int]] = {}
        for i, pl in enumerate(plans):
            if pl.mode == MODE_SHAPED_ONLY and pl.layout.layout_id in (
                LAYOUT_2D_TRANSPOSE,
                LAYOUT_2D_TRANSPOSE_DELTA,
                LAYOUT_3D_CUBE_PERMUTE,
                LAYOUT_3D_CUBE_PERMUTE_DELTA,
            ):
                layout_groups.setdefault(pl.layout, []).append(i)

        for lay, gids in layout_groups.items():
            if len(gids) < 2:
                continue
            # Pick natural macro tile size aligned to the layout geometry
            if lay.layout_id in (LAYOUT_2D_TRANSPOSE, LAYOUT_2D_TRANSPOSE_DELTA):
                stride = max(2, lay.d0)
                # Tile a small multiple of stride (up to 4096 bytes)
                tile_len = stride
                while tile_len * 2 <= min(1024, config.tree1_max_template_bytes):
                    tile_len *= 2
            else:
                plane = max(4, lay.d0 * lay.d1)
                tile_len = plane if plane <= config.tree1_max_template_bytes else max(64, lay.d0 * 64)
                tile_len = min(tile_len, config.tree1_max_template_bytes)

            tmpl = _build_group_template(raw_data, clumps, gids, tile_len)
            if tmpl.shape[0] == 0:
                continue

            # Evaluate each clump in group against template (raw - tmpl, then layout)
            beneficiaries: List[Tuple[int, int, float, np.ndarray]] = []
            total_saved_bits = 0.0
            for cid in gids:
                sp = clumps[cid]
                raw_c = raw_data[sp.start:sp.end]
                s_len = min(sp.length, 65536)
                best_op = OP_NONE
                best_sc = standalone_scores[cid]
                for op in (OP_SUB_MOD256, OP_XOR):
                    diff_s = apply_template_forward(raw_c[:s_len], tmpl, op)
                    sh_s = apply_layout_forward(diff_s, lay)
                    sc = float(fast_proxy_score(sh_s))
                    if sc < best_sc * 0.95:
                        best_sc = sc
                        best_op = op
                if best_op != OP_NONE:
                    saved_bits = (standalone_scores[cid] - best_sc) * sp.length
                    total_saved_bits += saved_bits
                    beneficiaries.append((cid, best_op, best_sc, raw_c))

            # Prune template branch unless total savings strictly beat template cost
            tmpl_cost_bits = float(fast_proxy_score(tmpl)) * tmpl.shape[0] + 512.0
            if len(beneficiaries) >= 2 and total_saved_bits > tmpl_cost_bits * 1.5:
                tid = len(templates)
                templates.append(tmpl.tobytes())
                for cid, op, new_sc, raw_c in beneficiaries:
                    diff_full = apply_template_forward(raw_c, tmpl, op)
                    payload = apply_layout_forward(diff_full, lay)
                    plans[cid].mode = MODE_TREE_TEMPLATE_DELTA
                    plans[cid].tree_parent = tid
                    plans[cid].tree_op = op
                    plans[cid].tree_depth = 1
                    plans[cid].payload = payload
                    standalone_scores[cid] = new_sc
                    tree1_edges += 1

    # -----------------------------------------------------------------------
    # Tree 3: Coarse-to-Fine Multi-Level Refinement Tree
    # Applies bounded macro-to-micro refinement on remaining SHAPED_ONLY clumps.
    # For delta-shaped clumps, also tests coarse-to-fine on the non-delta base
    # plane (avoids 2x noise amplification on noisy smooth 2D/3D numeric fields).
    # -----------------------------------------------------------------------
    tree3_l0 = 0
    tree3_l1 = 0
    if config.enable_tree3_coarse_fine:
        for i, pl in enumerate(plans):
            if pl.mode != MODE_SHAPED_ONLY or pl.payload is None:
                continue

            # Candidate A: coarse-to-fine on current shaped payload
            cand_lay = pl.layout
            cand_src = pl.payload
            accepted, n_levels, coarse_bytes, fine_res = coarse_fine_encode(cand_src, config)

            # Candidate B: if layout was *_DELTA, also test coarse-to-fine on non-delta layout
            # and accept if total bits strictly beat the delta-shaped proxy!
            if not accepted and pl.layout.layout_id in (LAYOUT_2D_TRANSPOSE_DELTA, LAYOUT_3D_CUBE_PERMUTE_DELTA):
                base_lid = (
                    LAYOUT_2D_TRANSPOSE
                    if pl.layout.layout_id == LAYOUT_2D_TRANSPOSE_DELTA
                    else LAYOUT_3D_CUBE_PERMUTE
                )
                base_lay = LayoutDescriptor(base_lid, pl.layout.d0, pl.layout.d1, pl.layout.d2)
                raw_c = raw_data[pl.start:pl.end]
                base_shaped = apply_layout_forward(raw_c, base_lay)
                ok_b, lvl_b, cb_b, fine_b = coarse_fine_encode(base_shaped, config)
                if ok_b:
                    delta_bits = float(fast_proxy_score(pl.payload)) * pl.payload.shape[0]
                    cb_arr = np.frombuffer(cb_b, dtype=np.uint8)
                    cf_bits = (
                        float(fast_proxy_score(fine_b)) * fine_b.shape[0]
                        + float(fast_proxy_score(cb_arr)) * cb_arr.shape[0]
                        + 192.0
                    )
                    if cf_bits < delta_bits * config.tree3_min_improvement:
                        accepted = True
                        n_levels = lvl_b
                        coarse_bytes = cb_b
                        fine_res = fine_b
                        cand_lay = base_lay

            if accepted:
                pl.mode = MODE_TREE_COARSE_FINE
                pl.layout = cand_lay
                pl.tree_parent = -1
                pl.tree_op = n_levels  # 1 = Level 0 only, 2 = Level 0 + Level 1
                pl.tree_depth = n_levels
                pl.coarse_band = coarse_bytes
                pl.payload = fine_res
                if n_levels == 1:
                    tree3_l0 += 1
                else:
                    tree3_l1 += 1

    return ForestResult(
        plans=plans,
        templates=templates,
        tree1_edges=tree1_edges,
        tree2_edges=tree2_edges,
        tree2_max_depth=tree2_max_depth,
        tree3_level0_count=tree3_l0,
        tree3_level1_count=tree3_l1,
    )
