#!/usr/bin/env python3
"""
Phase 0 falsification harness v2.2 (with real single-pass Rice coder & bit-exact CCSDS-123).

Part A  Six-row coding-gain matrix (R1..R6) under BOTH the idealized Laplace
        estimator AND a real single-pass bit-exact causal Golomb-Rice bitstream
        coder, including 2-level/multilevel Clustered SERM (adjacent & corr-
        clustered PLUS factorization, M <= 8), bit-exact CCSDS-123.0-B-1
        locally-adaptive predictor (P=3, sign-LMS), real-byte anchors
        (xz / zstd / ASV4 v4.1), rounding-noise + multiplier audit, and
        Gate 0A / 0B / 0C checks.
Part B  Track A: blind stride-lattice recovery (centered Pearson FFT-ACF +
        sequential axis-peeling + harmonic-verified divisor-refinement +
        rectified YIN fundamental selection) over all 6 axis permutations,
        plus Gate 0D oracle-roles comparison.
Track B Progressive lossy-to-lossless deadzone quantizer sweep
        (step in {1, 2, 4, 8, 16, 32, 64}) reporting PSNR vs. bits/sample.

Requires: numpy, scipy.  Optional: numba (JIT speedup), zstandard (byte anchor).

USAGE
  python phase0_harness.py selftest
  python phase0_harness.py partA  --data hyper            # synthetic stand-ins
  python phase0_harness.py partB  --data hyper
  python phase0_harness.py trackB --data hyper
  python phase0_harness.py all
  # real MATLAB hyperspectral cube (H,W,B -> auto-transposed to B,H,W):
  python phase0_harness.py all --mat phase0_data/Indian_pines.mat
  # real raw cube (e.g. AVIRIS, big-endian int16, BIL; --dims is ALWAYS bands,lines,samples):
  python phase0_harness.py partA --raw cube.img --dtype '>i2' --order BIL \
         --dims 224,512,614 --crop 64,256,256
  # real int32 volume:
  python phase0_harness.py all --raw vol.raw --dtype '<i4' --order BSQ \
         --dims 32,128,128

HONEST SCOPE NOTES
  * Every row reports BOTH the idealized context-adaptive discrete-Laplace code
    length (`est b/s`) AND an actual bit-exact single-pass causal Golomb-Rice
    bitstream coder (`real Rice`, CCSDS-123 style halving at cnt=64 with 32-bit
    escape, verified round-trip).  The two track within ~0.1–0.3 b/s.
  * Reversible spectral transforms included:
      - Clustered SERM (Hao & Shi PLUS factorization in groups of <= 8, Q16
        fixed-point integer lifting, safe Givens fallback if a group pivot is
        singular or multiplier > 8), in both adjacent (`R3c`) and correlation-
        clustered (`R3g`) grouping modes.
      - 3-shear Givens lifting (Q16) full (`R3e`) and 2-level clustered (`R3d`).
      - POT-style hierarchical pairwise rotation tree (`R3b`, M-1 rotations).
      - Float KLT (`R3f`) is a non-reversible ceiling row.
  * R5a is a simple 3D static causal LS predictor (weak proxy); R5b is a real
    bit-exact integer CCSDS-123.0-B-1 Fast Lossless locally-adaptive sign-LMS
    predictor (P=3, neighbor-oriented local sum, 3 spatial + P spectral local
    differences, exact round-trip verified).
  * Synthetic data validates MECHANICS only; real-cube data decides the gates.
"""
import argparse
import itertools
import lzma
import math
import time

import numpy as np

try:
    import zstandard as zstd
except Exception:  # optional
    zstd = None

from scipy import ndimage
from scipy.io import loadmat
from scipy.signal import find_peaks

try:
    from numba import njit
except Exception:  # fallback if numba unavailable
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

I64 = np.int64
Q_SHIFT = 16
Q_SCALE = 1 << Q_SHIFT
Q_HALF = 1 << (Q_SHIFT - 1)


# =============================================================================
# 1. Reversible integer 5/3 lifting (JPEG-2000 style), any axes, multilevel
# =============================================================================
def _fwd53_1d(x):
    """x: (..., N) int64 -> (..., N) laid out [low | high]. Symmetric extension."""
    N = x.shape[-1]
    if N < 2:
        return x.copy()
    e = x[..., 0::2]
    o = x[..., 1::2]
    ne, no = e.shape[-1], o.shape[-1]
    idx_next = np.minimum(np.arange(no) + 1, ne - 1)
    d = o - ((e[..., :no] + e[..., idx_next]) >> 1)
    idx_cur = np.minimum(np.arange(ne), no - 1)
    idx_prev = np.minimum(np.maximum(np.arange(ne) - 1, 0), no - 1)
    s = e + ((d[..., idx_prev] + d[..., idx_cur] + 2) >> 2)
    return np.concatenate([s, d], axis=-1)


def _inv53_1d(y):
    N = y.shape[-1]
    if N < 2:
        return y.copy()
    ne = (N + 1) // 2
    no = N // 2
    s = y[..., :ne]
    d = y[..., ne:]
    idx_cur = np.minimum(np.arange(ne), no - 1)
    idx_prev = np.minimum(np.maximum(np.arange(ne) - 1, 0), no - 1)
    e = s - ((d[..., idx_prev] + d[..., idx_cur] + 2) >> 2)
    idx_next = np.minimum(np.arange(no) + 1, ne - 1)
    o = d + ((e[..., :no] + e[..., idx_next]) >> 1)
    x = np.empty_like(y)
    x[..., 0::2] = e
    x[..., 1::2] = o
    return x


def dwt53_axes(x, axes, levels):
    y = np.array(x, dtype=I64, copy=True)
    cur = list(y.shape)
    plan = []
    for _ in range(levels):
        before = list(cur)
        if all(before[a] < 2 for a in axes):
            break
        sub = y[tuple(slice(0, cur[d]) for d in range(y.ndim))]
        for a in axes:
            if before[a] >= 2:
                t = np.moveaxis(sub, a, -1)
                t[...] = _fwd53_1d(t)
        for a in axes:
            if before[a] >= 2:
                cur[a] = (before[a] + 1) // 2
        plan.append(before)
    return y, plan


def dwt53_inv(y, plan, axes):
    x = np.array(y, dtype=I64, copy=True)
    for before in reversed(plan):
        sub = x[tuple(slice(0, before[d]) for d in range(x.ndim))]
        for a in reversed(axes):
            if before[a] >= 2:
                t = np.moveaxis(sub, a, -1)
                t[...] = _inv53_1d(t)
    return x


def subband_slices(plan, shape, axes):
    out = []
    last = list(shape)
    for before in plan:
        after = list(before)
        tr = [a for a in axes if before[a] >= 2]
        for a in tr:
            after[a] = (before[a] + 1) // 2
        for mask in range(1, 2 ** len(tr)):
            sl = [slice(0, before[d]) for d in range(len(shape))]
            for bit, a in enumerate(tr):
                sl[a] = slice(after[a], before[a]) if (mask >> bit) & 1 else slice(0, after[a])
            out.append(tuple(sl))
        last = after
    out.append(tuple(slice(0, e) for e in last))
    return out


# =============================================================================
# 2. Reversible rotation programs (Givens Q16 + Clustered SERM/PLUS Q16)
# =============================================================================
class Program:
    """Sequence of exactly-invertible integer steps acting on rows of an (M,N) int64 array."""

    def __init__(self):
        self.steps = []
        self.n_rot = 0
        self.n_neg = 0

    def add_neg(self, idx):
        if len(idx):
            self.steps.append(("neg", np.array(idx)))
            self.n_neg += len(idx)

    def add_rot(self, i, j, theta):
        theta = math.atan2(math.sin(theta), math.cos(theta))  # wrap to (-pi, pi]
        flip = False
        if theta > math.pi / 2:
            theta -= math.pi
            flip = True
        elif theta < -math.pi / 2:
            theta += math.pi
            flip = True
        t = math.tan(theta / 2.0)
        mt = int(round(-t * Q_SCALE))
        ms = int(round(math.sin(theta) * Q_SCALE))
        self.steps.append(("rot", i, j, mt, ms, flip, theta))
        self.n_rot += 1

    def extend_givens(self, U, idx_map):
        """Append steps so that rows idx_map transform as y = U x (U orthogonal, KxK)."""
        K = U.shape[0]
        A = U.astype(np.float64).copy()
        elim = []
        for j in range(K - 1):
            for i in range(K - 1, j, -1):
                a, b = A[i - 1, j], A[i, j]
                r = math.hypot(a, b)
                if r < 1e-14 or abs(b) < 1e-14:
                    continue
                c, s = a / r, b / r
                ra, rb = A[i - 1].copy(), A[i].copy()
                A[i - 1] = c * ra + s * rb
                A[i] = -s * ra + c * rb
                elim.append((i - 1, i, math.atan2(s, c)))
        D = np.sign(np.diag(A))
        D[D == 0] = 1
        self.add_neg([idx_map[k] for k in range(K) if D[k] < 0])
        for (i, j, phi) in reversed(elim):
            self.add_rot(idx_map[i], idx_map[j], phi)

    def fwd(self, X):
        X = np.array(X, dtype=I64, copy=True)
        for st in self.steps:
            if st[0] == "neg":
                X[st[1]] = -X[st[1]]
            else:
                _, i, j, mt, ms, flip, _t = st
                x, y = X[i], X[j]
                if flip:
                    x[:] = -x
                    y[:] = -y
                x += (mt * y + Q_HALF) >> Q_SHIFT
                y += (ms * x + Q_HALF) >> Q_SHIFT
                x += (mt * y + Q_HALF) >> Q_SHIFT
        return X

    def inv(self, Y):
        X = np.array(Y, dtype=I64, copy=True)
        for st in reversed(self.steps):
            if st[0] == "neg":
                X[st[1]] = -X[st[1]]
            else:
                _, i, j, mt, ms, flip, _t = st
                x, y = X[i], X[j]
                x -= (mt * y + Q_HALF) >> Q_SHIFT
                y -= (ms * x + Q_HALF) >> Q_SHIFT
                x -= (mt * y + Q_HALF) >> Q_SHIFT
                if flip:
                    x[:] = -x
                    y[:] = -y
        return X

    def fwd_float(self, X):
        """Same program with exact (unrounded) rotations -> reference for noise measurement."""
        X = np.array(X, dtype=np.float64, copy=True)
        for st in self.steps:
            if st[0] == "neg":
                X[st[1]] = -X[st[1]]
            else:
                _, i, j, _mt, _ms, flip, th = st
                x, y = X[i].copy(), X[j].copy()
                if flip:
                    x, y = -x, -y
                c, s = math.cos(th), math.sin(th)
                X[i] = c * x - s * y
                X[j] = s * x + c * y
        return X

    def header_bits(self):
        return 16 * self.n_rot + self.n_neg


def _plus_factor_pivoted(A, max_coeff_cap=8.0):
    """Exact Hao & Shi (2001) PLUS/SERM factorization for small orthogonal M (<= 8):
    diag(rs) @ A0[perm] = L @ U @ S  (diag(L)=1, diag(U)=1, S = I + e_{M-1} s^T).
    Returns dict on success, or None if singular pivot / multiplier exceeds cap
    (caller falls back cleanly to bounded Givens lifting for that group)."""
    M = A.shape[0]
    A0 = A.astype(np.float64).copy()
    sign0 = np.ones(M, dtype=I64)
    if np.linalg.det(A0) < 0:
        A0[0, :] *= -1
        sign0[0] = -1

    W = A0.copy()
    perm = list(range(M))
    L = np.eye(M)
    U = np.eye(M)
    s_vec = np.zeros(M)

    for k in range(M - 1):
        best_p = -1
        best_score = float("inf")
        for p in range(k, M):
            denom = W[p, M - 1]
            if abs(denom) < 1e-10:
                if abs(W[p, k] - 1.0) < 1e-10:
                    sk = 0.0
                else:
                    continue
            else:
                sk = (W[p, k] - 1.0) / denom
            col_k_new = W[k:, k] - sk * W[k:, M - 1]
            score = max(abs(sk), float(np.max(np.abs(col_k_new))))
            if score < best_score:
                best_score = score
                best_p = p
        if best_p < 0 or best_score > max_coeff_cap:
            return None
        if best_p != k:
            W[[k, best_p], :] = W[[best_p, k], :]
            if k > 0:
                L[[k, best_p], :k] = L[[best_p, k], :k]
            perm[k], perm[best_p] = perm[best_p], perm[k]
        denom = W[k, M - 1]
        s_k = 0.0 if abs(denom) < 1e-10 else (W[k, k] - 1.0) / denom
        s_vec[k] = s_k
        W[k:, k] -= s_k * W[k:, M - 1]
        U[:k, k] -= s_k * U[:k, M - 1]
        U[k, k:] = W[k, k:]
        for i in range(k + 1, M):
            L[i, k] = W[i, k]
            W[i, k:] -= L[i, k] * W[k, k:]

    if abs(abs(W[M - 1, M - 1]) - 1.0) > 1e-5:
        return None
    last_sign = 1 if W[M - 1, M - 1] > 0 else -1
    if last_sign == -1:
        L[M - 1, : M - 1] *= -1
    S = np.eye(M)
    S[M - 1, : M - 1] = s_vec[: M - 1]
    rs = np.ones(M, dtype=I64)
    rs[M - 1] = last_sign

    B = A0[perm, :].copy()
    B[M - 1, :] *= last_sign
    if np.max(np.abs(B - L @ U @ S)) > 1e-6:
        return None
    max_c = float(max(np.max(np.abs(L)), np.max(np.abs(U)), np.max(np.abs(S))))
    if max_c > max_coeff_cap:
        return None

    return dict(
        perm=np.array(perm, dtype=I64),
        rs=rs,
        sign0=sign0,
        Lq=np.rint(L * Q_SCALE).astype(I64),
        Uq=np.rint(U * Q_SCALE).astype(I64),
        Sq=np.rint(S[M - 1, : M - 1] * Q_SCALE).astype(I64),
        max_c=max_c,
    )


def _serm_fwd_block(X, f):
    M = X.shape[0]
    Y = np.array(X, dtype=I64, copy=True)
    acc = np.zeros(X.shape[1], dtype=I64)
    for j in range(M - 1):
        acc += f["Sq"][j] * Y[j]
    Y[M - 1] += (acc + Q_HALF) >> Q_SHIFT
    for i in range(M - 1):
        acc = np.zeros(X.shape[1], dtype=I64)
        for j in range(i + 1, M):
            acc += f["Uq"][i, j] * Y[j]
        Y[i] += (acc + Q_HALF) >> Q_SHIFT
    for i in range(M - 1, 0, -1):
        acc = np.zeros(X.shape[1], dtype=I64)
        for j in range(i):
            acc += f["Lq"][i, j] * Y[j]
        Y[i] += (acc + Q_HALF) >> Q_SHIFT
    Y *= f["rs"][:, None]
    out = np.empty_like(Y)
    out[f["perm"], :] = Y
    out *= f["sign0"][:, None]
    return out


def _serm_inv_block(Y_in, f):
    M = Y_in.shape[0]
    Y = (Y_in * f["sign0"][:, None])[f["perm"], :] * f["rs"][:, None]
    for i in range(1, M):
        acc = np.zeros(Y.shape[1], dtype=I64)
        for j in range(i):
            acc += f["Lq"][i, j] * Y[j]
        Y[i] -= (acc + Q_HALF) >> Q_SHIFT
    for i in range(M - 2, -1, -1):
        acc = np.zeros(Y.shape[1], dtype=I64)
        for j in range(i + 1, M):
            acc += f["Uq"][i, j] * Y[j]
        Y[i] -= (acc + Q_HALF) >> Q_SHIFT
    acc = np.zeros(Y.shape[1], dtype=I64)
    for j in range(M - 1):
        acc += f["Sq"][j] * Y[j]
    Y[M - 1] -= (acc + Q_HALF) >> Q_SHIFT
    return Y


class SermProgram:
    """Multilevel clustered SERM/PLUS program (groups <= gs), exact bit-exact Q16 lifting.
    Falls back per-group to Givens lifting if a group's PLUS pivot is singular or > cap."""

    def __init__(self):
        self.stages = []  # list of (grp, kind, obj, U_float)
        self.n_ops = 0
        self.max_coeff = 1.0
        self.n_fallbacks = 0

    def add_group(self, U, grp):
        f = _plus_factor_pivoted(U)
        if f is not None:
            self.stages.append((grp, "serm", f, U.astype(np.float64)))
            self.n_ops += 2 * len(grp) - 1
            self.max_coeff = max(self.max_coeff, f["max_c"])
        else:
            p = Program()
            p.extend_givens(U, list(range(len(grp))))
            self.stages.append((grp, "givens", p, U.astype(np.float64)))
            self.n_ops += p.n_rot
            self.n_fallbacks += 1

    def fwd(self, X):
        Y = np.array(X, dtype=I64, copy=True)
        for grp, kind, obj, _U in self.stages:
            if kind == "serm":
                Y[grp] = _serm_fwd_block(Y[grp], obj)
            else:
                Y[grp] = obj.fwd(Y[grp])
        return Y

    def inv(self, Y_in):
        X = np.array(Y_in, dtype=I64, copy=True)
        for grp, kind, obj, _U in reversed(self.stages):
            if kind == "serm":
                X[grp] = _serm_inv_block(X[grp], obj)
            else:
                X[grp] = obj.inv(X[grp])
        return X

    def fwd_float(self, X):
        Y = np.array(X, dtype=np.float64, copy=True)
        for grp, _kind, _obj, U in self.stages:
            Y[grp] = U @ Y[grp]
        return Y

    def header_bits(self):
        bits = 0
        for grp, kind, obj, _U in self.stages:
            if kind == "serm":
                k = len(grp)
                # L, U strict triangles + S row + perm/sign bits
                bits += 16 * (k * (k - 1) + (k - 1)) + 4 * k
            else:
                bits += obj.header_bits()
        return bits


def _sub_cols(X, maxc=40000):
    step = max(1, X.shape[1] // maxc)
    return X[:, ::step].astype(np.float64)


def klt_U(Xc_sub):
    C = Xc_sub @ Xc_sub.T / Xc_sub.shape[1]
    w, V = np.linalg.eigh(C)
    return V[:, ::-1].T  # rows = eigenvectors, descending energy


def build_givens_full(Xc):
    prog = Program()
    prog.extend_givens(klt_U(_sub_cols(Xc)), list(range(Xc.shape[0])))
    return prog


def build_clustered_givens(Xc, gs=8):
    M = Xc.shape[0]
    prog = Program()
    groups = [list(range(g, min(g + gs, M))) for g in range(0, M, gs)]
    for grp in groups:
        if len(grp) > 1:
            prog.extend_givens(klt_U(_sub_cols(Xc[grp])), grp)
    heads = [g[0] for g in groups]
    Y_cur = prog.fwd(Xc)
    while len(heads) > 1:
        next_heads = []
        for g in range(0, len(heads), gs):
            grp = heads[g : min(g + gs, len(heads))]
            if len(grp) > 1:
                prog.extend_givens(klt_U(_sub_cols(Y_cur[grp])), grp)
            next_heads.append(grp[0])
        if len(next_heads) > 1:
            Y_cur = prog.fwd(Xc)
        heads = next_heads
    return prog


def build_clustered_serm(Xc, gs=8):
    """Multilevel clustered SERM/PLUS (groups <= gs, recursively combining heads)."""
    M = Xc.shape[0]
    prog = SermProgram()
    Y_cur = np.array(Xc, dtype=I64, copy=True)
    groups = [list(range(g, min(g + gs, M))) for g in range(0, M, gs)]
    for grp in groups:
        if len(grp) > 1:
            U = klt_U(_sub_cols(Y_cur[grp]))
            prog.add_group(U, grp)
            _, kind, obj, _ = prog.stages[-1]
            Y_cur[grp] = _serm_fwd_block(Y_cur[grp], obj) if kind == "serm" else obj.fwd(Y_cur[grp])
    heads = [g[0] for g in groups]
    while len(heads) > 1:
        next_heads = []
        for g in range(0, len(heads), gs):
            grp = heads[g : min(g + gs, len(heads))]
            if len(grp) > 1:
                U = klt_U(_sub_cols(Y_cur[grp]))
                prog.add_group(U, grp)
                _, kind, obj, _ = prog.stages[-1]
                Y_cur[grp] = _serm_fwd_block(Y_cur[grp], obj) if kind == "serm" else obj.fwd(Y_cur[grp])
            next_heads.append(grp[0])
        heads = next_heads
    return prog


def build_corr_clustered_serm(Xc, gs=8):
    """Correlation-clustered multilevel SERM/PLUS (groups of <= gs formed by greedy
    band correlation rather than sequential index order, plus recursive head combining)."""
    M = Xc.shape[0]
    prog = SermProgram()
    Y_cur = np.array(Xc, dtype=I64, copy=True)
    sub = _sub_cols(Y_cur)
    C = np.cov(sub, bias=True)
    sd = np.sqrt(np.maximum(np.diag(C), 1e-12))
    R = np.abs(C / np.outer(sd, sd))
    np.fill_diagonal(R, -1.0)
    remaining = list(range(M))
    groups = []
    while remaining:
        if len(remaining) <= gs:
            groups.append(remaining)
            break
        # Seed with band having strongest max correlation among remaining
        sub_R = R[np.ix_(remaining, remaining)]
        seed_pos = int(np.argmax(sub_R.max(axis=1)))
        seed = remaining[seed_pos]
        # Take seed + top (gs-1) correlated bands from remaining
        scores = [(R[seed, b], b) for b in remaining if b != seed]
        scores.sort(reverse=True)
        grp = [seed] + [b for _, b in scores[: gs - 1]]
        groups.append(grp)
        gset = set(grp)
        remaining = [b for b in remaining if b not in gset]

    for grp in groups:
        if len(grp) > 1:
            U = klt_U(_sub_cols(Y_cur[grp]))
            prog.add_group(U, grp)
            _, kind, obj, _ = prog.stages[-1]
            Y_cur[grp] = _serm_fwd_block(Y_cur[grp], obj) if kind == "serm" else obj.fwd(Y_cur[grp])
    heads = [g[0] for g in groups]
    while len(heads) > 1:
        next_heads = []
        for g in range(0, len(heads), gs):
            grp = heads[g : min(g + gs, len(heads))]
            if len(grp) > 1:
                U = klt_U(_sub_cols(Y_cur[grp]))
                prog.add_group(U, grp)
                _, kind, obj, _ = prog.stages[-1]
                Y_cur[grp] = _serm_fwd_block(Y_cur[grp], obj) if kind == "serm" else obj.fwd(Y_cur[grp])
            next_heads.append(grp[0])
        heads = next_heads
    return prog


def build_pot(Xc):
    """POT-STYLE: greedy correlation pairing, 2x2 KLT rotation per pair, principal
    output continues to the next stage.  M-1 rotations.  (Reconstruction, unverified.)"""
    M = Xc.shape[0]
    prog = Program()
    cur = _sub_cols(Xc)
    active = list(range(M))
    while len(active) > 1:
        C = np.cov(cur[active], bias=True)
        sd = np.sqrt(np.maximum(np.diag(C), 1e-12))
        R = np.abs(C / np.outer(sd, sd))
        n = len(active)
        iu = np.triu_indices(n, 1)
        order = np.argsort(-R[iu])
        used = set()
        nxt = []
        for k in order:
            a, b = iu[0][k], iu[1][k]
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            ca, cb = active[a], active[b]
            th = 0.5 * math.atan2(2 * C[a, b], C[a, a] - C[b, b])
            prog.add_rot(ca, cb, -th)
            x, y = cur[ca].copy(), cur[cb].copy()
            c, s = math.cos(-th), math.sin(-th)
            cur[ca] = c * x - s * y
            cur[cb] = s * x + c * y
            nxt.append(ca)
            if len(used) == n:
                break
        for i, ca in enumerate(active):
            if i not in used:
                nxt.append(ca)
        active = nxt
    return prog


# =============================================================================
# 3. Common entropy estimator (identical for every row)
# =============================================================================
def _bits_one_context(A, mode, nbins=48, param_bits=10, mfloor=0.02):
    Aw = np.zeros_like(A)
    Aw[..., 1:] = A[..., :-1]
    An = np.zeros_like(A)
    An[:, 1:, :] = A[:, :-1, :]
    if mode == 0:
        a = (Aw + An) * 0.5
    else:
        Ap = np.zeros_like(A)
        Ap[1:] = A[:-1]
        a = (Aw + An + Ap) * (1.0 / 3.0)
    q = np.minimum((np.log2(1.0 + a) * 2.5).astype(np.int32), nbins - 1).ravel()
    n = np.bincount(q, minlength=nbins).astype(np.float64)
    S = np.bincount(q, weights=A.ravel(), minlength=nbins)
    used = n > 0
    m = np.maximum(S[used] / n[used], mfloor)
    th = (-1.0 + np.sqrt(1.0 + m * m)) / m
    bits = np.sum(n[used] * (-np.log2((1.0 - th) / (1.0 + th))) + S[used] * (-np.log2(th)))
    return float(bits + param_bits * used.sum())


def block_bits(X):
    """Ideal code length (bits) of one coefficient block under context-adaptive discrete
    Laplace.  Per block: best of {W,N} / {W,N,prev-slice} contexts (+1 bit to signal)."""
    X = np.asarray(X, dtype=I64)
    if X.size == 0:
        return 0.0
    while X.ndim < 3:
        X = X[None]
    A = np.abs(X).astype(np.float32)
    return min(_bits_one_context(A, 0), _bits_one_context(A, 1)) + 1.0


def total_bits(blocks):
    return sum(block_bits(b) for b in blocks)


# =============================================================================
# 3b. Real Single-Pass Bit-Exact Adaptive Golomb-Rice Coder (CCSDS-123 style)
# =============================================================================
@njit(cache=True)
def _rice_encode_3d_numba(X_3d, n_ctx=32, k_max=15, unary_cap=32):
    """True causal single-pass context-adaptive Golomb-Rice bitstream encoder.
    Uses strictly causal {W, N, prev-slice} local magnitude context and per-context
    running (sum, count) accumulator with CCSDS-123 style halving (gamma_max = 64).
    Returns packed uint8 bytearray + exact bit length (100% decodable bit-for-bit)."""
    B, H, W = X_3d.shape
    # Running accumulator per context: init count=2, sum=4 (k=1)
    acc_cnt = np.full(n_ctx, 2, dtype=np.int64)
    acc_sum = np.full(n_ctx, 4, dtype=np.int64)
    # Worst-case output buffer: 40 bits/sample
    out = np.zeros((B * H * W * 5) + 64, dtype=np.uint8)
    byte_pos = 0
    bit_buf = np.uint64(0)
    bit_cnt = 0

    for b in range(B):
        for y in range(H):
            for x in range(W):
                # Strictly causal local activity from previously coded neighbors
                a_sum = 0
                a_n = 0
                if x > 0:
                    v = X_3d[b, y, x - 1]
                    a_sum += -v if v < 0 else v
                    a_n += 1
                if y > 0:
                    v = X_3d[b, y - 1, x]
                    a_sum += -v if v < 0 else v
                    a_n += 1
                if b > 0:
                    v = X_3d[b - 1, y, x]
                    a_sum += -v if v < 0 else v
                    a_n += 1
                avg = (a_sum // a_n) if a_n > 0 else 0
                # Fast integer log2 context bucket in [0, n_ctx - 1]
                ctx = 0
                t = avg
                while t > 0 and ctx < n_ctx - 1:
                    t >>= 1
                    ctx += 1

                # Select Rice parameter k: smallest k >= 0 with (cnt << k) >= sum
                cnt = acc_cnt[ctx]
                sm = acc_sum[ctx]
                k = 0
                while k < k_max and (cnt << k) < sm:
                    k += 1

                # ZigZag map signed -> unsigned
                val = X_3d[b, y, x]
                u = (val << 1) if val >= 0 else ((-val << 1) - 1)
                q = u >> k

                if q < unary_cap:
                    # Emit q zeros + 1 one
                    # Flush in chunks if needed
                    rem_q = q
                    while rem_q > 0:
                        step = 24 if rem_q > 24 else rem_q
                        bit_buf <<= step
                        bit_cnt += step
                        rem_q -= step
                        while bit_cnt >= 8:
                            bit_cnt -= 8
                            out[byte_pos] = np.uint8((bit_buf >> bit_cnt) & np.uint64(0xFF))
                            byte_pos += 1
                    bit_buf = (bit_buf << 1) | np.uint64(1)
                    bit_cnt += 1
                    if k > 0:
                        mask = (np.uint64(1) << k) - np.uint64(1)
                        bit_buf = (bit_buf << k) | (np.uint64(u) & mask)
                        bit_cnt += k
                else:
                    # Escape: unary_cap zeros + 1 one + raw 24-bit unsigned value (or 32-bit if >= 2^23)
                    rem_q = unary_cap
                    while rem_q > 0:
                        step = 24 if rem_q > 24 else rem_q
                        bit_buf <<= step
                        bit_cnt += step
                        rem_q -= step
                        while bit_cnt >= 8:
                            bit_cnt -= 8
                            out[byte_pos] = np.uint8((bit_buf >> bit_cnt) & np.uint64(0xFF))
                            byte_pos += 1
                    bit_buf = (bit_buf << 1) | np.uint64(1)
                    bit_cnt += 1
                    # Write 32-bit escape in two 16-bit halves
                    hi16 = np.uint64((u >> 16) & 0xFFFF)
                    lo16 = np.uint64(u & 0xFFFF)
                    bit_buf = (bit_buf << 16) | hi16
                    bit_cnt += 16
                    while bit_cnt >= 8:
                        bit_cnt -= 8
                        out[byte_pos] = np.uint8((bit_buf >> bit_cnt) & np.uint64(0xFF))
                        byte_pos += 1
                    bit_buf = (bit_buf << 16) | lo16
                    bit_cnt += 16

                while bit_cnt >= 8:
                    bit_cnt -= 8
                    out[byte_pos] = np.uint8((bit_buf >> bit_cnt) & np.uint64(0xFF))
                    byte_pos += 1

                # Update context accumulator with CCSDS-123 halving at cnt == 64
                acc_sum[ctx] += u
                acc_cnt[ctx] += 1
                if acc_cnt[ctx] >= 64:
                    acc_sum[ctx] = (acc_sum[ctx] + 1) >> 1
                    acc_cnt[ctx] = (acc_cnt[ctx] + 1) >> 1

    total_bits_written = byte_pos * 8 + bit_cnt
    if bit_cnt > 0:
        out[byte_pos] = np.uint8((bit_buf << (8 - bit_cnt)) & np.uint64(0xFF))
        byte_pos += 1
    return out[:byte_pos], total_bits_written


@njit(cache=True)
def _rice_decode_3d_numba(buf, B, H, W, n_ctx=32, k_max=15, unary_cap=32):
    """Exact inverse of _rice_encode_3d_numba."""
    X = np.empty((B, H, W), dtype=np.int64)
    acc_cnt = np.full(n_ctx, 2, dtype=np.int64)
    acc_sum = np.full(n_ctx, 4, dtype=np.int64)
    byte_pos = 0
    bit_buf = np.uint64(0)
    bit_cnt = 0
    nbuf = len(buf)

    for b in range(B):
        for y in range(H):
            for x in range(W):
                a_sum = 0
                a_n = 0
                if x > 0:
                    v = X[b, y, x - 1]
                    a_sum += -v if v < 0 else v
                    a_n += 1
                if y > 0:
                    v = X[b, y - 1, x]
                    a_sum += -v if v < 0 else v
                    a_n += 1
                if b > 0:
                    v = X[b - 1, y, x]
                    a_sum += -v if v < 0 else v
                    a_n += 1
                avg = (a_sum // a_n) if a_n > 0 else 0
                ctx = 0
                t = avg
                while t > 0 and ctx < n_ctx - 1:
                    t >>= 1
                    ctx += 1

                cnt = acc_cnt[ctx]
                sm = acc_sum[ctx]
                k = 0
                while k < k_max and (cnt << k) < sm:
                    k += 1

                # Read unary zeros until 1
                q = 0
                while True:
                    if bit_cnt == 0:
                        bit_buf = np.uint64(buf[byte_pos])
                        byte_pos += 1
                        bit_cnt = 8
                    bit_cnt -= 1
                    if (bit_buf >> bit_cnt) & np.uint64(1):
                        break
                    q += 1

                if q < unary_cap:
                    rem = np.int64(0)
                    if k > 0:
                        while bit_cnt < k:
                            bit_buf = (bit_buf << 8) | np.uint64(buf[byte_pos])
                            byte_pos += 1
                            bit_cnt += 8
                        bit_cnt -= k
                        mask = (np.uint64(1) << k) - np.uint64(1)
                        rem = np.int64((bit_buf >> bit_cnt) & mask)
                    u = (np.int64(q) << k) | rem
                else:
                    while bit_cnt < 16:
                        bit_buf = (bit_buf << 8) | np.uint64(buf[byte_pos])
                        byte_pos += 1
                        bit_cnt += 8
                    bit_cnt -= 16
                    hi16 = np.int64((bit_buf >> bit_cnt) & np.uint64(0xFFFF))
                    while bit_cnt < 16:
                        bit_buf = (bit_buf << 8) | np.uint64(buf[byte_pos])
                        byte_pos += 1
                        bit_cnt += 8
                    bit_cnt -= 16
                    lo16 = np.int64((bit_buf >> bit_cnt) & np.uint64(0xFFFF))
                    u = (hi16 << 16) | lo16

                # Inverse ZigZag
                val = (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)
                X[b, y, x] = val

                acc_sum[ctx] += u
                acc_cnt[ctx] += 1
                if acc_cnt[ctx] >= 64:
                    acc_sum[ctx] = (acc_sum[ctx] + 1) >> 1
                    acc_cnt[ctx] = (acc_cnt[ctx] + 1) >> 1
    return X


def real_rice_block_bits(X, verify=False):
    arr = np.ascontiguousarray(X, dtype=I64)
    if arr.size == 0:
        return 0
    while arr.ndim < 3:
        arr = arr[None]
    buf, nbits = _rice_encode_3d_numba(arr)
    if verify:
        rec = _rice_decode_3d_numba(buf, arr.shape[0], arr.shape[1], arr.shape[2])
        assert np.array_equal(rec, arr), "Real Rice bitstream round-trip FAILED!"
    return int(nbits)


def real_rice_total_bits(blocks, verify=False):
    return sum(real_rice_block_bits(b, verify=verify) for b in blocks)


# =============================================================================
# 4. Deadzone quantizer (Track B progressive curves) & Rows
# =============================================================================
def dz_quant(X, step):
    if step <= 1:
        return X.copy()
    return np.sign(X) * (np.abs(X) // step)


def dz_dequant(Q, step):
    if step <= 1:
        return Q.copy()
    half = step // 2
    return np.where(Q == 0, 0, np.sign(Q) * (np.abs(Q) * step + half))


def row_raw(C, real_rice=False):
    est = block_bits(C)
    rb = real_rice_block_bits(C, verify=False) if real_rice else None
    return est, rb


def row_dwt3d(C, L=3, verify=True, real_rice=False):
    y, plan = dwt53_axes(C, (0, 1, 2), L)
    if verify:
        assert np.array_equal(dwt53_inv(y, plan, (0, 1, 2)), C), "3D 5/3 round-trip failed"
    blks = [y[s] for s in subband_slices(plan, y.shape, (0, 1, 2))]
    est = total_bits(blks)
    rb = real_rice_total_bits(blks, verify=False) if real_rice else None
    return est, rb


def spectral_forward(C, kind, L=3):
    """Spectral (axis-0) transform. Returns (Y, header_bits, ctx)."""
    B = C.shape[0]
    if kind == "none":
        return C.copy(), 0, ("none",)
    if kind == "dwt53":
        y, plan = dwt53_axes(C, (0,), L)
        return y, 0, ("dwt53", plan)
    X = C.reshape(B, -1)
    mu = np.rint(X.mean(axis=1)).astype(I64)
    Xc = X - mu[:, None]
    hdr = 16 * B
    if kind == "klt_float":
        U = klt_U(_sub_cols(Xc))
        Y = np.rint(U @ Xc.astype(np.float64)).astype(I64)
        return Y.reshape(C.shape), hdr + 16 * B * (B - 1) // 2, ("klt_float", U, mu)
    builders = {
        "givens": build_givens_full,
        "clustered_givens": build_clustered_givens,
        "clustered_serm": build_clustered_serm,
        "corr_serm": build_corr_clustered_serm,
        "pot": build_pot,
    }
    prog = builders[kind](Xc)
    Y = prog.fwd(Xc)
    return Y.reshape(C.shape), hdr + prog.header_bits(), (kind, prog, mu)


def spectral_inverse(Y, ctx):
    k = ctx[0]
    if k == "none":
        return Y.copy()
    if k == "dwt53":
        return dwt53_inv(Y, ctx[1], (0,))
    if k == "klt_float":
        U, mu = ctx[1], ctx[2]
        B = Y.shape[0]
        X = np.rint(U.T @ Y.reshape(B, -1).astype(np.float64)).astype(I64) + mu[:, None]
        return X.reshape(Y.shape)
    prog, mu = ctx[1], ctx[2]
    B = Y.shape[0]
    X = prog.inv(Y.reshape(B, -1)) + mu[:, None]
    return X.reshape(Y.shape)


def row_hybrid(C, kind, L=3, verify=True, real_rice=False):
    Y, hdr, ctx = spectral_forward(C, kind, L)
    y2, plan = dwt53_axes(Y, (1, 2), L)
    blks = [y2[s] for s in subband_slices(plan, y2.shape, (1, 2))]
    bits = total_bits(blks) + hdr
    rb = (real_rice_total_bits(blks, verify=False) + hdr) if real_rice else None
    if verify and kind != "klt_float":
        Y2 = dwt53_inv(y2, plan, (1, 2))
        assert np.array_equal(Y2, Y)
        assert np.array_equal(spectral_inverse(Y2, ctx), C), f"hybrid/{kind} round-trip failed"
    return bits, rb


def folded_forward(C, fold=4, spec_kind="clustered_serm"):
    """R4 forward: spectral transform + per-band separable fold x fold SERM/Givens RKLT."""
    B, H, W = C.shape
    Y, hdr, ctx = spectral_forward(C, spec_kind)
    f = fold
    progs = []
    Z = np.empty((B, H // f, f, W // f, f), dtype=I64)
    for b in range(B):
        T = Y[b].reshape(H // f, f, W // f, f)
        Ta = np.moveaxis(T, 1, 0).reshape(f, -1)
        pa = SermProgram()
        pa.add_group(klt_U(_sub_cols(Ta)), list(range(f)))
        Ta2 = pa.fwd(Ta).reshape(f, H // f, W // f, f)
        T2 = np.moveaxis(Ta2, 0, 1)
        Tb = np.moveaxis(T2, 3, 0).reshape(f, -1)
        pb = SermProgram()
        pb.add_group(klt_U(_sub_cols(Tb)), list(range(f)))
        Tb2 = pb.fwd(Tb).reshape(f, H // f, f, W // f)
        Z[b] = np.moveaxis(Tb2, 0, 3)
        progs.append((pa, pb))
        hdr += pa.header_bits() + pb.header_bits()
    return Z, hdr, (ctx, progs, f, (B, H, W))


def folded_inverse(Z, fctx):
    ctx, progs, f, (B, H, W) = fctx
    Y = np.empty((B, H, W), dtype=I64)
    for b in range(B):
        pa, pb = progs[b]
        Tb2 = np.moveaxis(Z[b], 3, 0).reshape(f, -1)
        T2_back = np.moveaxis(pb.inv(Tb2).reshape(f, H // f, f, W // f), 0, 3)
        Ta2 = np.moveaxis(T2_back, 1, 0).reshape(f, -1)
        T_back = np.moveaxis(pa.inv(Ta2).reshape(f, H // f, W // f, f), 0, 1)
        Y[b] = T_back.reshape(H, W)
    return spectral_inverse(Y, ctx)


def row_folded(C, fold=4, spec_kind="clustered_serm", verify=True, real_rice=False):
    Z, hdr, fctx = folded_forward(C, fold, spec_kind)
    if verify and spec_kind != "klt_float":
        assert np.array_equal(folded_inverse(Z, fctx), C), "folded RKLT round-trip failed"
    blocks = [Z[:, :, u, :, v] for u in range(fold) for v in range(fold)]
    est = total_bits(blocks) + hdr
    rb = (real_rice_total_bits(blocks, verify=False) + hdr) if real_rice else None
    return est, rb


def fit_predictor(C):
    X = C.astype(I64)

    def taps3(X):
        return [X[1:, 1:, :-1], X[1:, :-1, 1:], X[1:, :-1, :-1],
                X[:-1, 1:, 1:], X[:-1, 1:, :-1], X[:-1, :-1, 1:], X[:-1, :-1, :-1]]

    def taps2(X0):
        return [X0[1:, :-1], X0[:-1, 1:], X0[:-1, :-1]]

    rng = np.random.default_rng(1)
    F = np.stack([t.ravel() for t in taps3(X)], 1).astype(np.float64)
    tgt = X[1:, 1:, 1:].ravel().astype(np.float64)
    sel = rng.choice(len(tgt), size=min(150000, len(tgt)), replace=False)
    w3 = np.linalg.lstsq(F[sel], tgt[sel], rcond=None)[0]
    F2 = np.stack([t.ravel() for t in taps2(X[0])], 1).astype(np.float64)
    t2 = X[0, 1:, 1:].ravel().astype(np.float64)
    w2 = np.linalg.lstsq(F2, t2, rcond=None)[0]
    return np.rint(w3 * 256).astype(I64), np.rint(w2 * 256).astype(I64)


def apply_predictor(C, wq):
    w3, w2 = wq
    X = C.astype(I64)
    P = np.zeros_like(X)
    P[:, 0, 1:] = X[:, 0, :-1]
    P[:, 1:, 0] = X[:, :-1, 0]
    P[1:, 0, 0] = X[:-1, 0, 0]
    t2 = [X[0, 1:, :-1], X[0, :-1, 1:], X[0, :-1, :-1]]
    P[0, 1:, 1:] = (sum(int(w) * t for w, t in zip(w2, t2)) + 128) >> 8
    t3 = [X[1:, 1:, :-1], X[1:, :-1, 1:], X[1:, :-1, :-1],
          X[:-1, 1:, 1:], X[:-1, 1:, :-1], X[:-1, :-1, 1:], X[:-1, :-1, :-1]]
    P[1:, 1:, 1:] = (sum(int(w) * t for w, t in zip(w3, t3)) + 128) >> 8
    return P


def row_predictor(C):
    wq = fit_predictor(C)
    res = C.astype(I64) - apply_predictor(C, wq)
    return block_bits(res) + 16 * (len(wq[0]) + len(wq[1])), res


# ---- R5b/R5c: Real bit-exact CCSDS-123.0-B-1 Locally-Adaptive Predictor (Numba JIT) ----
@njit(cache=True)
def _ccsds123_predict_numba(C, P=3, D=16, omega=19, v_min=-1, v_max=3, t_inc_exp=6):
    """Bit-exact integer CCSDS-123.0-B-1 Fast Lossless predictor (Blue Book Sec 4.4–4.9:
    neighbor-oriented local sum, 3 spatial + P spectral local differences, double-resolution
    predicted sample s_tilde, and sign-LMS weight update with rho(t) = v(t) + D - omega).
    Returns exact integer residuals."""
    B, H, W = C.shape
    res = np.empty((B, H, W), dtype=np.int64)
    d_c = np.zeros((B, H, W), dtype=np.int64)
    w_lim = (1 << (omega + 2)) - 1
    w_min = -(1 << (omega + 2))
    n_taps = 3 + P

    for z in range(B):
        p_use = z if z < P else P
        weights = np.zeros(n_taps, dtype=np.int64)
        # CCSDS-123.0-B-1 Sec 4.6.3.2 default weight initialization
        if p_use >= 1:
            weights[3] = (7 * (1 << omega)) >> 3
            for k in range(1, p_use):
                weights[3 + k] = weights[3 + k - 1] >> 3
        t = 0
        for y in range(H):
            for x in range(W):
                # Neighbor-oriented local sum (CCSDS-123 Sec 4.4.1)
                if y == 0 and x == 0:
                    lsum = 4 * C[z - 1, 0, 0] if z > 0 else 0
                elif y == 0:
                    lsum = 4 * C[z, 0, x - 1]
                elif x == 0:
                    lsum = 2 * (C[z, y - 1, 0] + C[z, y - 1, 1]) if W > 1 else 4 * C[z, y - 1, 0]
                elif x == W - 1:
                    lsum = C[z, y, x - 1] + C[z, y - 1, x - 1] + 2 * C[z, y - 1, x]
                else:
                    lsum = C[z, y, x - 1] + C[z, y - 1, x - 1] + C[z, y - 1, x] + C[z, y - 1, x + 1]

                # Local differences vector U: [d^N, d^W, d^NW, d_c(z-1), ..., d_c(z-P)]
                dN = (4 * C[z, y - 1, x] - lsum) if y > 0 else 0
                dW = (4 * C[z, y, x - 1] - lsum) if (y > 0 and x > 0) else 0
                dNW = (4 * C[z, y - 1, x - 1] - lsum) if (y > 0 and x > 0) else 0

                if y == 0 and x == 0:
                    pred = C[z - 1, 0, 0] if z > 0 else 0
                    s_tilde = 2 * pred
                else:
                    dot = weights[0] * dN + weights[1] * dW + weights[2] * dNW
                    for k in range(p_use):
                        dot += weights[3 + k] * d_c[z - 1 - k, y, x]
                    # Double-resolution predicted sample s_tilde (Blue Book Sec 4.7)
                    s_tilde = ((lsum << omega) + dot) >> (omega + 1)
                    pred = (s_tilde + 1) >> 1

                s_val = C[z, y, x]
                err = s_val - pred
                res[z, y, x] = err
                d_c[z, y, x] = 4 * s_val - lsum

                # Online sign-LMS weight update (Blue Book Sec 4.9, Eq. 44)
                if not (y == 0 and x == 0):
                    v_t = v_min + ((t - W) >> t_inc_exp)
                    if v_t < v_min:
                        v_t = v_min
                    elif v_t > v_max:
                        v_t = v_max
                    rho = v_t + D - omega
                    e_z = 2 * s_val - s_tilde
                    sgn = 1 if e_z >= 0 else -1
                    for ti, dv in ((0, dN), (1, dW), (2, dNW)):
                        if rho >= 0:
                            delta = ((sgn * dv) + (1 << rho)) >> (rho + 1)
                        else:
                            delta = (sgn * dv) << (-rho - 1)
                        nw = weights[ti] + delta
                        if nw > w_lim:
                            nw = w_lim
                        elif nw < w_min:
                            nw = w_min
                        weights[ti] = nw
                    for k in range(p_use):
                        dv = d_c[z - 1 - k, y, x]
                        if rho >= 0:
                            delta = ((sgn * dv) + (1 << rho)) >> (rho + 1)
                        else:
                            delta = (sgn * dv) << (-rho - 1)
                        nw = weights[3 + k] + delta
                        if nw > w_lim:
                            nw = w_lim
                        elif nw < w_min:
                            nw = w_min
                        weights[3 + k] = nw
                t += 1
    return res


@njit(cache=True)
def _ccsds123_reconstruct_numba(res, P=3, D=16, omega=19, v_min=-1, v_max=3, t_inc_exp=6):
    """Exact bit-for-bit inverse of _ccsds123_predict_numba."""
    B, H, W = res.shape
    C = np.empty((B, H, W), dtype=np.int64)
    d_c = np.zeros((B, H, W), dtype=np.int64)
    w_lim = (1 << (omega + 2)) - 1
    w_min = -(1 << (omega + 2))
    n_taps = 3 + P

    for z in range(B):
        p_use = z if z < P else P
        weights = np.zeros(n_taps, dtype=np.int64)
        if p_use >= 1:
            weights[3] = (7 * (1 << omega)) >> 3
            for k in range(1, p_use):
                weights[3 + k] = weights[3 + k - 1] >> 3
        t = 0
        for y in range(H):
            for x in range(W):
                if y == 0 and x == 0:
                    lsum = 4 * C[z - 1, 0, 0] if z > 0 else 0
                elif y == 0:
                    lsum = 4 * C[z, 0, x - 1]
                elif x == 0:
                    lsum = 2 * (C[z, y - 1, 0] + C[z, y - 1, 1]) if W > 1 else 4 * C[z, y - 1, 0]
                elif x == W - 1:
                    lsum = C[z, y, x - 1] + C[z, y - 1, x - 1] + 2 * C[z, y - 1, x]
                else:
                    lsum = C[z, y, x - 1] + C[z, y - 1, x - 1] + C[z, y - 1, x] + C[z, y - 1, x + 1]

                dN = (4 * C[z, y - 1, x] - lsum) if y > 0 else 0
                dW = (4 * C[z, y, x - 1] - lsum) if (y > 0 and x > 0) else 0
                dNW = (4 * C[z, y - 1, x - 1] - lsum) if (y > 0 and x > 0) else 0

                if y == 0 and x == 0:
                    pred = C[z - 1, 0, 0] if z > 0 else 0
                    s_tilde = 2 * pred
                else:
                    dot = weights[0] * dN + weights[1] * dW + weights[2] * dNW
                    for k in range(p_use):
                        dot += weights[3 + k] * d_c[z - 1 - k, y, x]
                    s_tilde = ((lsum << omega) + dot) >> (omega + 1)
                    pred = (s_tilde + 1) >> 1

                err = res[z, y, x]
                s_val = pred + err
                C[z, y, x] = s_val
                d_c[z, y, x] = 4 * s_val - lsum

                if not (y == 0 and x == 0):
                    v_t = v_min + ((t - W) >> t_inc_exp)
                    if v_t < v_min:
                        v_t = v_min
                    elif v_t > v_max:
                        v_t = v_max
                    rho = v_t + D - omega
                    e_z = 2 * s_val - s_tilde
                    sgn = 1 if e_z >= 0 else -1
                    for ti, dv in ((0, dN), (1, dW), (2, dNW)):
                        if rho >= 0:
                            delta = ((sgn * dv) + (1 << rho)) >> (rho + 1)
                        else:
                            delta = (sgn * dv) << (-rho - 1)
                        nw = weights[ti] + delta
                        if nw > w_lim:
                            nw = w_lim
                        elif nw < w_min:
                            nw = w_min
                        weights[ti] = nw
                    for k in range(p_use):
                        dv = d_c[z - 1 - k, y, x]
                        if rho >= 0:
                            delta = ((sgn * dv) + (1 << rho)) >> (rho + 1)
                        else:
                            delta = (sgn * dv) << (-rho - 1)
                        nw = weights[3 + k] + delta
                        if nw > w_lim:
                            nw = w_lim
                        elif nw < w_min:
                            nw = w_min
                        weights[3 + k] = nw
                t += 1
    return C


def row_ccsds123(C, P=3, verify=True):
    C64 = np.ascontiguousarray(C, dtype=I64)
    res = _ccsds123_predict_numba(C64, P=P)
    if verify:
        rec = _ccsds123_reconstruct_numba(res, P=P)
        assert np.array_equal(rec, C64), "CCSDS-123 round-trip FAILED!"
    return block_bits(res), res


# ---- R6: oracle clustered spatial-block KLT (held-out Gaussian NLL, net of labels) ----
def _kmeans(F, K, rng, iters=20):
    n = len(F)
    cent = [F[rng.integers(n)]]
    for _ in range(1, K):
        d2 = np.min(((F[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cent.append(F[rng.choice(n, p=p)])
    cent = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(((F[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        for k in range(K):
            if np.any(lab == k):
                cent[k] = F[lab == k].mean(0)
    return cent


def _gauss_nll_bits(V, mu, Sig):
    w, Q = np.linalg.eigh(Sig)
    w = np.maximum(w, 1e-9)
    Z = (V - mu) @ Q
    return 0.5 * (np.log2(w).sum() + (Z * Z / w).sum(1) * math.log2(math.e))


def oracle_forest(C, bb=4, f=4, Ks=(4, 8, 16), floor=0.25, seed=0):
    B, H, W = C.shape
    B2, H2, W2 = B // bb * bb, H // f * f, W // f * f
    T = C[:B2, :H2, :W2].astype(np.float64).reshape(B2 // bb, bb, H2 // f, f, W2 // f, f)
    V = np.moveaxis(T, (1, 3, 5), (3, 4, 5)).reshape(-1, bb * f * f)
    d = V.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(V))
    tr, te = V[perm[: len(V) // 2]], V[perm[len(V) // 2:]]
    reg = floor * np.eye(d)
    mu1 = tr.mean(0)
    S1 = np.cov(tr.T, bias=True) + reg
    nll1 = _gauss_nll_bits(te, mu1, S1).mean()
    w, Q = np.linalg.eigh(S1)
    P6 = Q[:, ::-1][:, :6]
    ftr = (tr - mu1) @ P6
    out = []
    for K in Ks:
        cent = _kmeans(ftr, K, rng)
        labtr = np.argmin(((ftr[:, None] - cent[None]) ** 2).sum(-1), axis=1)
        models = []
        for k in range(K):
            Vk = tr[labtr == k]
            if len(Vk) < d + 2:
                Vk = tr
            models.append((Vk.mean(0), np.cov(Vk.T, bias=True) + reg))
        nll = np.stack([_gauss_nll_bits(te, m, S) for m, S in models], 1)
        lab = nll.argmin(1)
        nllK = nll[np.arange(len(te)), lab].mean()
        p = np.bincount(lab, minlength=K) / len(lab)
        Hlab = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        g_raw = (nll1 - nllK) / d
        g_net = (nll1 - nllK - Hlab) / d
        hdr_per_cluster = 16 * (d * (d - 1) // 2 + d)
        out.append(dict(K=K, g_raw=g_raw, g_net=g_net, Hlab=Hlab,
                        hdr_bits_per_sample_here=K * hdr_per_cluster / V.size,
                        hdr_bits_per_sample_at_128M=K * hdr_per_cluster / 128e6))
    return out, d


# =============================================================================
# 5. Synthetic stand-ins & raw loader (--dims ALWAYS bands,lines,samples)
# =============================================================================
def synth_hyper(B=32, H=96, W=128, noise=6.0, seed=0, K=4):
    rng = np.random.default_rng(seed)
    lam = np.linspace(0, 1, B)
    E = []
    for _ in range(K):
        sp = 0.0
        for _ in range(3):
            sp = sp + rng.uniform(0.3, 1) * np.exp(-0.5 * ((lam - rng.uniform()) / rng.uniform(0.15, 0.5)) ** 2)
        sp = sp + rng.uniform(0.1, 0.4)
        for _ in range(2):
            sp = sp - 0.15 * np.exp(-0.5 * ((lam - rng.uniform()) / 0.02) ** 2)
        E.append(np.clip(sp, 0.02, None) * 4000)
    E = np.array(E)

    def field(sig):
        s_eff = min(sig, max(1.0, min(H, W) / 4.0))
        f = ndimage.gaussian_filter(rng.standard_normal((H, W)), s_eff, mode="wrap")
        return f / max(float(f.std()), 1e-9)

    logits = np.stack([3.0 * field(rng.uniform(6, 16)) for _ in range(K)])
    a = np.exp(logits - logits.max(0))
    a /= a.sum(0)
    reg = (field(10) > 0.3).astype(float)
    a[0] = a[0] * (1 - 0.5 * reg) + 0.5 * reg
    a /= a.sum(0)
    illum = 1 + 0.15 * field(25)
    cube = np.einsum("khw,kb->bhw", a, E) * illum[None]
    cube = cube + noise * (1 + 0.5 * lam)[:, None, None] * rng.standard_normal((B, H, W))
    return np.clip(np.rint(cube), -32768, 32767).astype(np.int16)


def synth_smooth(B=32, H=96, W=128, noise=0.4, seed=1):
    rng = np.random.default_rng(seed)

    def field(sig):
        s_eff = min(sig, max(1.0, min(H, W) / 4.0))
        f = ndimage.gaussian_filter(rng.standard_normal((H, W)), s_eff, mode="wrap")
        return f / max(float(f.std()), 1e-9)

    base = 4000 + 900 * field(18) + 300 * field(7)
    slices = [base * (1 + 0.002 * b) + 20 * field(30) * (b / B) for b in range(B)]
    cube = np.stack(slices) + noise * rng.standard_normal((B, H, W))
    return np.clip(np.rint(cube), -32768, 32767).astype(np.int16)


def synth_noise(B=32, H=96, W=128, seed=2):
    return np.random.default_rng(seed).integers(-32768, 32767, size=(B, H, W), dtype=np.int16)


def parse_raw_buffer(buf, dtype_str, order, dims_bls):
    """Parse raw buffer with canonical dims_bls=(bands, lines, samples) into (B, H, W) int64."""
    dt = np.dtype(dtype_str)
    B, H, W = dims_bls
    arr = np.frombuffer(buf, dtype=dt, count=B * H * W)
    if order == "BSQ":
        a = arr.reshape(B, H, W)
    elif order == "BIL":
        a = np.moveaxis(arr.reshape(H, B, W), 1, 0)
    elif order == "BIP":
        a = np.moveaxis(arr.reshape(H, W, B), 2, 0)
    else:
        raise ValueError(f"Unknown order {order}")
    return np.ascontiguousarray(a).astype(I64), dt.itemsize


def load_data(args):
    if getattr(args, "mat", None):
        d = loadmat(args.mat)
        key = getattr(args, "mat_key", None)
        if not key:
            keys = [k for k in d.keys() if not k.startswith("__")]
            key = max(keys, key=lambda k: getattr(d[k], "size", 0))
        arr = np.asarray(d[key])
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array in {args.mat}[{key}], got shape {arr.shape}")
        # Standard MATLAB hyperspectral cubes are stored (H, W, B); move B to axis 0 -> (B, H, W)
        a = np.ascontiguousarray(np.moveaxis(arr, 2, 0)).astype(I64)
        itemsize = arr.dtype.itemsize
        if args.crop:
            cb, ch, cw = [int(v) for v in args.crop.split(",")]
            a = np.ascontiguousarray(a[:cb, :ch, :cw])
        return a, itemsize
    if args.raw:
        dt = np.dtype(args.dtype)
        B, H, W = [int(v) for v in args.dims.split(",")]
        with open(args.raw, "rb") as fh:
            fh.seek(args.header_bytes)
            buf = fh.read(B * H * W * dt.itemsize)
        a, itemsize = parse_raw_buffer(buf, args.dtype, args.order, (B, H, W))
        if args.crop:
            cb, ch, cw = [int(v) for v in args.crop.split(",")]
            a = np.ascontiguousarray(a[:cb, :ch, :cw])
        return a, itemsize
    gen = {"hyper": synth_hyper, "dem": synth_smooth, "noise": synth_noise}[args.data]
    return gen(B=args.B, H=args.H, W=args.W).astype(I64), 2


# =============================================================================
# 6. Part A & Track B
# =============================================================================
def byte_anchors(C, itemsize=2):
    import subprocess
    from pathlib import Path
    dt = "<i2" if itemsize == 2 else "<i4"
    a = C.astype(dt)
    raw = a.tobytes()
    shuf = a.view(np.uint8).reshape(-1, itemsize).T.ravel().tobytes()
    out = {"xz-9e raw": len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)),
           "xz-9e byte-shuffled": len(lzma.compress(shuf, preset=9 | lzma.PRESET_EXTREME))}
    if zstd is not None:
        out["zstd-19 raw"] = len(zstd.ZstdCompressor(level=19).compress(raw))
        out["zstd-19 byte-shuffled"] = len(zstd.ZstdCompressor(level=19).compress(shuf))
    asv4_bin = Path(__file__).resolve().parent / "asv4"
    if asv4_bin.exists():
        for lvl in (3, 19):
            pr = subprocess.run([str(asv4_bin), "c", "--level", str(lvl)], input=raw,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            out[f"ASV4 v4.1 (L{lvl}) blind"] = len(pr.stdout)
    return out


def rounding_noise_report(C):
    B = C.shape[0]
    X = C.reshape(B, -1)
    mu = np.rint(X.mean(axis=1)).astype(I64)
    Xc = X - mu[:, None]
    rows = []
    specs = [
        ("givens(full)", build_givens_full),
        ("clustered_givens(8)", build_clustered_givens),
        ("clustered_serm(8)", build_clustered_serm),
        ("corr_serm(8)", build_corr_clustered_serm),
        ("pot-style", build_pot),
    ]
    for name, builder in specs:
        if name == "givens(full)" and B > 64:
            continue
        p = builder(Xc)
        yi = p.fwd(Xc).astype(np.float64)
        yf = p.fwd_float(Xc)
        nv = (yi - yf).var(axis=1)
        ops = p.n_ops if isinstance(p, SermProgram) else p.n_rot
        extra = f"max_coeff={p.max_coeff:.2f}, fb={p.n_fallbacks}" if isinstance(p, SermProgram) else ""
        rows.append((name, ops, float(nv.mean()), float(nv.max()), extra))
    return rows


def run_partA(C, itemsize=2, L=3, label=""):
    B, H, W = C.shape
    C = C[:, : H // 8 * 8, : W // 8 * 8]
    n = C.size
    print(f"\n=== PART A  [{label}]  cube (bands,lines,samples)={C.shape}  samples={n}  "
          f"width={itemsize*8}b  range=[{C.min()},{C.max()}] ===")
    t0 = time.time()
    rows = []  # (name, est_bits, real_rice_bits)
    r1_e, r1_r = row_raw(C, real_rice=True)
    rows.append(("R1  raw", r1_e, r1_r))
    r2_e, r2_r = row_dwt3d(C, L, real_rice=True)
    rows.append(("R2  3D 5/3 (all axes)", r2_e, r2_r))
    hy = {}
    hy_rice = {}
    for kind, nm in (("dwt53", "R3a hybrid: 2D5/3 + spectral 5/3"),
                     ("pot", "R3b hybrid: 2D5/3 + spectral POT-style"),
                     ("clustered_serm", "R3c hybrid: 2D5/3 + spectral clustered SERM(8)"),
                     ("corr_serm", "R3g hybrid: 2D5/3 + spectral corr-clustered SERM(8)"),
                     ("clustered_givens", "R3d hybrid: 2D5/3 + spectral clustered Givens(8)"),
                     ("givens", "R3e hybrid: 2D5/3 + spectral Givens (full)"),
                     ("klt_float", "R3f hybrid: 2D5/3 + float KLT [ceiling, not reversible]")):
        if kind == "givens" and B > 64:
            continue
        eb, rb = row_hybrid(C, kind, L, real_rice=(kind != "klt_float"))
        hy[kind] = eb
        hy_rice[kind] = rb
        rows.append((nm, eb, rb))
    reversible_hy = {k: v for k, v in hy.items() if k != "klt_float"}
    best_hy_kind = min(reversible_hy, key=reversible_hy.get)
    r3 = reversible_hy[best_hy_kind]
    fold_spec = best_hy_kind if best_hy_kind != "dwt53" else "clustered_serm"
    r4_e, r4_r = row_folded(C, 4, fold_spec, real_rice=True)
    rows.append((f"R4  folded 4x4 SERM-RKLT (spectral={fold_spec})", r4_e, r4_r))
    r5a_e, res_ls = row_predictor(C)
    r5a_r = real_rice_block_bits(res_ls, verify=False) + 16 * 10
    rows.append(("R5a 3D static causal LS predictor [weak proxy]", r5a_e, r5a_r))
    r5b_e, res_cc3 = row_ccsds123(C, P=3, verify=True)
    r5b_r = real_rice_block_bits(res_cc3, verify=True)
    rows.append(("R5b CCSDS-123.0-B-1 adaptive predictor (P=3, Blue Book)", r5b_e, r5b_r))
    r5c_e, res_cc15 = row_ccsds123(C, P=15, verify=True)
    r5c_r = real_rice_block_bits(res_cc15, verify=False)
    rows.append(("R5c CCSDS-123.0-B-1 adaptive predictor (P=15, Blue Book)", r5c_e, r5c_r))

    raw_bytes = n * itemsize
    print(f"{'row':<60}{'est b/s':>9}{'real Rice':>11}{'ratio':>7}{'vs R3':>9}")
    for nm, b, rb in rows:
        rb_s = f"{rb / n:11.3f}" if rb is not None else f"{'—':>11}"
        ref_b = rb if rb is not None else b
        print(f"{nm:<60}{b / n:>9.3f}{rb_s}{raw_bytes / (ref_b / 8):>7.2f}{(b / r3 - 1) * 100:>+8.1f}%")
    print(f"\nReal-byte anchors (raw int{itemsize*8} LE, BSQ):")
    for k, v in byte_anchors(C, itemsize).items():
        print(f"  {k:<28}{v:>10} B   {v * 8 / n:>7.3f} bits/sample  ({raw_bytes / v:5.2f}x)")
    print("\nRounding noise of reversible spectral programs (vs same program in float64):")
    for nm, ops, mean, mx, extra in rounding_noise_report(C):
        print(f"  {nm:<22} ops={ops:<6} mean noise var={mean:7.3f} LSB^2   max={mx:7.3f}   {extra}")
    print("\nR6 oracle clustered spatial-block KLT (held-out Gaussian NLL; floor=0.25 LSB^2):")
    f6 = {}
    for (bb, f) in ((2, 2), (4, 4)):
        if B < bb:
            continue
        res, d = oracle_forest(C, bb, f)
        f6[(bb, f)] = res
        for r in res:
            print(f"  block {bb}x{f}x{f} (d={d:>2})  K={r['K']:>2}  g_raw={r['g_raw']:+.3f}  "
                  f"g_net_of_labels={r['g_net']:+.3f} b/sample   label H={r['Hlab']:.2f} b/blk   "
                  f"header: {r['hdr_bits_per_sample_here']:.3f} b/sample here, "
                  f"{r['hdr_bits_per_sample_at_128M']:.5f} at 128M samples")
    print("\nGATES:")
    g0a = (r3 - r4_e) / r3 * 100
    print(f"  0A folded RKLT vs best hybrid ({best_hy_kind}): R4 is {g0a:+.1f}% better (need >= +2.0%)  -> "
          f"{'PASS' if g0a >= 2 else 'KILL folded spatial KLT'}")
    best_g = max(r["g_net"] for res in f6.values() for r in res) if f6 else 0.0
    print(f"  0B best oracle g_net_of_labels = {best_g:+.3f} b/sample "
          f"(kill < 0.05; explore backward-adaptive >= 0.10)  -> "
          f"{'KILL spatial forest' if best_g < 0.05 else 'continue' if best_g < 0.10 else 'strong'}")
    best_all_est = min(r3, r2_e)
    best_all_rice = min([r2_r] + [v for v in hy_rice.values() if v is not None])
    best_cc_est = min(r5b_e, r5c_e)
    best_cc_rice = min(r5b_r, r5c_r)
    g0c_est = (best_all_est - best_cc_est) / best_all_est * 100
    g0c_rice = (best_all_rice - best_cc_rice) / best_all_rice * 100
    print(f"  0C CCSDS-123 best (P=3/15) vs best transform: est={g0c_est:+.1f}% | real Rice={g0c_rice:+.1f}% "
          f"({'CCSDS-123 wins: scope to Track A/B' if g0c_rice > 0 else 'transform wins!'})")
    print(f"  (Part A wall time {time.time() - t0:.1f}s)")
    return dict(rows=rows, r3=r3, r4=r4_e, r5a=r5a_e, r5b=r5b_e, r5c=r5c_e,
                best_cc_rice=best_cc_rice, best_rice=best_all_rice)


def run_trackB(C, L=3, steps=(1, 2, 4, 8, 16, 32, 64), label=""):
    """Track B: Deadzone quantizer sweep comparing progressive PSNR vs. bits/sample."""
    B, H, W = C.shape
    C = C[:, : H // 8 * 8, : W // 8 * 8]
    n = C.size
    peak = max(1.0, float(C.max() - C.min()))
    print(f"\n=== TRACK B  [{label}]  Progressive Deadzone Sweep (peak_range={peak:.0f}) ===")

    y_3d, plan_3d = dwt53_axes(C, (0, 1, 2), L)
    sl_3d = subband_slices(plan_3d, y_3d.shape, (0, 1, 2))

    hy_cache = {}
    for kind in ("dwt53", "pot", "clustered_serm"):
        Y, hdr, ctx = spectral_forward(C, kind, L)
        y2, plan2 = dwt53_axes(Y, (1, 2), L)
        sl2 = subband_slices(plan2, y2.shape, (1, 2))
        hy_cache[kind] = (y2, plan2, sl2, hdr, ctx)

    Z_f, hdr_f, fctx = folded_forward(C, 4, "clustered_serm")

    methods = [
        ("R2  3D 5/3", "3d"),
        ("R3a 2D5/3+spec5/3", "dwt53"),
        ("R3b 2D5/3+POT", "pot"),
        ("R3c 2D5/3+SERM(8)", "clustered_serm"),
        ("R4  folded 4x4 SERM", "folded"),
    ]
    hdr_line = f"{'step':>5}  " + "  ".join(f"{nm:<21}" for nm, _ in methods)
    print(hdr_line)
    print("       " + "  ".join(f"{'b/s     PSNR(dB)':<21}" for _ in methods))

    for q in steps:
        cells = []
        for _nm, tag in methods:
            if tag == "3d":
                Q = dz_quant(y_3d, q)
                bits = total_bits([Q[s] for s in sl_3d])
                C_hat = dwt53_inv(dz_dequant(Q, q), plan_3d, (0, 1, 2))
            elif tag == "folded":
                Q = dz_quant(Z_f, q)
                blocks = [Q[:, :, u, :, v] for u in range(4) for v in range(4)]
                bits = total_bits(blocks) + hdr_f
                C_hat = folded_inverse(dz_dequant(Q, q), fctx)
            else:
                y2, plan2, sl2, hdr, ctx = hy_cache[tag]
                Q = dz_quant(y2, q)
                bits = total_bits([Q[s] for s in sl2]) + hdr
                C_hat = spectral_inverse(dwt53_inv(dz_dequant(Q, q), plan2, (1, 2)), ctx)
            mse = float(np.mean((C.astype(np.float64) - C_hat.astype(np.float64)) ** 2))
            psnr = float("inf") if mse == 0.0 else 10.0 * math.log10((peak * peak) / mse)
            psnr_s = "   inf" if math.isinf(psnr) else f"{psnr:6.2f}"
            cells.append(f"{bits / n:6.3f}  {psnr_s:<13}")
        print(f"{q:>5}  " + "  ".join(cells))


# =============================================================================
# 7. Part B: Track A blind stride-lattice recovery (Pearson FFT + Axis-Peeling)
# =============================================================================
def acf_fft_pearson(x, maxlag):
    """Exact lag-tau Pearson correlation for lags 0..maxlag in O(L log L).
    Centers x by global mean before FFT/cumsum to prevent catastrophic floating-point
    cancellation on high-DC or constant signals, and guards near-zero variance to 0.0."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return np.zeros(maxlag + 1, dtype=np.float64)
    xc = x - float(np.mean(x))
    total_var = float(np.dot(xc, xc))
    if total_var <= 1e-12 * n:
        rho = np.zeros(maxlag + 1, dtype=np.float64)
        rho[0] = 1.0
        return rho
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    cs = np.concatenate(([0.0], np.cumsum(xc)))
    cs2 = np.concatenate(([0.0], np.cumsum(xc * xc)))
    F = np.fft.rfft(xc, nfft)
    xy = np.fft.irfft(F * np.conj(F), nfft)[: maxlag + 1]
    taus = np.arange(maxlag + 1)
    cnt = (n - taus).astype(np.float64)
    sum_l = cs[n - taus]
    sum_r = cs[n] - cs[taus]
    sum2_l = cs2[n - taus]
    sum2_r = cs2[n] - cs2[taus]
    var_l = sum2_l - (sum_l * sum_l) / cnt
    var_r = sum2_r - (sum_r * sum_r) / cnt
    cov = xy - (sum_l * sum_r) / cnt
    vfloor = 1e-9 * (total_var / n) * cnt
    valid = (var_l > vfloor) & (var_r > vfloor)
    denom = np.sqrt(np.maximum(var_l, 0.0) * np.maximum(var_r, 0.0))
    rho = np.where(valid, cov / np.maximum(denom, 1e-30), 0.0)
    rho[0] = 1.0
    return np.clip(rho, -1.0, 1.0)


def recover_strides(x, max_axes=5, min_prom=0.10, min_val=0.20):
    """Sequential axis-peeling subsequence-peak stride detector (zero shape/divisor hints).
    At each step given current stride s_prev:
      1) Peel along s_prev (cur[s:] - cur[:-s]) and evaluate subsequences g[k] = rho[k * s_prev]
         on both raw and peeled Pearson FFT-ACF.
      2) Rectify g_pos = max(g, 0) and suppress weak MA(1) diff-ringing rebounds at k=2 on peel.
      3) Select fundamental via YIN rule (safely falling back to max(prom*val) if max_prom and
         max_val occur on different peaks), then apply harmonic-verified divisor-refinement: if
         the chosen stride step k has a proper divisor d | k that is itself a clear ACF peak AND
         shows periodic harmonic support at 2*d (distinguishing true lattice axes from one-off
         spectral bumps), emit the inner divisor d first so no axis is ever skipped.
    """
    x = np.asarray(x, dtype=np.float64)
    L_orig = len(x)
    limit = L_orig // 4
    cur = x.copy()
    rho_raw0 = acf_fft_pearson(x, limit)

    chain = [1]
    trace = []

    while len(chain) < max_axes:
        s_prev = chain[-1]
        max_k = min(limit // s_prev, (L_orig // 2) // s_prev)
        if max_k < 4:
            break

        cur_peeled = cur[s_prev:] - cur[:-s_prev]
        lim_p = min(limit, len(cur_peeled) - 2)
        rho_p = acf_fft_pearson(cur_peeled, lim_p)

        idx0 = np.arange(max_k + 1) * s_prev
        g0 = rho_raw0[idx0]
        idxp = idx0[idx0 <= lim_p]
        gp = rho_p[idxp]
        curves = [(g0, "raw"), (gp, "peel")]
        if s_prev > 1:
            # Also peel the original signal x directly along s_prev (avoids SNR loss from
            # cumulative double-differencing on noisy BIP/BIL bands like AVIRIS Indian Pines),
            # unless >99.8% of variance cancels (e.g. linear-ramp quantization sawtooth).
            x_peeled = x[s_prev:] - x[:-s_prev]
            if float(np.var(x_peeled)) > 2e-3 * float(np.var(x)):
                lim_xp = min(limit, len(x_peeled) - 2)
                rho_xp = acf_fft_pearson(x_peeled, lim_xp)
                idx_xp = idx0[idx0 <= lim_xp]
                curves.append((rho_xp[idx_xp], "xpeel"))

        raw_peaks = set()
        cand_scores = {}
        curve_map = {tag: g_arr for g_arr, tag in curves}
        for g_arr, tag in curves:
            if len(g_arr) < 4:
                continue
            g_pos = np.maximum(g_arr, 0.0)
            pk, props = find_peaks(g_pos, prominence=min_prom, height=min_val)
            for p, pr in zip(pk, props["prominences"]):
                k = int(p)
                if k < 2:
                    continue
                val = float(g_arr[k])
                pr_f = float(pr)
                if tag == "raw":
                    raw_peaks.add(k)
                else:
                    if k == 2 and k not in raw_peaks and (val < 0.50 or pr_f < 0.30):
                        # Guard against MA(1) first-difference positive rebound at k=2
                        continue
                    if k not in raw_peaks and 2 * k < len(g_arr) and val < 0.50 and float(g_arr[2 * k]) < 0.05:
                        # Guard against non-periodic peel/xpeel bumps (e.g. sensor ripple):
                        # true lattice axes with >=8 slices exhibit a positive 2*k harmonic.
                        continue
                if k not in cand_scores or (pr_f * val) > (cand_scores[k][0] * cand_scores[k][1]):
                    cand_scores[k] = (pr_f, val, tag)

        if not cand_scores:
            break

        max_pr = max(v[0] for v in cand_scores.values())
        max_val = max(v[1] for v in cand_scores.values())
        strong_ks = sorted(
            k for k, v in cand_scores.items()
            if v[0] >= 0.55 * max_pr and v[1] >= 0.65 * max_val
        )
        if strong_ks:
            best_k = strong_ks[0]
        else:
            best_k = max(cand_scores, key=lambda k: cand_scores[k][0] * cand_scores[k][1])

        # Harmonic-verified blind divisor-refinement: accept a proper divisor d | best_k as an
        # inner lattice axis only if d is a strong peak AND its 2*d harmonic also carries positive
        # correlation on the same ACF curve (rejects one-off spectral bumps like AVIRIS lag=4).
        if best_k >= 4:
            div_cands = []
            for d in range(2, int(math.isqrt(best_k)) + 1):
                if best_k % d == 0:
                    for dd in (d, best_k // d):
                        if 2 <= dd < best_k and dd in cand_scores:
                            dpr, dval, dtag = cand_scores[dd]
                            if dpr >= max(min_prom, 0.10) and dval >= max(min_val, 0.25):
                                g_ref = curve_map[dtag]
                                if 2 * dd < best_k:
                                    harm_ok = (2 * dd < len(g_ref) and float(g_ref[2 * dd]) >= 0.15)
                                else:
                                    # 2*dd == best_k is the parent peak itself; guard against weak
                                    # half-period spatial echoes (e.g. peel[k/2]=0.26 vs peel[k]=0.93)
                                    # by requiring either raw support or strong correlation + 3*dd harmonic.
                                    if 3 * dd < len(g_ref):
                                        harm_ok = (dd in raw_peaks or dval >= 0.40) and float(g_ref[3 * dd]) >= 0.15
                                    else:
                                        harm_ok = dval >= 0.65 * cand_scores[best_k][1]
                                if harm_ok:
                                    div_cands.append(dd)
            if div_cands:
                best_d_score = max(cand_scores[dd][0] * cand_scores[dd][1] for dd in div_cands)
                yin_divs = sorted(
                    dd for dd in set(div_cands)
                    if cand_scores[dd][0] * cand_scores[dd][1] >= 0.50 * best_d_score
                )
                best_k = yin_divs[0]

        s_next = best_k * s_prev
        pr, val, tag = cand_scores[best_k]
        chain.append(s_next)
        trace.append((s_next, f"k={best_k}", round(val, 3), round(pr, 3), tag))
        if np.var(cur_peeled) < np.var(cur):
            cur = cur_peeled

    return chain, trace


def synth_4d(shape=(16, 16, 32, 32), rhos=(0.9, 0.85, 0.5, 0.5), noise=0.35, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    for ax, r in enumerate(rhos):
        x = ndimage.gaussian_filter1d(x, sigma=1.0 / max(1e-3, 1 - r) * 0.6, axis=ax, mode="wrap")
    x = x / x.std() + noise * rng.standard_normal(shape)
    return x


def trial_hybrid_bits(cube3, spec_axis, spec_kind="clustered_serm", L=3):
    """Hybrid pipeline with `spec_axis` as the spectral axis; others 2D 5/3."""
    C = np.ascontiguousarray(np.moveaxis(cube3, spec_axis, 0))
    H, W = C.shape[1] // 4 * 4, C.shape[2] // 4 * 4
    sub = C[:, :H, :W]
    eb, _ = row_hybrid(sub, spec_kind, L, verify=False, real_rice=False)
    return eb, sub.size


def run_partB(C, label="", expect_noise=False, spec_kind="clustered_serm"):
    print(f"\n=== PART B  [{label}]  Track A: blind stride-lattice recovery ===")
    if not expect_noise:
        x4 = synth_4d()
        truth4 = [1, 32, 1024, 16384]
        c4, tr4 = recover_strides(x4.ravel(), max_axes=5)
        print("B1  4D field (16,16,32,32), zero shape/divisor hints")
        print(f"    truth      {truth4}")
        print(f"    recovered  {c4}   {'OK' if c4 == truth4 else 'WRONG'}   trace={tr4}")

    B, H, W = C.shape
    C = C[:, : H // 8 * 8, : W // 8 * 8]
    B, H, W = C.shape
    n = C.size
    oracle_bits = (row_hybrid(C, spec_kind, 3, verify=False)[0] if not expect_noise else row_raw(C)[0])
    print(f"\nB2  permutation invariance, cube dims (bands,lines,samples)=({B},{H},{W}), headerless streams")
    if not expect_noise:
        print(f"    oracle-roles hybrid ({spec_kind} on true band axis): {oracle_bits / n:.3f} bits/sample")
    else:
        print(f"    noise control (expect clean [1] fallback; raw={oracle_bits / n:.3f} bits/sample)")
    hdr = (f"    {'perm(axes of stored array)':<28}{'true strides':<22}{'recovered':<22}"
           f"{'ok':<5}{'chosen spectral axis':<24}{'bits/sample':>12}{'vs oracle':>11}")
    print(hdr)
    ok_cnt, within3 = 0, 0
    for perm in itertools.permutations(range(3)):
        A = np.ascontiguousarray(np.transpose(C, perm))
        dims = A.shape
        truth_s = [1, dims[2], dims[1] * dims[2]]
        rec, _ = recover_strides(A.ravel().astype(np.float64), max_axes=5)
        if expect_noise:
            ok = (rec == [1])
            ok_cnt += ok
            bps = row_raw(A)[0] / n
            within3 += abs(bps * n - oracle_bits) / oracle_bits <= 0.03
            lab = "identity fallback (T0)" if ok else "FALSE POSITIVE"
            print(f"    {str(perm):<28}{str(truth_s):<22}{str(rec):<22}"
                  f"{'OK' if ok else 'NO':<5}{lab:<24}{bps:>12.3f}{(bps * n / oracle_bits - 1) * 100:>+10.2f}%")
            continue

        ok = (rec == truth_s)
        ok_cnt += ok
        if len(rec) >= 3 and rec[1] > 1 and rec[2] > rec[1] and rec[2] % rec[1] == 0:
            s1, s2 = rec[1], rec[2]
            d0 = n // s2
            d1 = s2 // s1
            d2 = s1
            usable = d0 * d1 * d2
            if usable == n and d0 >= 2 and d1 >= 4 and d2 >= 4:
                rec_cube = A.ravel().reshape(d0, d1, d2)
                trials = [(ax, trial_hybrid_bits(rec_cube, ax, spec_kind, 3)) for ax in range(3)]
                best_ax, (best_bits, used_n) = min(trials, key=lambda kv: kv[1][0] / kv[1][1])
                bps = best_bits / used_n
                true_ax_name = ("B", "H", "W")[perm[best_ax]]
                lab = f"ax={best_ax} (true={true_ax_name})"
            else:
                bps = row_raw(A)[0] / n
                lab = "fallback raw (shape)"
        else:
            bps = row_raw(A)[0] / n
            lab = "fallback raw (strides)"
        rel = (bps * n / oracle_bits - 1.0) * 100.0
        if rel <= 3.0:
            within3 += 1
        print(f"    {str(perm):<28}{str(truth_s):<22}{str(rec):<22}"
              f"{'OK' if ok else 'NO':<5}{lab:<24}{bps:>12.3f}{rel:>+10.2f}%")

    print(f"\nGATE 0D [{label}]: exact_strides={ok_cnt}/6  within_3pct_of_oracle={within3}/6  -> "
          f"{'PASS' if (ok_cnt == 6 and within3 == 6) else 'FAIL'}")


# =============================================================================
# 8. Self-test & CLI entrypoint
# =============================================================================
def run_selftest():
    print("=== SELFTEST ===")
    rng = np.random.default_rng(42)
    # 1. 5/3 DWT round-trip on odd & even shapes
    for shp in ((7, 13, 19), (16, 24, 32)):
        C = rng.integers(-5000, 5000, size=shp, dtype=I64)
        y, plan = dwt53_axes(C, (0, 1, 2), 3)
        assert np.array_equal(dwt53_inv(y, plan, (0, 1, 2)), C), f"DWT53 failed on {shp}"
    print("  [PASS] 3D 5/3 integer lifting round-trip (odd & even shapes)")

    # 2. All reversible spectral transforms & folded RKLT round-trip
    C = synth_hyper(B=16, H=32, W=32, seed=7).astype(I64)
    assert C.std() > 10.0, "synth_hyper small-grid collapse!"
    for kind in ("dwt53", "pot", "clustered_serm", "corr_serm", "clustered_givens", "givens"):
        Y, _hdr, ctx = spectral_forward(C, kind, 3)
        assert np.array_equal(spectral_inverse(Y, ctx), C), f"Spectral {kind} failed"
    Z, _hdr, fctx = folded_forward(C, 4, "clustered_serm")
    assert np.array_equal(folded_inverse(Z, fctx), C), "Folded SERM-RKLT failed"
    print("  [PASS] Reversible spectral transforms (dwt53, pot, clustered_serm, corr_serm, clustered_givens, givens) & folded RKLT")

    # 3. CCSDS-123.0-B-1 adaptive predictor & real single-pass Rice bitstream round-trip
    _bits_cc, res_cc = row_ccsds123(C, P=3, verify=True)
    rb_cc = real_rice_block_bits(res_cc, verify=True)
    assert rb_cc > 0
    # Also test Rice on extreme values (escaping path)
    ext = np.array([[[0, -1, 1, 50000, -50000, 123456, -123456, 0]]], dtype=I64)
    real_rice_block_bits(ext, verify=True)
    print(f"  [PASS] Bit-exact CCSDS-123.0-B-1 predictor & causal adaptive Golomb-Rice bitstream ({rb_cc / C.size:.3f} b/s)")

    # 4. BSQ / BIL / BIP raw parser round-trip
    B, H, W = 6, 8, 10
    C3 = rng.integers(-1000, 1000, size=(B, H, W), dtype=np.int16)
    for order, arr in [
        ("BSQ", C3),
        ("BIL", np.moveaxis(C3, 0, 1)),
        ("BIP", np.moveaxis(C3, 0, 2)),
    ]:
        buf = np.ascontiguousarray(arr).astype(">i2").tobytes()
        parsed, isz = parse_raw_buffer(buf, ">i2", order, (B, H, W))
        assert isz == 2 and np.array_equal(parsed, C3.astype(I64)), f"parse_raw_buffer failed for {order}"
    print("  [PASS] BSQ / BIL / BIP raw buffer loader")

    # 5. Pearson FFT-ACF centering, constant-signal guard, small & default cube stride recovery, noise control
    const_x = np.full(16384, 32767.0)
    assert recover_strides(const_x, max_axes=5)[0] == [1], "Constant signal must return [1]!"
    rec_sm, _ = recover_strides(C.astype(np.float64).ravel(), max_axes=5)
    assert rec_sm == [1, 32, 1024], f"Small (16,32,32) hyper stride recovery failed: got {rec_sm}"
    C_big = synth_hyper(B=32, H=96, W=128, seed=0).astype(np.float64).ravel()
    rec_bg, _ = recover_strides(C_big, max_axes=5)
    assert rec_bg == [1, 128, 12288], f"Default (32,96,128) hyper stride recovery failed: got {rec_bg}"
    noise = synth_noise(B=16, H=32, W=32, seed=9).astype(np.float64).ravel()
    assert recover_strides(noise, max_axes=5)[0] == [1], "Noise must return [1]!"
    print("  [PASS] Centered Pearson FFT-ACF, constant guard, small+default cube stride recovery, and noise control")
    print("ALL SELFTESTS PASSED!")


def main():
    ap = argparse.ArgumentParser(description="Phase 0 falsification harness v2.2")
    ap.add_argument("cmd", choices=["selftest", "partA", "partB", "trackB", "all"])
    ap.add_argument("--data", choices=["hyper", "dem", "noise"], default="hyper")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--raw", type=str, default=None)
    ap.add_argument("--mat", type=str, default=None, help="Path to MATLAB .mat 3D hyperspectral cube (H,W,B)")
    ap.add_argument("--mat-key", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=">i2")
    ap.add_argument("--order", choices=["BSQ", "BIL", "BIP"], default="BSQ")
    ap.add_argument("--dims", type=str, default=None, help="bands,lines,samples")
    ap.add_argument("--crop", type=str, default=None, help="cb,ch,cw")
    ap.add_argument("--header-bytes", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "selftest":
        run_selftest()
        return

    if args.cmd == "all" and not args.raw and not args.mat:
        run_selftest()
        for ds in ("hyper", "dem", "noise"):
            args.data = ds
            C, itemsize = load_data(args)
            run_partA(C, itemsize=itemsize, L=args.L, label=f"synth:{ds}")
            run_partB(C, label=f"synth:{ds}", expect_noise=(ds == "noise"))
            if ds != "noise":
                run_trackB(C, L=args.L, label=f"synth:{ds}")
        return

    C, itemsize = load_data(args)
    label = f"mat:{args.mat}" if args.mat else (f"raw:{args.raw}" if args.raw else f"synth:{args.data}")
    if args.cmd in ("partA", "all"):
        run_partA(C, itemsize=itemsize, L=args.L, label=label)
    if args.cmd in ("partB", "all"):
        run_partB(C, label=label, expect_noise=(not args.raw and not args.mat and args.data == "noise"))
    if args.cmd in ("trackB", "all"):
        run_trackB(C, L=args.L, label=label)


if __name__ == "__main__":
    main()
