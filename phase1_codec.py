#!/usr/bin/env python3
"""
Phase 1 End-to-End Hyperspectral & N-D Reversible Codec (.hnd1 container).

Architecture (directly derived from Phase 0 & Gate 1A/1B falsification):
  * Spectral transform: Complete-pivoted LDU-RKLT (exact P A Q = L D U with
    |L_ij|, |U_ij| <= 1 in Q15 int16 and balanced Daubechies-Sweldens 3-rounding
    diagonal ladder -> zero Givens fallbacks, O(1) rounding noise).
    Automatically selects Full B x B LDU-RKLT when header amortizes < 0.04 b/s,
    or Multi-PC Hierarchical Clustered LDU-RKLT (gs=16, keep_pcs=3) on smaller cubes.
  * Spatial transform: Arbitrary-dimension reversible integer 5/3 lifting (L=3).
  * Entropy coder: Real bit-exact 32-bit rANS (12-bit precision, 16-bit renorm)
    with 16-bin 3D causal context (2D spatial W/N/NW/NE + coarser-scale parent +
    EMA-normalized cross-band activity) + compact packed extra-bits stream.
  * Container (.hnd1): Coarse-to-fine subband framing (LL -> L3 -> L2 -> L1)
    enabling true prefix progressive decode from a single lossless bitstream,
    L2-weighted lossy quantization (q_base > 1), SHA-256 verification, and
    Track A blind headerless 1D-stream encode/decode via recover_strides().
"""
import argparse
import hashlib
import itertools
import math
import struct
import time
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.io import loadmat

import phase0_harness as p0

MAGIC = b"HND1"
VERSION = 1
FLAG_BLIND = 1 << 0
FLAG_LOSSY = 1 << 1
FLAG_RAW = 1 << 2

DTYPE_TO_CODE = {np.dtype("<i2"): 1, np.dtype("<u2"): 2, np.dtype("<i4"): 3}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}

SPEC_NONE = 0
SPEC_FULL_LDU = 1
SPEC_CLUSTERED_LDU = 2

# Q15 fixed point for |L_ij|, |U_ij| <= 1 -> fits cleanly in int16!
LDU_SHIFT = 15
LDU_SCALE = 1 << LDU_SHIFT       # 32768 (we clamp coeffs to [-32768..32767] -> int16!)
LDU_HALF = 1 << (LDU_SHIFT - 1)  # 16384

RANS_BITS = 12
RANS_M = 1 << RANS_BITS  # 4096
RANS_L = 1 << 16         # 65536
N_SYMS = 32
N_CTXS = 16


# =============================================================================
# 1. Complete-Pivoted LDU-RKLT (Q15 int16 L/U, Q15 int32 balanced diagonal ladder)
# =============================================================================
def complete_pivoted_ldu(A_in):
    """Factor any orthogonal M x M matrix A into exact reversible integer lifting:
       P * diag(sign0) * A * Q = L * diag(dsgn) * D_ladder * U
    with |L_ij| <= 1, |U_ij| <= 1, and greedy log-balanced diagonal ladder."""
    M = A_in.shape[0]
    A = A_in.astype(np.float64).copy()
    sign0 = np.ones(M, dtype=np.int8)
    if np.linalg.det(A) < 0:
        A[0] *= -1.0
        sign0[0] = -1

    rperm = np.arange(M, dtype=np.int32)
    cperm = np.arange(M, dtype=np.int32)
    L = np.eye(M, dtype=np.float64)
    U = np.eye(M, dtype=np.float64)
    d = np.zeros(M, dtype=np.float64)
    W = A.copy()
    for k in range(M):
        sub = np.abs(W[k:, k:])
        idx = int(np.argmax(sub))
        p = k + idx // (M - k)
        q = k + idx % (M - k)
        if p != k:
            W[[k, p], :] = W[[p, k], :]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]
            rperm[k], rperm[p] = rperm[p], rperm[k]
        if q != k:
            W[:, [k, q]] = W[:, [q, k]]
            if k > 0:
                U[:k, [k, q]] = U[:k, [q, k]]
            cperm[k], cperm[q] = cperm[q], cperm[k]
        piv = W[k, k]
        d[k] = piv
        if k + 1 < M:
            L[k + 1 :, k] = W[k + 1 :, k] / piv
            U[k, k + 1 :] = W[k, k + 1 :] / piv
            W[k + 1 :, k + 1 :] -= np.outer(L[k + 1 :, k], W[k, k + 1 :])

    dsgn = np.where(d >= 0, 1, -1).astype(np.int8)
    d_abs = np.abs(d)

    # Greedy log-balanced ordering keeps cumulative diagonal products near 1.0
    rem = list(range(M))
    dperm = []
    log_cur = 0.0
    log_d = np.log(np.maximum(d_abs, 1e-15))
    while rem:
        best_i = min(rem, key=lambda idx: abs(log_cur + log_d[idx]))
        dperm.append(best_i)
        log_cur += log_d[best_i]
        rem.remove(best_i)
    dperm = np.array(dperm, dtype=np.int32)
    c_chain = np.cumprod(d_abs[dperm])

    # Clamp L, U strict triangles to int16 [-32767, 32767] (exact Q15: 32767/32768 ~ 0.99997)
    Lq = np.clip(np.rint(L * LDU_SCALE), -32767, 32767).astype(np.int16)
    Uq = np.clip(np.rint(U * LDU_SCALE), -32767, 32767).astype(np.int16)
    cq = np.rint(c_chain[:-1] * LDU_SCALE).astype(np.int32)
    inv_cq_m1 = np.rint((1.0 / c_chain[:-1] - 1.0) * LDU_SCALE).astype(np.int32)
    cq_m1 = np.rint((c_chain[:-1] - 1.0) * LDU_SCALE).astype(np.int32)
    return dict(
        M=M, rperm=rperm, cperm=cperm, sign0=sign0, dsgn=dsgn, dperm=dperm,
        Lq=Lq, Uq=Uq, cq=cq, inv_cq_m1=inv_cq_m1, cq_m1=cq_m1,
    )


@njit(parallel=True, cache=True)
def _ldu_fwd_numba(X_in, rperm, cperm, sign0, dsgn, dperm, Lq, Uq, cq, inv_cq_m1, cq_m1):
    M, N = X_in.shape
    out = np.empty((M, N), dtype=np.int64)
    BLK = 1024
    n_blks = (N + BLK - 1) // BLK
    for bi in prange(n_blks):
        n0 = bi * BLK
        n1 = min(n0 + BLK, N)
        nb = n1 - n0
        Y = np.empty((M, nb), dtype=np.int64)
        for i in range(M):
            ci = cperm[i]
            for t in range(nb):
                Y[i, t] = X_in[ci, n0 + t]
        for i in range(M - 1):
            for t in range(nb):
                acc = np.int64(0)
                for j in range(i + 1, M):
                    acc += np.int64(Uq[i, j]) * Y[j, t]
                Y[i, t] += (acc + LDU_HALF) >> LDU_SHIFT
        for k in range(M - 1):
            a = dperm[k]
            b = dperm[k + 1]
            c_q = np.int64(cq[k])
            icm1 = np.int64(inv_cq_m1[k])
            cm1 = np.int64(cq_m1[k])
            for t in range(nb):
                x = Y[a, t]
                y = Y[b, t]
                y -= (c_q * x + LDU_HALF) >> LDU_SHIFT
                x += (icm1 * y + LDU_HALF) >> LDU_SHIFT
                y += x
                x += (cm1 * y + LDU_HALF) >> LDU_SHIFT
                Y[a, t] = x
                Y[b, t] = y
        for i in range(M):
            if dsgn[i] < 0:
                for t in range(nb):
                    Y[i, t] = -Y[i, t]
        for i in range(M - 1, 0, -1):
            for t in range(nb):
                acc = np.int64(0)
                for j in range(i):
                    acc += np.int64(Lq[i, j]) * Y[j, t]
                Y[i, t] += (acc + LDU_HALF) >> LDU_SHIFT
        for i in range(M):
            ri = rperm[i]
            s = np.int64(sign0[ri])
            for t in range(nb):
                out[ri, n0 + t] = Y[i, t] * s
    return out


@njit(parallel=True, cache=True)
def _ldu_inv_numba(Y_in, rperm, cperm, sign0, dsgn, dperm, Lq, Uq, cq, inv_cq_m1, cq_m1):
    M, N = Y_in.shape
    out = np.empty((M, N), dtype=np.int64)
    BLK = 1024
    n_blks = (N + BLK - 1) // BLK
    for bi in prange(n_blks):
        n0 = bi * BLK
        n1 = min(n0 + BLK, N)
        nb = n1 - n0
        Y = np.empty((M, nb), dtype=np.int64)
        for i in range(M):
            ri = rperm[i]
            s = np.int64(sign0[ri])
            for t in range(nb):
                Y[i, t] = Y_in[ri, n0 + t] * s
        for i in range(1, M):
            for t in range(nb):
                acc = np.int64(0)
                for j in range(i):
                    acc += np.int64(Lq[i, j]) * Y[j, t]
                Y[i, t] -= (acc + LDU_HALF) >> LDU_SHIFT
        for i in range(M):
            if dsgn[i] < 0:
                for t in range(nb):
                    Y[i, t] = -Y[i, t]
        for k in range(M - 2, -1, -1):
            a = dperm[k]
            b = dperm[k + 1]
            c_q = np.int64(cq[k])
            icm1 = np.int64(inv_cq_m1[k])
            cm1 = np.int64(cq_m1[k])
            for t in range(nb):
                x = Y[a, t]
                y = Y[b, t]
                x -= (cm1 * y + LDU_HALF) >> LDU_SHIFT
                y -= x
                x -= (icm1 * y + LDU_HALF) >> LDU_SHIFT
                y += (c_q * x + LDU_HALF) >> LDU_SHIFT
                Y[a, t] = x
                Y[b, t] = y
        for i in range(M - 2, -1, -1):
            for t in range(nb):
                acc = np.int64(0)
                for j in range(i + 1, M):
                    acc += np.int64(Uq[i, j]) * Y[j, t]
                Y[i, t] -= (acc + LDU_HALF) >> LDU_SHIFT
        for i in range(M):
            ci = cperm[i]
            for t in range(nb):
                out[ci, n0 + t] = Y[i, t]
    return out


def pack_ldu_stage(grp, f):
    """Serialize one LDU stage (group indices + perms + packed signs + strict L/U int16 + ladder)."""
    M = f["M"]
    buf = bytearray()
    buf += struct.pack("<H", M)
    buf += np.asarray(grp, dtype="<u2").tobytes()
    buf += f["rperm"].astype("<u2").tobytes()
    buf += f["cperm"].astype("<u2").tobytes()
    buf += f["dperm"].astype("<u2").tobytes()
    # Pack sign0 and dsgn as bit arrays
    s0_bits = np.packbits((f["sign0"] < 0).astype(np.uint8)).tobytes()
    ds_bits = np.packbits((f["dsgn"] < 0).astype(np.uint8)).tobytes()
    buf += s0_bits + ds_bits
    # Strict lower triangle of Lq and strict upper triangle of Uq (exact int16)
    li = np.tril_indices(M, k=-1)
    ui = np.triu_indices(M, k=1)
    buf += f["Lq"][li].astype("<i2").tobytes()
    buf += f["Uq"][ui].astype("<i2").tobytes()
    buf += f["cq"].astype("<i4").tobytes()
    buf += f["inv_cq_m1"].astype("<i4").tobytes()
    buf += f["cq_m1"].astype("<i4").tobytes()
    return bytes(buf)


def unpack_ldu_stage(mv, pos):
    M = struct.unpack_from("<H", mv, pos)[0]
    pos += 2
    grp = np.frombuffer(mv, dtype="<u2", count=M, offset=pos).astype(np.int64); pos += 2 * M
    rperm = np.frombuffer(mv, dtype="<u2", count=M, offset=pos).astype(np.int32); pos += 2 * M
    cperm = np.frombuffer(mv, dtype="<u2", count=M, offset=pos).astype(np.int32); pos += 2 * M
    dperm = np.frombuffer(mv, dtype="<u2", count=M, offset=pos).astype(np.int32); pos += 2 * M
    nb = (M + 7) // 8
    s0_b = np.unpackbits(np.frombuffer(mv, dtype=np.uint8, count=nb, offset=pos))[:M]; pos += nb
    ds_b = np.unpackbits(np.frombuffer(mv, dtype=np.uint8, count=nb, offset=pos))[:M]; pos += nb
    sign0 = np.where(s0_b == 1, -1, 1).astype(np.int8)
    dsgn = np.where(ds_b == 1, -1, 1).astype(np.int8)
    ntri = M * (M - 1) // 2
    Lq = np.zeros((M, M), dtype=np.int16)
    Uq = np.zeros((M, M), dtype=np.int16)
    if ntri > 0:
        Lq[np.tril_indices(M, k=-1)] = np.frombuffer(mv, dtype="<i2", count=ntri, offset=pos); pos += 2 * ntri
        Uq[np.triu_indices(M, k=1)] = np.frombuffer(mv, dtype="<i2", count=ntri, offset=pos); pos += 2 * ntri
    cq = np.frombuffer(mv, dtype="<i4", count=M - 1, offset=pos).astype(np.int32); pos += 4 * (M - 1)
    inv_cq_m1 = np.frombuffer(mv, dtype="<i4", count=M - 1, offset=pos).astype(np.int32); pos += 4 * (M - 1)
    cq_m1 = np.frombuffer(mv, dtype="<i4", count=M - 1, offset=pos).astype(np.int32); pos += 4 * (M - 1)
    f = dict(M=M, rperm=rperm, cperm=cperm, sign0=sign0, dsgn=dsgn, dperm=dperm,
             Lq=Lq, Uq=Uq, cq=cq, inv_cq_m1=inv_cq_m1, cq_m1=cq_m1)
    return grp, f, pos


def spectral_ldu_forward(C, mode="auto", gs=16, keep_pcs=3):
    """Apply mean-centering + complete-pivoted LDU-RKLT (full or multi-PC clustered).
    Returns (Y_cube, spec_mode_code, serialized_spectral_bytes)."""
    B = C.shape[0]
    N = C.size
    X = C.reshape(B, -1)
    mu = np.rint(X.mean(axis=1)).astype(np.int64)
    Xc = X - mu[:, None]

    if B <= 1 or mode == "none":
        return C.copy(), SPEC_NONE, b""

    if mode == "auto":
        # Choose full LDU-RKLT when full matrix header amortizes < 0.04 bits/sample
        full_hdr_bits = 16.0 * (B * (B - 1) + 12 * B)
        use_full = (B <= 16) or (full_hdr_bits / max(1, N) < 0.04)
        spec_code = SPEC_FULL_LDU if use_full else SPEC_CLUSTERED_LDU
    elif mode == "full":
        spec_code = SPEC_FULL_LDU
    elif mode == "clustered":
        spec_code = SPEC_CLUSTERED_LDU
    else:
        raise ValueError(f"Unknown spec mode: {mode}")

    stages = []
    Y_cur = np.ascontiguousarray(Xc, dtype=np.int64)

    def apply_grp(grp):
        if len(grp) <= 1:
            return
        sub = np.ascontiguousarray(Y_cur[grp])
        U = p0.klt_U(p0._sub_cols(sub))
        f = complete_pivoted_ldu(U)
        Y_cur[grp] = _ldu_fwd_numba(sub, f["rperm"], f["cperm"], f["sign0"], f["dsgn"], f["dperm"],
                                    f["Lq"], f["Uq"], f["cq"], f["inv_cq_m1"], f["cq_m1"])
        stages.append((grp, f))

    if spec_code == SPEC_FULL_LDU:
        apply_grp(list(range(B)))
    else:
        groups = [list(range(g, min(g + gs, B))) for g in range(0, B, gs)]
        for grp in groups:
            apply_grp(grp)
        for r in range(keep_pcs):
            heads = [g[r] for g in groups if len(g) > r]
            while len(heads) > 1:
                next_heads = []
                for g in range(0, len(heads), gs):
                    grp = heads[g : min(g + gs, len(heads))]
                    apply_grp(grp)
                    next_heads.append(grp[0])
                heads = next_heads

    buf = bytearray()
    buf += struct.pack("<HH", B, len(stages))
    buf += mu.astype("<i4").tobytes()
    for grp, f in stages:
        buf += pack_ldu_stage(grp, f)
    return Y_cur.reshape(C.shape), spec_code, bytes(buf)


def spectral_ldu_inverse(Y_cube, spec_code, mv, pos):
    if spec_code == SPEC_NONE:
        return Y_cube.copy(), pos
    B, n_stages = struct.unpack_from("<HH", mv, pos)
    pos += 4
    mu = np.frombuffer(mv, dtype="<i4", count=B, offset=pos).astype(np.int64); pos += 4 * B
    stages = []
    for _ in range(n_stages):
        grp, f, pos = unpack_ldu_stage(mv, pos)
        stages.append((grp, f))
    X = np.ascontiguousarray(Y_cube.reshape(B, -1), dtype=np.int64)
    for grp, f in reversed(stages):
        sub = np.ascontiguousarray(X[grp])
        X[grp] = _ldu_inv_numba(sub, f["rperm"], f["cperm"], f["sign0"], f["dsgn"], f["dperm"],
                                f["Lq"], f["Uq"], f["cq"], f["inv_cq_m1"], f["cq_m1"])
    X += mu[:, None]
    return X.reshape(Y_cube.shape), pos


# =============================================================================
# 2. Real Bit-Exact 32-bit rANS + 3D Causal Context Entropy Coder
# =============================================================================
@njit(cache=True)
def _val_to_tok_eb(v):
    u = (v << 1) if v >= 0 else ((-v) << 1) - 1
    if u < 4:
        return u, 0, 0
    if u >= (1 << 15):
        return 31, 32, u
    m = 0
    tmp = u
    while tmp > 1:
        tmp >>= 1
        m += 1
    b = (u >> (m - 1)) & 1
    tok = 2 * m + b
    eb = m - 1
    rem = u & ((1 << eb) - 1)
    return tok, eb, rem


@njit(cache=True)
def _tok_eb_to_val(tok, rem):
    if tok < 4:
        u = tok
    elif tok == 31:
        u = rem
    else:
        m = tok >> 1
        b = tok & 1
        eb = m - 1
        u = ((2 + b) << eb) | rem
    return (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)


@njit(cache=True)
def _act_to_ctx(act):
    if act <= 0: return 0
    if act == 1: return 1
    if act == 2: return 2
    if act == 3: return 3
    if act <= 5: return 4
    if act <= 8: return 5
    if act <= 12: return 6
    if act <= 18: return 7
    if act <= 27: return 8
    if act <= 42: return 9
    if act <= 64: return 10
    if act <= 100: return 11
    if act <= 160: return 12
    if act <= 260: return 13
    if act <= 450: return 14
    return 15


@njit(cache=True)
def _extract_3d_tokens(blk, parent, has_parent):
    B, H, W = blk.shape
    N = B * H * W
    toks = np.empty(N, dtype=np.uint8)
    ctxs = np.empty(N, dtype=np.uint8)
    ebs = np.empty(N, dtype=np.uint8)
    rems = np.empty(N, dtype=np.int64)
    counts = np.zeros((N_CTXS, N_SYMS), dtype=np.int64)
    pH = parent.shape[1] if has_parent else 0
    pW = parent.shape[2] if has_parent else 0

    idx = 0
    prev_ema = 16
    for b in range(B):
        cur_ema = 16
        for y in range(H):
            py = y >> 1
            if has_parent and py >= pH:
                py = pH - 1
            for x in range(W):
                sp_sum = 0
                sp_w = 0
                if x > 0:
                    sp_sum += 2 * abs(blk[b, y, x - 1])
                    sp_w += 2
                if y > 0:
                    sp_sum += 2 * abs(blk[b, y - 1, x])
                    sp_w += 2
                    if x > 0:
                        sp_sum += abs(blk[b, y - 1, x - 1])
                        sp_w += 1
                    if x + 1 < W:
                        sp_sum += abs(blk[b, y - 1, x + 1])
                        sp_w += 1
                a_sp = (sp_sum // sp_w) if sp_w > 0 else 0
                a_par = 0
                if has_parent:
                    px = x >> 1
                    if px >= pW:
                        px = pW - 1
                    a_par = abs(parent[b, py, px]) >> 1
                a_xb = 0
                if b > 0:
                    a_xb = (abs(blk[b - 1, y, x]) * cur_ema) // prev_ema

                if has_parent and b > 0:
                    act = (3 * a_sp + a_par + a_xb) // 5
                elif has_parent:
                    act = (3 * a_sp + a_par) >> 2
                elif b > 0:
                    act = (3 * a_sp + a_xb) >> 2
                else:
                    act = a_sp

                c = _act_to_ctx(act)
                v = blk[b, y, x]
                tok, eb, rem = _val_to_tok_eb(v)
                toks[idx] = tok
                ctxs[idx] = c
                ebs[idx] = eb
                rems[idx] = rem
                counts[c, tok] += 1
                idx += 1

                av = abs(v) << 4
                cur_ema = cur_ema + ((av - cur_ema) >> 6)
                if cur_ema < 4:
                    cur_ema = 4
        prev_ema = cur_ema
    return toks, ctxs, ebs, rems, counts


@njit(cache=True)
def _rans_encode_numba(toks, ctxs, ebs, rems, freqs, cdfs):
    N = len(toks)
    words = np.empty(N + 16, dtype=np.uint16)
    wpos = 0
    state = np.uint64(RANS_L)
    shift_lim = np.uint64((RANS_L >> RANS_BITS) << 16)

    for i in range(N - 1, -1, -1):
        c = ctxs[i]
        s = toks[i]
        f = np.uint64(freqs[c, s])
        start = np.uint64(cdfs[c, s])
        x_max = shift_lim * f
        while state >= x_max:
            words[wpos] = np.uint16(state & np.uint64(0xFFFF))
            wpos += 1
            state >>= np.uint64(16)
        state = ((state // f) << np.uint64(RANS_BITS)) + (state % f) + start

    words[wpos] = np.uint16(state & np.uint64(0xFFFF))
    wpos += 1
    words[wpos] = np.uint16((state >> np.uint64(16)) & np.uint64(0xFFFF))
    wpos += 1

    tot_eb = 0
    for i in range(N):
        tot_eb += ebs[i]
    eb_bytes = np.zeros((tot_eb + 7) // 8 + 1, dtype=np.uint8)
    bpos = 0
    cur = 0
    nb = 0
    for i in range(N):
        eb = ebs[i]
        if eb > 0:
            val = rems[i]
            for k in range(eb - 1, -1, -1):
                cur = (cur << 1) | ((val >> k) & 1)
                nb += 1
                if nb == 8:
                    eb_bytes[bpos] = cur
                    bpos += 1
                    cur = 0
                    nb = 0
    if nb > 0:
        eb_bytes[bpos] = cur << (8 - nb)
        bpos += 1
    return words[:wpos], eb_bytes[:bpos]


@njit(cache=True)
def _rans_decode_3d_numba(B, H, W, parent, has_parent, words, eb_bytes, freqs, cdfs):
    lut = np.empty((N_CTXS, RANS_M), dtype=np.uint8)
    for c in range(N_CTXS):
        for s in range(N_SYMS):
            lo = cdfs[c, s]
            hi = cdfs[c, s + 1]
            for slot in range(lo, hi):
                lut[c, slot] = s

    out = np.empty((B, H, W), dtype=np.int64)
    wpos = len(words) - 1
    hi_w = np.uint64(words[wpos]); wpos -= 1
    lo_w = np.uint64(words[wpos]); wpos -= 1
    state = (hi_w << np.uint64(16)) | lo_w

    bpos = 0
    bit_i = 0
    pH = parent.shape[1] if has_parent else 0
    pW = parent.shape[2] if has_parent else 0
    mask_m = np.uint64(RANS_M - 1)

    prev_ema = 16
    for b in range(B):
        cur_ema = 16
        for y in range(H):
            py = y >> 1
            if has_parent and py >= pH:
                py = pH - 1
            for x in range(W):
                sp_sum = 0
                sp_w = 0
                if x > 0:
                    sp_sum += 2 * abs(out[b, y, x - 1])
                    sp_w += 2
                if y > 0:
                    sp_sum += 2 * abs(out[b, y - 1, x])
                    sp_w += 2
                    if x > 0:
                        sp_sum += abs(out[b, y - 1, x - 1])
                        sp_w += 1
                    if x + 1 < W:
                        sp_sum += abs(out[b, y - 1, x + 1])
                        sp_w += 1
                a_sp = (sp_sum // sp_w) if sp_w > 0 else 0
                a_par = 0
                if has_parent:
                    px = x >> 1
                    if px >= pW:
                        px = pW - 1
                    a_par = abs(parent[b, py, px]) >> 1
                a_xb = 0
                if b > 0:
                    a_xb = (abs(out[b - 1, y, x]) * cur_ema) // prev_ema

                if has_parent and b > 0:
                    act = (3 * a_sp + a_par + a_xb) // 5
                elif has_parent:
                    act = (3 * a_sp + a_par) >> 2
                elif b > 0:
                    act = (3 * a_sp + a_xb) >> 2
                else:
                    act = a_sp

                c = _act_to_ctx(act)
                slot = state & mask_m
                s = lut[c, slot]
                f = np.uint64(freqs[c, s])
                start = np.uint64(cdfs[c, s])
                state = f * (state >> np.uint64(RANS_BITS)) + slot - start
                while state < np.uint64(RANS_L) and wpos >= 0:
                    state = (state << np.uint64(16)) | np.uint64(words[wpos])
                    wpos -= 1

                if s < 4:
                    rem = np.int64(0)
                else:
                    eb = 32 if s == 31 else ((s >> 1) - 1)
                    rem = np.int64(0)
                    for _ in range(eb):
                        bit = (int(eb_bytes[bpos]) >> (7 - bit_i)) & 1
                        rem = (rem << 1) | bit
                        bit_i += 1
                        if bit_i == 8:
                            bit_i = 0
                            bpos += 1
                v = _tok_eb_to_val(s, rem)
                out[b, y, x] = v

                av = abs(v) << 4
                cur_ema = cur_ema + ((av - cur_ema) >> 6)
                if cur_ema < 4:
                    cur_ema = 4
        prev_ema = cur_ema
    return out


def _normalize_and_pack_tables(counts):
    freqs = np.zeros((N_CTXS, N_SYMS), dtype=np.int32)
    cdfs = np.zeros((N_CTXS, N_SYMS + 1), dtype=np.int32)
    tbuf = bytearray()
    active_ctx_mask = 0
    ctx_payloads = []
    for c in range(N_CTXS):
        row = counts[c]
        tot = int(row.sum())
        if tot == 0:
            freqs[c, 0] = RANS_M
            cdfs[c, 1:] = RANS_M
            continue
        active_ctx_mask |= (1 << c)
        nz = np.nonzero(row)[0]
        f = np.maximum(1, np.rint(row[nz] * (RANS_M / tot)).astype(np.int32))
        diff = int(f.sum()) - RANS_M
        while diff != 0:
            if diff > 0:
                cand = np.where(f > 1)[0]
                j = cand[int(np.argmax(f[cand]))]
                f[j] -= 1
                diff -= 1
            else:
                j = int(np.argmax(row[nz]))
                f[j] += 1
                diff += 1
        freqs[c, nz] = f
        cdfs[c, 1:] = np.cumsum(freqs[c])
        sym_mask = 0
        for s in nz:
            sym_mask |= (1 << int(s))
        # Pack nonzero frequencies (last one implied = RANS_M - sum(prev)) as uint16
        # (compact & exact; max 31 * 2 = 62 bytes per active context)
        p = bytearray(struct.pack("<I", sym_mask))
        if len(nz) > 1:
            p += f[:-1].astype("<u2").tobytes()
        ctx_payloads.append(bytes(p))
    tbuf += struct.pack("<H", active_ctx_mask)
    for p in ctx_payloads:
        tbuf += p
    return freqs, cdfs, bytes(tbuf)


def _unpack_tables(mv, pos):
    freqs = np.zeros((N_CTXS, N_SYMS), dtype=np.int32)
    cdfs = np.zeros((N_CTXS, N_SYMS + 1), dtype=np.int32)
    active_ctx_mask = struct.unpack_from("<H", mv, pos)[0]
    pos += 2
    for c in range(N_CTXS):
        if not ((active_ctx_mask >> c) & 1):
            freqs[c, 0] = RANS_M
            cdfs[c, 1:] = RANS_M
            continue
        sym_mask = struct.unpack_from("<I", mv, pos)[0]
        pos += 4
        nz = [s for s in range(N_SYMS) if (sym_mask >> s) & 1]
        if len(nz) == 1:
            freqs[c, nz[0]] = RANS_M
        else:
            vals = np.frombuffer(mv, dtype="<u2", count=len(nz) - 1, offset=pos).astype(np.int32)
            pos += 2 * (len(nz) - 1)
            for s, fv in zip(nz[:-1], vals):
                freqs[c, s] = fv
            freqs[c, nz[-1]] = RANS_M - int(vals.sum())
        cdfs[c, 1:] = np.cumsum(freqs[c])
    return freqs, cdfs, pos


def pack_rans_subband(blk, parent=None):
    blk64 = np.ascontiguousarray(blk, dtype=np.int64)
    has_p = parent is not None
    par64 = np.ascontiguousarray(parent, dtype=np.int64) if has_p else np.zeros((1, 1, 1), dtype=np.int64)
    toks, ctxs, ebs, rems, counts = _extract_3d_tokens(blk64, par64, has_p)
    freqs, cdfs, tbuf = _normalize_and_pack_tables(counts)
    words, eb_bytes = _rans_encode_numba(toks, ctxs, ebs, rems, freqs, cdfs)
    payload = tbuf + struct.pack("<II", len(words), len(eb_bytes)) + words.astype("<u2").tobytes() + eb_bytes.tobytes()
    return struct.pack("<I", len(payload)) + payload


def unpack_rans_subband(mv, pos, shape, parent=None):
    plen = struct.unpack_from("<I", mv, pos)[0]
    pos += 4
    end_pos = pos + plen
    freqs, cdfs, pos = _unpack_tables(mv, pos)
    n_words, n_eb = struct.unpack_from("<II", mv, pos)
    pos += 8
    words = np.frombuffer(mv, dtype="<u2", count=n_words, offset=pos).copy()
    pos += 2 * n_words
    eb_bytes = np.frombuffer(mv, dtype=np.uint8, count=n_eb, offset=pos).copy()
    pos += n_eb
    assert pos == end_pos, f"Subband framing mismatch: {pos} != {end_pos}"
    has_p = parent is not None
    par64 = np.ascontiguousarray(parent, dtype=np.int64) if has_p else np.zeros((1, 1, 1), dtype=np.int64)
    out = _rans_decode_3d_numba(shape[0], shape[1], shape[2], par64, has_p, words, eb_bytes, freqs, cdfs)
    return out, end_pos


# =============================================================================
# 3. Exact 2D 5/3 L2 Synthesis Norms & Coarse-to-Fine Subband Ordering
# =============================================================================
_L2_GAIN_CACHE = {}


def get_2d53_l2_gains(L):
    """Exact L2 synthesis norm for each subband in coarse-to-fine order:
    [LL, L_L(LH,HL,HH), ..., L1(LH,HL,HH)]."""
    if L in _L2_GAIN_CACHE:
        return _L2_GAIN_CACHE[L]
    if L == 0:
        _L2_GAIN_CACHE[0] = [1.0]
        return [1.0]
    H = W = max(64, 1 << (L + 4))
    dummy = np.zeros((1, H, W), dtype=np.int64)
    _, plan = p0.dwt53_axes(dummy, (1, 2), L)
    slices_f2c = p0.subband_slices(plan, (1, H, W), (1, 2))
    # Convert fine-to-coarse [L1(3), L2(3), ..., LL] -> coarse-to-fine [LL, L_L(3), ..., L1(3)]
    c2f_slices = [slices_f2c[-1]]
    for lv in range(len(plan) - 1, -1, -1):
        c2f_slices.extend(slices_f2c[3 * lv : 3 * lv + 3])
    M = 10000
    gains = []
    for sl in c2f_slices:
        z = np.zeros((1, H, W), dtype=np.int64)
        sub = z[sl]
        z_sub = sub
        z_sub[0, sub.shape[1] // 2, sub.shape[2] // 2] = M
        rec = p0.dwt53_inv(z, plan, (1, 2)).astype(np.float64) / M
        gains.append(math.sqrt(float(np.sum(rec * rec))))
    _L2_GAIN_CACHE[L] = gains
    return gains


def _build_plan_from_shape(shape, L):
    """Deterministic 2D 5/3 plan from cube shape (B, H, W) and level count L."""
    cur = list(shape)
    plan = []
    for _ in range(L):
        before = list(cur)
        if before[1] < 2 and before[2] < 2:
            break
        if before[1] >= 2:
            cur[1] = (before[1] + 1) // 2
        if before[2] >= 2:
            cur[2] = (before[2] + 1) // 2
        plan.append(before)
    return plan


def _c2f_subband_schedule(plan, shape):
    """Return list of (slice_tuple, level_idx, parent_idx_or_None) in coarse-to-fine order.
    level_idx: 0 = LL, 1 = coarsest detail (L_max), ..., n_lv = finest detail (L1)."""
    slices_f2c = p0.subband_slices(plan, shape, (1, 2))
    n_lv = len(plan)
    if n_lv == 0:
        return [(slices_f2c[0], 0, None)]
    # Check each level produced 3 detail bands (both H>=2 and W>=2)
    # Handle general case safely using exact slice grouping per level
    per_lv = []
    idx = 0
    for before in plan:
        n_tr = (1 if before[1] >= 2 else 0) + (1 if before[2] >= 2 else 0)
        cnt = (1 << n_tr) - 1
        per_lv.append(slices_f2c[idx : idx + cnt])
        idx += cnt
    ll_sl = slices_f2c[idx]
    sched = [(ll_sl, 0, None)]
    prev_start = 0
    prev_cnt = 1
    for lv_rev in range(n_lv - 1, -1, -1):
        lv_num = n_lv - lv_rev  # 1 .. n_lv
        grp = per_lv[lv_rev]
        cur_start = len(sched)
        for oi, sl in enumerate(grp):
            par_idx = 0 if prev_cnt == 1 else (prev_start + min(oi, prev_cnt - 1))
            sched.append((sl, lv_num, par_idx))
        prev_start = cur_start
        prev_cnt = len(grp)
    return sched


# =============================================================================
# 4. End-to-End Encode / Decode API (Explicit 3D + Blind 1D Track A + Progressive)
# =============================================================================
def encode_cube(C, spec_mode="auto", L=3, q_base=1, spec_axis=0, flags=0):
    """Encode a 3D numpy array C (any integer dtype in int16/uint16/int32, arbitrary shape)
    into a real bit-exact .hnd1 binary byte stream."""
    C = np.ascontiguousarray(C)
    if C.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {C.shape}")
    dt = C.dtype.newbyteorder("<")
    if dt not in DTYPE_TO_CODE:
        raise ValueError(f"Unsupported dtype {C.dtype}; use int16, uint16, or int32")
    raw_bytes = C.astype(dt).tobytes()
    sha = hashlib.sha256(raw_bytes).digest()
    orig_shape = C.shape  # (D0, D1, D2)

    # Move spectral axis to 0
    C_work = np.ascontiguousarray(np.moveaxis(C, spec_axis, 0)).astype(np.int64)
    if q_base > 1:
        flags |= FLAG_LOSSY

    Y_spec, spec_code, spec_bytes = spectral_ldu_forward(C_work, mode=spec_mode)
    y2, plan = p0.dwt53_axes(Y_spec, (1, 2), L)
    sched = _c2f_subband_schedule(plan, y2.shape)
    gains = get_2d53_l2_gains(len(plan)) if len(sched) == 1 + 3 * len(plan) else [1.0] * len(sched)

    # Quantize coarse-to-fine & encode each subband with exact causal parent reference!
    # IMPORTANT: In lossy mode, parent must be the RECONSTRUCTED (dequantized) parent so
    # encoder and decoder compute identical parent contexts!
    sub_blobs = []
    decoded_subs = []
    for si, (sl, _lv, par_idx) in enumerate(sched):
        sub = y2[sl]
        if q_base > 1:
            qs = max(1, int(round(q_base / gains[si])))
            qsub = p0.dz_quant(sub, qs)
            recon_sub = p0.dz_dequant(qsub, qs)
        else:
            qsub = sub
            recon_sub = sub
        par = decoded_subs[par_idx] if par_idx is not None else None
        sub_blobs.append(pack_rans_subband(qsub, parent=par))
        decoded_subs.append(recon_sub)

    hdr = struct.pack(
        "<4sBBBBBBHIII32sI",
        MAGIC, VERSION, flags, DTYPE_TO_CODE[dt], spec_code,
        len(plan), spec_axis, q_base,
        orig_shape[0], orig_shape[1], orig_shape[2],
        sha, len(spec_bytes),
    )
    out = bytearray(hdr)
    out += spec_bytes
    for blob in sub_blobs:
        out += blob
    return bytes(out)


def decode_cube(stream, max_level=None, verify_sha=True):
    """Decode a .hnd1 binary byte stream back into a 3D numpy array.
    If max_level is set (0=LL only, 1=LL+L3, ..., L=full), performs true prefix
    progressive decode without reading finer subbands."""
    mv = memoryview(stream)
    hdr_fmt = "<4sBBBBBBHIII32sI"
    hdr_sz = struct.calcsize(hdr_fmt)
    (magic, ver, flags, dt_code, spec_code, n_lv, spec_axis, q_base,
     d0, d1, d2, sha_exp, spec_len) = struct.unpack_from(hdr_fmt, mv, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError(f"Invalid HND1 stream (magic={magic}, ver={ver})")
    dt = CODE_TO_DTYPE[dt_code]

    if flags & FLAG_RAW:
        arr = np.frombuffer(mv, dtype=dt, count=d0 * d1 * d2, offset=hdr_sz).reshape(d0, d1, d2).copy()
        if verify_sha and hashlib.sha256(arr.tobytes()).digest() != sha_exp:
            raise ValueError("SHA-256 mismatch on raw HND1 stream!")
        return arr

    pos = hdr_sz
    spec_end = pos + spec_len
    orig_shape = (d0, d1, d2)
    # Shape after moveaxis(spec_axis -> 0)
    work_dims = list(orig_shape)
    b_dim = work_dims.pop(spec_axis)
    work_shape = (b_dim, work_dims[0], work_dims[1])

    plan = _build_plan_from_shape(work_shape, n_lv)
    sched = _c2f_subband_schedule(plan, work_shape)
    gains = get_2d53_l2_gains(len(plan)) if len(sched) == 1 + 3 * len(plan) else [1.0] * len(sched)
    lv_cap = n_lv if max_level is None else min(max_level, n_lv)

    y2_rec = np.zeros(work_shape, dtype=np.int64)
    decoded_subs = []
    sub_pos = spec_end
    for si, (sl, lv, par_idx) in enumerate(sched):
        sub_shape = y2_rec[sl].shape
        if lv <= lv_cap:
            par = decoded_subs[par_idx] if par_idx is not None else None
            qsub, sub_pos = unpack_rans_subband(mv, sub_pos, sub_shape, parent=par)
            if q_base > 1:
                qs = max(1, int(round(q_base / gains[si])))
                recon_sub = p0.dz_dequant(qsub, qs)
            else:
                recon_sub = qsub
            y2_rec[sl] = recon_sub
            decoded_subs.append(recon_sub)
        else:
            decoded_subs.append(np.zeros(sub_shape, dtype=np.int64))

    Y_rec = p0.dwt53_inv(y2_rec, plan, (1, 2))
    C_work_rec, check_pos = spectral_ldu_inverse(Y_rec, spec_code, mv, pos)
    assert check_pos == spec_end, f"Spectral header length mismatch: {check_pos} != {spec_end}"

    C_rec = np.ascontiguousarray(np.moveaxis(C_work_rec, 0, spec_axis))
    # Clamp only in lossy/progressive mode (lossless is already exact!)
    if (flags & FLAG_LOSSY) or (lv_cap < n_lv):
        info = np.iinfo(dt)
        C_rec = np.clip(C_rec, info.min, info.max)
    out = C_rec.astype(dt)
    if verify_sha and not (flags & FLAG_LOSSY) and lv_cap == n_lv:
        if hashlib.sha256(out.tobytes()).digest() != sha_exp:
            raise ValueError("SHA-256 verification FAILED on lossless HND1 decode!")
    return out


def _choose_spectral_axis_fast(cube3):
    """Fast schema-free spectral-axis role picker (tests 1-level 2D 5/3 + lag-1 spectral
    correlation / variance score on a spatial subsample -> picks true band axis in ~5ms)."""
    # Use exact Phase 0 criterion on a fast cropped subcube (up to 64x64 spatial) so it
    # generalizes to any hyperspectral/scientific cube without scanning 22M samples 3 times!
    best_ax = 0
    best_bps = 1e18
    for ax in range(3):
        C = np.moveaxis(cube3, ax, 0)
        B, H, W = C.shape
        if B < 2 or H < 4 or W < 4:
            continue
        # Subsample spatially to <= 64x64 (contiguous crop in center to preserve local correlation!)
        ch = min(H, 64) // 4 * 4
        cw = min(W, 64) // 4 * 4
        y0 = (H - ch) // 2
        x0 = (W - cw) // 2
        sub = np.ascontiguousarray(C[:, y0 : y0 + ch, x0 : x0 + cw]).astype(np.int64)
        eb, _ = p0.row_hybrid(sub, "clustered_serm", L=2, verify=False, real_rice=False)
        bps = eb / sub.size
        if bps < best_bps:
            best_bps = bps
            best_ax = ax
    return best_ax


def encode_blind_1d(arr_1d, spec_mode="auto", L=3, q_base=1):
    """Track A schema-free 1D stream encoder: takes a flat 1D integer array with ZERO
    shape or axis-order hints, discovers its 3D stride lattice via recover_strides(),
    identifies the spectral axis, and writes a self-describing .hnd1 binary stream."""
    x = np.ascontiguousarray(arr_1d).ravel()
    dt = x.dtype.newbyteorder("<")
    if dt not in DTYPE_TO_CODE:
        raise ValueError(f"Unsupported dtype {x.dtype}")
    n = len(x)
    rec, _trace = p0.recover_strides(x.astype(np.float64), max_axes=5)

    if len(rec) >= 3 and rec[1] > 1 and rec[2] > rec[1] and rec[2] % rec[1] == 0:
        s1, s2 = rec[1], rec[2]
        d0 = n // s2
        d1 = s2 // s1
        d2 = s1
        if d0 * d1 * d2 == n and d0 >= 2 and d1 >= 4 and d2 >= 4:
            cube3 = x.reshape(d0, d1, d2)
            spec_ax = _choose_spectral_axis_fast(cube3)
            return encode_cube(cube3, spec_mode=spec_mode, L=L, q_base=q_base,
                               spec_axis=spec_ax, flags=FLAG_BLIND), rec, spec_ax

    # Clean raw fallback (e.g. incompressible noise or non-lattice 1D stream -> 0% blowup!)
    raw_bytes = x.astype(dt).tobytes()
    sha = hashlib.sha256(raw_bytes).digest()
    hdr = struct.pack(
        "<4sBBBBBBHIII32sI",
        MAGIC, VERSION, FLAG_BLIND | FLAG_RAW, DTYPE_TO_CODE[dt], SPEC_NONE,
        0, 0, 1, 1, 1, n, sha, 0,
    )
    return hdr + raw_bytes, rec, None


def decode_blind_1d(stream, max_level=None, verify_sha=True):
    """Decode a blind .hnd1 stream back to the exact original flat 1D array."""
    return decode_cube(stream, max_level=max_level, verify_sha=verify_sha).ravel()


# =============================================================================
# 5. Benchmark Suite & CLI
# =============================================================================
def _load_mat_cube(path):
    d = loadmat(str(path))
    key = max([k for k in d if not k.startswith("__")], key=lambda k: getattr(d[k], "size", 0))
    arr = np.asarray(d[key])
    return np.ascontiguousarray(np.moveaxis(arr, 2, 0))


def run_benchmark(data_dir):
    cubes = [
        ("Indian_pines", Path(data_dir) / "Indian_pines.mat"),
        ("Salinas", Path(data_dir) / "Salinas.mat"),
        ("PaviaU", Path(data_dir) / "PaviaU.mat"),
    ]
    print("=== PHASE 1 END-TO-END .hnd1 REAL-BINARY BENCHMARK ===")
    for name, path in cubes:
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue
        C = _load_mat_cube(path)
        raw_bytes = C.size * C.dtype.itemsize
        print(f"\n--- {name}  full native shape={C.shape}  dtype={C.dtype}  raw={raw_bytes:,} B ---")

        # 1. Explicit 3D Lossless (auto, clustered, full) with full SHA-256 decode verification
        for mode in ("auto", "clustered", "full"):
            t0 = time.time()
            blob = encode_cube(C, spec_mode=mode, L=3, q_base=1)
            t_enc = time.time() - t0
            t1 = time.time()
            rec = decode_cube(blob, verify_sha=True)
            t_dec = time.time() - t1
            assert np.array_equal(rec, C), f"{name}/{mode} decode mismatch!"
            bps = len(blob) * 8 / C.size
            ratio = raw_bytes / len(blob)
            print(f"  Lossless [{mode:<9s}]  {len(blob):>10,} B  {bps:6.3f} b/s  ({ratio:4.2f}x)  "
                  f"enc={raw_bytes/1e6/t_enc:5.1f} MB/s  dec={raw_bytes/1e6/t_dec:5.1f} MB/s  SHA256=PASS")

        # 2. True Prefix Progressive Decode from SINGLE lossless auto stream!
        blob_auto = encode_cube(C, spec_mode="auto", L=3, q_base=1)
        peak = float(int(C.max()) - int(C.min()))
        print("  Prefix progressive decode (from single lossless .hnd1 stream):")
        # Measure exact byte offset for each level prefix by parsing framing
        mv = memoryview(blob_auto)
        hdr_sz = struct.calcsize("<4sBBBBBBHIII32sI")
        spec_len = struct.unpack_from("<I", mv, hdr_sz - 4)[0]
        plan = _build_plan_from_shape(C.shape, 3)
        sched = _c2f_subband_schedule(plan, C.shape)
        pos = hdr_sz + spec_len
        lv_end = {}
        for _sl, lv, _p in sched:
            plen = struct.unpack_from("<I", mv, pos)[0]
            pos += 4 + plen
            lv_end[lv] = pos
        for lv in range(4):
            rec_lv = decode_cube(blob_auto[: lv_end[lv]], max_level=lv, verify_sha=(lv == 3))
            mse = float(np.mean((C.astype(np.float64) - rec_lv.astype(np.float64)) ** 2))
            psnr = "   inf" if mse == 0 else f"{10*math.log10(peak*peak/mse):6.2f} dB"
            prefix_b = lv_end[lv]
            print(f"    level<={lv} ({'LL only' if lv==0 else 'LL+L3' if lv==1 else 'LL+L3+L2' if lv==2 else 'FULL lossless':<13s}): "
                  f"{prefix_b:>10,} B ({prefix_b*8/C.size:5.3f} b/s, {prefix_b/len(blob_auto)*100:5.1f}% of stream)  PSNR={psnr}")

        # 3. L2-Weighted Quality-Targeted Lossy .hnd1 Sweep (real binary bytes!)
        print("  L2-weighted lossy .hnd1 sweep (real binary bytes):")
        for q in (2, 4, 8, 16, 32, 64):
            qlob = encode_cube(C, spec_mode="auto", L=3, q_base=q)
            qrec = decode_cube(qlob, verify_sha=False)
            mse = float(np.mean((C.astype(np.float64) - qrec.astype(np.float64)) ** 2))
            psnr = 10.0 * math.log10(peak * peak / max(mse, 1e-12))
            print(f"    q={q:<2d}: {len(qlob):>10,} B  {len(qlob)*8/C.size:5.3f} b/s  ({raw_bytes/len(qlob):5.1f}x)  PSNR={psnr:6.2f} dB")

        # 4. Blind 1D Track A end-to-end encode/decode (BSQ, BIL, BIP — zero shape hints!)
        # Crop slightly only to keep multi-perm demo fast on 45MB cubes if needed; test native Indian_pines!
        C_bl = C if C.size <= 10_000_000 else np.ascontiguousarray(C[:64, :256, :216])
        print(f"  Blind Track A 1D-stream round-trip on {C_bl.shape} (BSQ, BIL, BIP, zero shape hints):")
        for tag, perm in [("BSQ (0,1,2)", (0, 1, 2)), ("BIL (1,0,2)", (1, 0, 2)), ("BIP (1,2,0)", (1, 2, 0))]:
            A = np.ascontiguousarray(np.transpose(C_bl, perm))
            flat = A.ravel()
            b_stream, rec_s, sp_ax = encode_blind_1d(flat, spec_mode="auto", L=3)
            flat_rec = decode_blind_1d(b_stream, verify_sha=True)
            assert np.array_equal(flat_rec, flat), f"Blind 1D mismatch on {name} {tag}!"
            true_s = [1, A.shape[2], A.shape[1] * A.shape[2]]
            true_ax = ("B", "H", "W")[perm[sp_ax]]
            print(f"    {tag}: strides={rec_s} ({'OK' if rec_s==true_s else 'NO'})  "
                  f"spec_ax={sp_ax} (true={true_ax})  {len(b_stream)*8/flat.size:6.3f} b/s  SHA256=PASS")


def main():
    ap = argparse.ArgumentParser(description="Phase 1 End-to-End .hnd1 Codec")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="Run full real-cube .hnd1 benchmark")
    b.add_argument("--data-dir", default="phase0_data")
    e = sub.add_parser("encode", help="Encode .mat cube to .hnd1 binary file")
    e.add_argument("input_mat")
    e.add_argument("output_hnd1")
    e.add_argument("--mode", choices=["auto", "clustered", "full"], default="auto")
    e.add_argument("--q", type=int, default=1)
    e.add_argument("--blind", action="store_true")
    d = sub.add_parser("decode", help="Decode .hnd1 binary file and verify integrity")
    d.add_argument("input_hnd1")
    d.add_argument("--max-level", type=int, default=None)
    args = ap.parse_args()

    if args.cmd == "bench":
        run_benchmark(args.data_dir)
    elif args.cmd == "encode":
        C = _load_mat_cube(args.input_mat)
        if args.blind:
            blob, rec_s, sp_ax = encode_blind_1d(C.ravel(), spec_mode=args.mode, q_base=args.q)
            print(f"Blind strides={rec_s}, spec_axis={sp_ax}")
        else:
            blob = encode_cube(C, spec_mode=args.mode, q_base=args.q)
        Path(args.output_hnd1).write_bytes(blob)
        print(f"Encoded {C.shape} ({C.size*C.dtype.itemsize:,} B) -> {args.output_hnd1} ({len(blob):,} B, {len(blob)*8/C.size:.3f} b/s)")
    elif args.cmd == "decode":
        blob = Path(args.input_hnd1).read_bytes()
        C = decode_cube(blob, max_level=args.max_level, verify_sha=True)
        print(f"Decoded {args.input_hnd1} -> shape={C.shape}, dtype={C.dtype}, SHA256=PASS")


if __name__ == "__main__":
    main()
