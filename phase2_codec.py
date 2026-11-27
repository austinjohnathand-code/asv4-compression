#!/usr/bin/env python3
"""
Phase 2 Unified .hnd2 Codec (Python + Numba Reference Implementation).

Wire-compatible bit-for-bit with the C++20 NEON production engine (cpp/hnd2.cpp -> ./hnd2).
Supports BOTH:
  * base_k = 0: One-shot fastest/tightest lossless (or L2-weighted lossy q_base > 1)
  * base_k = 6: Single-stream 7-layer L2-aligned dyadic quality-embedded lossless
                (truncating at layer k=6..0 decodes 53 -> 59 -> 63 -> 68 -> 73 -> 76 dB -> EXACT!)
  * Track A blind headerless 1D-stream encode/decode via recover_strides().
"""
import argparse
import hashlib
import math
import struct
import time
from pathlib import Path

import numpy as np
from numba import njit

import phase0_harness as p0
import phase1_codec as p1

MAGIC_HND2 = b"HND2"
VERSION_HND2 = 2
HDR_FMT = "<4sBBBBBBBBHIII32sI"
HDR_SIZE = struct.calcsize(HDR_FMT)  # 62 bytes

RANS_BITS = 12
RANS_M = 1 << RANS_BITS
RANS_L = 1 << 16


def get_subband_offsets(sched, n_lv):
    """Exact deterministic L2 bitplane offsets matching cpp/hnd2.cpp identically.
    For L=3 (10 subbands): [2, 2, 2, 1, 1, 1, 0, 0, 0, 0]."""
    offsets = []
    # Track orientation index within each level
    lv_seen = {}
    for _sl, lv, _par in sched:
        if lv == 0:
            offsets.append(max(0, n_lv - 1))
        else:
            oi = lv_seen.get(lv, 0)
            lv_seen[lv] = oi + 1
            if oi == 2:  # HH
                offsets.append(max(0, n_lv - 1 - lv))
            else:        # LH, HL
                offsets.append(max(0, n_lv - lv))
    return offsets


def dequant_midpoint(cur_sub, p):
    if p <= 0:
        return cur_sub
    sgn = np.sign(cur_sub)
    mag = np.abs(cur_sub)
    return np.where(mag == 0, 0, sgn * ((mag << p) + (1 << (p - 1))))


@njit(cache=True)
def _sig_ctx(prev_sub, cur_sub, parent_sub, has_par, b, y, x, B, H, W):
    cur_nz = 0
    if x > 0 and cur_sub[b, y, x - 1] != 0: cur_nz += 2
    if y > 0 and cur_sub[b, y - 1, x] != 0: cur_nz += 2
    if y > 0 and x > 0 and cur_sub[b, y - 1, x - 1] != 0: cur_nz += 1
    if y > 0 and x + 1 < W and cur_sub[b, y - 1, x + 1] != 0: cur_nz += 1

    prev_nz = 0
    if x + 1 < W and prev_sub[b, y, x + 1] != 0: prev_nz += 2
    if y + 1 < H and prev_sub[b, y + 1, x] != 0: prev_nz += 2
    if y + 1 < H and x > 0 and prev_sub[b, y + 1, x - 1] != 0: prev_nz += 1
    if y + 1 < H and x + 1 < W and prev_sub[b, y + 1, x + 1] != 0: prev_nz += 1

    par_sig = 0
    if has_par:
        py = y >> 1
        px = x >> 1
        if py >= parent_sub.shape[1]: py = parent_sub.shape[1] - 1
        if px >= parent_sub.shape[2]: px = parent_sub.shape[2] - 1
        if parent_sub[b, py, px] != 0:
            par_sig = 1

    xb_sig = 0
    if b > 0 and cur_sub[b - 1, y, x] != 0:
        xb_sig = 1
    elif b + 1 < B and prev_sub[b + 1, y, x] != 0:
        xb_sig = 1

    sp = cur_nz + prev_nz
    if sp == 0:
        return (par_sig << 1) + xb_sig
    if sp <= 2:
        return 4 + par_sig
    if sp <= 4:
        return 6 + par_sig
    if sp <= 7:
        return 8 + par_sig
    return 10 + par_sig


@njit(cache=True)
def _extract_refine_layer(full_sub, prev_sub, parent_sub, has_par, p):
    B, H, W = full_sub.shape
    N = B * H * W
    syms = np.empty(N, dtype=np.uint8)
    ctxs = np.empty(N, dtype=np.uint8)
    counts = np.zeros((16, 3), dtype=np.int64)
    cur_sub = np.empty((B, H, W), dtype=np.int64)

    idx = 0
    for b in range(B):
        for y in range(H):
            for x in range(W):
                pv = prev_sub[b, y, x]
                fv = full_sub[b, y, x]
                amag = fv if fv >= 0 else -fv
                bit = (amag >> p) & 1
                if pv == 0:
                    c = _sig_ctx(prev_sub, cur_sub, parent_sub, has_par, b, y, x, B, H, W)
                    if bit == 0:
                        s = 0; nv = 0
                    elif fv > 0:
                        s = 1; nv = 1
                    else:
                        s = 2; nv = -1
                else:
                    pmag = pv if pv >= 0 else -pv
                    if pmag == 1: c = 12
                    elif pmag == 2: c = 13
                    elif pmag <= 5: c = 14
                    else: c = 15
                    s = bit
                    nv = (pv << 1) + bit if pv > 0 else -((pmag << 1) + bit)
                cur_sub[b, y, x] = nv
                syms[idx] = s
                ctxs[idx] = c
                counts[c, s] += 1
                idx += 1
    return syms, ctxs, counts, cur_sub


@njit(cache=True)
def _rans_encode_small(syms, ctxs, freqs, cdfs):
    N = len(syms)
    words = np.empty(N // 2 + 64, dtype=np.uint16)
    wpos = 0
    state = np.uint64(RANS_L)
    shift_lim = np.uint64((RANS_L >> RANS_BITS) << 16)
    for i in range(N - 1, -1, -1):
        c = ctxs[i]
        s = syms[i]
        f = np.uint64(freqs[c, s])
        start = np.uint64(cdfs[c, s])
        x_max = shift_lim * f
        while state >= x_max:
            words[wpos] = np.uint16(state & np.uint64(0xFFFF))
            wpos += 1
            state >>= np.uint64(16)
        state = ((state // f) << np.uint64(RANS_BITS)) + (state % f) + start
    words[wpos] = np.uint16(state & np.uint64(0xFFFF)); wpos += 1
    words[wpos] = np.uint16((state >> np.uint64(16)) & np.uint64(0xFFFF)); wpos += 1
    return words[:wpos]


@njit(cache=True)
def _rans_decode_refine(B, H, W, prev_sub, parent_sub, has_par, words, freqs, cdfs):
    lut = np.empty((16, RANS_M), dtype=np.uint8)
    for c in range(16):
        for s in range(3):
            for slot in range(cdfs[c, s], cdfs[c, s + 1]):
                lut[c, slot] = s
    cur_sub = np.empty((B, H, W), dtype=np.int64)
    wpos = len(words) - 1
    hi_w = np.uint64(words[wpos]); wpos -= 1
    lo_w = np.uint64(words[wpos]); wpos -= 1
    state = (hi_w << np.uint64(16)) | lo_w
    mask_m = np.uint64(RANS_M - 1)
    for b in range(B):
        for y in range(H):
            for x in range(W):
                pv = prev_sub[b, y, x]
                if pv == 0:
                    c = _sig_ctx(prev_sub, cur_sub, parent_sub, has_par, b, y, x, B, H, W)
                else:
                    pmag = pv if pv >= 0 else -pv
                    if pmag == 1: c = 12
                    elif pmag == 2: c = 13
                    elif pmag <= 5: c = 14
                    else: c = 15
                slot = state & mask_m
                s = lut[c, slot]
                f = np.uint64(freqs[c, s])
                start = np.uint64(cdfs[c, s])
                state = f * (state >> np.uint64(RANS_BITS)) + slot - start
                while state < np.uint64(RANS_L) and wpos >= 0:
                    state = (state << np.uint64(16)) | np.uint64(words[wpos])
                    wpos -= 1
                if pv == 0:
                    nv = 0 if s == 0 else (1 if s == 1 else -1)
                else:
                    pmag = pv if pv >= 0 else -pv
                    nv = (pv << 1) + s if pv > 0 else -((pmag << 1) + s)
                cur_sub[b, y, x] = nv
    return cur_sub


def _pack_small_tables(counts):
    freqs = np.zeros((16, 3), dtype=np.int32)
    cdfs = np.zeros((16, 4), dtype=np.int32)
    active_mask = 0
    parts = []
    for c in range(16):
        row = counts[c]
        tot = int(row.sum())
        if tot == 0:
            freqs[c, 0] = RANS_M
            cdfs[c, 1:] = RANS_M
            continue
        active_mask |= (1 << c)
        nz = np.nonzero(row)[0]
        f = np.maximum(1, np.rint(row[nz] * (RANS_M / tot)).astype(np.int32))
        diff = int(f.sum()) - RANS_M
        while diff != 0:
            if diff > 0:
                cand = np.where(f > 1)[0]
                j = cand[int(np.argmax(f[cand]))]
                f[j] -= 1; diff -= 1
            else:
                j = int(np.argmax(row[nz]))
                f[j] += 1; diff += 1
        freqs[c, nz] = f
        cdfs[c, 1:] = np.cumsum(freqs[c])
        sm = 0
        for s in nz:
            sm |= (1 << int(s))
        p = bytearray(struct.pack("<B", sm))
        if len(nz) > 1:
            p += f[:-1].astype("<u2").tobytes()
        parts.append(bytes(p))
    tbuf = bytearray(struct.pack("<H", active_mask))
    for p in parts:
        tbuf += p
    return freqs, cdfs, bytes(tbuf)


def _unpack_small_tables(mv, pos):
    freqs = np.zeros((16, 3), dtype=np.int32)
    cdfs = np.zeros((16, 4), dtype=np.int32)
    active_mask = struct.unpack_from("<H", mv, pos)[0]
    pos += 2
    for c in range(16):
        if not ((active_mask >> c) & 1):
            freqs[c, 0] = RANS_M
            cdfs[c, 1:] = RANS_M
            continue
        sm = struct.unpack_from("<B", mv, pos)[0]
        pos += 1
        nz = [s for s in range(3) if (sm >> s) & 1]
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


def pack_refine_subband(full_sub, prev_sub, parent_sub, p_cur):
    f64 = np.ascontiguousarray(full_sub, dtype=np.int64)
    pr64 = np.ascontiguousarray(prev_sub, dtype=np.int64)
    has_p = parent_sub is not None
    pa64 = np.ascontiguousarray(parent_sub, dtype=np.int64) if has_p else np.zeros((1, 1, 1), dtype=np.int64)
    syms, ctxs, counts, new_sub = _extract_refine_layer(f64, pr64, pa64, has_p, p_cur)
    freqs, cdfs, tbuf = _pack_small_tables(counts)
    words = _rans_encode_small(syms, ctxs, freqs, cdfs)
    payload = tbuf + struct.pack("<I", len(words)) + words.astype("<u2").tobytes()
    return struct.pack("<I", len(payload)) + payload, new_sub


def unpack_refine_subband(mv, pos, shape, prev_sub, parent_sub):
    plen = struct.unpack_from("<I", mv, pos)[0]
    pos += 4
    end_pos = pos + plen
    freqs, cdfs, pos = _unpack_small_tables(mv, pos)
    n_words = struct.unpack_from("<I", mv, pos)[0]
    pos += 4
    words = np.frombuffer(mv, dtype="<u2", count=n_words, offset=pos).copy()
    pos += 2 * n_words
    assert pos == end_pos, f"Refine framing mismatch: {pos} != {end_pos}"
    pr64 = np.ascontiguousarray(prev_sub, dtype=np.int64)
    has_p = parent_sub is not None
    pa64 = np.ascontiguousarray(parent_sub, dtype=np.int64) if has_p else np.zeros((1, 1, 1), dtype=np.int64)
    new_sub = _rans_decode_refine(shape[0], shape[1], shape[2], pr64, pa64, has_p, words, freqs, cdfs)
    return new_sub, end_pos


def encode_hnd2(C, spec_mode="auto", L=3, base_k=6, q_base=1, spec_axis=0, flags=0):
    """Encode a 3D cube into the unified .hnd2 binary format (wire-compatible with ./hnd2 C++)."""
    C = np.ascontiguousarray(C)
    if C.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {C.shape}")
    dt = C.dtype.newbyteorder("<")
    if dt not in p1.DTYPE_TO_CODE:
        raise ValueError(f"Unsupported dtype {C.dtype}")
    if base_k > 0 and q_base != 1:
        raise ValueError("Use base_k > 0 (embedded lossless) OR q_base > 1 (one-shot lossy), not both")
    raw_bytes = C.astype(dt).tobytes()
    sha = hashlib.sha256(raw_bytes).digest()
    orig_shape = C.shape

    C_work = np.ascontiguousarray(np.moveaxis(C, spec_axis, 0)).astype(np.int64)
    if q_base > 1:
        flags |= p1.FLAG_LOSSY

    Y_spec, spec_code, spec_bytes = p1.spectral_ldu_forward(C_work, mode=spec_mode)
    y2, plan = p0.dwt53_axes(Y_spec, (1, 2), L)
    sched = p1._c2f_subband_schedule(plan, y2.shape)
    offsets = get_subband_offsets(sched, len(plan))
    gains = p1.get_2d53_l2_gains(len(plan)) if len(sched) == 1 + 3 * len(plan) else [1.0] * len(sched)

    out = bytearray(struct.pack(
        HDR_FMT,
        MAGIC_HND2, VERSION_HND2, flags, p1.DTYPE_TO_CODE[dt], spec_code,
        len(plan), spec_axis, base_k, 0, q_base,
        orig_shape[0], orig_shape[1], orig_shape[2],
        sha, len(spec_bytes),
    ))
    out += spec_bytes

    # Base layer (k = base_k)
    cur_subs = []
    for si, (sl, _lv, par_idx) in enumerate(sched):
        sub = y2[sl]
        if base_k > 0:
            p = max(0, base_k - offsets[si])
            c_base = np.sign(sub) * (np.abs(sub) >> p) if p > 0 else sub.copy()
            par = dequant_midpoint(cur_subs[par_idx], max(0, base_k - offsets[par_idx])) if par_idx is not None else None
            out += p1.pack_rans_subband(c_base, parent=par)
            cur_subs.append(c_base)
        else:
            if q_base > 1:
                qs = max(1, int(round(q_base / gains[si])))
                qsub = p0.dz_quant(sub, qs)
                recon = p0.dz_dequant(qsub, qs)
            else:
                qsub = sub
                recon = sub
            par = cur_subs[par_idx] if par_idx is not None else None
            out += p1.pack_rans_subband(qsub, parent=par)
            cur_subs.append(recon)

    # Refinement layers (k = base_k - 1 .. 0)
    for k in range(base_k - 1, -1, -1):
        for si, (sl, _lv, par_idx) in enumerate(sched):
            p_prev = max(0, (k + 1) - offsets[si])
            p_cur = max(0, k - offsets[si])
            if p_cur == p_prev:
                continue
            par = cur_subs[par_idx] if par_idx is not None else None
            blob, new_sub = pack_refine_subband(y2[sl], cur_subs[si], par, p_cur)
            out += blob
            cur_subs[si] = new_sub

    return bytes(out)


def decode_hnd2(stream, stop_layer=0, max_level=None, verify_sha=True):
    """Decode a .hnd2 binary stream (produced by Python or C++ ./hnd2).
    * When base_k > 0: stop_layer in [base_k..0] decodes prefix down to quality layer stop_layer
      (stop_layer=0 is bit-exact lossless; stop_layer=base_k decodes base layer only!).
    * When base_k == 0: max_level in [0..n_lv] decodes resolution prefix."""
    mv = memoryview(stream)
    (magic, ver, flags, dt_code, spec_code, n_lv, spec_axis, base_k, _res, q_base,
     d0, d1, d2, sha_exp, spec_len) = struct.unpack_from(HDR_FMT, mv, 0)
    if magic != MAGIC_HND2 or ver != VERSION_HND2:
        raise ValueError(f"Invalid HND2 stream (magic={magic}, ver={ver})")
    dt = p1.CODE_TO_DTYPE[dt_code]

    if flags & p1.FLAG_RAW:
        arr = np.frombuffer(mv, dtype=dt, count=d0 * d1 * d2, offset=HDR_SIZE).reshape(d0, d1, d2).copy()
        if verify_sha and hashlib.sha256(arr.tobytes()).digest() != sha_exp:
            raise ValueError("SHA-256 mismatch on raw HND2 stream!")
        return arr

    pos = HDR_SIZE
    spec_end = pos + spec_len
    orig_shape = (d0, d1, d2)
    work_dims = list(orig_shape)
    b_dim = work_dims.pop(spec_axis)
    work_shape = (b_dim, work_dims[0], work_dims[1])

    plan = p1._build_plan_from_shape(work_shape, n_lv)
    sched = p1._c2f_subband_schedule(plan, work_shape)
    offsets = get_subband_offsets(sched, len(plan))
    gains = p1.get_2d53_l2_gains(len(plan)) if len(sched) == 1 + 3 * len(plan) else [1.0] * len(sched)

    sub_pos = spec_end
    cur_subs = []
    lv_cap = n_lv if max_level is None else min(max_level, n_lv)
    k_target = max(0, min(stop_layer, base_k))

    # Decode base layer
    for si, (sl, lv, par_idx) in enumerate(sched):
        sub_shape = tuple(s.stop - s.start for s in sl)
        if base_k > 0:
            par = dequant_midpoint(cur_subs[par_idx], max(0, base_k - offsets[par_idx])) if par_idx is not None else None
            c_base, sub_pos = p1.unpack_rans_subband(mv, sub_pos, sub_shape, parent=par)
            cur_subs.append(c_base)
        else:
            if lv <= lv_cap:
                par = cur_subs[par_idx] if par_idx is not None else None
                qsub, sub_pos = p1.unpack_rans_subband(mv, sub_pos, sub_shape, parent=par)
                if q_base > 1:
                    qs = max(1, int(round(q_base / gains[si])))
                    cur_subs.append(p0.dz_dequant(qsub, qs))
                else:
                    cur_subs.append(qsub)
            else:
                cur_subs.append(np.zeros(sub_shape, dtype=np.int64))

    # Decode refinement layers down to k_target
    for k in range(base_k - 1, k_target - 1, -1):
        for si, (sl, _lv, par_idx) in enumerate(sched):
            p_prev = max(0, (k + 1) - offsets[si])
            p_cur = max(0, k - offsets[si])
            if p_cur == p_prev:
                continue
            sub_shape = tuple(s.stop - s.start for s in sl)
            par = cur_subs[par_idx] if par_idx is not None else None
            cur_subs[si], sub_pos = unpack_refine_subband(mv, sub_pos, sub_shape, cur_subs[si], par)

    # Reconstruct wavelet cube
    y2_rec = np.empty(work_shape, dtype=np.int64)
    for si, (sl, _lv, _p) in enumerate(sched):
        if base_k > 0:
            y2_rec[sl] = dequant_midpoint(cur_subs[si], max(0, k_target - offsets[si]))
        else:
            y2_rec[sl] = cur_subs[si]

    Y_rec = p0.dwt53_inv(y2_rec, plan, (1, 2))
    C_work_rec, check_pos = p1.spectral_ldu_inverse(Y_rec, spec_code, mv, pos)
    assert check_pos == spec_end
    C_rec = np.ascontiguousarray(np.moveaxis(C_work_rec, 0, spec_axis))

    is_exact = (not (flags & p1.FLAG_LOSSY)) and (k_target == 0) and (lv_cap == n_lv)
    if not is_exact:
        info = np.iinfo(dt)
        C_rec = np.clip(C_rec, info.min, info.max)
    out = C_rec.astype(dt)
    if verify_sha and is_exact:
        if hashlib.sha256(out.tobytes()).digest() != sha_exp:
            raise ValueError("SHA-256 verification FAILED on lossless HND2 decode!")
    return out


def encode_hnd2_blind_1d(arr_1d, spec_mode="auto", L=3, base_k=6):
    x = np.ascontiguousarray(arr_1d).ravel()
    dt = x.dtype.newbyteorder("<")
    n = len(x)
    rec, _trace = p0.recover_strides(x.astype(np.float64), max_axes=5)
    if len(rec) >= 3 and rec[1] > 1 and rec[2] > rec[1] and rec[2] % rec[1] == 0:
        s1, s2 = rec[1], rec[2]
        d0, d1, d2 = n // s2, s2 // s1, s1
        if d0 * d1 * d2 == n and d0 >= 2 and d1 >= 4 and d2 >= 4:
            cube3 = x.reshape(d0, d1, d2)
            spec_ax = p1._choose_spectral_axis_fast(cube3)
            return encode_hnd2(cube3, spec_mode=spec_mode, L=L, base_k=base_k,
                               spec_axis=spec_ax, flags=p1.FLAG_BLIND), rec, spec_ax
    raw_bytes = x.astype(dt).tobytes()
    sha = hashlib.sha256(raw_bytes).digest()
    hdr = struct.pack(
        HDR_FMT, MAGIC_HND2, VERSION_HND2, p1.FLAG_BLIND | p1.FLAG_RAW,
        p1.DTYPE_TO_CODE[dt], p1.SPEC_NONE, 0, 0, 0, 0, 1, 1, 1, n, sha, 0,
    )
    return hdr + raw_bytes, rec, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 2 Unified HND2 Reference Codec")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("input_mat")
    e.add_argument("output_hnd2")
    e.add_argument("--base-k", type=int, default=6)
    e.add_argument("--mode", choices=["auto", "clustered", "full"], default="auto")
    d = sub.add_parser("decode")
    d.add_argument("input_hnd2")
    d.add_argument("--stop-layer", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "encode":
        C = p1._load_mat_cube(args.input_mat)
        t0 = time.time()
        blob = encode_hnd2(C, spec_mode=args.mode, base_k=args.base_k)
        Path(args.output_hnd2).write_bytes(blob)
        print(f"Encoded {C.shape} -> {args.output_hnd2} ({len(blob):,} B, {len(blob)*8/C.size:.3f} b/s, {time.time()-t0:.2f}s)")
    elif args.cmd == "decode":
        t0 = time.time()
        C = decode_hnd2(Path(args.input_hnd2).read_bytes(), stop_layer=args.stop_layer, verify_sha=True)
        print(f"Decoded {C.shape} dtype={C.dtype} stop_layer={args.stop_layer} ({time.time()-t0:.2f}s)")
