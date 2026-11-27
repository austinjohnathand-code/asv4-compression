#!/usr/bin/env python3
"""
Phase 3 Unified Band-Striped .hnd3 Reference Codec (Python + Numba).

Wire-compatible bit-for-bit with cpp/hnd3.cpp (./hnd3):
  * Band-stripe parallel 32-bit rANS (n_stripes = min(B, 16) independent band
    stripes per subband, each with its own compact 12-bit frequency table).
  * Supports base_k=0 (one-shot) and base_k=6 (7-layer quality-embedded).
"""
import hashlib
import struct

import numpy as np

import phase0_harness as p0
import phase1_codec as p1
import phase2_codec as p2

MAGIC_HND3 = b"HND3"
VERSION_HND3 = 3
HDR_FMT = "<4sBBBBBBBBHIII32sI"
HDR_SIZE = struct.calcsize(HDR_FMT)  # 62 bytes


def _stripe_bounds(B, n_stripes, gi):
    return (gi * B) // n_stripes, ((gi + 1) * B) // n_stripes


def encode_hnd3(C, spec_mode="auto", L=3, base_k=6, n_stripes=16, spec_axis=0, flags=0):
    C = np.ascontiguousarray(C)
    if C.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {C.shape}")
    dt = C.dtype.newbyteorder("<")
    if dt not in p1.DTYPE_TO_CODE:
        raise ValueError(f"Unsupported dtype {C.dtype}")
    raw_bytes = C.astype(dt).tobytes()
    sha = hashlib.sha256(raw_bytes).digest()
    orig_shape = C.shape

    C_work = np.ascontiguousarray(np.moveaxis(C, spec_axis, 0)).astype(np.int64)
    B = C_work.shape[0]
    n_stripes = max(1, min(int(n_stripes), B))

    Y_spec, spec_code, spec_bytes = p1.spectral_ldu_forward(C_work, mode=spec_mode)
    y2, plan = p0.dwt53_axes(Y_spec, (1, 2), L)
    sched = p1._c2f_subband_schedule(plan, y2.shape)
    offsets = p2.get_subband_offsets(sched, len(plan))

    out = bytearray(struct.pack(
        HDR_FMT,
        MAGIC_HND3, VERSION_HND3, flags, p1.DTYPE_TO_CODE[dt], spec_code,
        len(plan), spec_axis, base_k, n_stripes, 1,
        orig_shape[0], orig_shape[1], orig_shape[2],
        sha, len(spec_bytes),
    ))
    out += spec_bytes

    # Base layer (subband-major, then stripe gi = 0..n_stripes-1)
    cur_subs = []
    par_mids = []
    for si, (sl, _lv, par_idx) in enumerate(sched):
        sub = y2[sl]
        p = max(0, base_k - offsets[si])
        if base_k > 0 and p > 0:
            c_base = np.sign(sub) * (np.abs(sub) >> p)
            pmid = p2.dequant_midpoint(c_base, p)
        else:
            c_base = sub.copy()
            pmid = c_base
        cur_subs.append(c_base)
        par_mids.append(pmid)
        for gi in range(n_stripes):
            b0, b1 = _stripe_bounds(B, n_stripes, gi)
            par = par_mids[par_idx][b0:b1] if par_idx is not None else None
            out += p1.pack_rans_subband(c_base[b0:b1], parent=par)

    # Refinement layers k = base_k - 1 .. 0 (level-major, matching cpp/hnd3.cpp!)
    for k in range(base_k - 1, -1, -1):
        for lv in range(len(plan) + 1):
            lv_subs = []
            for si, (_sl, slv, _par) in enumerate(sched):
                if slv != lv:
                    continue
                if max(0, k - offsets[si]) != max(0, (k + 1) - offsets[si]):
                    lv_subs.append(si)
            if not lv_subs:
                continue
            for si in lv_subs:
                sl, _slv, par_idx = sched[si]
                p_cur = max(0, k - offsets[si])
                new_full = np.empty_like(cur_subs[si])
                for gi in range(n_stripes):
                    b0, b1 = _stripe_bounds(B, n_stripes, gi)
                    par = cur_subs[par_idx][b0:b1] if par_idx is not None else None
                    blob, new_s = p2.pack_refine_subband(y2[sl][b0:b1], cur_subs[si][b0:b1], par, p_cur)
                    new_full[b0:b1] = new_s
                    out += blob
                cur_subs[si] = new_full
    return bytes(out)


def decode_hnd3(stream, stop_layer=0, max_level=None, verify_sha=True):
    mv = memoryview(stream)
    (magic, ver, flags, dt_code, spec_code, n_lv, spec_axis, base_k, n_stripes, q_base,
     d0, d1, d2, sha_exp, spec_len) = struct.unpack_from(HDR_FMT, mv, 0)
    if magic != MAGIC_HND3 or ver != VERSION_HND3:
        raise ValueError(f"Invalid HND3 stream (magic={magic}, ver={ver})")
    dt = p1.CODE_TO_DTYPE[dt_code]
    n_stripes = max(1, n_stripes)

    if flags & p1.FLAG_RAW:
        arr = np.frombuffer(mv, dtype=dt, count=d0 * d1 * d2, offset=HDR_SIZE).reshape(d0, d1, d2).copy()
        if verify_sha and hashlib.sha256(arr.tobytes()).digest() != sha_exp:
            raise ValueError("SHA-256 mismatch on raw HND3 stream!")
        return arr

    pos = HDR_SIZE
    spec_end = pos + spec_len
    orig_shape = (d0, d1, d2)
    work_dims = list(orig_shape)
    B = work_dims.pop(spec_axis)
    work_shape = (B, work_dims[0], work_dims[1])

    plan = p1._build_plan_from_shape(work_shape, n_lv)
    sched = p1._c2f_subband_schedule(plan, work_shape)
    offsets = p2.get_subband_offsets(sched, len(plan))
    lv_cap = n_lv if max_level is None else min(max_level, n_lv)
    k_target = max(0, min(stop_layer, base_k))

    sub_pos = spec_end
    cur_subs = []
    par_mids = []
    for si, (sl, lv, par_idx) in enumerate(sched):
        sub_shape = tuple(s.stop - s.start for s in sl)
        arr = np.zeros(sub_shape, dtype=np.int64)
        pmid = np.zeros(sub_shape, dtype=np.int64)
        if base_k > 0 or lv <= lv_cap:
            p = max(0, base_k - offsets[si])
            for gi in range(n_stripes):
                b0, b1 = _stripe_bounds(B, n_stripes, gi)
                par = par_mids[par_idx][b0:b1] if par_idx is not None else None
                stripe, sub_pos = p1.unpack_rans_subband(mv, sub_pos, (b1 - b0, sub_shape[1], sub_shape[2]), parent=par)
                arr[b0:b1] = stripe
                pmid[b0:b1] = p2.dequant_midpoint(stripe, p) if (base_k > 0 and p > 0) else stripe
        cur_subs.append(arr)
        par_mids.append(pmid)

    for k in range(base_k - 1, k_target - 1, -1):
        for lv in range(len(plan) + 1):
            for si, (sl, slv, par_idx) in enumerate(sched):
                if slv != lv:
                    continue
                if max(0, k - offsets[si]) == max(0, (k + 1) - offsets[si]):
                    continue
                sub_shape = tuple(s.stop - s.start for s in sl)
                new_full = np.empty(sub_shape, dtype=np.int64)
                for gi in range(n_stripes):
                    b0, b1 = _stripe_bounds(B, n_stripes, gi)
                    par = cur_subs[par_idx][b0:b1] if par_idx is not None else None
                    ns, sub_pos = p2.unpack_refine_subband(
                        mv, sub_pos, (b1 - b0, sub_shape[1], sub_shape[2]), cur_subs[si][b0:b1], par
                    )
                    new_full[b0:b1] = ns
                cur_subs[si] = new_full

    y2_rec = np.empty(work_shape, dtype=np.int64)
    for si, (sl, _lv, _p) in enumerate(sched):
        y2_rec[sl] = p2.dequant_midpoint(cur_subs[si], max(0, k_target - offsets[si])) if base_k > 0 else cur_subs[si]

    Y_rec = p0.dwt53_inv(y2_rec, plan, (1, 2))
    C_work_rec, check_pos = p1.spectral_ldu_inverse(Y_rec, spec_code, mv, pos)
    assert check_pos == spec_end
    C_rec = np.ascontiguousarray(np.moveaxis(C_work_rec, 0, spec_axis))

    is_exact = (not (flags & p1.FLAG_LOSSY)) and (q_base == 1) and (k_target == 0) and (lv_cap == n_lv)
    if not is_exact:
        info = np.iinfo(dt)
        C_rec = np.clip(C_rec, info.min, info.max)
    out = C_rec.astype(dt)
    if verify_sha and is_exact:
        if hashlib.sha256(out.tobytes()).digest() != sha_exp:
            raise ValueError("SHA-256 verification FAILED on lossless HND3 decode!")
    return out
