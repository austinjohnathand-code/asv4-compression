#pragma once
#include "hnd3_core.hpp"

namespace hnd3 {

// 5. Band-Stripe Parallel 32-bit rANS + 64-bit Extra-Bit Reader/Writer
// =============================================================================
static inline void val_to_tok_eb(int64_t v, uint8_t& tok, uint8_t& eb, uint64_t& rem) {
    uint64_t u = (v >= 0) ? (static_cast<uint64_t>(v) << 1)
                          : ((static_cast<uint64_t>(-v) << 1) - 1ULL);
    if (u < 4ULL) { tok = static_cast<uint8_t>(u); eb = 0; rem = 0; return; }
    if (u < (1ULL << 15)) {
        int m = 63 - __builtin_clzll(u);
        int b = static_cast<int>((u >> (m - 1)) & 1ULL);
        tok = static_cast<uint8_t>(2 * m + b);
        eb  = static_cast<uint8_t>(m - 1);
        rem = u & ((1ULL << eb) - 1ULL);
        return;
    }
    if (u < (1ULL << 16)) {
        tok = 30; eb = 15; rem = u & ((1ULL << 15) - 1ULL); return;
    }
    int m = 63 - __builtin_clzll(u);
    tok = 31; eb = static_cast<uint8_t>(m); rem = u & ((1ULL << m) - 1ULL);
}

static inline int64_t tok_eb_to_val(uint8_t tok, uint64_t rem) {
    uint64_t u;
    if (tok < 4) u = tok;
    else if (tok == 30) u = (1ULL << 15) | rem;
    else if (tok == 31) u = rem;
    else {
        int m = tok >> 1;
        int b = tok & 1;
        int eb = m - 1;
        u = (static_cast<uint64_t>(2 + b) << eb) | rem;
    }
    return ((u & 1ULL) == 0) ? static_cast<int64_t>(u >> 1)
                             : -static_cast<int64_t>((u + 1ULL) >> 1);
}

static inline uint8_t act_to_ctx(int64_t act) {
    if (act <= 0) return 0;
    if (act == 1) return 1;
    if (act == 2) return 2;
    if (act == 3) return 3;
    if (act <= 5) return 4;
    if (act <= 8) return 5;
    if (act <= 12) return 6;
    if (act <= 18) return 7;
    if (act <= 27) return 8;
    if (act <= 42) return 9;
    if (act <= 64) return 10;
    if (act <= 100) return 11;
    if (act <= 160) return 12;
    if (act <= 260) return 13;
    if (act <= 450) return 14;
    return 15;
}

static void normalize_counts(const int64_t* counts, int n_ctxs, int n_syms,
                             std::vector<int32_t>& freqs, std::vector<int32_t>& cdfs,
                             std::vector<uint8_t>& tbuf, bool small_sym_mask) {
    freqs.assign(n_ctxs * n_syms, 0);
    cdfs.assign(n_ctxs * (n_syms + 1), 0);
    uint16_t active_mask = 0;
    std::vector<uint8_t> body;
    for (int c = 0; c < n_ctxs; ++c) {
        const int64_t* row = counts + c * n_syms;
        int64_t tot = 0;
        for (int s = 0; s < n_syms; ++s) tot += row[s];
        if (tot == 0) {
            freqs[c * n_syms + 0] = RANS_M;
            for (int s = 1; s <= n_syms; ++s) cdfs[c * (n_syms + 1) + s] = RANS_M;
            continue;
        }
        active_mask |= static_cast<uint16_t>(1U << c);
        std::vector<int> nz;
        for (int s = 0; s < n_syms; ++s) if (row[s] > 0) nz.push_back(s);
        std::vector<int32_t> f(nz.size());
        int32_t sum_f = 0;
        for (size_t i = 0; i < nz.size(); ++i) {
            int32_t fv = static_cast<int32_t>(std::llround(static_cast<double>(row[nz[i]]) * RANS_M / tot));
            if (fv < 1) fv = 1;
            f[i] = fv; sum_f += fv;
        }
        while (sum_f != static_cast<int32_t>(RANS_M)) {
            if (sum_f > static_cast<int32_t>(RANS_M)) {
                size_t best = 0; int32_t bv = -1;
                for (size_t i = 0; i < f.size(); ++i) if (f[i] > 1 && f[i] > bv) { bv = f[i]; best = i; }
                f[best]--; sum_f--;
            } else {
                size_t best = 0; int64_t bv = -1;
                for (size_t i = 0; i < nz.size(); ++i) if (row[nz[i]] > bv) { bv = row[nz[i]]; best = i; }
                f[best]++; sum_f++;
            }
        }
        uint32_t sm = 0;
        for (size_t i = 0; i < nz.size(); ++i) { freqs[c * n_syms + nz[i]] = f[i]; sm |= (1U << nz[i]); }
        int32_t acc = 0; cdfs[c * (n_syms + 1) + 0] = 0;
        for (int s = 0; s < n_syms; ++s) { acc += freqs[c * n_syms + s]; cdfs[c * (n_syms + 1) + s + 1] = acc; }
        if (small_sym_mask) body.push_back(static_cast<uint8_t>(sm));
        else write_le<uint32_t>(body, sm);
        for (size_t i = 0; i + 1 < nz.size(); ++i) write_le<uint16_t>(body, static_cast<uint16_t>(f[i]));
    }
    tbuf.clear();
    write_le<uint16_t>(tbuf, active_mask);
    tbuf.insert(tbuf.end(), body.begin(), body.end());
}

static void unpack_tables(const uint8_t* p, size_t& pos, size_t end_pos, int n_ctxs, int n_syms, bool small_sym_mask,
                          std::vector<int32_t>& freqs, std::vector<int32_t>& cdfs) {
    if (pos + 2 > end_pos) throw std::runtime_error("Truncated rANS table header");
    freqs.assign(n_ctxs * n_syms, 0);
    cdfs.assign(n_ctxs * (n_syms + 1), 0);
    uint16_t active_mask = read_le<uint16_t>(p + pos); pos += 2;
    for (int c = 0; c < n_ctxs; ++c) {
        if (!((active_mask >> c) & 1U)) {
            freqs[c * n_syms + 0] = RANS_M;
            for (int s = 1; s <= n_syms; ++s) cdfs[c * (n_syms + 1) + s] = RANS_M;
            continue;
        }
        size_t msz = small_sym_mask ? 1 : 4;
        if (pos + msz > end_pos) throw std::runtime_error("Truncated rANS sym_mask");
        uint32_t sm = small_sym_mask ? p[pos++] : read_le<uint32_t>(p + pos);
        if (!small_sym_mask) pos += 4;
        std::vector<int> nz;
        for (int s = 0; s < n_syms; ++s) if ((sm >> s) & 1U) nz.push_back(s);
        if (nz.empty()) throw std::runtime_error("Empty active rANS context");
        if (nz.size() == 1) {
            freqs[c * n_syms + nz[0]] = RANS_M;
        } else {
            if (pos + (nz.size() - 1) * 2 > end_pos) throw std::runtime_error("Truncated rANS freqs");
            int32_t sum = 0;
            for (size_t i = 0; i + 1 < nz.size(); ++i) {
                int32_t fv = read_le<uint16_t>(p + pos); pos += 2;
                if (fv < 1 || fv >= static_cast<int32_t>(RANS_M)) throw std::runtime_error("Invalid rANS freq");
                freqs[c * n_syms + nz[i]] = fv; sum += fv;
            }
            int32_t last_f = static_cast<int32_t>(RANS_M) - sum;
            if (last_f < 1) throw std::runtime_error("Invalid rANS freq sum");
            freqs[c * n_syms + nz.back()] = last_f;
        }
        int32_t acc = 0; cdfs[c * (n_syms + 1) + 0] = 0;
        for (int s = 0; s < n_syms; ++s) { acc += freqs[c * n_syms + s]; cdfs[c * (n_syms + 1) + s + 1] = acc; }
        if (acc != static_cast<int32_t>(RANS_M)) throw std::runtime_error("Corrupt rANS CDF sum");
    }
}

// Fast 64-bit MSB-first bit reader (wire-identical to byte-by-byte MSB packing!)
struct FastBitReader {
    const uint8_t* ptr;
    const uint8_t* end;
    uint64_t buf = 0;
    int bits = 0;
    FastBitReader(const uint8_t* p, size_t len) : ptr(p), end(p + len) {}
    inline void refill() {
        while (bits <= 56 && ptr < end) {
            buf |= static_cast<uint64_t>(*ptr++) << (56 - bits);
            bits += 8;
        }
    }
    inline uint64_t read(int n) {
        if (bits < n) refill();
        if (bits < n) throw std::runtime_error("Truncated extra-bit stream");
        uint64_t v = buf >> (64 - n);
        buf <<= n;
        bits -= n;
        return v;
    }
};

// Fast 64-bit MSB-first bit writer
struct FastBitWriter {
    std::vector<uint8_t> out;
    uint64_t buf = 0;
    int bits = 0;
    inline void write(uint64_t val, int n) {
        buf = (buf << n) | (val & ((n == 64) ? ~0ULL : ((1ULL << n) - 1ULL)));
        bits += n;
        while (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xFFu));
        }
    }
    inline void flush() {
        if (bits > 0) {
            out.push_back(static_cast<uint8_t>((buf << (8 - bits)) & 0xFFu));
            bits = 0;
        }
    }
};

// Pack ONE band stripe [0..Bs) of a base subband (Bs, H, W)
static std::vector<uint8_t> pack_rans_stripe(const int64_t* blk, int Bs, int H, int W,
                                             const int64_t* par, int pH, int pW) {
    size_t N = static_cast<size_t>(Bs) * H * W;
    std::vector<uint8_t> toks(N), ctxs(N), ebs(N);
    std::vector<uint64_t> rems(N);
    int64_t counts[N_CTXS * N_SYMS]{};
    bool has_p = (par != nullptr);
    size_t HW = static_cast<size_t>(H) * W;
    size_t pHW = static_cast<size_t>(pH) * pW;

    size_t idx = 0;
    int64_t prev_ema = 16;
    for (int b = 0; b < Bs; ++b) {
        int64_t cur_ema = 16;
        const int64_t* pl = blk + static_cast<size_t>(b) * HW;
        const int64_t* pl_prev = (b > 0) ? (blk + static_cast<size_t>(b - 1) * HW) : nullptr;
        const int64_t* pl_par  = has_p ? (par + static_cast<size_t>(b) * pHW) : nullptr;
        for (int y = 0; y < H; ++y) {
            int py = has_p ? std::min(y >> 1, pH - 1) : 0;
            for (int x = 0; x < W; ++x) {
                int64_t sp_sum = 0; int sp_w = 0;
                if (x > 0) { sp_sum += 2 * std::abs(pl[y * W + x - 1]); sp_w += 2; }
                if (y > 0) {
                    sp_sum += 2 * std::abs(pl[(y - 1) * W + x]); sp_w += 2;
                    if (x > 0)     { sp_sum += std::abs(pl[(y - 1) * W + x - 1]); sp_w += 1; }
                    if (x + 1 < W) { sp_sum += std::abs(pl[(y - 1) * W + x + 1]); sp_w += 1; }
                }
                int64_t a_sp  = (sp_w > 0) ? (sp_sum / sp_w) : 0;
                int64_t a_par = has_p ? (std::abs(pl_par[py * pW + std::min(x >> 1, pW - 1)]) >> 1) : 0;
                int64_t a_xb  = (b > 0) ? ((std::abs(pl_prev[y * W + x]) * cur_ema) / prev_ema) : 0;
                int64_t act;
                if (has_p && b > 0) act = (3 * a_sp + a_par + a_xb) / 5;
                else if (has_p)     act = (3 * a_sp + a_par) >> 2;
                else if (b > 0)     act = (3 * a_sp + a_xb) >> 2;
                else                act = a_sp;

                uint8_t c = act_to_ctx(act);
                int64_t v = pl[y * W + x];
                uint8_t tok, eb; uint64_t rem;
                val_to_tok_eb(v, tok, eb, rem);
                toks[idx] = tok; ctxs[idx] = c; ebs[idx] = eb; rems[idx] = rem;
                counts[c * N_SYMS + tok]++;
                idx++;

                int64_t av = std::abs(v) << 4;
                cur_ema = cur_ema + floor_shr(av - cur_ema, 6);
                if (cur_ema < 4) cur_ema = 4;
            }
        }
        prev_ema = cur_ema;
    }

    std::vector<int32_t> freqs, cdfs;
    std::vector<uint8_t> tbuf;
    normalize_counts(counts, N_CTXS, N_SYMS, freqs, cdfs, tbuf, false);

    std::vector<uint16_t> words;
    words.reserve(N / 2 + 16);
    uint64_t state = RANS_L;
    static constexpr uint64_t SHIFT_LIM = (RANS_L >> RANS_BITS) << 16;
    for (size_t i = N; i-- > 0;) {
        uint8_t c = ctxs[i], s = toks[i];
        uint64_t f = static_cast<uint64_t>(freqs[c * N_SYMS + s]);
        uint64_t start = static_cast<uint64_t>(cdfs[c * (N_SYMS + 1) + s]);
        uint64_t x_max = SHIFT_LIM * f;
        while (state >= x_max) { words.push_back(static_cast<uint16_t>(state & 0xFFFFu)); state >>= 16; }
        state = ((state / f) << RANS_BITS) + (state % f) + start;
    }
    words.push_back(static_cast<uint16_t>(state & 0xFFFFu));
    words.push_back(static_cast<uint16_t>((state >> 16) & 0xFFFFu));

    FastBitWriter bw;
    bw.out.reserve(N / 2 + 16);
    for (size_t i = 0; i < N; ++i) {
        if (toks[i] == 31) {
            int m = ebs[i];
            bw.write(static_cast<uint64_t>(m - 16), 6);
            if (m <= 32) {
                bw.write(rems[i], m);
            } else {
                bw.write(rems[i] >> 32, m - 32);
                bw.write(rems[i] & 0xFFFFFFFFULL, 32);
            }
        } else if (ebs[i] > 0) {
            bw.write(rems[i], ebs[i]);
        }
    }
    bw.flush();

    uint32_t plen = static_cast<uint32_t>(tbuf.size() + 8 + words.size() * 2 + bw.out.size());
    std::vector<uint8_t> out;
    out.reserve(4 + plen);
    write_le<uint32_t>(out, plen);
    out.insert(out.end(), tbuf.begin(), tbuf.end());
    write_le<uint32_t>(out, static_cast<uint32_t>(words.size()));
    write_le<uint32_t>(out, static_cast<uint32_t>(bw.out.size()));
    size_t wp = out.size();
    out.resize(wp + words.size() * 2);
    std::memcpy(out.data() + wp, words.data(), words.size() * 2);
    out.insert(out.end(), bw.out.begin(), bw.out.end());
    return out;
}

static void unpack_rans_stripe(const uint8_t* p, size_t payload_pos, size_t payload_len,
                               int64_t* out, int Bs, int H, int W,
                               const int64_t* par, int pH, int pW) {
    size_t pos = payload_pos, end_pos = payload_pos + payload_len;
    std::vector<int32_t> freqs, cdfs;
    unpack_tables(p, pos, end_pos, N_CTXS, N_SYMS, false, freqs, cdfs);
    if (pos + 8 > end_pos) throw std::runtime_error("Truncated stripe base lengths");
    uint32_t n_words = read_le<uint32_t>(p + pos); pos += 4;
    uint32_t n_eb    = read_le<uint32_t>(p + pos); pos += 4;
    if (n_words < 2 || static_cast<size_t>(n_words) * 2 > end_pos - pos)
        throw std::runtime_error("Invalid stripe base n_words");
    const uint8_t* w_ptr  = p + pos; pos += static_cast<size_t>(n_words) * 2;
    if (n_eb > end_pos - pos) throw std::runtime_error("Invalid stripe base n_eb");
    const uint8_t* eb_ptr = p + pos; pos += n_eb;
    if (pos != end_pos) throw std::runtime_error("Stripe base framing mismatch");

    uint8_t lut[N_CTXS * RANS_M];
    for (int c = 0; c < N_CTXS; ++c) {
        for (int s = 0; s < N_SYMS; ++s) {
            int lo = cdfs[c * (N_SYMS + 1) + s];
            int hi = cdfs[c * (N_SYMS + 1) + s + 1];
            std::memset(lut + c * RANS_M + lo, s, hi - lo);
        }
    }

    int64_t wpos = static_cast<int64_t>(n_words) - 1;
    uint64_t hi_w = read_le<uint16_t>(w_ptr + wpos * 2); wpos--;
    uint64_t lo_w = read_le<uint16_t>(w_ptr + wpos * 2); wpos--;
    uint64_t state = (hi_w << 16) | lo_w;
    if (state < RANS_L) throw std::runtime_error("Invalid initial rANS state");

    FastBitReader br(eb_ptr, n_eb);
    bool has_p = (par != nullptr);
    constexpr uint64_t MASK_M = RANS_M - 1;
    size_t HW = static_cast<size_t>(H) * W;
    size_t pHW = static_cast<size_t>(pH) * pW;
    int64_t prev_ema = 16;

    for (int b = 0; b < Bs; ++b) {
        int64_t cur_ema = 16;
        int64_t* pl = out + static_cast<size_t>(b) * HW;
        const int64_t* pl_prev = (b > 0) ? (out + static_cast<size_t>(b - 1) * HW) : nullptr;
        const int64_t* pl_par  = has_p ? (par + static_cast<size_t>(b) * pHW) : nullptr;
        for (int y = 0; y < H; ++y) {
            int py = has_p ? std::min(y >> 1, pH - 1) : 0;
            for (int x = 0; x < W; ++x) {
                unsigned __int128 sp_sum = 0; int sp_w = 0;
                if (x > 0) { sp_sum += static_cast<unsigned __int128>(uabs64(pl[y * W + x - 1])) * 2u; sp_w += 2; }
                if (y > 0) {
                    sp_sum += static_cast<unsigned __int128>(uabs64(pl[(y - 1) * W + x])) * 2u; sp_w += 2;
                    if (x > 0)     { sp_sum += uabs64(pl[(y - 1) * W + x - 1]); sp_w += 1; }
                    if (x + 1 < W) { sp_sum += uabs64(pl[(y - 1) * W + x + 1]); sp_w += 1; }
                }
                unsigned __int128 a_sp  = (sp_w > 0) ? (sp_sum / static_cast<unsigned>(sp_w)) : 0;
                unsigned __int128 a_par = has_p ? (uabs64(pl_par[py * pW + std::min(x >> 1, pW - 1)]) >> 1) : 0;
                unsigned __int128 a_xb  = (b > 0) ? ((static_cast<unsigned __int128>(uabs64(pl_prev[y * W + x])) * static_cast<uint64_t>(cur_ema)) / static_cast<uint64_t>(prev_ema)) : 0;
                unsigned __int128 act_u;
                if (has_p && b > 0) act_u = (3 * a_sp + a_par + a_xb) / 5;
                else if (has_p)     act_u = (3 * a_sp + a_par) >> 2;
                else if (b > 0)     act_u = (3 * a_sp + a_xb) >> 2;
                else                act_u = a_sp;
                int64_t act = (act_u > 1000000ULL) ? 1000000LL : static_cast<int64_t>(act_u);

                uint8_t c = act_to_ctx(act);
                uint64_t slot = state & MASK_M;
                uint8_t s = lut[c * RANS_M + slot];
                uint64_t f = static_cast<uint64_t>(freqs[c * N_SYMS + s]);
                uint64_t start = static_cast<uint64_t>(cdfs[c * (N_SYMS + 1) + s]);
                state = f * (state >> RANS_BITS) + slot - start;
                while (state < RANS_L && wpos >= 0) {
                    state = (state << 16) | read_le<uint16_t>(w_ptr + wpos * 2);
                    wpos--;
                }
                if (state < RANS_L) throw std::runtime_error("Truncated rANS bitstream");
                uint64_t rem = 0;
                if (s >= 4) {
                    if (s == 30) {
                        rem = br.read(15);
                    } else if (s == 31) {
                        uint64_t w = br.read(6);
                        if (w > 46) throw std::runtime_error("Invalid escape bit width");
                        int m = 16 + static_cast<int>(w);
                        uint64_t r;
                        if (m <= 32) {
                            r = br.read(m);
                        } else {
                            uint64_t hi = br.read(m - 32);
                            uint64_t lo = br.read(32);
                            r = (hi << 32) | lo;
                        }
                        rem = (1ULL << m) | r;
                    } else {
                        int eb = (s >> 1) - 1;
                        rem = br.read(eb);
                    }
                }
                int64_t v = tok_eb_to_val(s, rem);
                pl[y * W + x] = v;

                uint64_t uv = uabs64(v);
                int64_t av = (uv <= (1ULL << 58)) ? static_cast<int64_t>(uv << 4) : (1LL << 62);
                cur_ema = cur_ema + floor_shr128(static_cast<__int128>(av) - cur_ema, 6);
                if (cur_ema < 4) cur_ema = 4;
            }
        }
        prev_ema = cur_ema;
    }
    if (state != RANS_L || wpos != -1)
        throw std::runtime_error("Corrupt base rANS final state");
}

static inline uint8_t sig_ctx(const int64_t* prev_sub, const int64_t* cur_sub,
                              const int64_t* par_sub, bool has_p, int pH, int pW,
                              int b, int y, int x, int Bs, int H, int W) {
    size_t pl_sz = static_cast<size_t>(H) * W;
    const int64_t* cp = cur_sub  + static_cast<size_t>(b) * pl_sz;
    const int64_t* pp = prev_sub + static_cast<size_t>(b) * pl_sz;
    int cur_nz = 0;
    if (x > 0 && cp[y * W + x - 1] != 0) cur_nz += 2;
    if (y > 0 && cp[(y - 1) * W + x] != 0) cur_nz += 2;
    if (y > 0 && x > 0 && cp[(y - 1) * W + x - 1] != 0) cur_nz += 1;
    if (y > 0 && x + 1 < W && cp[(y - 1) * W + x + 1] != 0) cur_nz += 1;

    int prev_nz = 0;
    if (x + 1 < W && pp[y * W + x + 1] != 0) prev_nz += 2;
    if (y + 1 < H && pp[(y + 1) * W + x] != 0) prev_nz += 2;
    if (y + 1 < H && x > 0 && pp[(y + 1) * W + x - 1] != 0) prev_nz += 1;
    if (y + 1 < H && x + 1 < W && pp[(y + 1) * W + x + 1] != 0) prev_nz += 1;

    int par_sig = 0;
    if (has_p) {
        int py = std::min(y >> 1, pH - 1);
        int px = std::min(x >> 1, pW - 1);
        if (par_sub[static_cast<size_t>(b) * pH * pW + py * pW + px] != 0) par_sig = 1;
    }
    int xb_sig = 0;
    if (b > 0 && cur_sub[static_cast<size_t>(b - 1) * pl_sz + y * W + x] != 0) xb_sig = 1;
    else if (b + 1 < Bs && prev_sub[static_cast<size_t>(b + 1) * pl_sz + y * W + x] != 0) xb_sig = 1;

    int sp = cur_nz + prev_nz;
    if (sp == 0) return static_cast<uint8_t>((par_sig << 1) + xb_sig);
    if (sp <= 2) return static_cast<uint8_t>(4 + par_sig);
    if (sp <= 4) return static_cast<uint8_t>(6 + par_sig);
    if (sp <= 7) return static_cast<uint8_t>(8 + par_sig);
    return static_cast<uint8_t>(10 + par_sig);
}

static std::vector<uint8_t> pack_refine_stripe(const int64_t* full_sub, const int64_t* prev_sub,
                                               int64_t* new_sub, int Bs, int H, int W,
                                               const int64_t* par_sub, int pH, int pW, int p_cur) {
    size_t N = static_cast<size_t>(Bs) * H * W;
    std::vector<uint8_t> syms(N), ctxs(N);
    int64_t counts[16 * 3]{};
    bool has_p = (par_sub != nullptr);
    size_t idx = 0;
    for (int b = 0; b < Bs; ++b) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int64_t pv = prev_sub[idx], fv = full_sub[idx];
                int64_t amag = (fv >= 0) ? fv : -fv;
                int bit = static_cast<int>((amag >> p_cur) & 1LL);
                uint8_t c, s; int64_t nv;
                if (pv == 0) {
                    c = sig_ctx(prev_sub, new_sub, par_sub, has_p, pH, pW, b, y, x, Bs, H, W);
                    if (bit == 0)    { s = 0; nv = 0; }
                    else if (fv > 0) { s = 1; nv = 1; }
                    else             { s = 2; nv = -1; }
                } else {
                    int64_t pmag = (pv >= 0) ? pv : -pv;
                    c = (pmag == 1) ? 12 : ((pmag == 2) ? 13 : ((pmag <= 5) ? 14 : 15));
                    s = static_cast<uint8_t>(bit);
                    nv = (pv > 0) ? ((pv << 1) + bit) : -((pmag << 1) + bit);
                }
                new_sub[idx] = nv; syms[idx] = s; ctxs[idx] = c; counts[c * 3 + s]++; idx++;
            }
        }
    }
    std::vector<int32_t> freqs, cdfs; std::vector<uint8_t> tbuf;
    normalize_counts(counts, 16, 3, freqs, cdfs, tbuf, true);

    std::vector<uint16_t> words;
    words.reserve(N / 4 + 32);
    uint64_t state = RANS_L;
    static constexpr uint64_t SHIFT_LIM = (RANS_L >> RANS_BITS) << 16;
    for (size_t i = N; i-- > 0;) {
        uint8_t c = ctxs[i], s = syms[i];
        uint64_t f = static_cast<uint64_t>(freqs[c * 3 + s]);
        uint64_t start = static_cast<uint64_t>(cdfs[c * 4 + s]);
        uint64_t x_max = SHIFT_LIM * f;
        while (state >= x_max) { words.push_back(static_cast<uint16_t>(state & 0xFFFFu)); state >>= 16; }
        state = ((state / f) << RANS_BITS) + (state % f) + start;
    }
    words.push_back(static_cast<uint16_t>(state & 0xFFFFu));
    words.push_back(static_cast<uint16_t>((state >> 16) & 0xFFFFu));

    uint32_t plen = static_cast<uint32_t>(tbuf.size() + 4 + words.size() * 2);
    std::vector<uint8_t> out; out.reserve(4 + plen);
    write_le<uint32_t>(out, plen);
    out.insert(out.end(), tbuf.begin(), tbuf.end());
    write_le<uint32_t>(out, static_cast<uint32_t>(words.size()));
    size_t wp = out.size(); out.resize(wp + words.size() * 2);
    std::memcpy(out.data() + wp, words.data(), words.size() * 2);
    return out;
}

static void unpack_refine_stripe(const uint8_t* p, size_t payload_pos, size_t payload_len,
                                 const int64_t* prev_sub, int64_t* cur_sub, int Bs, int H, int W,
                                 const int64_t* par_sub, int pH, int pW) {
    size_t pos = payload_pos, end_pos = payload_pos + payload_len;
    std::vector<int32_t> freqs, cdfs;
    unpack_tables(p, pos, end_pos, 16, 3, true, freqs, cdfs);
    if (pos + 4 > end_pos) throw std::runtime_error("Truncated stripe refine n_words");
    uint32_t n_words = read_le<uint32_t>(p + pos); pos += 4;
    if (n_words < 2 || static_cast<size_t>(n_words) * 2 != end_pos - pos)
        throw std::runtime_error("Stripe refine framing mismatch");
    const uint8_t* w_ptr = p + pos;

    uint8_t lut[16 * RANS_M];
    for (int c = 0; c < 16; ++c)
        for (int s = 0; s < 3; ++s) {
            int lo = cdfs[c * 4 + s], hi = cdfs[c * 4 + s + 1];
            std::memset(lut + c * RANS_M + lo, s, hi - lo);
        }

    int64_t wpos = static_cast<int64_t>(n_words) - 1;
    uint64_t hi_w = read_le<uint16_t>(w_ptr + wpos * 2); wpos--;
    uint64_t lo_w = read_le<uint16_t>(w_ptr + wpos * 2); wpos--;
    uint64_t state = (hi_w << 16) | lo_w;
    if (state < RANS_L) throw std::runtime_error("Invalid initial rANS state");
    constexpr uint64_t MASK_M = RANS_M - 1;
    bool has_p = (par_sub != nullptr);

    size_t idx = 0;
    for (int b = 0; b < Bs; ++b) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int64_t pv = prev_sub[idx];
                uint8_t c;
                if (pv == 0) {
                    c = sig_ctx(prev_sub, cur_sub, par_sub, has_p, pH, pW, b, y, x, Bs, H, W);
                } else {
                    uint64_t pmag = uabs64(pv);
                    c = (pmag == 1) ? 12 : ((pmag == 2) ? 13 : ((pmag <= 5) ? 14 : 15));
                }
                uint64_t slot = state & MASK_M;
                uint8_t s = lut[c * RANS_M + slot];
                uint64_t f = static_cast<uint64_t>(freqs[c * 3 + s]);
                uint64_t start = static_cast<uint64_t>(cdfs[c * 4 + s]);
                state = f * (state >> RANS_BITS) + slot - start;
                while (state < RANS_L && wpos >= 0) {
                    state = (state << 16) | read_le<uint16_t>(w_ptr + wpos * 2);
                    wpos--;
                }
                if (state < RANS_L) throw std::runtime_error("Truncated rANS bitstream");
                int64_t nv;
                if (pv == 0) nv = (s == 0) ? 0 : ((s == 1) ? 1 : -1);
                else {
                    uint64_t nmag = (uabs64(pv) << 1) + static_cast<uint64_t>(s);
                    nv = (pv > 0) ? static_cast<int64_t>(nmag) : static_cast<int64_t>(0ULL - nmag);
                }
                cur_sub[idx] = nv;
                idx++;
            }
        }
    }
    if (state != RANS_L || wpos != -1)
        throw std::runtime_error("Corrupt refine rANS final state");
}

static inline int64_t dequant_mid(int64_t v, int p) {
    if (p <= 0 || v == 0) return v;
    uint64_t mag = uabs64(v);
    uint64_t dq = (p < 63) ? ((mag << p) + (1ULL << (p - 1))) : 0ULL;
    return (v > 0) ? static_cast<int64_t>(dq) : static_cast<int64_t>(0ULL - dq);
}

static inline void stripe_bounds(int B, int n_stripes, int gi, int& b0, int& b1) {
    b0 = (gi * B) / n_stripes;
    b1 = ((gi + 1) * B) / n_stripes;
}

// =============================================================================
// 6. Axis Helpers, Fast Native Spectral Axis Picker, & HND3 Encode/Decode
// =============================================================================
static void move_axis_to_front(const int64_t* src, int64_t* dst, uint32_t d0, uint32_t d1, uint32_t d2, int ax) {
    size_t tot = static_cast<size_t>(d0) * d1 * d2;
    if (ax == 0) { std::memcpy(dst, src, tot * sizeof(int64_t)); return; }
    if (ax == 1) {
        for (uint32_t i = 0; i < d0; ++i)
            for (uint32_t j = 0; j < d1; ++j)
                std::memcpy(dst + (static_cast<size_t>(j) * d0 + i) * d2,
                            src + (static_cast<size_t>(i) * d1 + j) * d2,
                            static_cast<size_t>(d2) * sizeof(int64_t));
    } else {
        size_t hw = static_cast<size_t>(d0) * d1;
        for (uint32_t i = 0; i < d0; ++i)
            for (uint32_t j = 0; j < d1; ++j) {
                size_t p = static_cast<size_t>(i) * d1 + j;
                const int64_t* sp = src + p * d2;
                for (uint32_t k = 0; k < d2; ++k) dst[static_cast<size_t>(k) * hw + p] = sp[k];
            }
    }
}

static void move_axis_from_front(const int64_t* src, int64_t* dst, uint32_t d0, uint32_t d1, uint32_t d2, int ax) {
    size_t tot = static_cast<size_t>(d0) * d1 * d2;
    if (ax == 0) { std::memcpy(dst, src, tot * sizeof(int64_t)); return; }
    if (ax == 1) {
        for (uint32_t i = 0; i < d0; ++i)
            for (uint32_t j = 0; j < d1; ++j)
                std::memcpy(dst + (static_cast<size_t>(i) * d1 + j) * d2,
                            src + (static_cast<size_t>(j) * d0 + i) * d2,
                            static_cast<size_t>(d2) * sizeof(int64_t));
    } else {
        size_t hw = static_cast<size_t>(d0) * d1;
        for (uint32_t i = 0; i < d0; ++i)
            for (uint32_t j = 0; j < d1; ++j) {
                size_t p = static_cast<size_t>(i) * d1 + j;
                int64_t* dp = dst + p * d2;
                for (uint32_t k = 0; k < d2; ++k) dp[k] = src[static_cast<size_t>(k) * hw + p];
            }
    }
}

// Fast native C++20 spectral-axis role picker: compares all 3 candidate spectral axes AND plain 2D 5/3
// (SPEC_NONE, returns -1) on the EXACT SAME centered 3D subcube so voxel subsets are apples-to-apples!
static int choose_spectral_axis_cpp(const int64_t* orig, uint32_t d0, uint32_t d1, uint32_t d2, int n_threads) {
    uint32_t dims[3] = {d0, d1, d2};
    uint32_t sd[3];
    uint32_t o0[3];
    for (int i = 0; i < 3; ++i) {
        sd[i] = (dims[i] >= 8) ? (std::min<uint32_t>(dims[i], 48) / 4 * 4) : dims[i];
        o0[i] = (dims[i] - sd[i]) / 2;
    }

    auto score_sub = [&](std::vector<int64_t>& sub, int B, int H, int W, bool do_ldu) -> double {
        size_t cHW = static_cast<size_t>(H) * W;
        if (do_ldu && B > 1) {
            for (int b = 0; b < B; ++b) {
                int64_t* row = sub.data() + static_cast<size_t>(b) * cHW;
                double s = 0.0; for (size_t i = 0; i < cHW; ++i) s += static_cast<double>(row[i]);
                int64_t m = std::llround(s / static_cast<double>(cHW));
                for (size_t i = 0; i < cHW; ++i) row[i] -= m;
            }
            std::vector<uint16_t> heads;
            for (int g = 0; g < B; g += 8) {
                std::vector<uint16_t> grp;
                for (int i = g; i < std::min(g + 8, B); ++i) grp.push_back(static_cast<uint16_t>(i));
                if (grp.size() > 1) {
                    auto U = klt_u_jacobi(sub.data(), static_cast<int>(grp.size()), cHW, grp);
                    auto st = complete_pivoted_ldu(U, grp);
                    ldu_apply_stage(sub.data(), cHW, st, false, false, 1);
                }
                heads.push_back(grp[0]);
            }
            while (heads.size() > 1) {
                std::vector<uint16_t> nxt;
                for (size_t g = 0; g < heads.size(); g += 8) {
                    std::vector<uint16_t> grp(heads.begin() + g, heads.begin() + std::min(g + 8, heads.size()));
                    if (grp.size() > 1) {
                        auto U = klt_u_jacobi(sub.data(), static_cast<int>(grp.size()), cHW, grp);
                        auto st = complete_pivoted_ldu(U, grp);
                        ldu_apply_stage(sub.data(), cHW, st, false, false, 1);
                    }
                    nxt.push_back(grp[0]);
                }
                heads = std::move(nxt);
            }
        }
        auto pl = build_dwt_plan(H, W, 2);
        dwt53_2d_cube(sub.data(), B, H, W, pl, false, std::min(n_threads, 4));
        double sc = 0.0;
        for (int64_t v : sub) sc += std::log2(static_cast<double>(std::abs(v)) + 1.0);
        return sc / static_cast<double>(sub.size());
    };

    auto extract_ax = [&](int ax) -> std::vector<int64_t> {
        int B = static_cast<int>(sd[ax]);
        int H = static_cast<int>(sd[(ax == 0) ? 1 : 0]);
        int W = static_cast<int>(sd[(ax == 2) ? 1 : 2]);
        size_t cHW = static_cast<size_t>(H) * W;
        std::vector<int64_t> sub(static_cast<size_t>(B) * cHW);
        for (int b = 0; b < B; ++b) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    uint32_t i0, i1, i2;
                    if (ax == 0)      { i0 = o0[0] + b; i1 = o0[1] + y; i2 = o0[2] + x; }
                    else if (ax == 1) { i0 = o0[0] + y; i1 = o0[1] + b; i2 = o0[2] + x; }
                    else              { i0 = o0[0] + y; i1 = o0[1] + x; i2 = o0[2] + b; }
                    sub[static_cast<size_t>(b) * cHW + y * W + x] =
                        orig[(static_cast<size_t>(i0) * d1 + i1) * d2 + i2];
                }
            }
        }
        return sub;
    };

    int best_ax = 0;
    double best_score = 1e30;
    for (int ax = 0; ax < 3; ++ax) {
        int B = static_cast<int>(sd[ax]);
        int H = static_cast<int>(sd[(ax == 0) ? 1 : 0]);
        int W = static_cast<int>(sd[(ax == 2) ? 1 : 2]);
        if (B < 2 || H < 4 || W < 4) continue;
        auto sub = extract_ax(ax);
        double sc = score_sub(sub, B, H, W, true);
        if (sc < best_score) { best_score = sc; best_ax = ax; }
    }
    // Also score SPEC_NONE (plain 2D 5/3 without LDU on axis 0) so non-spectral 3D volumes skip harmful LDU
    if (sd[1] >= 4 && sd[2] >= 4) {
        auto sub_none = extract_ax(0);
        double sc_none = score_sub(sub_none, static_cast<int>(sd[0]), static_cast<int>(sd[1]), static_cast<int>(sd[2]), false);
        if (sc_none < best_score) return -1;
    }
    return best_ax;
}


} // namespace hnd3
