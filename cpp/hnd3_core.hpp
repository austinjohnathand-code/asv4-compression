#pragma once
#include "hnd3.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace hnd3 {

static constexpr uint8_t MAGIC[4]   = {'H', 'N', 'D', '3'};
static constexpr uint8_t VERSION    = 3;
static constexpr uint8_t FLAG_BLIND = 1 << 0;
static constexpr uint8_t FLAG_LOSSY = 1 << 1;
static constexpr uint8_t FLAG_RAW   = 1 << 2;

static constexpr uint8_t DT_I16 = 1;
static constexpr uint8_t DT_U16 = 2;
static constexpr uint8_t DT_I32 = 3;

static constexpr uint8_t SPEC_NONE          = 0;
static constexpr uint8_t SPEC_FULL_LDU      = 1;
static constexpr uint8_t SPEC_CLUSTERED_LDU = 2;

static constexpr int     LDU_SHIFT = 15;
static constexpr int64_t LDU_SCALE = 1LL << LDU_SHIFT;
static constexpr int64_t LDU_HALF  = 1LL << (LDU_SHIFT - 1);

static constexpr int      RANS_BITS = 12;
static constexpr uint32_t RANS_M    = 1U << RANS_BITS;
static constexpr uint64_t RANS_L    = 1ULL << 16;
static constexpr int      N_SYMS    = 32;
static constexpr int      N_CTXS    = 16;
static constexpr size_t   HDR_SIZE  = 62;

static_assert(std::endian::native == std::endian::little, "HND3/HND4 requires a little-endian host");

template <typename T>
inline T read_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

template <typename T>
inline void write_le(std::vector<uint8_t>& buf, T v) {
    size_t pos = buf.size();
    buf.resize(pos + sizeof(T));
    std::memcpy(buf.data() + pos, &v, sizeof(T));
}

inline int64_t floor_shr(int64_t x, int s) { return x >> s; }

// =============================================================================
// 1. Built-in FIPS 180-4 SHA-256
// =============================================================================
class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        total_ = 0; buf_len_ = 0;
    }
    void update(const uint8_t* data, size_t len) {
        total_ += len;
        size_t i = 0;
        if (buf_len_ > 0) {
            while (i < len && buf_len_ < 64) buf_[buf_len_++] = data[i++];
            if (buf_len_ == 64) { transform(buf_); buf_len_ = 0; }
        }
        while (i + 64 <= len) { transform(data + i); i += 64; }
        while (i < len) buf_[buf_len_++] = data[i++];
    }
    std::array<uint8_t, 32> digest() {
        uint64_t bits = total_ * 8ULL;
        uint8_t pad = 0x80; update(&pad, 1);
        uint8_t zero = 0;
        while (buf_len_ != 56) update(&zero, 1);
        uint8_t len_be[8];
        for (int k = 0; k < 8; ++k) len_be[7 - k] = static_cast<uint8_t>((bits >> (k * 8)) & 0xFFu);
        update(len_be, 8);
        std::array<uint8_t, 32> out{};
        for (int k = 0; k < 8; ++k) {
            out[k*4+0] = static_cast<uint8_t>((h_[k] >> 24) & 0xFFu);
            out[k*4+1] = static_cast<uint8_t>((h_[k] >> 16) & 0xFFu);
            out[k*4+2] = static_cast<uint8_t>((h_[k] >> 8) & 0xFFu);
            out[k*4+3] = static_cast<uint8_t>(h_[k] & 0xFFu);
        }
        return out;
    }
private:
    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    void transform(const uint8_t blk[64]) {
        static constexpr uint32_t K[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(blk[i*4])<<24)|(uint32_t(blk[i*4+1])<<16)|(uint32_t(blk[i*4+2])<<8)|uint32_t(blk[i*4+3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],hh=h_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t ch = (e&f)^((~e)&g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a,2)^rotr(a,13)^rotr(a,22);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h_[0]+=a; h_[1]+=b; h_[2]+=c; h_[3]+=d; h_[4]+=e; h_[5]+=f; h_[6]+=g; h_[7]+=hh;
    }
    std::array<uint32_t, 8> h_{};
    uint64_t total_ = 0;
    uint8_t buf_[64]{};
    size_t buf_len_ = 0;
};

static std::array<uint8_t, 32> sha256_bytes(const uint8_t* p, size_t n) {
    Sha256 s; s.update(p, n); return s.digest();
}

// =============================================================================
// 1b. Castagnoli CRC32C (0x82F63B78, 256-entry lookup table, zero deps)
// =============================================================================
inline uint32_t crc32c_impl(const uint8_t* data, size_t len, uint32_t init = 0u) {
    static const std::array<uint32_t, 256> tbl = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    uint32_t c = ~init;
    for (size_t i = 0; i < len; ++i) c = tbl[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    return ~c;
}

static constexpr uint8_t MAGIC4[4]          = {'H', 'N', 'D', '4'};
static constexpr uint8_t VERSION4           = 4;
static constexpr uint8_t FLAG_TILED         = 1 << 3;
static constexpr uint8_t FLAG_CRC32C        = 1 << 4;
static constexpr uint8_t FLAG_NEAR_LOSSLESS = 1 << 5;
static constexpr uint8_t FLAG_HIST_LUT      = 1 << 6;
static constexpr size_t  HDR4_SIZE          = 84;
static constexpr size_t  TILE_ENTRY_SIZE    = 8 + 4 + 16 * 4 + 16 * 4; // 140 B

template <typename Fn>
static void parallel_for(size_t count, int n_threads, Fn&& fn) {
    if (count == 0) return;
    int nt = std::max(1, std::min(n_threads, static_cast<int>(count)));
    if (nt == 1) { fn(0, count); return; }
    std::vector<std::thread> workers;
    workers.reserve(nt);
    std::mutex err_mtx;
    std::exception_ptr first_err = nullptr;
    size_t chunk = (count + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        size_t lo = static_cast<size_t>(t) * chunk;
        size_t hi = std::min(count, lo + chunk);
        if (lo >= hi) break;
        workers.emplace_back([&, lo, hi]() {
            try { fn(lo, hi); }
            catch (...) {
                std::lock_guard<std::mutex> lk(err_mtx);
                if (!first_err) first_err = std::current_exception();
            }
        });
    }
    for (auto& w : workers) w.join();
    if (first_err) std::rethrow_exception(first_err);
}

// =============================================================================
// 2. Native C++20 Blind Track A (Radix-2 FFT Centered Pearson ACF + Axis-Peeling)
// =============================================================================
static void fft_radix2(std::vector<double>& re, std::vector<double>& im, bool inverse) {
    size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    constexpr double PI = 3.14159265358979323846264338327950288;
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = (inverse ? 2.0 : -2.0) * PI / static_cast<double>(len);
        double wlen_re = std::cos(ang), wlen_im = std::sin(ang);
        size_t half = len >> 1;
        // Precompute twiddles for this butterfly stage (fast & accurate)
        std::vector<double> tw_re(half), tw_im(half);
        double wr = 1.0, wi = 0.0;
        for (size_t k = 0; k < half; ++k) {
            tw_re[k] = wr; tw_im[k] = wi;
            double nwr = wr * wlen_re - wi * wlen_im;
            wi = wr * wlen_im + wi * wlen_re;
            wr = nwr;
        }
        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < half; ++j) {
                double ur = re[i + j], ui = im[i + j];
                double vr = re[i + j + half] * tw_re[j] - im[i + j + half] * tw_im[j];
                double vi = re[i + j + half] * tw_im[j] + im[i + j + half] * tw_re[j];
                re[i + j]        = ur + vr;
                im[i + j]        = ui + vi;
                re[i + j + half] = ur - vr;
                im[i + j + half] = ui - vi;
            }
        }
    }
    if (inverse) {
        double inv_n = 1.0 / static_cast<double>(n);
        for (size_t i = 0; i < n; ++i) { re[i] *= inv_n; im[i] *= inv_n; }
    }
}

static std::vector<double> acf_fft_pearson(const double* x, size_t n, size_t maxlag) {
    std::vector<double> rho(maxlag + 1, 0.0);
    if (n < 4) return rho;
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += x[i];
    mean /= static_cast<double>(n);

    std::vector<double> cs(n + 1, 0.0), cs2(n + 1, 0.0);
    double total_var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double xc = x[i] - mean;
        cs[i + 1]  = cs[i] + xc;
        cs2[i + 1] = cs2[i] + xc * xc;
        total_var += xc * xc;
    }
    if (total_var <= 1e-12 * static_cast<double>(n)) { rho[0] = 1.0; return rho; }

    size_t nfft = 1;
    while (nfft < 2 * n) nfft <<= 1;
    std::vector<double> re(nfft, 0.0), im(nfft, 0.0);
    for (size_t i = 0; i < n; ++i) re[i] = x[i] - mean;
    fft_radix2(re, im, false);
    for (size_t i = 0; i < nfft; ++i) {
        re[i] = re[i] * re[i] + im[i] * im[i];
        im[i] = 0.0;
    }
    fft_radix2(re, im, true);

    double var_scale = 1e-9 * (total_var / static_cast<double>(n));
    rho[0] = 1.0;
    for (size_t tau = 1; tau <= maxlag; ++tau) {
        double cnt = static_cast<double>(n - tau);
        double sum_l = cs[n - tau];
        double sum_r = cs[n] - cs[tau];
        double sum2_l = cs2[n - tau];
        double sum2_r = cs2[n] - cs2[tau];
        double var_l = sum2_l - (sum_l * sum_l) / cnt;
        double var_r = sum2_r - (sum_r * sum_r) / cnt;
        double vfloor = var_scale * cnt;
        if (var_l > vfloor && var_r > vfloor) {
            double cov = re[tau] - (sum_l * sum_r) / cnt;
            double r = cov / std::max(std::sqrt(var_l * var_r), 1e-30);
            rho[tau] = std::clamp(r, -1.0, 1.0);
        }
    }
    return rho;
}

// SciPy-equivalent local peak finder with prominence & height thresholds
struct Peak { size_t idx; double val; double prom; };

static std::vector<Peak> find_peaks_prom(const std::vector<double>& g_pos, double min_prom, double min_val) {
    size_t m = g_pos.size();
    std::vector<Peak> out;
    if (m < 3) return out;
    for (size_t i = 1; i + 1 < m; ++i) {
        double v = g_pos[i];
        if (v < min_val) continue;
        if (!(v > g_pos[i - 1] && v >= g_pos[i + 1])) continue;
        // Handle plateau
        size_t j = i;
        while (j + 1 < m && g_pos[j + 1] == v) j++;
        if (j + 1 < m && g_pos[j + 1] > v) { i = j; continue; }
        size_t pk = (i + j) / 2;
        // Left & right prominence search
        double lmin = v;
        for (size_t l = pk; l-- > 0;) {
            if (g_pos[l] < lmin) lmin = g_pos[l];
            if (g_pos[l] > v) break;
        }
        double rmin = v;
        for (size_t r = pk + 1; r < m; ++r) {
            if (g_pos[r] < rmin) rmin = g_pos[r];
            if (g_pos[r] > v) break;
        }
        double prom = v - std::max(lmin, rmin);
        if (prom >= min_prom) out.push_back({pk, v, prom});
        i = j;
    }
    return out;
}

static double vec_var(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double s = 0.0;
    for (double x : v) { double d = x - mean; s += d * d; }
    return s / static_cast<double>(v.size());
}

static std::vector<size_t> recover_strides_cpp(const std::vector<double>& x, size_t N_full, int max_axes = 5,
                                               double min_prom = 0.10, double min_val = 0.20) {
    size_t L_orig = x.size();
    size_t limit = L_orig / 4;
    if (limit < 4) return {1};

    std::vector<double> cur = x;
    double var_x = vec_var(x);
    double var_cur = var_x;
    auto rho_raw0 = acf_fft_pearson(x.data(), L_orig, limit);

    std::vector<size_t> chain = {1};
    while (static_cast<int>(chain.size()) < max_axes) {
        size_t s_prev = chain.back();
        size_t max_k = std::min(limit / s_prev, (L_orig / 2) / s_prev);
        if (max_k < 4) break;

        size_t cp_len = cur.size() - s_prev;
        std::vector<double> cur_peeled(cp_len);
        for (size_t i = 0; i < cp_len; ++i) cur_peeled[i] = cur[i + s_prev] - cur[i];
        size_t lim_p = std::min(limit, cp_len - 2);
        auto rho_p = acf_fft_pearson(cur_peeled.data(), cp_len, lim_p);

        struct Curve { std::vector<double> g; int tag; }; // 0=raw, 1=peel, 2=xpeel
        std::vector<Curve> curves;
        {
            std::vector<double> g0(max_k + 1), gp;
            for (size_t k = 0; k <= max_k; ++k) g0[k] = rho_raw0[k * s_prev];
            for (size_t k = 0; k <= max_k && k * s_prev <= lim_p; ++k) gp.push_back(rho_p[k * s_prev]);
            curves.push_back({std::move(g0), 0});
            curves.push_back({std::move(gp), 1});
        }
        if (s_prev > 1 && L_orig > s_prev + 4) {
            size_t xp_len = L_orig - s_prev;
            std::vector<double> x_peeled(xp_len);
            for (size_t i = 0; i < xp_len; ++i) x_peeled[i] = x[i + s_prev] - x[i];
            if (vec_var(x_peeled) > 2e-3 * var_x) {
                size_t lim_xp = std::min(limit, xp_len - 2);
                auto rho_xp = acf_fft_pearson(x_peeled.data(), xp_len, lim_xp);
                std::vector<double> gxp;
                for (size_t k = 0; k <= max_k && k * s_prev <= lim_xp; ++k) gxp.push_back(rho_xp[k * s_prev]);
                curves.push_back({std::move(gxp), 2});
            }
        }

        struct Cand { size_t k; double pr; double val; int tag; };
        std::vector<uint8_t> is_raw_pk(max_k + 1, 0);
        std::vector<Cand> cands;
        auto find_cand = [&](size_t k) -> Cand* {
            for (auto& c : cands) if (c.k == k) return &c;
            return nullptr;
        };
        auto get_curve = [&](int tag) -> const std::vector<double>& {
            for (const auto& cv : curves) if (cv.tag == tag) return cv.g;
            return curves[0].g;
        };

        for (const auto& cv : curves) {
            if (cv.g.size() < 4) continue;
            std::vector<double> g_pos(cv.g.size());
            for (size_t i = 0; i < cv.g.size(); ++i) g_pos[i] = std::max(cv.g[i], 0.0);
            auto pks = find_peaks_prom(g_pos, min_prom, min_val);
            for (const auto& pk : pks) {
                size_t k = pk.idx;
                if (k < 2) continue;
                double val = cv.g[k], pr_f = pk.prom;
                if (cv.tag == 0) {
                    is_raw_pk[k] = 1;
                } else {
                    if (k == 2 && !is_raw_pk[k] && (val < 0.50 || pr_f < 0.30)) continue;
                    if (!is_raw_pk[k] && 2 * k < cv.g.size() && val < 0.50 && cv.g[2 * k] < 0.05) continue;
                }
                if (auto* ex = find_cand(k)) {
                    if (pr_f * val > ex->pr * ex->val) { ex->pr = pr_f; ex->val = val; ex->tag = cv.tag; }
                } else {
                    cands.push_back({k, pr_f, val, cv.tag});
                }
            }
        }
        if (cands.empty()) break;

        double max_pr = 0.0, max_val = 0.0;
        for (const auto& c : cands) { max_pr = std::max(max_pr, c.pr); max_val = std::max(max_val, c.val); }
        size_t best_k = 0;
        for (const auto& c : cands) {
            if (c.pr >= 0.55 * max_pr && c.val >= 0.65 * max_val) {
                if (best_k == 0 || c.k < best_k) best_k = c.k;
            }
        }
        if (best_k == 0) {
            double bs = -1.0;
            for (const auto& c : cands) if (c.pr * c.val > bs) { bs = c.pr * c.val; best_k = c.k; }
        }

        // Harmonic-verified divisor-refinement (exact Phase 0 v2.2 rule)
        if (best_k >= 4) {
            const Cand* bc = find_cand(best_k);
            std::vector<size_t> div_cands;
            size_t sq = static_cast<size_t>(std::sqrt(static_cast<double>(best_k)));
            for (size_t d = 2; d <= sq; ++d) {
                if (best_k % d != 0) continue;
                size_t pair_d[2] = {d, best_k / d};
                for (size_t dd : pair_d) {
                    if (dd < 2 || dd >= best_k) continue;
                    const Cand* dc = find_cand(dd);
                    if (!dc) continue;
                    if (dc->pr >= std::max(min_prom, 0.10) && dc->val >= std::max(min_val, 0.25)) {
                        const auto& g_ref = get_curve(dc->tag);
                        bool harm_ok = false;
                        if (2 * dd < best_k) {
                            harm_ok = (2 * dd < g_ref.size() && g_ref[2 * dd] >= 0.15);
                        } else {
                            if (3 * dd < g_ref.size())
                                harm_ok = (is_raw_pk[dd] || dc->val >= 0.40) && (g_ref[3 * dd] >= 0.15);
                            else
                                harm_ok = (dc->val >= 0.65 * bc->val);
                        }
                        if (harm_ok) div_cands.push_back(dd);
                    }
                }
            }
            if (!div_cands.empty()) {
                double best_ds = 0.0;
                for (size_t dd : div_cands) {
                    const Cand* dc = find_cand(dd);
                    best_ds = std::max(best_ds, dc->pr * dc->val);
                }
                size_t pick = best_k;
                for (size_t dd : div_cands) {
                    const Cand* dc = find_cand(dd);
                    if (dc->pr * dc->val >= 0.50 * best_ds && dd < pick) pick = dd;
                }
                best_k = pick;
            }
        }

        size_t s_next = best_k * s_prev;
        // Once a full 3D lattice [1, s1, s2] is recovered, reject any extra 4th+ candidate
        // that does not divide the stream length N_full (eliminates non-lattice urban echoes).
        if (chain.size() >= 3 && (N_full % s_next != 0)) break;
        chain.push_back(s_next);
        double var_p = vec_var(cur_peeled);
        if (var_p < var_cur) { cur = std::move(cur_peeled); var_cur = var_p; }
    }
    return chain;
}

// =============================================================================
// 3. Complete-Pivoted LDU-RKLT with True ARM64 NEON vmlal_s32 Widening FMA
// =============================================================================
struct LduStage {
    int M = 0;
    std::vector<uint16_t> grp, rperm, cperm, dperm;
    std::vector<int8_t> sign0, dsgn;
    std::vector<int16_t> Lq, Uq;
    std::vector<int32_t> cq, inv_cq_m1, cq_m1;
};

static std::vector<double> klt_u_jacobi(const int64_t* X, int M, size_t N, const std::vector<uint16_t>& grp) {
    size_t step = std::max<size_t>(1, N / 40000);
    size_t ns = (N + step - 1) / step;
    std::vector<double> C(static_cast<size_t>(M) * M, 0.0);
    std::vector<double> col(M);
    for (size_t n = 0; n < N; n += step) {
        for (int i = 0; i < M; ++i) col[i] = static_cast<double>(X[static_cast<size_t>(grp[i]) * N + n]);
        for (int i = 0; i < M; ++i) {
            double ci = col[i];
            for (int j = i; j < M; ++j) C[i * M + j] += ci * col[j];
        }
    }
    double inv_ns = 1.0 / static_cast<double>(ns);
    for (int i = 0; i < M; ++i)
        for (int j = i; j < M; ++j) { double v = C[i * M + j] * inv_ns; C[i * M + j] = C[j * M + i] = v; }

    std::vector<double> V(static_cast<size_t>(M) * M, 0.0);
    for (int i = 0; i < M; ++i) V[i * M + i] = 1.0;
    double scale = 0.0;
    for (int i = 0; i < M; ++i) scale = std::max(scale, std::abs(C[i * M + i]));
    double tol = 1e-13 * std::max(scale, 1.0);

    for (int sweep = 0; sweep < 50; ++sweep) {
        double max_off = 0.0;
        for (int p = 0; p < M - 1; ++p) {
            for (int q = p + 1; q < M; ++q) {
                double apq = C[p * M + q];
                max_off = std::max(max_off, std::abs(apq));
                if (std::abs(apq) <= tol) continue;
                double app = C[p * M + p], aqq = C[q * M + q];
                double tau = (aqq - app) / (2.0 * apq);
                double t = (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                                        : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
                double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                C[p * M + p] = app - t * apq; C[q * M + q] = aqq + t * apq;
                C[p * M + q] = C[q * M + p] = 0.0;
                for (int r = 0; r < M; ++r) {
                    if (r == p || r == q) continue;
                    double arp = C[r * M + p], arq = C[r * M + q];
                    C[r * M + p] = C[p * M + r] = c * arp - s * arq;
                    C[r * M + q] = C[q * M + r] = s * arp + c * arq;
                }
                for (int r = 0; r < M; ++r) {
                    double vrp = V[r * M + p], vrq = V[r * M + q];
                    V[r * M + p] = c * vrp - s * vrq;
                    V[r * M + q] = s * vrp + c * vrq;
                }
            }
        }
        if (max_off <= tol) break;
    }
    std::vector<int> order(M);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return C[a * M + a] > C[b * M + b]; });
    std::vector<double> U(static_cast<size_t>(M) * M);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j) U[i * M + j] = V[j * M + order[i]];
    return U;
}

static LduStage complete_pivoted_ldu(const std::vector<double>& A_in, const std::vector<uint16_t>& grp) {
    int M = static_cast<int>(grp.size());
    std::vector<double> W = A_in;
    LduStage st; st.M = M; st.grp = grp; st.sign0.assign(M, 1);
    st.rperm.resize(M); st.cperm.resize(M);

    auto factor_once = [&](std::vector<double> Wm, std::vector<uint16_t>& rp, std::vector<uint16_t>& cp,
                           std::vector<double>& L, std::vector<double>& U, std::vector<double>& d) -> double {
        L.assign(static_cast<size_t>(M) * M, 0.0);
        U.assign(static_cast<size_t>(M) * M, 0.0);
        d.assign(M, 0.0);
        for (int i = 0; i < M; ++i) { rp[i] = i; cp[i] = i; L[i*M+i] = 1.0; U[i*M+i] = 1.0; }
        int swaps = 0;
        for (int k = 0; k < M; ++k) {
            double best = -1.0; int bp = k, bq = k;
            for (int i = k; i < M; ++i)
                for (int j = k; j < M; ++j) {
                    double v = std::abs(Wm[i * M + j]);
                    if (v > best) { best = v; bp = i; bq = j; }
                }
            if (bp != k) {
                for (int j = 0; j < M; ++j) std::swap(Wm[k * M + j], Wm[bp * M + j]);
                for (int j = 0; j < k; ++j) std::swap(L[k * M + j], L[bp * M + j]);
                std::swap(rp[k], rp[bp]); swaps++;
            }
            if (bq != k) {
                for (int i = 0; i < M; ++i) std::swap(Wm[i * M + k], Wm[i * M + bq]);
                for (int i = 0; i < k; ++i) std::swap(U[i * M + k], U[i * M + bq]);
                std::swap(cp[k], cp[bq]); swaps++;
            }
            double piv = Wm[k * M + k];
            if (!std::isfinite(piv) || std::abs(piv) < 1e-12) {
                // Degenerate/singular fallback: set remaining block to identity so LDU stays finite & bijective
                for (int i = k; i < M; ++i) {
                    d[i] = 1.0;
                    for (int j = k; j < i; ++j) L[i * M + j] = 0.0;
                    for (int j = i + 1; j < M; ++j) U[i * M + j] = 0.0;
                }
                break;
            }
            d[k] = piv;
            if (k + 1 < M) {
                for (int i = k + 1; i < M; ++i) L[i * M + k] = Wm[i * M + k] / piv;
                for (int j = k + 1; j < M; ++j) U[k * M + j] = Wm[k * M + j] / piv;
                for (int i = k + 1; i < M; ++i) {
                    double lik = L[i * M + k];
                    for (int j = k + 1; j < M; ++j) Wm[i * M + j] -= lik * Wm[k * M + j];
                }
            }
        }
        double det = (swaps & 1) ? -1.0 : 1.0;
        for (int i = 0; i < M; ++i) det *= d[i];
        return det;
    };

    std::vector<double> L, U, d;
    if (factor_once(W, st.rperm, st.cperm, L, U, d) < 0.0) {
        st.sign0[0] = -1;
        for (int j = 0; j < M; ++j) W[j] = -W[j];
        factor_once(W, st.rperm, st.cperm, L, U, d);
    }

    st.dsgn.resize(M);
    std::vector<double> d_abs(M), log_d(M);
    for (int i = 0; i < M; ++i) {
        st.dsgn[i] = (d[i] >= 0.0) ? 1 : -1;
        d_abs[i] = std::max(std::abs(d[i]), 1e-12);
        log_d[i] = std::log(d_abs[i]);
    }
    std::vector<uint8_t> used(M, 0);
    st.dperm.resize(M);
    double log_cur = 0.0;
    for (int step = 0; step < M; ++step) {
        int best_i = -1; double best_v = 1e30;
        for (int i = 0; i < M; ++i) if (!used[i]) {
            double v = std::abs(log_cur + log_d[i]);
            if (v < best_v) { best_v = v; best_i = i; }
        }
        used[best_i] = 1; st.dperm[step] = static_cast<uint16_t>(best_i); log_cur += log_d[best_i];
    }

    st.Lq.assign(static_cast<size_t>(M) * M, 0);
    st.Uq.assign(static_cast<size_t>(M) * M, 0);
    auto clamp_i16 = [](double v) -> int16_t {
        if (!std::isfinite(v)) return 0;
        long long q = std::llround(std::clamp(v * static_cast<double>(LDU_SCALE), -32767.0, 32767.0));
        return static_cast<int16_t>(std::clamp<long long>(q, -32767, 32767));
    };
    auto clamp_i32 = [](double v) -> int32_t {
        if (!std::isfinite(v)) return 0;
        double s = std::clamp(v * static_cast<double>(LDU_SCALE), -2147483520.0, 2147483520.0);
        return static_cast<int32_t>(std::llround(s));
    };
    for (int i = 1; i < M; ++i)
        for (int j = 0; j < i; ++j) st.Lq[i * M + j] = clamp_i16(L[i * M + j]);
    for (int i = 0; i < M - 1; ++i)
        for (int j = i + 1; j < M; ++j) st.Uq[i * M + j] = clamp_i16(U[i * M + j]);

    st.cq.resize(M - 1); st.inv_cq_m1.resize(M - 1); st.cq_m1.resize(M - 1);
    double prod = 1.0;
    for (int k = 0; k < M - 1; ++k) {
        prod = std::clamp(prod * d_abs[st.dperm[k]], 1e-6, 1e6);
        st.cq[k]        = clamp_i32(prod);
        st.inv_cq_m1[k] = clamp_i32(1.0 / prod - 1.0);
        st.cq_m1[k]     = clamp_i32(prod - 1.0);
    }
    return st;
}

// Overflow-safe modular integer helpers (compile to identical 1-cycle instructions; zero signed-overflow UB on fuzzed bitstreams)
static inline int32_t wrap_add32(int32_t a, int32_t b) { return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b)); }
static inline int32_t wrap_sub32(int32_t a, int32_t b) { return static_cast<int32_t>(static_cast<uint32_t>(a) - static_cast<uint32_t>(b)); }
static inline int32_t wrap_neg32(int32_t a)            { return static_cast<int32_t>(0u - static_cast<uint32_t>(a)); }
static inline int64_t wrap_add64(int64_t a, int64_t b) { return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b)); }
static inline int64_t wrap_sub64(int64_t a, int64_t b) { return static_cast<int64_t>(static_cast<uint64_t>(a) - static_cast<uint64_t>(b)); }
static inline int64_t wrap_neg64(int64_t a)            { return static_cast<int64_t>(0ULL - static_cast<uint64_t>(a)); }
static inline uint64_t uabs64(int64_t v)               { return (v >= 0) ? static_cast<uint64_t>(v) : (0ULL - static_cast<uint64_t>(v)); }

static inline int64_t floor_shr128(__int128 x, int k) {
    return static_cast<int64_t>(x >> k);
}

static inline int64_t mul_shr_ldu(int64_t c, int64_t x) {
    __int128 p = static_cast<__int128>(c) * static_cast<__int128>(x) + static_cast<__int128>(LDU_HALF);
    return static_cast<int64_t>(p >> LDU_SHIFT);
}

// True ARM64 NEON 32x32->64 widening FMA path (exact bit-for-bit with int64 path when |Y| < 2^28!)
static inline void ldu_row_madd_i32_neon(int32_t* dst, const int32_t* Y, int nb,
                                         const int16_t* coeffs, int j_lo, int j_hi, int sign_mul) {
    int t = 0;
#if defined(__aarch64__) && !defined(HND3_FORCE_SCALAR)
    const int64x2_t vhalf = vdupq_n_s64(LDU_HALF);
    for (; t + 7 < nb; t += 8) {
        int64x2_t a0 = vdupq_n_s64(0), a1 = vdupq_n_s64(0);
        int64x2_t a2 = vdupq_n_s64(0), a3 = vdupq_n_s64(0);
        for (int j = j_lo; j < j_hi; ++j) {
            int32_t c = coeffs[j];
            if (c == 0) continue;
            int32x2_t vc = vdup_n_s32(c);
            const int32_t* row = Y + static_cast<size_t>(j) * nb + t;
            int32x4_t v01 = vld1q_s32(row);
            int32x4_t v23 = vld1q_s32(row + 4);
            a0 = vmlal_s32(a0, vget_low_s32(v01), vc);
            a1 = vmlal_high_s32(a1, v01, vdupq_n_s32(c));
            a2 = vmlal_s32(a2, vget_low_s32(v23), vc);
            a3 = vmlal_high_s32(a3, v23, vdupq_n_s32(c));
        }
        int32x2_t d0 = vmovn_s64(vshrq_n_s64(vaddq_s64(a0, vhalf), LDU_SHIFT));
        int32x2_t d1 = vmovn_s64(vshrq_n_s64(vaddq_s64(a1, vhalf), LDU_SHIFT));
        int32x2_t d2 = vmovn_s64(vshrq_n_s64(vaddq_s64(a2, vhalf), LDU_SHIFT));
        int32x2_t d3 = vmovn_s64(vshrq_n_s64(vaddq_s64(a3, vhalf), LDU_SHIFT));
        int32x4_t d01 = vcombine_s32(d0, d1);
        int32x4_t d23 = vcombine_s32(d2, d3);
        int32x4_t c01 = vld1q_s32(dst + t);
        int32x4_t c23 = vld1q_s32(dst + t + 4);
        if (sign_mul > 0) {
            vst1q_s32(dst + t,     vaddq_s32(c01, d01));
            vst1q_s32(dst + t + 4, vaddq_s32(c23, d23));
        } else {
            vst1q_s32(dst + t,     vsubq_s32(c01, d01));
            vst1q_s32(dst + t + 4, vsubq_s32(c23, d23));
        }
    }
#endif
    for (; t < nb; ++t) {
        __int128 acc = 0;
        for (int j = j_lo; j < j_hi; ++j)
            acc += static_cast<__int128>(coeffs[j]) * static_cast<__int128>(Y[static_cast<size_t>(j) * nb + t]);
        int32_t delta = static_cast<int32_t>(floor_shr128(acc + LDU_HALF, LDU_SHIFT));
        dst[t] = (sign_mul > 0) ? wrap_add32(dst[t], delta) : wrap_sub32(dst[t], delta);
    }
}

static inline void ldu_row_madd_i64(int64_t* dst, const int64_t* Y, int nb,
                                    const int16_t* coeffs, int j_lo, int j_hi, int sign_mul) {
    for (int t = 0; t < nb; ++t) {
        __int128 acc = 0;
        for (int j = j_lo; j < j_hi; ++j)
            acc += static_cast<__int128>(coeffs[j]) * static_cast<__int128>(Y[static_cast<size_t>(j) * nb + t]);
        int64_t delta = floor_shr128(acc + LDU_HALF, LDU_SHIFT);
        dst[t] = (sign_mul > 0) ? wrap_add64(dst[t], delta) : wrap_sub64(dst[t], delta);
    }
}

static void ldu_apply_stage(int64_t* X, size_t N, const LduStage& st, bool inverse, bool use_i32_neon, int n_threads) {
    const int M = st.M;
    static constexpr size_t BLK = 1024;
    size_t n_blks = (N + BLK - 1) / BLK;
    parallel_for(n_blks, n_threads, [&](size_t b_lo, size_t b_hi) {
        if (use_i32_neon) {
            std::vector<int32_t> Y(static_cast<size_t>(M) * BLK);
            for (size_t bi = b_lo; bi < b_hi; ++bi) {
                size_t n0 = bi * BLK;
                int nb = static_cast<int>(std::min(BLK, N - n0));
                if (!inverse) {
                    for (int i = 0; i < M; ++i) {
                        const int64_t* src = X + static_cast<size_t>(st.grp[st.cperm[i]]) * N + n0;
                        int32_t* dst = Y.data() + static_cast<size_t>(i) * nb;
                        for (int t = 0; t < nb; ++t) dst[t] = static_cast<int32_t>(src[t]);
                    }
                    for (int i = 0; i < M - 1; ++i)
                        ldu_row_madd_i32_neon(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Uq.data() + i * M, i + 1, M, +1);
                    for (int k = 0; k < M - 1; ++k) {
                        int32_t* ya = Y.data() + static_cast<size_t>(st.dperm[k]) * nb;
                        int32_t* yb = Y.data() + static_cast<size_t>(st.dperm[k + 1]) * nb;
                        int64_t cq = st.cq[k], icm1 = st.inv_cq_m1[k], cm1 = st.cq_m1[k];
                        for (int t = 0; t < nb; ++t) {
                            int64_t x = ya[t], y = yb[t];
                            y = wrap_sub64(y, mul_shr_ldu(cq, x));
                            x = wrap_add64(x, mul_shr_ldu(icm1, y));
                            y = wrap_add64(y, x);
                            x = wrap_add64(x, mul_shr_ldu(cm1, y));
                            ya[t] = static_cast<int32_t>(x); yb[t] = static_cast<int32_t>(y);
                        }
                    }
                    for (int i = 0; i < M; ++i) if (st.dsgn[i] < 0) {
                        int32_t* row = Y.data() + static_cast<size_t>(i) * nb;
                        for (int t = 0; t < nb; ++t) row[t] = wrap_neg32(row[t]);
                    }
                    for (int i = M - 1; i >= 1; --i)
                        ldu_row_madd_i32_neon(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Lq.data() + i * M, 0, i, +1);
                    for (int i = 0; i < M; ++i) {
                        int ri = st.rperm[i]; int64_t s = st.sign0[ri];
                        int64_t* dst = X + static_cast<size_t>(st.grp[ri]) * N + n0;
                        const int32_t* src = Y.data() + static_cast<size_t>(i) * nb;
                        if (s > 0) for (int t = 0; t < nb; ++t) dst[t] = src[t];
                        else       for (int t = 0; t < nb; ++t) dst[t] = wrap_neg64(static_cast<int64_t>(src[t]));
                    }
                } else {
                    for (int i = 0; i < M; ++i) {
                        int ri = st.rperm[i]; int64_t s = st.sign0[ri];
                        const int64_t* src = X + static_cast<size_t>(st.grp[ri]) * N + n0;
                        int32_t* dst = Y.data() + static_cast<size_t>(i) * nb;
                        if (s > 0) for (int t = 0; t < nb; ++t) dst[t] = static_cast<int32_t>(src[t]);
                        else       for (int t = 0; t < nb; ++t) dst[t] = static_cast<int32_t>(wrap_neg64(src[t]));
                    }
                    for (int i = 1; i < M; ++i)
                        ldu_row_madd_i32_neon(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Lq.data() + i * M, 0, i, -1);
                    for (int i = 0; i < M; ++i) if (st.dsgn[i] < 0) {
                        int32_t* row = Y.data() + static_cast<size_t>(i) * nb;
                        for (int t = 0; t < nb; ++t) row[t] = wrap_neg32(row[t]);
                    }
                    for (int k = M - 2; k >= 0; --k) {
                        int32_t* ya = Y.data() + static_cast<size_t>(st.dperm[k]) * nb;
                        int32_t* yb = Y.data() + static_cast<size_t>(st.dperm[k + 1]) * nb;
                        int64_t cq = st.cq[k], icm1 = st.inv_cq_m1[k], cm1 = st.cq_m1[k];
                        for (int t = 0; t < nb; ++t) {
                            int64_t x = ya[t], y = yb[t];
                            x = wrap_sub64(x, mul_shr_ldu(cm1, y));
                            y = wrap_sub64(y, x);
                            x = wrap_sub64(x, mul_shr_ldu(icm1, y));
                            y = wrap_add64(y, mul_shr_ldu(cq, x));
                            ya[t] = static_cast<int32_t>(x); yb[t] = static_cast<int32_t>(y);
                        }
                    }
                    for (int i = M - 2; i >= 0; --i)
                        ldu_row_madd_i32_neon(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Uq.data() + i * M, i + 1, M, -1);
                    for (int i = 0; i < M; ++i) {
                        int64_t* dst = X + static_cast<size_t>(st.grp[st.cperm[i]]) * N + n0;
                        const int32_t* src = Y.data() + static_cast<size_t>(i) * nb;
                        for (int t = 0; t < nb; ++t) dst[t] = src[t];
                    }
                }
            }
        } else {
            std::vector<int64_t> Y(static_cast<size_t>(M) * BLK);
            for (size_t bi = b_lo; bi < b_hi; ++bi) {
                size_t n0 = bi * BLK;
                int nb = static_cast<int>(std::min(BLK, N - n0));
                if (!inverse) {
                    for (int i = 0; i < M; ++i)
                        std::memcpy(Y.data() + static_cast<size_t>(i) * nb,
                                    X + static_cast<size_t>(st.grp[st.cperm[i]]) * N + n0,
                                    static_cast<size_t>(nb) * sizeof(int64_t));
                    for (int i = 0; i < M - 1; ++i)
                        ldu_row_madd_i64(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Uq.data() + i * M, i + 1, M, +1);
                    for (int k = 0; k < M - 1; ++k) {
                        int64_t* ya = Y.data() + static_cast<size_t>(st.dperm[k]) * nb;
                        int64_t* yb = Y.data() + static_cast<size_t>(st.dperm[k + 1]) * nb;
                        int64_t cq = st.cq[k], icm1 = st.inv_cq_m1[k], cm1 = st.cq_m1[k];
                        for (int t = 0; t < nb; ++t) {
                            int64_t x = ya[t], y = yb[t];
                            y = wrap_sub64(y, mul_shr_ldu(cq, x));
                            x = wrap_add64(x, mul_shr_ldu(icm1, y));
                            y = wrap_add64(y, x);
                            x = wrap_add64(x, mul_shr_ldu(cm1, y));
                            ya[t] = x; yb[t] = y;
                        }
                    }
                    for (int i = 0; i < M; ++i) if (st.dsgn[i] < 0)
                        for (int t = 0; t < nb; ++t) Y[static_cast<size_t>(i) * nb + t] = wrap_neg64(Y[static_cast<size_t>(i) * nb + t]);
                    for (int i = M - 1; i >= 1; --i)
                        ldu_row_madd_i64(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Lq.data() + i * M, 0, i, +1);
                    for (int i = 0; i < M; ++i) {
                        int ri = st.rperm[i]; int64_t s = st.sign0[ri];
                        int64_t* dst = X + static_cast<size_t>(st.grp[ri]) * N + n0;
                        const int64_t* src = Y.data() + static_cast<size_t>(i) * nb;
                        if (s > 0) std::memcpy(dst, src, static_cast<size_t>(nb) * sizeof(int64_t));
                        else for (int t = 0; t < nb; ++t) dst[t] = wrap_neg64(src[t]);
                    }
                } else {
                    for (int i = 0; i < M; ++i) {
                        int ri = st.rperm[i]; int64_t s = st.sign0[ri];
                        const int64_t* src = X + static_cast<size_t>(st.grp[ri]) * N + n0;
                        int64_t* dst = Y.data() + static_cast<size_t>(i) * nb;
                        if (s > 0) std::memcpy(dst, src, static_cast<size_t>(nb) * sizeof(int64_t));
                        else for (int t = 0; t < nb; ++t) dst[t] = wrap_neg64(src[t]);
                    }
                    for (int i = 1; i < M; ++i)
                        ldu_row_madd_i64(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Lq.data() + i * M, 0, i, -1);
                    for (int i = 0; i < M; ++i) if (st.dsgn[i] < 0)
                        for (int t = 0; t < nb; ++t) Y[static_cast<size_t>(i) * nb + t] = wrap_neg64(Y[static_cast<size_t>(i) * nb + t]);
                    for (int k = M - 2; k >= 0; --k) {
                        int64_t* ya = Y.data() + static_cast<size_t>(st.dperm[k]) * nb;
                        int64_t* yb = Y.data() + static_cast<size_t>(st.dperm[k + 1]) * nb;
                        int64_t cq = st.cq[k], icm1 = st.inv_cq_m1[k], cm1 = st.cq_m1[k];
                        for (int t = 0; t < nb; ++t) {
                            int64_t x = ya[t], y = yb[t];
                            x = wrap_sub64(x, mul_shr_ldu(cm1, y));
                            y = wrap_sub64(y, x);
                            x = wrap_sub64(x, mul_shr_ldu(icm1, y));
                            y = wrap_add64(y, mul_shr_ldu(cq, x));
                            ya[t] = x; yb[t] = y;
                        }
                    }
                    for (int i = M - 2; i >= 0; --i)
                        ldu_row_madd_i64(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Uq.data() + i * M, i + 1, M, -1);
                    for (int i = 0; i < M; ++i)
                        std::memcpy(X + static_cast<size_t>(st.grp[st.cperm[i]]) * N + n0,
                                    Y.data() + static_cast<size_t>(i) * nb,
                                    static_cast<size_t>(nb) * sizeof(int64_t));
                }
            }
        }
    });
}

static void pack_ldu_stage(std::vector<uint8_t>& buf, const LduStage& st) {
    int M = st.M;
    write_le<uint16_t>(buf, static_cast<uint16_t>(M));
    for (int i = 0; i < M; ++i) write_le<uint16_t>(buf, st.grp[i]);
    for (int i = 0; i < M; ++i) write_le<uint16_t>(buf, st.rperm[i]);
    for (int i = 0; i < M; ++i) write_le<uint16_t>(buf, st.cperm[i]);
    for (int i = 0; i < M; ++i) write_le<uint16_t>(buf, st.dperm[i]);
    int nb = (M + 7) / 8;
    for (int b = 0; b < nb; ++b) {
        uint8_t byte = 0;
        for (int bit = 0; bit < 8 && b * 8 + bit < M; ++bit)
            if (st.sign0[b * 8 + bit] < 0) byte |= static_cast<uint8_t>(1U << (7 - bit));
        buf.push_back(byte);
    }
    for (int b = 0; b < nb; ++b) {
        uint8_t byte = 0;
        for (int bit = 0; bit < 8 && b * 8 + bit < M; ++bit)
            if (st.dsgn[b * 8 + bit] < 0) byte |= static_cast<uint8_t>(1U << (7 - bit));
        buf.push_back(byte);
    }
    for (int i = 1; i < M; ++i) for (int j = 0; j < i; ++j) write_le<int16_t>(buf, st.Lq[i * M + j]);
    for (int i = 0; i < M - 1; ++i) for (int j = i + 1; j < M; ++j) write_le<int16_t>(buf, st.Uq[i * M + j]);
    for (int k = 0; k < M - 1; ++k) write_le<int32_t>(buf, st.cq[k]);
    for (int k = 0; k < M - 1; ++k) write_le<int32_t>(buf, st.inv_cq_m1[k]);
    for (int k = 0; k < M - 1; ++k) write_le<int32_t>(buf, st.cq_m1[k]);
}

static LduStage unpack_ldu_stage(const uint8_t* p, size_t& pos, size_t end_pos, int B_max) {
    if (pos + 2 > end_pos) throw std::runtime_error("Truncated LDU stage M");
    LduStage st;
    int M = read_le<uint16_t>(p + pos); pos += 2;
    if (M < 2 || M > B_max) throw std::runtime_error("Invalid LDU stage M");
    size_t nb = static_cast<size_t>((M + 7) / 8);
    size_t need = static_cast<size_t>(M) * 8 + 2 * nb + static_cast<size_t>(M) * (M - 1) * 2
                + static_cast<size_t>(M - 1) * 12;
    if (pos + need > end_pos) throw std::runtime_error("Truncated LDU stage payload");
    st.M = M;
    st.grp.resize(M); st.rperm.resize(M); st.cperm.resize(M); st.dperm.resize(M);
    for (int i = 0; i < M; ++i) {
        st.grp[i] = read_le<uint16_t>(p + pos); pos += 2;
        if (st.grp[i] >= static_cast<uint16_t>(B_max)) throw std::runtime_error("OOB LDU grp");
    }
    for (int i = 0; i < M; ++i) {
        st.rperm[i] = read_le<uint16_t>(p + pos); pos += 2;
        if (st.rperm[i] >= static_cast<uint16_t>(M)) throw std::runtime_error("OOB LDU rperm");
    }
    for (int i = 0; i < M; ++i) {
        st.cperm[i] = read_le<uint16_t>(p + pos); pos += 2;
        if (st.cperm[i] >= static_cast<uint16_t>(M)) throw std::runtime_error("OOB LDU cperm");
    }
    for (int i = 0; i < M; ++i) {
        st.dperm[i] = read_le<uint16_t>(p + pos); pos += 2;
        if (st.dperm[i] >= static_cast<uint16_t>(M)) throw std::runtime_error("OOB LDU dperm");
    }
    st.sign0.assign(M, 1); st.dsgn.assign(M, 1);
    for (int i = 0; i < M; ++i) if ((p[pos + (i >> 3)] >> (7 - (i & 7))) & 1) st.sign0[i] = -1;
    pos += nb;
    for (int i = 0; i < M; ++i) if ((p[pos + (i >> 3)] >> (7 - (i & 7))) & 1) st.dsgn[i] = -1;
    pos += nb;
    st.Lq.assign(static_cast<size_t>(M) * M, 0);
    st.Uq.assign(static_cast<size_t>(M) * M, 0);
    for (int i = 1; i < M; ++i) for (int j = 0; j < i; ++j) { st.Lq[i * M + j] = read_le<int16_t>(p + pos); pos += 2; }
    for (int i = 0; i < M - 1; ++i) for (int j = i + 1; j < M; ++j) { st.Uq[i * M + j] = read_le<int16_t>(p + pos); pos += 2; }
    st.cq.resize(M - 1); st.inv_cq_m1.resize(M - 1); st.cq_m1.resize(M - 1);
    for (int k = 0; k < M - 1; ++k) { st.cq[k]        = read_le<int32_t>(p + pos); pos += 4; }
    for (int k = 0; k < M - 1; ++k) { st.inv_cq_m1[k] = read_le<int32_t>(p + pos); pos += 4; }
    for (int k = 0; k < M - 1; ++k) { st.cq_m1[k]     = read_le<int32_t>(p + pos); pos += 4; }
    return st;
}

// =============================================================================
// 4. Multi-Threaded 2D 5/3 Integer Lifting
// =============================================================================
static inline void fwd53_1d(const int64_t* in, int64_t* out, int N, int64_t* tmp_e, int64_t* tmp_d) {
    if (N < 2) { if (N == 1) out[0] = in[0]; return; }
    int ne = (N + 1) >> 1, no = N >> 1;
    for (int i = 0; i < ne; ++i) tmp_e[i] = in[i * 2];
    for (int i = 0; i < no; ++i) {
        int nxt = (i + 1 < ne) ? (i + 1) : (ne - 1);
        int64_t pred = floor_shr128(static_cast<__int128>(tmp_e[i]) + tmp_e[nxt], 1);
        tmp_d[i] = wrap_sub64(in[i * 2 + 1], pred);
    }
    for (int i = 0; i < ne; ++i) {
        int cur = (i < no) ? i : (no - 1);
        int prv = (i > 0) ? ((i - 1 < no) ? (i - 1) : (no - 1)) : 0;
        int64_t upd = floor_shr128(static_cast<__int128>(tmp_d[prv]) + tmp_d[cur] + 2, 2);
        out[i] = wrap_add64(tmp_e[i], upd);
    }
    for (int i = 0; i < no; ++i) out[ne + i] = tmp_d[i];
}

static inline void inv53_1d(const int64_t* in, int64_t* out, int N, int64_t* tmp_e) {
    if (N < 2) { if (N == 1) out[0] = in[0]; return; }
    int ne = (N + 1) >> 1, no = N >> 1;
    const int64_t* s = in; const int64_t* d = in + ne;
    for (int i = 0; i < ne; ++i) {
        int cur = (i < no) ? i : (no - 1);
        int prv = (i > 0) ? ((i - 1 < no) ? (i - 1) : (no - 1)) : 0;
        int64_t upd = floor_shr128(static_cast<__int128>(d[prv]) + d[cur] + 2, 2);
        tmp_e[i] = wrap_sub64(s[i], upd);
    }
    for (int i = 0; i < ne; ++i) out[i * 2] = tmp_e[i];
    for (int i = 0; i < no; ++i) {
        int nxt = (i + 1 < ne) ? (i + 1) : (ne - 1);
        int64_t pred = floor_shr128(static_cast<__int128>(tmp_e[i]) + tmp_e[nxt], 1);
        out[i * 2 + 1] = wrap_add64(d[i], pred);
    }
}

struct LevelPlan { int H, W; };

static std::vector<LevelPlan> build_dwt_plan(int H, int W, int L) {
    std::vector<LevelPlan> plan;
    int ch = H, cw = W;
    for (int lv = 0; lv < L; ++lv) {
        if (ch < 2 && cw < 2) break;
        plan.push_back({ch, cw});
        if (ch >= 2) ch = (ch + 1) >> 1;
        if (cw >= 2) cw = (cw + 1) >> 1;
    }
    return plan;
}

static void dwt53_2d_cube(int64_t* cube, int B, int H, int W, const std::vector<LevelPlan>& plan,
                          bool inverse, int n_threads) {
    size_t plane_sz = static_cast<size_t>(H) * W;
    int max_dim = std::max(H, W);
    parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b_lo, size_t b_hi) {
        std::vector<int64_t> col_in(max_dim), col_out(max_dim), tmp_e(max_dim), tmp_d(max_dim), row_tmp(max_dim);
        for (size_t b = b_lo; b < b_hi; ++b) {
            int64_t* plane = cube + b * plane_sz;
            if (!inverse) {
                for (const auto& lp : plan) {
                    if (lp.H >= 2) {
                        for (int x = 0; x < lp.W; ++x) {
                            for (int y = 0; y < lp.H; ++y) col_in[y] = plane[static_cast<size_t>(y) * W + x];
                            fwd53_1d(col_in.data(), col_out.data(), lp.H, tmp_e.data(), tmp_d.data());
                            for (int y = 0; y < lp.H; ++y) plane[static_cast<size_t>(y) * W + x] = col_out[y];
                        }
                    }
                    if (lp.W >= 2) {
                        for (int y = 0; y < lp.H; ++y) {
                            int64_t* row = plane + static_cast<size_t>(y) * W;
                            fwd53_1d(row, row_tmp.data(), lp.W, tmp_e.data(), tmp_d.data());
                            std::memcpy(row, row_tmp.data(), static_cast<size_t>(lp.W) * sizeof(int64_t));
                        }
                    }
                }
            } else {
                for (int li = static_cast<int>(plan.size()) - 1; li >= 0; --li) {
                    const auto& lp = plan[li];
                    if (lp.W >= 2) {
                        for (int y = 0; y < lp.H; ++y) {
                            int64_t* row = plane + static_cast<size_t>(y) * W;
                            inv53_1d(row, row_tmp.data(), lp.W, tmp_e.data());
                            std::memcpy(row, row_tmp.data(), static_cast<size_t>(lp.W) * sizeof(int64_t));
                        }
                    }
                    if (lp.H >= 2) {
                        for (int x = 0; x < lp.W; ++x) {
                            for (int y = 0; y < lp.H; ++y) col_in[y] = plane[static_cast<size_t>(y) * W + x];
                            inv53_1d(col_in.data(), col_out.data(), lp.H, tmp_e.data());
                            for (int y = 0; y < lp.H; ++y) plane[static_cast<size_t>(y) * W + x] = col_out[y];
                        }
                    }
                }
            }
        }
    });
}

struct SubbandDesc {
    int y0, y1, x0, x1, lv, par_idx, offset;
    int h() const { return y1 - y0; }
    int w() const { return x1 - x0; }
};

static std::vector<SubbandDesc> build_c2f_schedule(int H, int W, const std::vector<LevelPlan>& plan) {
    int n_lv = static_cast<int>(plan.size());
    if (n_lv == 0) return {{0, H, 0, W, 0, -1, 0}};
    struct Rect { int y0, y1, x0, x1; };
    std::vector<std::vector<Rect>> per_lv(n_lv);
    int last_h = H, last_w = W;
    for (int li = 0; li < n_lv; ++li) {
        int bh = plan[li].H, bw = plan[li].W;
        int ah = (bh >= 2) ? ((bh + 1) >> 1) : bh;
        int aw = (bw >= 2) ? ((bw + 1) >> 1) : bw;
        std::vector<int> tr;
        if (bh >= 2) tr.push_back(1);
        if (bw >= 2) tr.push_back(2);
        int cnt = (1 << tr.size()) - 1;
        for (int mask = 1; mask <= cnt; ++mask) {
            int y0 = 0, y1 = bh, x0 = 0, x1 = bw;
            for (size_t bit = 0; bit < tr.size(); ++bit) {
                bool hi = ((mask >> bit) & 1) != 0;
                if (tr[bit] == 1) { y0 = hi ? ah : 0; y1 = hi ? bh : ah; }
                else              { x0 = hi ? aw : 0; x1 = hi ? bw : aw; }
            }
            per_lv[li].push_back({y0, y1, x0, x1});
        }
        last_h = ah; last_w = aw;
    }
    std::vector<SubbandDesc> sched;
    sched.push_back({0, last_h, 0, last_w, 0, -1, std::max(0, n_lv - 1)});
    int prev_start = 0, prev_cnt = 1;
    for (int lr = n_lv - 1; lr >= 0; --lr) {
        int lv_num = n_lv - lr;
        const auto& grp = per_lv[lr];
        int cur_start = static_cast<int>(sched.size());
        for (int oi = 0; oi < static_cast<int>(grp.size()); ++oi) {
            int par = (prev_cnt == 1) ? 0 : (prev_start + std::min(oi, prev_cnt - 1));
            int off = (oi == 2) ? std::max(0, n_lv - 1 - lv_num) : std::max(0, n_lv - lv_num);
            sched.push_back({grp[oi].y0, grp[oi].y1, grp[oi].x0, grp[oi].x1, lv_num, par, off});
        }
        prev_start = cur_start; prev_cnt = static_cast<int>(grp.size());
    }
    return sched;
}

// =============================================================================

} // namespace hnd3
