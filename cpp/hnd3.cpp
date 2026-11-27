// =============================================================================
// HND3 — Phase 3 Production C++20 ARM64 NEON Hyperspectral & N-D Codec
//
// Phase 3 upgrades over HND2:
//   1. Band-Stripe Parallel 32-bit rANS (up to 16 independent band stripes per
//      subband, each with its own compact 12-bit frequency table -> unlocks ALL
//      CPU cores on both base & refinement layers while improving energy-tier
//      entropy adaptation!).
//   2. True ARM64 NEON 32x32->64 widening FMA (vmlal_s32 / vmlal_high_s32) for
//      LDU-RKLT lifting with automatic overflow-safe int64 fallback.
//   3. 64-bit bit-buffer extra-bits packing/unpacking (branch-minimal inner loop).
//   4. Native zero-dependency C++20 Blind Track A (--blind): iterative radix-2
//      FFT centered Pearson ACF + sequential axis-peeling recover_strides() +
//      fast spectral-axis role picker + zero-tax raw fallback for noise.
//
// Zero external dependencies. Wire-compatible bit-for-bit with phase3_codec.py.
// =============================================================================
#include <algorithm>
#include <array>
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
        d_abs[i] = std::abs(d[i]);
        log_d[i] = std::log(std::max(d_abs[i], 1e-15));
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
        long long q = std::llround(v * static_cast<double>(LDU_SCALE));
        return static_cast<int16_t>(std::clamp<long long>(q, -32767, 32767));
    };
    for (int i = 1; i < M; ++i)
        for (int j = 0; j < i; ++j) st.Lq[i * M + j] = clamp_i16(L[i * M + j]);
    for (int i = 0; i < M - 1; ++i)
        for (int j = i + 1; j < M; ++j) st.Uq[i * M + j] = clamp_i16(U[i * M + j]);

    st.cq.resize(M - 1); st.inv_cq_m1.resize(M - 1); st.cq_m1.resize(M - 1);
    double prod = 1.0;
    for (int k = 0; k < M - 1; ++k) {
        prod *= d_abs[st.dperm[k]];
        st.cq[k]        = static_cast<int32_t>(std::llround(prod * static_cast<double>(LDU_SCALE)));
        st.inv_cq_m1[k] = static_cast<int32_t>(std::llround((1.0 / prod - 1.0) * static_cast<double>(LDU_SCALE)));
        st.cq_m1[k]     = static_cast<int32_t>(std::llround((prod - 1.0) * static_cast<double>(LDU_SCALE)));
    }
    return st;
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
        int64_t acc = 0;
        for (int j = j_lo; j < j_hi; ++j)
            acc += static_cast<int64_t>(coeffs[j]) * static_cast<int64_t>(Y[static_cast<size_t>(j) * nb + t]);
        int32_t delta = static_cast<int32_t>(floor_shr(acc + LDU_HALF, LDU_SHIFT));
        dst[t] += (sign_mul > 0) ? delta : -delta;
    }
}

static inline void ldu_row_madd_i64(int64_t* dst, const int64_t* Y, int nb,
                                    const int16_t* coeffs, int j_lo, int j_hi, int sign_mul) {
    for (int t = 0; t < nb; ++t) {
        int64_t acc = 0;
        for (int j = j_lo; j < j_hi; ++j) acc += static_cast<int64_t>(coeffs[j]) * Y[static_cast<size_t>(j) * nb + t];
        int64_t delta = floor_shr(acc + LDU_HALF, LDU_SHIFT);
        dst[t] += (sign_mul > 0) ? delta : -delta;
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
                            y -= floor_shr(cq * x + LDU_HALF, LDU_SHIFT);
                            x += floor_shr(icm1 * y + LDU_HALF, LDU_SHIFT);
                            y += x;
                            x += floor_shr(cm1 * y + LDU_HALF, LDU_SHIFT);
                            ya[t] = static_cast<int32_t>(x); yb[t] = static_cast<int32_t>(y);
                        }
                    }
                    for (int i = 0; i < M; ++i) if (st.dsgn[i] < 0) {
                        int32_t* row = Y.data() + static_cast<size_t>(i) * nb;
                        for (int t = 0; t < nb; ++t) row[t] = -row[t];
                    }
                    for (int i = M - 1; i >= 1; --i)
                        ldu_row_madd_i32_neon(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Lq.data() + i * M, 0, i, +1);
                    for (int i = 0; i < M; ++i) {
                        int ri = st.rperm[i]; int64_t s = st.sign0[ri];
                        int64_t* dst = X + static_cast<size_t>(st.grp[ri]) * N + n0;
                        const int32_t* src = Y.data() + static_cast<size_t>(i) * nb;
                        if (s > 0) for (int t = 0; t < nb; ++t) dst[t] = src[t];
                        else       for (int t = 0; t < nb; ++t) dst[t] = -static_cast<int64_t>(src[t]);
                    }
                } else {
                    for (int i = 0; i < M; ++i) {
                        int ri = st.rperm[i]; int64_t s = st.sign0[ri];
                        const int64_t* src = X + static_cast<size_t>(st.grp[ri]) * N + n0;
                        int32_t* dst = Y.data() + static_cast<size_t>(i) * nb;
                        if (s > 0) for (int t = 0; t < nb; ++t) dst[t] = static_cast<int32_t>(src[t]);
                        else       for (int t = 0; t < nb; ++t) dst[t] = static_cast<int32_t>(-src[t]);
                    }
                    for (int i = 1; i < M; ++i)
                        ldu_row_madd_i32_neon(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Lq.data() + i * M, 0, i, -1);
                    for (int i = 0; i < M; ++i) if (st.dsgn[i] < 0) {
                        int32_t* row = Y.data() + static_cast<size_t>(i) * nb;
                        for (int t = 0; t < nb; ++t) row[t] = -row[t];
                    }
                    for (int k = M - 2; k >= 0; --k) {
                        int32_t* ya = Y.data() + static_cast<size_t>(st.dperm[k]) * nb;
                        int32_t* yb = Y.data() + static_cast<size_t>(st.dperm[k + 1]) * nb;
                        int64_t cq = st.cq[k], icm1 = st.inv_cq_m1[k], cm1 = st.cq_m1[k];
                        for (int t = 0; t < nb; ++t) {
                            int64_t x = ya[t], y = yb[t];
                            x -= floor_shr(cm1 * y + LDU_HALF, LDU_SHIFT);
                            y -= x;
                            x -= floor_shr(icm1 * y + LDU_HALF, LDU_SHIFT);
                            y += floor_shr(cq * x + LDU_HALF, LDU_SHIFT);
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
                            y -= floor_shr(cq * x + LDU_HALF, LDU_SHIFT);
                            x += floor_shr(icm1 * y + LDU_HALF, LDU_SHIFT);
                            y += x;
                            x += floor_shr(cm1 * y + LDU_HALF, LDU_SHIFT);
                            ya[t] = x; yb[t] = y;
                        }
                    }
                    for (int i = 0; i < M; ++i) if (st.dsgn[i] < 0)
                        for (int t = 0; t < nb; ++t) Y[static_cast<size_t>(i) * nb + t] = -Y[static_cast<size_t>(i) * nb + t];
                    for (int i = M - 1; i >= 1; --i)
                        ldu_row_madd_i64(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Lq.data() + i * M, 0, i, +1);
                    for (int i = 0; i < M; ++i) {
                        int ri = st.rperm[i]; int64_t s = st.sign0[ri];
                        int64_t* dst = X + static_cast<size_t>(st.grp[ri]) * N + n0;
                        const int64_t* src = Y.data() + static_cast<size_t>(i) * nb;
                        if (s > 0) std::memcpy(dst, src, static_cast<size_t>(nb) * sizeof(int64_t));
                        else for (int t = 0; t < nb; ++t) dst[t] = -src[t];
                    }
                } else {
                    for (int i = 0; i < M; ++i) {
                        int ri = st.rperm[i]; int64_t s = st.sign0[ri];
                        const int64_t* src = X + static_cast<size_t>(st.grp[ri]) * N + n0;
                        int64_t* dst = Y.data() + static_cast<size_t>(i) * nb;
                        if (s > 0) std::memcpy(dst, src, static_cast<size_t>(nb) * sizeof(int64_t));
                        else for (int t = 0; t < nb; ++t) dst[t] = -src[t];
                    }
                    for (int i = 1; i < M; ++i)
                        ldu_row_madd_i64(Y.data() + static_cast<size_t>(i) * nb, Y.data(), nb, st.Lq.data() + i * M, 0, i, -1);
                    for (int i = 0; i < M; ++i) if (st.dsgn[i] < 0)
                        for (int t = 0; t < nb; ++t) Y[static_cast<size_t>(i) * nb + t] = -Y[static_cast<size_t>(i) * nb + t];
                    for (int k = M - 2; k >= 0; --k) {
                        int64_t* ya = Y.data() + static_cast<size_t>(st.dperm[k]) * nb;
                        int64_t* yb = Y.data() + static_cast<size_t>(st.dperm[k + 1]) * nb;
                        int64_t cq = st.cq[k], icm1 = st.inv_cq_m1[k], cm1 = st.cq_m1[k];
                        for (int t = 0; t < nb; ++t) {
                            int64_t x = ya[t], y = yb[t];
                            x -= floor_shr(cm1 * y + LDU_HALF, LDU_SHIFT);
                            y -= x;
                            x -= floor_shr(icm1 * y + LDU_HALF, LDU_SHIFT);
                            y += floor_shr(cq * x + LDU_HALF, LDU_SHIFT);
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
        tmp_d[i] = in[i * 2 + 1] - floor_shr(tmp_e[i] + tmp_e[nxt], 1);
    }
    for (int i = 0; i < ne; ++i) {
        int cur = (i < no) ? i : (no - 1);
        int prv = (i > 0) ? ((i - 1 < no) ? (i - 1) : (no - 1)) : 0;
        out[i] = tmp_e[i] + floor_shr(tmp_d[prv] + tmp_d[cur] + 2, 2);
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
        tmp_e[i] = s[i] - floor_shr(d[prv] + d[cur] + 2, 2);
    }
    for (int i = 0; i < ne; ++i) out[i * 2] = tmp_e[i];
    for (int i = 0; i < no; ++i) {
        int nxt = (i + 1 < ne) ? (i + 1) : (ne - 1);
        out[i * 2 + 1] = d[i] + floor_shr(tmp_e[i] + tmp_e[nxt], 1);
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

                int64_t av = std::abs(v) << 4;
                cur_ema = cur_ema + floor_shr(av - cur_ema, 6);
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
                    int64_t pmag = (pv >= 0) ? pv : -pv;
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
                    int64_t pmag = (pv >= 0) ? pv : -pv;
                    nv = (pv > 0) ? ((pv << 1) + s) : -((pmag << 1) + s);
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
    int64_t mag = (v > 0) ? v : -v;
    int64_t dq = (mag << p) + (1LL << (p - 1));
    return (v > 0) ? dq : -dq;
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

static std::vector<uint8_t> encode_hnd3(const uint8_t* raw_bytes, uint32_t d0, uint32_t d1, uint32_t d2,
                                        uint8_t dt_code, const std::string& spec_mode,
                                        uint8_t n_lv, uint8_t spec_axis, uint8_t base_k,
                                        uint8_t n_stripes_req, uint8_t flags, int n_threads) {
    size_t tot = static_cast<size_t>(d0) * d1 * d2;
    size_t itemsize = (dt_code == DT_I32) ? 4 : 2;
    auto sha = sha256_bytes(raw_bytes, tot * itemsize);

    std::vector<int64_t> orig(tot);
    if (dt_code == DT_I16)      for (size_t i = 0; i < tot; ++i) orig[i] = read_le<int16_t>(raw_bytes + i * 2);
    else if (dt_code == DT_U16) for (size_t i = 0; i < tot; ++i) orig[i] = read_le<uint16_t>(raw_bytes + i * 2);
    else                        for (size_t i = 0; i < tot; ++i) orig[i] = read_le<int32_t>(raw_bytes + i * 4);

    uint32_t dims[3] = {d0, d1, d2};
    int B = static_cast<int>(dims[spec_axis]);
    int H = static_cast<int>(dims[(spec_axis == 0) ? 1 : 0]);
    int W = static_cast<int>(dims[(spec_axis == 2) ? 1 : 2]);
    size_t HW = static_cast<size_t>(H) * W;
    uint8_t n_stripes = static_cast<uint8_t>(std::max(1, std::min<int>(n_stripes_req, B)));
#if defined(HND3_FORCE_SCALAR)
    bool use_i32_neon = false;
#else
    bool use_i32_neon = (dt_code != DT_I32);
#endif

    std::vector<int64_t> work(tot);
    move_axis_to_front(orig.data(), work.data(), d0, d1, d2, spec_axis);
    orig.clear(); orig.shrink_to_fit();

    uint8_t spec_code = SPEC_NONE;
    std::vector<uint8_t> spec_bytes;
    if (B > 1 && spec_mode != "none") {
        if (spec_mode == "auto") {
            double full_hdr_bits = 16.0 * (double(B) * (B - 1) + 12.0 * B);
            spec_code = (B <= 16 || full_hdr_bits / double(tot) < 0.04) ? SPEC_FULL_LDU : SPEC_CLUSTERED_LDU;
        } else if (spec_mode == "full") spec_code = SPEC_FULL_LDU;
        else spec_code = SPEC_CLUSTERED_LDU;

        std::vector<int32_t> mu(B);
        parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b) {
                int64_t* row = work.data() + b * HW;
                double s = 0.0;
                for (size_t i = 0; i < HW; ++i) s += static_cast<double>(row[i]);
                int32_t m = static_cast<int32_t>(std::llround(s / static_cast<double>(HW)));
                mu[b] = m;
                for (size_t i = 0; i < HW; ++i) row[i] -= m;
            }
        });

        std::vector<LduStage> stages;
        auto apply_grp = [&](const std::vector<uint16_t>& grp) {
            if (grp.size() <= 1) return;
            auto U = klt_u_jacobi(work.data(), static_cast<int>(grp.size()), HW, grp);
            auto st = complete_pivoted_ldu(U, grp);
            ldu_apply_stage(work.data(), HW, st, false, use_i32_neon, n_threads);
            stages.push_back(std::move(st));
        };
        if (spec_code == SPEC_FULL_LDU) {
            std::vector<uint16_t> all(B); std::iota(all.begin(), all.end(), 0);
            apply_grp(all);
        } else {
            const int gs = 16, keep_pcs = 3;
            std::vector<std::vector<uint16_t>> groups;
            for (int g = 0; g < B; g += gs) {
                std::vector<uint16_t> grp;
                for (int i = g; i < std::min(g + gs, B); ++i) grp.push_back(static_cast<uint16_t>(i));
                apply_grp(grp); groups.push_back(std::move(grp));
            }
            for (int r = 0; r < keep_pcs; ++r) {
                std::vector<uint16_t> heads;
                for (const auto& g : groups) if (static_cast<int>(g.size()) > r) heads.push_back(g[r]);
                while (heads.size() > 1) {
                    std::vector<uint16_t> next_heads;
                    for (size_t g = 0; g < heads.size(); g += gs) {
                        std::vector<uint16_t> grp(heads.begin() + g, heads.begin() + std::min(g + gs, heads.size()));
                        apply_grp(grp); next_heads.push_back(grp[0]);
                    }
                    heads = std::move(next_heads);
                }
            }
        }
        write_le<uint16_t>(spec_bytes, static_cast<uint16_t>(B));
        write_le<uint16_t>(spec_bytes, static_cast<uint16_t>(stages.size()));
        for (int b = 0; b < B; ++b) write_le<int32_t>(spec_bytes, mu[b]);
        for (const auto& st : stages) pack_ldu_stage(spec_bytes, st);
    }

    auto plan = build_dwt_plan(H, W, n_lv);
    dwt53_2d_cube(work.data(), B, H, W, plan, false, n_threads);
    auto sched = build_c2f_schedule(H, W, plan);
    size_t n_subs = sched.size();

    std::vector<std::vector<int64_t>> full_subs(n_subs);
    for (size_t si = 0; si < n_subs; ++si) {
        const auto& sd = sched[si];
        int sh = sd.h(), sw = sd.w();
        full_subs[si].resize(static_cast<size_t>(B) * sh * sw);
        parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b) {
                const int64_t* pl = work.data() + b * HW;
                int64_t* dst = full_subs[si].data() + b * sh * sw;
                for (int y = 0; y < sh; ++y)
                    std::memcpy(dst + static_cast<size_t>(y) * sw, pl + static_cast<size_t>(sd.y0 + y) * W + sd.x0,
                                static_cast<size_t>(sw) * sizeof(int64_t));
            }
        });
    }

    // Header (62 bytes exact: byte 11 = n_stripes!)
    std::vector<uint8_t> out;
    out.insert(out.end(), MAGIC, MAGIC + 4);
    out.push_back(VERSION);
    out.push_back(flags);
    out.push_back(dt_code);
    out.push_back(spec_code);
    out.push_back(static_cast<uint8_t>(plan.size()));
    out.push_back(spec_axis);
    out.push_back(base_k);
    out.push_back(n_stripes); // byte 11: band-stripe parallelism count!
    write_le<uint16_t>(out, 1); // q_base = 1
    write_le<uint32_t>(out, d0);
    write_le<uint32_t>(out, d1);
    write_le<uint32_t>(out, d2);
    out.insert(out.end(), sha.begin(), sha.end());
    write_le<uint32_t>(out, static_cast<uint32_t>(spec_bytes.size()));
    out.insert(out.end(), spec_bytes.begin(), spec_bytes.end());

    // Prepare base-layer quantized arrays + parent midpoints
    std::vector<std::vector<int64_t>> cur_subs(n_subs), par_mid(n_subs);
    for (size_t si = 0; si < n_subs; ++si) {
        int p = std::max(0, static_cast<int>(base_k) - sched[si].offset);
        const auto& fs = full_subs[si];
        cur_subs[si].resize(fs.size());
        if (base_k > 0 && p > 0) {
            par_mid[si].resize(fs.size());
            for (size_t i = 0; i < fs.size(); ++i) {
                int64_t v = fs[i]; int64_t q = ((v >= 0) ? v : -v) >> p;
                cur_subs[si][i] = (v >= 0) ? q : -q;
                par_mid[si][i] = dequant_mid(cur_subs[si][i], p);
            }
        } else {
            cur_subs[si] = fs;
            par_mid[si] = fs;
        }
    }

    // Encode ALL (subband, stripe) tasks in ONE flat parallel_for across all 16 cores!
    size_t n_tasks = n_subs * n_stripes;
    std::vector<std::vector<uint8_t>> base_blobs(n_tasks);
    parallel_for(n_tasks, n_threads, [&](size_t t0, size_t t1) {
        for (size_t ti = t0; ti < t1; ++ti) {
            size_t si = ti / n_stripes;
            int gi = static_cast<int>(ti % n_stripes);
            const auto& sd = sched[si];
            int b0, b1; stripe_bounds(B, n_stripes, gi, b0, b1);
            int Bs = b1 - b0;
            size_t shw = static_cast<size_t>(sd.h()) * sd.w();
            const int64_t* par = nullptr; int pH = 0, pW = 0;
            if (sd.par_idx >= 0) {
                pH = sched[sd.par_idx].h(); pW = sched[sd.par_idx].w();
                par = par_mid[sd.par_idx].data() + static_cast<size_t>(b0) * pH * pW;
            }
            base_blobs[ti] = pack_rans_stripe(cur_subs[si].data() + static_cast<size_t>(b0) * shw,
                                              Bs, sd.h(), sd.w(), par, pH, pW);
        }
    });
    for (size_t ti = 0; ti < n_tasks; ++ti) out.insert(out.end(), base_blobs[ti].begin(), base_blobs[ti].end());
    par_mid.clear();

    // Refinement layers k = base_k - 1 .. 0: every (subband at level lv, stripe) runs in parallel across all cores!
    for (int k = static_cast<int>(base_k) - 1; k >= 0; --k) {
        for (int lv = 0; lv <= static_cast<int>(plan.size()); ++lv) {
            std::vector<size_t> lv_subs;
            for (size_t si = 0; si < n_subs; ++si) {
                if (sched[si].lv != lv) continue;
                int p_prev = std::max(0, (k + 1) - sched[si].offset);
                int p_cur  = std::max(0, k - sched[si].offset);
                if (p_cur != p_prev) lv_subs.push_back(si);
            }
            if (lv_subs.empty()) continue;
            std::vector<std::vector<int64_t>> next_subs(lv_subs.size());
            for (size_t j = 0; j < lv_subs.size(); ++j) next_subs[j].resize(full_subs[lv_subs[j]].size());
            size_t rt = lv_subs.size() * n_stripes;
            std::vector<std::vector<uint8_t>> rblobs(rt);
            parallel_for(rt, n_threads, [&](size_t t0, size_t t1) {
                for (size_t ti = t0; ti < t1; ++ti) {
                    size_t j = ti / n_stripes;
                    int gi = static_cast<int>(ti % n_stripes);
                    size_t si = lv_subs[j];
                    const auto& sd = sched[si];
                    int b0, b1; stripe_bounds(B, n_stripes, gi, b0, b1);
                    int Bs = b1 - b0;
                    int p_cur = std::max(0, k - sd.offset);
                    size_t shw = static_cast<size_t>(sd.h()) * sd.w();
                    const int64_t* par = nullptr; int pH = 0, pW = 0;
                    if (sd.par_idx >= 0) {
                        pH = sched[sd.par_idx].h(); pW = sched[sd.par_idx].w();
                        par = cur_subs[sd.par_idx].data() + static_cast<size_t>(b0) * pH * pW;
                    }
                    rblobs[ti] = pack_refine_stripe(
                        full_subs[si].data() + static_cast<size_t>(b0) * shw,
                        cur_subs[si].data()  + static_cast<size_t>(b0) * shw,
                        next_subs[j].data()  + static_cast<size_t>(b0) * shw,
                        Bs, sd.h(), sd.w(), par, pH, pW, p_cur);
                }
            });
            for (size_t j = 0; j < lv_subs.size(); ++j) cur_subs[lv_subs[j]] = std::move(next_subs[j]);
            for (size_t ti = 0; ti < rt; ++ti) out.insert(out.end(), rblobs[ti].begin(), rblobs[ti].end());
        }
    }
    return out;
}

// Native C++20 Blind 1D Track A Encoder (zero shape hints!)
struct BlindResult {
    std::vector<uint8_t> blob;
    std::vector<size_t> strides;
    int spec_axis = -1;
    uint32_t d0 = 1, d1 = 1, d2 = 0;
};

static BlindResult encode_hnd3_blind(const uint8_t* raw_bytes, size_t n_bytes, uint8_t dt_code,
                                     const std::string& spec_mode, uint8_t n_lv, uint8_t base_k,
                                     uint8_t n_stripes_req, int n_threads) {
    size_t itemsize = (dt_code == DT_I32) ? 4 : 2;
    if (n_bytes % itemsize != 0) throw std::runtime_error("Input byte size not divisible by dtype itemsize");
    size_t n = n_bytes / itemsize;

    // Window FFT to <= 4M samples (covers all 3D strides up to 1,048,576 in <= 128 MB RAM & ~1.5s!)
    size_t w_win = std::min<size_t>(n, 4194304ULL);
    std::vector<double> xf(w_win);
    if (dt_code == DT_I16)      for (size_t i = 0; i < w_win; ++i) xf[i] = read_le<int16_t>(raw_bytes + i * 2);
    else if (dt_code == DT_U16) for (size_t i = 0; i < w_win; ++i) xf[i] = read_le<uint16_t>(raw_bytes + i * 2);
    else                        for (size_t i = 0; i < w_win; ++i) xf[i] = read_le<int32_t>(raw_bytes + i * 4);

    auto rec = recover_strides_cpp(xf, n, 5);
    xf.clear(); xf.shrink_to_fit();

    BlindResult br;
    br.strides = rec;
    if (rec.size() >= 3 && rec[1] > 1 && rec[2] > rec[1] && (rec[2] % rec[1] == 0)) {
        size_t s1 = rec[1], s2 = rec[2];
        size_t d0 = n / s2, d1 = s2 / s1, d2 = s1;
        if (d0 * d1 * d2 == n && d0 >= 2 && d1 >= 4 && d2 >= 4) {
            std::vector<int64_t> xi(n);
            if (dt_code == DT_I16)      for (size_t i = 0; i < n; ++i) xi[i] = read_le<int16_t>(raw_bytes + i * 2);
            else if (dt_code == DT_U16) for (size_t i = 0; i < n; ++i) xi[i] = read_le<uint16_t>(raw_bytes + i * 2);
            else                        for (size_t i = 0; i < n; ++i) xi[i] = read_le<int32_t>(raw_bytes + i * 4);
            int sp_ax = choose_spectral_axis_cpp(xi.data(), static_cast<uint32_t>(d0),
                                                 static_cast<uint32_t>(d1), static_cast<uint32_t>(d2), n_threads);
            xi.clear(); xi.shrink_to_fit();
            br.d0 = static_cast<uint32_t>(d0); br.d1 = static_cast<uint32_t>(d1); br.d2 = static_cast<uint32_t>(d2);
            br.spec_axis = sp_ax;
            std::string eff_mode = (sp_ax < 0) ? "none" : spec_mode;
            uint8_t eff_ax = (sp_ax < 0) ? 0 : static_cast<uint8_t>(sp_ax);
            br.blob = encode_hnd3(raw_bytes, br.d0, br.d1, br.d2, dt_code, eff_mode,
                                  n_lv, eff_ax, base_k, n_stripes_req, FLAG_BLIND, n_threads);
            return br;
        }
    }

    // Clean raw fallback (zero blowup on incompressible noise!)
    auto sha = sha256_bytes(raw_bytes, n_bytes);
    br.d0 = 1; br.d1 = 1; br.d2 = static_cast<uint32_t>(n); br.spec_axis = -1;
    std::vector<uint8_t> out;
    out.reserve(HDR_SIZE + n_bytes);
    out.insert(out.end(), MAGIC, MAGIC + 4);
    out.push_back(VERSION);
    out.push_back(FLAG_BLIND | FLAG_RAW);
    out.push_back(dt_code);
    out.push_back(SPEC_NONE);
    out.push_back(0); out.push_back(0); out.push_back(0); out.push_back(1);
    write_le<uint16_t>(out, 1);
    write_le<uint32_t>(out, 1); write_le<uint32_t>(out, 1); write_le<uint32_t>(out, static_cast<uint32_t>(n));
    out.insert(out.end(), sha.begin(), sha.end());
    write_le<uint32_t>(out, 0);
    out.insert(out.end(), raw_bytes, raw_bytes + n_bytes);
    br.blob = std::move(out);
    return br;
}

struct DecodedCube {
    std::vector<uint8_t> raw_bytes;
    uint32_t d0 = 0, d1 = 0, d2 = 0;
    uint8_t dt_code = 0;
    bool sha_ok = false;
    bool sha_checked = false;
};

static DecodedCube decode_hnd3(const uint8_t* data, size_t len, int stop_layer, int max_level,
                               bool verify_sha, int n_threads, uint64_t max_samples = 512000000ULL) {
    if (len < HDR_SIZE || std::memcmp(data, MAGIC, 4) != 0 || data[4] != VERSION)
        throw std::runtime_error("Invalid HND3 header");
    uint8_t flags     = data[5];
    uint8_t dt_code   = data[6];
    uint8_t spec_code = data[7];
    uint8_t n_lv      = data[8];
    uint8_t spec_axis = data[9];
    uint8_t base_k    = data[10];
    uint8_t n_stripes = data[11];
    uint16_t q_base   = read_le<uint16_t>(data + 12);
    uint32_t d0       = read_le<uint32_t>(data + 14);
    uint32_t d1       = read_le<uint32_t>(data + 18);
    uint32_t d2       = read_le<uint32_t>(data + 22);
    std::array<uint8_t, 32> sha_exp{};
    std::memcpy(sha_exp.data(), data + 26, 32);
    uint32_t spec_len = read_le<uint32_t>(data + 58);

    if (dt_code != DT_I16 && dt_code != DT_U16 && dt_code != DT_I32)
        throw std::runtime_error("Invalid dt_code");
    if (spec_code > SPEC_CLUSTERED_LDU || spec_axis > 2 || n_lv > 8 || base_k > 15 || n_stripes == 0)
        throw std::runtime_error("Invalid HND3 header field");
    if (d0 == 0 || d1 == 0 || d2 == 0)
        throw std::runtime_error("Zero dimension in HND3 header");
    uint64_t tot64 = static_cast<uint64_t>(d0) * d1;
    if (tot64 > max_samples || tot64 * d2 > max_samples)
        throw std::runtime_error("HND3 cube dimensions exceed sample safety cap");
    size_t tot = static_cast<size_t>(tot64 * d2);
    size_t itemsize = (dt_code == DT_I32) ? 4 : 2;
    DecodedCube res; res.d0 = d0; res.d1 = d1; res.d2 = d2; res.dt_code = dt_code;

    if (flags & FLAG_RAW) {
        if (tot * itemsize > len - HDR_SIZE) throw std::runtime_error("Truncated raw HND3 stream");
        res.raw_bytes.assign(data + HDR_SIZE, data + HDR_SIZE + tot * itemsize);
        res.sha_checked = verify_sha;
        res.sha_ok = (!verify_sha) || (sha256_bytes(res.raw_bytes.data(), res.raw_bytes.size()) == sha_exp);
        if (verify_sha && !res.sha_ok) throw std::runtime_error("SHA-256 mismatch on raw HND3!");
        return res;
    }

    if (spec_len > len - HDR_SIZE) throw std::runtime_error("Truncated HND3 spectral header");
    uint32_t dims[3] = {d0, d1, d2};
    int B = static_cast<int>(dims[spec_axis]);
    int H = static_cast<int>(dims[(spec_axis == 0) ? 1 : 0]);
    int W = static_cast<int>(dims[(spec_axis == 2) ? 1 : 2]);
    if (n_stripes > B) throw std::runtime_error("n_stripes > B in HND3 header");

    // Pre-parse and validate the full spectral header BEFORE allocating cube buffers
    std::vector<int32_t> spec_mu;
    std::vector<LduStage> spec_stages;
    if (spec_code != SPEC_NONE) {
        size_t sp = HDR_SIZE, sp_end = HDR_SIZE + spec_len;
        if (sp + 4 > sp_end) throw std::runtime_error("Truncated spectral header");
        int Bs = read_le<uint16_t>(data + sp); sp += 2;
        int n_stages = read_le<uint16_t>(data + sp); sp += 2;
        if (Bs != B) throw std::runtime_error("Spectral band count mismatch");
        if (static_cast<size_t>(B) * 4 > sp_end - sp) throw std::runtime_error("Truncated spectral mu");
        spec_mu.resize(B);
        for (int b = 0; b < B; ++b) { spec_mu[b] = read_le<int32_t>(data + sp); sp += 4; }
        spec_stages.reserve(std::min(n_stages, 4096));
        for (int s = 0; s < n_stages; ++s) spec_stages.push_back(unpack_ldu_stage(data, sp, sp_end, B));
        if (sp != sp_end) throw std::runtime_error("Spectral header length mismatch");
    }
    size_t HW = static_cast<size_t>(H) * W;
#if defined(HND3_FORCE_SCALAR)
    bool use_i32_neon = false;
#else
    bool use_i32_neon = (dt_code != DT_I32);
#endif

    auto plan = build_dwt_plan(H, W, n_lv);
    auto sched = build_c2f_schedule(H, W, plan);
    size_t n_subs = sched.size();
    int lv_cap = (max_level < 0) ? static_cast<int>(plan.size()) : std::min(max_level, static_cast<int>(plan.size()));
    int k_target = std::max(0, std::min(stop_layer, static_cast<int>(base_k)));

    // Scan base-layer (subband x stripe) byte offsets with full bounds verification
    size_t pos = HDR_SIZE + spec_len;
    size_t n_tasks = n_subs * n_stripes;
    std::vector<size_t> b_pos(n_tasks, 0), b_len(n_tasks, 0);
    for (size_t si = 0; si < n_subs; ++si) {
        if (base_k == 0 && sched[si].lv > lv_cap) continue;
        for (int gi = 0; gi < n_stripes; ++gi) {
            size_t ti = si * n_stripes + gi;
            if (pos + 4 > len) throw std::runtime_error("Truncated base stripe length");
            uint32_t plen = read_le<uint32_t>(data + pos); pos += 4;
            if (plen > len - pos) throw std::runtime_error("Truncated base stripe payload");
            b_pos[ti] = pos; b_len[ti] = plen; pos += plen;
        }
    }

    // Decode base layer level-by-level with LAZY per-level allocation (LL decoded first ->
    // forged headers fail before fine subbands are allocated, while valid >256x streams succeed!)
    std::vector<std::vector<int64_t>> cur_subs(n_subs), par_mid(n_subs);
    for (int lv = 0; lv <= static_cast<int>(plan.size()); ++lv) {
        if (base_k == 0 && lv > lv_cap) break;
        std::vector<size_t> lv_subs;
        for (size_t si = 0; si < n_subs; ++si) {
            if (sched[si].lv == lv) {
                lv_subs.push_back(si);
                size_t ssz = static_cast<size_t>(B) * sched[si].h() * sched[si].w();
                cur_subs[si].assign(ssz, 0);
                par_mid[si].assign(ssz, 0);
            }
        }
        size_t lt = lv_subs.size() * n_stripes;
        parallel_for(lt, n_threads, [&](size_t t0, size_t t1) {
            for (size_t k_idx = t0; k_idx < t1; ++k_idx) {
                size_t si = lv_subs[k_idx / n_stripes];
                int gi = static_cast<int>(k_idx % n_stripes);
                size_t ti = si * n_stripes + gi;
                const auto& sd = sched[si];
                int b0, b1; stripe_bounds(B, n_stripes, gi, b0, b1);
                int Bs = b1 - b0;
                size_t shw = static_cast<size_t>(sd.h()) * sd.w();
                int64_t* dst = cur_subs[si].data() + static_cast<size_t>(b0) * shw;
                const int64_t* par = nullptr; int pH = 0, pW = 0;
                if (sd.par_idx >= 0) {
                    pH = sched[sd.par_idx].h(); pW = sched[sd.par_idx].w();
                    par = par_mid[sd.par_idx].data() + static_cast<size_t>(b0) * pH * pW;
                }
                unpack_rans_stripe(data, b_pos[ti], b_len[ti], dst, Bs, sd.h(), sd.w(), par, pH, pW);
                int64_t* pdst = par_mid[si].data() + static_cast<size_t>(b0) * shw;
                size_t cnt = static_cast<size_t>(Bs) * shw;
                if (base_k > 0) {
                    int p = std::max(0, static_cast<int>(base_k) - sd.offset);
                    for (size_t i = 0; i < cnt; ++i) pdst[i] = dequant_mid(dst[i], p);
                } else {
                    std::memcpy(pdst, dst, cnt * sizeof(int64_t));
                }
            }
        });
    }
    par_mid.clear();
    // Zero-fill any un-decoded high levels when max_level < plan.size() on base_k == 0
    for (size_t si = 0; si < n_subs; ++si) {
        if (cur_subs[si].empty()) {
            cur_subs[si].assign(static_cast<size_t>(B) * sched[si].h() * sched[si].w(), 0);
        }
    }

    // Decode refinement layers k = base_k - 1 .. k_target across ALL cores!
    for (int k = static_cast<int>(base_k) - 1; k >= k_target; --k) {
        for (int lv = 0; lv <= static_cast<int>(plan.size()); ++lv) {
            std::vector<size_t> lv_subs;
            for (size_t si = 0; si < n_subs; ++si) {
                if (sched[si].lv != lv) continue;
                int p_prev = std::max(0, (k + 1) - sched[si].offset);
                int p_cur  = std::max(0, k - sched[si].offset);
                if (p_cur != p_prev) lv_subs.push_back(si);
            }
            if (lv_subs.empty()) continue;
            size_t rt = lv_subs.size() * n_stripes;
            std::vector<size_t> r_pos(rt), r_len(rt);
            for (size_t ti = 0; ti < rt; ++ti) {
                if (pos + 4 > len) throw std::runtime_error("Truncated refine stripe length");
                uint32_t plen = read_le<uint32_t>(data + pos); pos += 4;
                if (plen > len - pos) throw std::runtime_error("Truncated refine stripe payload");
                r_pos[ti] = pos; r_len[ti] = plen; pos += plen;
            }
            std::vector<std::vector<int64_t>> next_subs(lv_subs.size());
            for (size_t j = 0; j < lv_subs.size(); ++j) next_subs[j].resize(cur_subs[lv_subs[j]].size());
            parallel_for(rt, n_threads, [&](size_t t0, size_t t1) {
                for (size_t ti = t0; ti < t1; ++ti) {
                    size_t j = ti / n_stripes;
                    int gi = static_cast<int>(ti % n_stripes);
                    size_t si = lv_subs[j];
                    const auto& sd = sched[si];
                    int b0, b1; stripe_bounds(B, n_stripes, gi, b0, b1);
                    int Bs = b1 - b0;
                    size_t shw = static_cast<size_t>(sd.h()) * sd.w();
                    const int64_t* par = nullptr; int pH = 0, pW = 0;
                    if (sd.par_idx >= 0) {
                        pH = sched[sd.par_idx].h(); pW = sched[sd.par_idx].w();
                        par = cur_subs[sd.par_idx].data() + static_cast<size_t>(b0) * pH * pW;
                    }
                    unpack_refine_stripe(data, r_pos[ti], r_len[ti],
                                         cur_subs[si].data() + static_cast<size_t>(b0) * shw,
                                         next_subs[j].data() + static_cast<size_t>(b0) * shw,
                                         Bs, sd.h(), sd.w(), par, pH, pW);
                }
            });
            for (size_t j = 0; j < lv_subs.size(); ++j) cur_subs[lv_subs[j]] = std::move(next_subs[j]);
        }
    }

    // Reassemble wavelet cube in parallel across bands!
    std::vector<int64_t> work(tot);
    parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) {
            int64_t* pl = work.data() + b * HW;
            for (size_t si = 0; si < n_subs; ++si) {
                const auto& sd = sched[si];
                int sh = sd.h(), sw = sd.w();
                int p = (base_k > 0) ? std::max(0, k_target - sd.offset) : 0;
                const int64_t* src = cur_subs[si].data() + b * sh * sw;
                for (int y = 0; y < sh; ++y) {
                    int64_t* dst = pl + static_cast<size_t>(sd.y0 + y) * W + sd.x0;
                    const int64_t* srow = src + static_cast<size_t>(y) * sw;
                    if (p == 0) std::memcpy(dst, srow, static_cast<size_t>(sw) * sizeof(int64_t));
                    else for (int x = 0; x < sw; ++x) dst[x] = dequant_mid(srow[x], p);
                }
            }
        }
    });
    cur_subs.clear();

    dwt53_2d_cube(work.data(), B, H, W, plan, true, n_threads);

    if (spec_code != SPEC_NONE) {
        for (int s = static_cast<int>(spec_stages.size()) - 1; s >= 0; --s)
            ldu_apply_stage(work.data(), HW, spec_stages[s], true, use_i32_neon, n_threads);
        parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b) {
                int64_t* row = work.data() + b * HW;
                int64_t m = spec_mu[b];
                for (size_t i = 0; i < HW; ++i) row[i] += m;
            }
        });
    }

    std::vector<int64_t> out64(tot);
    move_axis_from_front(work.data(), out64.data(), d0, d1, d2, spec_axis);
    bool is_exact = !(flags & FLAG_LOSSY) && (q_base == 1) && (k_target == 0) && (lv_cap == static_cast<int>(plan.size()));

    res.raw_bytes.resize(tot * itemsize);
    if (dt_code == DT_I16) {
        for (size_t i = 0; i < tot; ++i) {
            int16_t sv = static_cast<int16_t>(is_exact ? out64[i] : std::clamp<int64_t>(out64[i], -32768, 32767));
            std::memcpy(res.raw_bytes.data() + i * 2, &sv, 2);
        }
    } else if (dt_code == DT_U16) {
        for (size_t i = 0; i < tot; ++i) {
            uint16_t uv = static_cast<uint16_t>(is_exact ? out64[i] : std::clamp<int64_t>(out64[i], 0, 65535));
            std::memcpy(res.raw_bytes.data() + i * 2, &uv, 2);
        }
    } else {
        for (size_t i = 0; i < tot; ++i) {
            int32_t iv = static_cast<int32_t>(is_exact ? out64[i] : std::clamp<int64_t>(
                out64[i], std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()));
            std::memcpy(res.raw_bytes.data() + i * 4, &iv, 4);
        }
    }

    if (verify_sha && is_exact) {
        res.sha_checked = true;
        res.sha_ok = (sha256_bytes(res.raw_bytes.data(), res.raw_bytes.size()) == sha_exp);
        if (!res.sha_ok) throw std::runtime_error("SHA-256 verification FAILED on lossless HND3 decode!");
    } else {
        res.sha_checked = false;
        res.sha_ok = false;
    }
    return res;
}

} // namespace hnd3

static std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open input file: " + path);
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    if (sz > 0) f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    return buf;
}

static void write_file_bytes(const std::string& path, const std::vector<uint8_t>& buf) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open output file: " + path);
    if (!buf.empty()) f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  hnd3 enc <in.raw> <out.hnd3> (--shape D0,D1,D2 | --blind) --dtype u16|i16|i32\n"
                      << "           [--base-k 0..15] [--mode auto|full|clustered|none] [--levels 0..8]\n"
                      << "           [--stripes 1..64] [--spec-axis 0|1|2] [--threads N] [--no-verify]\n"
                      << "  hnd3 dec <in.hnd3> [out.raw] [--stop-layer K] [--max-level L] [--threads N] [--bench]\n";
            return 1;
        }
        std::string cmd = argv[1];
        int n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

        if (cmd == "enc") {
            if (argc < 4) throw std::runtime_error("enc requires <in.raw> <out.hnd3>");
            std::string in_path = argv[2], out_path = argv[3];
            uint32_t d0 = 0, d1 = 0, d2 = 0;
            bool has_shape = false, has_dtype = false, blind = false, verify_rt = true;
            uint8_t dt_code = hnd3::DT_U16;
            std::string mode = "auto";
            int base_k_i = 6, spec_axis_i = 0, n_lv_i = 3, n_stripes_i = 16;
            for (int i = 4; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--shape" && i + 1 < argc) {
                    if (std::sscanf(argv[++i], "%u,%u,%u", &d0, &d1, &d2) != 3) throw std::runtime_error("Bad --shape");
                    has_shape = true;
                } else if (a == "--blind") {
                    blind = true;
                } else if (a == "--dtype" && i + 1 < argc) {
                    std::string dt = argv[++i];
                    if (dt == "i16") dt_code = hnd3::DT_I16;
                    else if (dt == "u16") dt_code = hnd3::DT_U16;
                    else if (dt == "i32") dt_code = hnd3::DT_I32;
                    else throw std::runtime_error("Bad --dtype (expected u16|i16|i32)");
                    has_dtype = true;
                } else if (a == "--base-k" && i + 1 < argc) {
                    base_k_i = std::atoi(argv[++i]);
                } else if (a == "--mode" && i + 1 < argc) {
                    mode = argv[++i];
                } else if (a == "--stripes" && i + 1 < argc) {
                    n_stripes_i = std::atoi(argv[++i]);
                } else if (a == "--spec-axis" && i + 1 < argc) {
                    spec_axis_i = std::atoi(argv[++i]);
                } else if (a == "--levels" && i + 1 < argc) {
                    n_lv_i = std::atoi(argv[++i]);
                } else if (a == "--threads" && i + 1 < argc) {
                    n_threads = std::max(1, std::atoi(argv[++i]));
                } else if (a == "--verify") {
                    verify_rt = true;
                } else if (a == "--no-verify") {
                    verify_rt = false;
                } else {
                    throw std::runtime_error("Unknown or incomplete option: " + a);
                }
            }
            if (!has_dtype) throw std::runtime_error("Missing required --dtype u16|i16|i32");
            if (!blind && !has_shape) throw std::runtime_error("Must pass either --shape D0,D1,D2 or --blind");
            if (blind && has_shape) throw std::runtime_error("Cannot pass both --shape and --blind");
            if (base_k_i < 0 || base_k_i > 15) throw std::runtime_error("--base-k must be in 0..15");
            if (spec_axis_i < 0 || spec_axis_i > 2) throw std::runtime_error("--spec-axis must be in 0..2");
            if (n_lv_i < 0 || n_lv_i > 8) throw std::runtime_error("--levels must be in 0..8");
            if (n_stripes_i < 1 || n_stripes_i > 64) throw std::runtime_error("--stripes must be in 1..64");
            if (mode != "auto" && mode != "full" && mode != "clustered" && mode != "none")
                throw std::runtime_error("Bad --mode (expected auto|full|clustered|none)");

            uint8_t base_k = static_cast<uint8_t>(base_k_i);
            uint8_t spec_axis = static_cast<uint8_t>(spec_axis_i);
            uint8_t n_lv = static_cast<uint8_t>(n_lv_i);
            uint8_t n_stripes = static_cast<uint8_t>(n_stripes_i);
            size_t itemsize = (dt_code == hnd3::DT_I32) ? 4 : 2;

            auto raw = read_file_bytes(in_path);
            if (!blind) {
                if (d0 == 0 || d1 == 0 || d2 == 0) throw std::runtime_error("Zero dimension in --shape");
                uint64_t tot64 = static_cast<uint64_t>(d0) * d1;
                if (tot64 > 512000000ULL || tot64 * d2 > 512000000ULL)
                    throw std::runtime_error("Shape exceeds 512M sample safety cap");
                if (tot64 * d2 * itemsize != raw.size())
                    throw std::runtime_error("Input file size (" + std::to_string(raw.size()) +
                                             " B) does not match shape*itemsize (" +
                                             std::to_string(tot64 * d2 * itemsize) + " B)");
            } else {
                if (raw.empty() || (raw.size() % itemsize != 0))
                    throw std::runtime_error("Input byte size must be non-empty and divisible by dtype itemsize");
            }

            auto t0 = std::chrono::steady_clock::now();
            if (blind) {
                auto br = hnd3::encode_hnd3_blind(raw.data(), raw.size(), dt_code, mode, n_lv, base_k, n_stripes, n_threads);
                double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (verify_rt) {
                    auto chk = hnd3::decode_hnd3(br.blob.data(), br.blob.size(), 0, -1, true, n_threads);
                    if (chk.raw_bytes != raw) throw std::runtime_error("Encode round-trip verification FAILED!");
                }
                write_file_bytes(out_path, br.blob);
                double n_samp = double(raw.size() / itemsize);
                std::printf("BLIND-ENC strides=[");
                for (size_t i = 0; i < br.strides.size(); ++i) std::printf("%s%zu", i ? "," : "", br.strides[i]);
                std::printf("] shape=(%u,%u,%u) spec_ax=%d base_k=%u: %zu -> %zu B (%.3f b/s, %.2fx) %.1f MB/s (%.3fs) VERIFY=%s\n",
                            br.d0, br.d1, br.d2, br.spec_axis, base_k, raw.size(), br.blob.size(),
                            br.blob.size() * 8.0 / n_samp, double(raw.size()) / br.blob.size(),
                            raw.size() / 1e6 / std::max(sec, 1e-6), sec, verify_rt ? "PASS" : "SKIP");
            } else {
                auto blob = hnd3::encode_hnd3(raw.data(), d0, d1, d2, dt_code, mode, n_lv, spec_axis, base_k, n_stripes, 0, n_threads);
                double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (verify_rt) {
                    auto chk = hnd3::decode_hnd3(blob.data(), blob.size(), 0, -1, true, n_threads);
                    if (chk.raw_bytes != raw) throw std::runtime_error("Encode round-trip verification FAILED!");
                }
                write_file_bytes(out_path, blob);
                double n_samp = double(d0) * d1 * d2;
                std::printf("ENC (%u,%u,%u) base_k=%u stripes=%u mode=%s threads=%d: %zu -> %zu B (%.3f b/s, %.2fx) %.1f MB/s (%.3fs) VERIFY=%s\n",
                            d0, d1, d2, base_k, n_stripes, mode.c_str(), n_threads, raw.size(), blob.size(),
                            blob.size() * 8.0 / n_samp, double(raw.size()) / blob.size(),
                            raw.size() / 1e6 / std::max(sec, 1e-6), sec, verify_rt ? "PASS" : "SKIP");
            }
            return 0;
        } else if (cmd == "dec") {
            std::string in_path = argv[2];
            std::string out_path;
            int stop_layer = 0, max_level = -1;
            bool bench = false;
            for (int i = 3; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--stop-layer" && i + 1 < argc) stop_layer = std::atoi(argv[++i]);
                else if (a == "--max-level" && i + 1 < argc) max_level = std::atoi(argv[++i]);
                else if (a == "--threads" && i + 1 < argc) n_threads = std::max(1, std::atoi(argv[++i]));
                else if (a == "--bench") bench = true;
                else if (out_path.empty() && a[0] != '-') out_path = a;
                else throw std::runtime_error("Unknown or incomplete option: " + a);
            }
            auto blob = read_file_bytes(in_path);
            int reps = bench ? 3 : 1;
            double best_sec = 1e30;
            hnd3::DecodedCube dec;
            for (int r = 0; r < reps; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                dec = hnd3::decode_hnd3(blob.data(), blob.size(), stop_layer, max_level, true, n_threads);
                double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (sec < best_sec) best_sec = sec;
            }
            if (!out_path.empty()) write_file_bytes(out_path, dec.raw_bytes);
            const char* sha_str = dec.sha_checked ? (dec.sha_ok ? "PASS" : "FAIL") : "N/A (preview)";
            std::printf("DEC (%u,%u,%u) stop_layer=%d threads=%d: %zu -> %zu B  %.1f MB/s (%.3fs)  SHA256=%s\n",
                        dec.d0, dec.d1, dec.d2, stop_layer, n_threads, blob.size(), dec.raw_bytes.size(),
                        dec.raw_bytes.size() / 1e6 / std::max(best_sec, 1e-6), best_sec, sha_str);
            return 0;
#ifdef HND3_ENABLE_FUZZ
        } else if (cmd == "fuzz") {
            auto seed = read_file_bytes(argv[2]);
            int iters = (argc >= 4) ? std::atoi(argv[3]) : 10000;
            uint64_t rng = 0x9e3779b97f4a7c15ULL;
            auto nxt = [&]() -> uint64_t {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng;
            };
            int rejected = 0, accepted = 0;
            for (int it = 0; it < iters; ++it) {
                std::vector<uint8_t> mut = seed;
                uint64_t mode = nxt() % 5;
                if (mode == 0 && !mut.empty()) {
                    size_t cut = (nxt() % 2 == 0) ? (nxt() % std::min<size_t>(mut.size(), 512)) : (nxt() % mut.size());
                    mut.resize(cut);
                } else if (mode == 1 && !mut.empty()) {
                    int nf = 1 + static_cast<int>(nxt() % 8);
                    for (int f = 0; f < nf; ++f) {
                        size_t p = (nxt() % 2 == 0) ? (nxt() % std::min<size_t>(mut.size(), 512)) : (nxt() % mut.size());
                        mut[p] ^= static_cast<uint8_t>(1U << (nxt() & 7));
                    }
                } else if (mode == 2 && mut.size() >= 62) {
                    size_t p = 4 + (nxt() % 58);
                    mut[p] = static_cast<uint8_t>(nxt() & 0xFF);
                } else if (mode == 3) {
                    size_t sz = nxt() % 1024;
                    mut.resize(sz);
                    for (size_t k = 0; k < sz; ++k) mut[k] = static_cast<uint8_t>(nxt() & 0xFF);
                    if (sz >= 5 && (nxt() & 1)) { std::memcpy(mut.data(), hnd3::MAGIC, 4); mut[4] = hnd3::VERSION; }
                } else if (!mut.empty()) {
                    size_t p = nxt() % mut.size();
                    mut[p] = static_cast<uint8_t>(nxt() & 0xFF);
                }
                try {
                    auto d = hnd3::decode_hnd3(mut.data(), mut.size(), 0, -1, true, 1, 8000000ULL);
                    (void)d;
                    accepted++;
                } catch (const std::exception&) {
                    rejected++;
                }
            }
            std::string san;
#if defined(__SANITIZE_ADDRESS__)
            san += "ASan+";
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
            san += "ASan+";
#endif
#endif
#if defined(_LIBCPP_HARDENING_MODE)
            san += "libc++_hardened+";
#endif
            san += "UBSan/bounds";
            std::printf("FUZZ PASS: %d iterations (%d cleanly rejected, %d accepted, 0 crashes/%s)\n",
                        iters, rejected, accepted, san.c_str());
            return 0;
#endif
        } else {
            throw std::runtime_error("Unknown command: " + cmd);
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
