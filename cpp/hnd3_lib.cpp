// =============================================================================
// cpp/hnd3_lib.cpp — HND3 (v3) & HND4 (v4) Library Implementation + C ABI
//
// Features:
//   * HND3 v3: Single-tile lossless / 7-layer progressive (wire-identical to
//     phase3_codec.py), blind 3D stride/spectral-axis recovery.
//   * HND4 v4: Streaming spatial tiling (O(B * T^2) Peak RSS on multi-GB files),
//     random-access single-tile decoding, per-header & per-layer Castagnoli
//     CRC32C (verifies even lossy previews!), and guaranteed-L_inf near-lossless
//     coding (--max-error E: |x - x_hat| <= E for every sample).
// =============================================================================
#include "hnd3_rans.hpp"

namespace hnd3 {

uint32_t crc32c(const uint8_t* data, size_t len, uint32_t init) {
    return crc32c_impl(data, len, init);
}

static inline int resolve_threads(int req) {
    return (req > 0) ? req : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

static inline size_t dtype_size(uint8_t dt_code) {
    return (dt_code == DT_I32) ? 4u : 2u;
}

static inline int64_t read_sample_le(const uint8_t* p, uint8_t dt_code) {
    if (dt_code == DT_I16) return read_le<int16_t>(p);
    if (dt_code == DT_U16) return read_le<uint16_t>(p);
    return read_le<int32_t>(p);
}

static inline void write_sample_le(uint8_t* p, int64_t v, uint8_t dt_code, bool clamp_val) {
    if (dt_code == DT_I16) {
        int16_t sv = static_cast<int16_t>(clamp_val ? std::clamp<int64_t>(v, -32768, 32767) : v);
        std::memcpy(p, &sv, 2);
    } else if (dt_code == DT_U16) {
        uint16_t uv = static_cast<uint16_t>(clamp_val ? std::clamp<int64_t>(v, 0, 65535) : v);
        std::memcpy(p, &uv, 2);
    } else {
        int32_t iv = static_cast<int32_t>(clamp_val ? std::clamp<int64_t>(
            v, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()) : v);
        std::memcpy(p, &iv, 4);
    }
}

// Guaranteed-L_inf uniform centered quantizer: |x - dequant(quant(x, E), E)| <= E for ALL x!
static inline int64_t nl_quant(int64_t x, uint32_t max_err) {
    if (max_err == 0) return x;
    int64_t e = static_cast<int64_t>(max_err);
    int64_t step = 2LL * e + 1LL;
    return (x >= 0) ? ((x + e) / step) : -((-x + e) / step);
}

static inline int64_t nl_dequant(int64_t q, uint32_t max_err, uint8_t dt_code) {
    if (max_err == 0) return q;
    __int128 v = static_cast<__int128>(q) * (static_cast<__int128>(2) * max_err + 1);
    if (dt_code == DT_I16) return static_cast<int64_t>(std::clamp<__int128>(v, -32768, 32767));
    if (dt_code == DT_U16) return static_cast<int64_t>(std::clamp<__int128>(v, 0, 65535));
    return static_cast<int64_t>(std::clamp<__int128>(
        v, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()));
}

static void validate_encode_opts(const EncodeOptions& opt) {
    uint8_t dt = static_cast<uint8_t>(opt.dtype);
    if (dt != DT_I16 && dt != DT_U16 && dt != DT_I32)
        throw std::runtime_error("Invalid dtype");
    if (opt.spec_axis > 2) throw std::runtime_error("--spec-axis must be in 0..2");
    if (opt.base_k > 15) throw std::runtime_error("--base-k must be in 0..15");
    if (opt.levels > 8) throw std::runtime_error("--levels must be in 0..8");
    if (opt.stripes < 1 || opt.stripes > 64) throw std::runtime_error("--stripes must be in 1..64");
    if (opt.mode != "auto" && opt.mode != "full" && opt.mode != "clustered" && opt.mode != "none")
        throw std::runtime_error("Bad --mode (expected auto|full|clustered|none)");
    if (opt.max_error > 1073741823u) throw std::runtime_error("--max-error out of range");
    if (opt.hist_lut < -1 || opt.hist_lut > 1) throw std::runtime_error("Invalid hist_lut option");
}

// =============================================================================
// Reversible Monotonic Histogram Compaction LUT (HND4 FLAG_HIST_LUT)
// =============================================================================
struct HistLut {
    bool enabled = false;
    uint8_t dt_code = DT_I16;
    uint32_t K = 0;
    std::vector<int32_t> inv;   // dense idx [0..K-1] -> original i16/u16 value
    std::vector<uint16_t> fwd;  // 65536-entry O(1) map (encoder only)
    std::vector<uint8_t> bytes; // wire-serialized LUT prefix
};

static inline uint16_t hist_key(int64_t v, uint8_t dt_code) {
    if (dt_code == DT_U16) return static_cast<uint16_t>(v & 0xFFFF);
    return static_cast<uint16_t>(v + 32768);
}

static inline int32_t hist_unkey(uint16_t k, uint8_t dt_code) {
    if (dt_code == DT_U16) return static_cast<int32_t>(k);
    return static_cast<int32_t>(k) - 32768;
}

static void scan_hist_bits_mem(const uint8_t* raw, size_t n_bytes, uint8_t dt_code, std::array<uint8_t, 8192>& bits) {
    bits.fill(0);
    if (dt_code != DT_I16 && dt_code != DT_U16) return;
    size_t n = n_bytes / 2;
    for (size_t i = 0; i < n; ++i) {
        uint16_t k = hist_key(read_sample_le(raw + i * 2, dt_code), dt_code);
        bits[k >> 3] |= static_cast<uint8_t>(1u << (k & 7));
    }
}

static HistLut build_hist_lut_from_bits(const std::array<uint8_t, 8192>& bits, uint8_t dt_code,
                                        size_t tot_samples, int8_t hist_opt, uint32_t max_error) {
    HistLut lut;
    lut.dt_code = dt_code;
    if ((dt_code != DT_I16 && dt_code != DT_U16) || max_error != 0 || hist_opt == 0 || tot_samples == 0)
        return lut;

    std::vector<uint16_t> keys;
    keys.reserve(1024);
    uint32_t adj_pairs = 0;
    int prev_k = -2;
    for (uint32_t b = 0; b < 8192; ++b) {
        uint8_t byte = bits[b];
        if (!byte) continue;
        for (uint32_t bit = 0; bit < 8; ++bit) {
            if ((byte >> bit) & 1u) {
                int k = static_cast<int>((b << 3) | bit);
                if (k == prev_k + 1) adj_pairs++;
                prev_k = k;
                keys.push_back(static_cast<uint16_t>(k));
            }
        }
    }
    uint32_t K = static_cast<uint32_t>(keys.size());
    if (K == 0) return lut;
    uint32_t span = static_cast<uint32_t>(keys.back()) - static_cast<uint32_t>(keys.front()) + 1u;
    size_t wire_bytes = (K <= 4096u) ? (3u + 2u * K) : (1u + 8192u);

    if (hist_opt < 0) {
        // Auto-gate: activate ONLY on genuinely sparse 16-bit histograms (immune to single-pixel outliers
        // and tiny test cubes; 0% trigger rate on dense hyperspectral cubes!).
        if (K < 2) return lut;
        if (K * 2u >= span) return lut;
        if (adj_pairs * 2u >= (K - 1u)) return lut;
        if (static_cast<double>(wire_bytes) * 8.0 / static_cast<double>(tot_samples) >= 0.15) return lut;
    }

    lut.enabled = true;
    lut.K = K;
    lut.inv.resize(K);
    lut.fwd.assign(65536, 0);
    for (uint32_t i = 0; i < K; ++i) {
        lut.inv[i] = hist_unkey(keys[i], dt_code);
        lut.fwd[keys[i]] = static_cast<uint16_t>(i);
    }
    lut.bytes.reserve(wire_bytes);
    if (K <= 4096u) {
        lut.bytes.push_back(1u);
        write_le<uint16_t>(lut.bytes, static_cast<uint16_t>(K));
        for (uint32_t i = 0; i < K; ++i) {
            if (dt_code == DT_U16) write_le<uint16_t>(lut.bytes, static_cast<uint16_t>(lut.inv[i]));
            else                   write_le<int16_t>(lut.bytes, static_cast<int16_t>(lut.inv[i]));
        }
    } else {
        lut.bytes.push_back(2u);
        lut.bytes.insert(lut.bytes.end(), bits.begin(), bits.end());
    }
    return lut;
}

static HistLut parse_hist_lut(const uint8_t* data, size_t& pos, size_t end_pos, uint8_t dt_code) {
    if (dt_code != DT_I16 && dt_code != DT_U16)
        throw std::runtime_error("FLAG_HIST_LUT invalid for non-16-bit dtype");
    if (pos >= end_pos) throw std::runtime_error("Truncated HND4 histogram LUT header");
    uint8_t fmt = data[pos++];
    HistLut lut;
    lut.enabled = true;
    lut.dt_code = dt_code;
    if (fmt == 1u) {
        if (pos + 2 > end_pos) throw std::runtime_error("Truncated HND4 sparse LUT count");
        uint16_t K = read_le<uint16_t>(data + pos); pos += 2;
        if (K < 1 || K > 4096) throw std::runtime_error("Invalid HND4 sparse LUT K");
        if (static_cast<size_t>(K) * 2 > end_pos - pos) throw std::runtime_error("Truncated HND4 sparse LUT values");
        lut.K = K;
        lut.inv.resize(K);
        for (uint16_t i = 0; i < K; ++i) {
            int32_t v = (dt_code == DT_U16) ? static_cast<int32_t>(read_le<uint16_t>(data + pos))
                                            : static_cast<int32_t>(read_le<int16_t>(data + pos));
            pos += 2;
            if (i > 0 && v <= lut.inv[i - 1])
                throw std::runtime_error("Non-monotonic HND4 sparse LUT");
            lut.inv[i] = v;
        }
        return lut;
    }
    if (fmt == 2u) {
        if (8192u > end_pos - pos) throw std::runtime_error("Truncated HND4 bitmap LUT");
        lut.inv.reserve(16384);
        for (uint32_t b = 0; b < 8192u; ++b) {
            uint8_t byte = data[pos + b];
            if (!byte) continue;
            for (uint32_t bit = 0; bit < 8u; ++bit) {
                if ((byte >> bit) & 1u)
                    lut.inv.push_back(hist_unkey(static_cast<uint16_t>((b << 3) | bit), dt_code));
            }
        }
        pos += 8192u;
        if (lut.inv.empty()) throw std::runtime_error("Empty HND4 bitmap LUT");
        lut.K = static_cast<uint32_t>(lut.inv.size());
        return lut;
    }
    throw std::runtime_error("Unknown HND4 histogram LUT format");
}

static inline int64_t apply_hist_lut_fwd(int64_t v, const HistLut& lut) {
    if (!lut.enabled) return v;
    return static_cast<int64_t>(lut.fwd[hist_key(v, lut.dt_code)]);
}

static inline int64_t apply_hist_lut_inv(int64_t v, const HistLut& lut) {
    if (!lut.enabled) return v;
    int64_t idx = std::clamp<int64_t>(v, 0, static_cast<int64_t>(lut.K) - 1);
    return static_cast<int64_t>(lut.inv[static_cast<size_t>(idx)]);
}

// =============================================================================
// Shared Spectral Model Fit & Pack/Unpack
// =============================================================================
struct SpectralModel {
    uint8_t spec_code = SPEC_NONE;
    std::vector<int32_t> mu;
    std::vector<LduStage> stages;
    std::vector<uint8_t> bytes;
};

// Fit shared global LDU-RKLT from sample matrix X (B x N_samp, row-major int64), modifying X in-place
static SpectralModel fit_spectral_model(int64_t* X, int B, size_t N_samp, size_t total_cube_samples,
                                        const std::string& spec_mode, bool use_i32_neon, int n_threads) {
    SpectralModel m;
    if (B <= 1 || spec_mode == "none" || N_samp == 0) {
        m.spec_code = SPEC_NONE;
        return m;
    }
    if (spec_mode == "auto") {
        double full_hdr_bits = 16.0 * (double(B) * (B - 1) + 12.0 * B);
        m.spec_code = (B <= 16 || full_hdr_bits / double(std::max<size_t>(1, total_cube_samples)) < 0.04)
                          ? SPEC_FULL_LDU
                          : SPEC_CLUSTERED_LDU;
    } else if (spec_mode == "full") {
        m.spec_code = SPEC_FULL_LDU;
    } else {
        m.spec_code = SPEC_CLUSTERED_LDU;
    }

    m.mu.resize(B);
    parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) {
            int64_t* row = X + b * N_samp;
            double s = 0.0;
            for (size_t i = 0; i < N_samp; ++i) s += static_cast<double>(row[i]);
            int32_t mean_v = static_cast<int32_t>(std::llround(s / static_cast<double>(N_samp)));
            m.mu[b] = mean_v;
            for (size_t i = 0; i < N_samp; ++i) row[i] -= mean_v;
        }
    });

    auto apply_grp = [&](const std::vector<uint16_t>& grp) {
        if (grp.size() <= 1) return;
        auto U = klt_u_jacobi(X, static_cast<int>(grp.size()), N_samp, grp);
        auto st = complete_pivoted_ldu(U, grp);
        ldu_apply_stage(X, N_samp, st, false, use_i32_neon, n_threads);
        m.stages.push_back(std::move(st));
    };

    if (m.spec_code == SPEC_FULL_LDU) {
        std::vector<uint16_t> all(B);
        std::iota(all.begin(), all.end(), 0);
        apply_grp(all);
    } else {
        const int gs = 16, keep_pcs = 3;
        std::vector<std::vector<uint16_t>> groups;
        for (int g = 0; g < B; g += gs) {
            std::vector<uint16_t> grp;
            for (int i = g; i < std::min(g + gs, B); ++i) grp.push_back(static_cast<uint16_t>(i));
            apply_grp(grp);
            groups.push_back(std::move(grp));
        }
        for (int r = 0; r < keep_pcs; ++r) {
            std::vector<uint16_t> heads;
            for (const auto& g : groups) if (static_cast<int>(g.size()) > r) heads.push_back(g[r]);
            while (heads.size() > 1) {
                std::vector<uint16_t> next_heads;
                for (size_t g = 0; g < heads.size(); g += gs) {
                    std::vector<uint16_t> grp(heads.begin() + g, heads.begin() + std::min(g + gs, heads.size()));
                    apply_grp(grp);
                    next_heads.push_back(grp[0]);
                }
                heads = std::move(next_heads);
            }
        }
    }

    write_le<uint16_t>(m.bytes, static_cast<uint16_t>(B));
    write_le<uint16_t>(m.bytes, static_cast<uint16_t>(m.stages.size()));
    for (int b = 0; b < B; ++b) write_le<int32_t>(m.bytes, m.mu[b]);
    for (const auto& st : m.stages) pack_ldu_stage(m.bytes, st);
    return m;
}

static SpectralModel parse_spectral_model(const uint8_t* data, size_t sp_off, size_t spec_len,
                                          uint8_t spec_code, int B) {
    SpectralModel m;
    m.spec_code = spec_code;
    if (spec_code == SPEC_NONE) return m;
    size_t sp = sp_off, sp_end = sp_off + spec_len;
    if (sp + 4 > sp_end) throw std::runtime_error("Truncated spectral header");
    int Bs = read_le<uint16_t>(data + sp); sp += 2;
    int n_stages = read_le<uint16_t>(data + sp); sp += 2;
    if (Bs != B) throw std::runtime_error("Spectral band count mismatch");
    if (static_cast<size_t>(B) * 4 > sp_end - sp) throw std::runtime_error("Truncated spectral mu");
    m.mu.resize(B);
    for (int b = 0; b < B; ++b) { m.mu[b] = read_le<int32_t>(data + sp); sp += 4; }
    m.stages.reserve(std::min(n_stages, 4096));
    for (int s = 0; s < n_stages; ++s) m.stages.push_back(unpack_ldu_stage(data, sp, sp_end, B));
    if (sp != sp_end) throw std::runtime_error("Spectral header length mismatch");
    return m;
}

static void apply_spectral_forward(int64_t* tile, int B, size_t tHW, const SpectralModel& m,
                                   bool use_i32_neon, int n_threads) {
    if (m.spec_code == SPEC_NONE) return;
    parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) {
            int64_t* row = tile + b * tHW;
            int64_t mu = m.mu[b];
            for (size_t i = 0; i < tHW; ++i) row[i] -= mu;
        }
    });
    for (const auto& st : m.stages) ldu_apply_stage(tile, tHW, st, false, use_i32_neon, n_threads);
}

static void apply_spectral_inverse(int64_t* tile, int B, size_t tHW, const SpectralModel& m,
                                   bool use_i32_neon, int n_threads) {
    if (m.spec_code == SPEC_NONE) return;
    for (int s = static_cast<int>(m.stages.size()) - 1; s >= 0; --s)
        ldu_apply_stage(tile, tHW, m.stages[s], true, use_i32_neon, n_threads);
    parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) {
            int64_t* row = tile + b * tHW;
            int64_t mu = m.mu[b];
            for (size_t i = 0; i < tHW; ++i) row[i] = wrap_add64(row[i], mu);
        }
    });
}

// =============================================================================
// Per-Tile Wavelet + 16-Stripe rANS + Per-Layer CRC32C Encode/Decode
// =============================================================================
struct EncodedTile {
    std::vector<uint8_t> payload;
    std::array<uint32_t, 16> layer_bytes{};
    std::array<uint32_t, 16> layer_crc{};
};

static EncodedTile encode_tile_layers(int64_t* tile, int B, int th, int tw,
                                      uint8_t n_lv, uint8_t base_k, uint8_t n_stripes, int n_threads) {
    EncodedTile et;
    size_t tHW = static_cast<size_t>(th) * tw;
    auto plan = build_dwt_plan(th, tw, n_lv);
    dwt53_2d_cube(tile, B, th, tw, plan, false, n_threads);
    auto sched = build_c2f_schedule(th, tw, plan);
    size_t n_subs = sched.size();

    std::vector<std::vector<int64_t>> full_subs(n_subs);
    for (size_t si = 0; si < n_subs; ++si) {
        const auto& sd = sched[si];
        int sh = sd.h(), sw = sd.w();
        full_subs[si].resize(static_cast<size_t>(B) * sh * sw);
        parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b) {
                const int64_t* pl = tile + b * tHW;
                int64_t* dst = full_subs[si].data() + b * sh * sw;
                for (int y = 0; y < sh; ++y)
                    std::memcpy(dst + static_cast<size_t>(y) * sw, pl + static_cast<size_t>(sd.y0 + y) * tw + sd.x0,
                                static_cast<size_t>(sw) * sizeof(int64_t));
            }
        });
    }

    std::vector<std::vector<int64_t>> cur_subs(n_subs), par_mid(n_subs);
    for (size_t si = 0; si < n_subs; ++si) {
        int p = std::max(0, static_cast<int>(base_k) - sched[si].offset);
        const auto& fs = full_subs[si];
        cur_subs[si].resize(fs.size());
        if (base_k > 0 && p > 0) {
            par_mid[si].resize(fs.size());
            for (size_t i = 0; i < fs.size(); ++i) {
                int64_t v = fs[i];
                int64_t q = ((v >= 0) ? v : -v) >> p;
                cur_subs[si][i] = (v >= 0) ? q : -q;
                par_mid[si][i] = dequant_mid(cur_subs[si][i], p);
            }
        } else {
            cur_subs[si] = fs;
            par_mid[si] = fs;
        }
    }

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
    size_t base_start = et.payload.size();
    for (size_t ti = 0; ti < n_tasks; ++ti)
        et.payload.insert(et.payload.end(), base_blobs[ti].begin(), base_blobs[ti].end());
    size_t base_len = et.payload.size() - base_start;
    et.layer_bytes[base_k] = static_cast<uint32_t>(base_len);
    et.layer_crc[base_k] = crc32c_impl(et.payload.data() + base_start, base_len);
    par_mid.clear();

    for (int k = static_cast<int>(base_k) - 1; k >= 0; --k) {
        size_t l_start = et.payload.size();
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
            for (size_t ti = 0; ti < rt; ++ti)
                et.payload.insert(et.payload.end(), rblobs[ti].begin(), rblobs[ti].end());
        }
        size_t l_len = et.payload.size() - l_start;
        et.layer_bytes[k] = static_cast<uint32_t>(l_len);
        et.layer_crc[k] = crc32c_impl(et.payload.data() + l_start, l_len);
    }
    return et;
}

static std::vector<int64_t> decode_tile_layers(
    const uint8_t* data, size_t len, int B, int th, int tw,
    uint8_t n_lv, uint8_t base_k, uint8_t n_stripes,
    int stop_layer, int max_level,
    const uint32_t* expected_layer_bytes, const uint32_t* expected_layer_crc,
    bool verify_crc, size_t& layers_crc_ok, int n_threads) {

    auto plan = build_dwt_plan(th, tw, n_lv);
    auto sched = build_c2f_schedule(th, tw, plan);
    size_t n_subs = sched.size();
    int lv_cap = (max_level < 0) ? static_cast<int>(plan.size()) : std::min(max_level, static_cast<int>(plan.size()));
    int k_target = std::max(0, std::min(stop_layer, static_cast<int>(base_k)));
    size_t tHW = static_cast<size_t>(th) * tw;

    // Verify per-layer CRC32C up through k_target BEFORE decoding!
    if (verify_crc && expected_layer_bytes && expected_layer_crc) {
        size_t lpos = 0;
        for (int k = static_cast<int>(base_k); k >= k_target; --k) {
            uint32_t lb = expected_layer_bytes[k];
            if (lb > len - lpos) throw std::runtime_error("Truncated HND4 tile layer");
            uint32_t got = crc32c_impl(data + lpos, lb);
            if (got != expected_layer_crc[k])
                throw std::runtime_error("CRC32C mismatch on HND4 tile layer k=" + std::to_string(k));
            lpos += lb;
            layers_crc_ok++;
        }
    }

    size_t pos = 0;
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
    for (size_t si = 0; si < n_subs; ++si) {
        if (cur_subs[si].empty())
            cur_subs[si].assign(static_cast<size_t>(B) * sched[si].h() * sched[si].w(), 0);
    }

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

    std::vector<int64_t> work(static_cast<size_t>(B) * tHW);
    parallel_for(static_cast<size_t>(B), n_threads, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) {
            int64_t* pl = work.data() + b * tHW;
            for (size_t si = 0; si < n_subs; ++si) {
                const auto& sd = sched[si];
                int sh = sd.h(), sw = sd.w();
                int p = (base_k > 0) ? std::max(0, k_target - sd.offset) : 0;
                const int64_t* src = cur_subs[si].data() + b * sh * sw;
                for (int y = 0; y < sh; ++y) {
                    int64_t* dst = pl + static_cast<size_t>(sd.y0 + y) * tw + sd.x0;
                    const int64_t* srow = src + static_cast<size_t>(y) * sw;
                    if (p == 0) std::memcpy(dst, srow, static_cast<size_t>(sw) * sizeof(int64_t));
                    else for (int x = 0; x < sw; ++x) dst[x] = dequant_mid(srow[x], p);
                }
            }
        }
    });
    cur_subs.clear();
    dwt53_2d_cube(work.data(), B, th, tw, plan, true, n_threads);
    return work;
}

// =============================================================================
// HND3 v3 Encode/Decode (100% Wire-Compatible with phase3_codec.py!)
// =============================================================================
static std::vector<uint8_t> encode_hnd3_v3(const uint8_t* raw_bytes, uint32_t d0, uint32_t d1, uint32_t d2,
                                           uint8_t dt_code, const std::string& spec_mode,
                                           uint8_t n_lv, uint8_t spec_axis, uint8_t base_k,
                                           uint8_t n_stripes_req, uint8_t flags, int n_threads) {
    size_t tot = static_cast<size_t>(d0) * d1 * d2;
    size_t itemsize = dtype_size(dt_code);
    auto sha = sha256_bytes(raw_bytes, tot * itemsize);

    std::vector<int64_t> orig(tot);
    for (size_t i = 0; i < tot; ++i) orig[i] = read_sample_le(raw_bytes + i * itemsize, dt_code);

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

    auto sm = fit_spectral_model(work.data(), B, HW, tot, spec_mode, use_i32_neon, n_threads);
    auto plan = build_dwt_plan(H, W, n_lv);
    auto et = encode_tile_layers(work.data(), B, H, W, static_cast<uint8_t>(plan.size()), base_k, n_stripes, n_threads);

    std::vector<uint8_t> out;
    out.reserve(HDR_SIZE + sm.bytes.size() + et.payload.size());
    out.insert(out.end(), MAGIC, MAGIC + 4);
    out.push_back(VERSION);
    out.push_back(flags);
    out.push_back(dt_code);
    out.push_back(sm.spec_code);
    out.push_back(static_cast<uint8_t>(plan.size()));
    out.push_back(spec_axis);
    out.push_back(base_k);
    out.push_back(n_stripes);
    write_le<uint16_t>(out, 1);
    write_le<uint32_t>(out, d0);
    write_le<uint32_t>(out, d1);
    write_le<uint32_t>(out, d2);
    out.insert(out.end(), sha.begin(), sha.end());
    write_le<uint32_t>(out, static_cast<uint32_t>(sm.bytes.size()));
    out.insert(out.end(), sm.bytes.begin(), sm.bytes.end());
    out.insert(out.end(), et.payload.begin(), et.payload.end());
    return out;
}

static DecodedCube decode_hnd3_v3(const uint8_t* data, size_t len, const DecodeOptions& opt) {
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
    if (d0 == 0 || d1 == 0 || d2 == 0) throw std::runtime_error("Zero dimension in HND3 header");
    uint64_t tot64 = static_cast<uint64_t>(d0) * d1;
    if (tot64 > opt.max_samples || tot64 * d2 > opt.max_samples)
        throw std::runtime_error("HND3 cube dimensions exceed sample safety cap");
    size_t tot = static_cast<size_t>(tot64 * d2);
    size_t itemsize = dtype_size(dt_code);
    int n_threads = resolve_threads(opt.threads);

    DecodedCube res;
    res.version = 3; res.d0 = d0; res.d1 = d1; res.d2 = d2; res.dt_code = dt_code;

    if (flags & FLAG_RAW) {
        if (tot * itemsize > len - HDR_SIZE) throw std::runtime_error("Truncated raw HND3 stream");
        res.raw_bytes.assign(data + HDR_SIZE, data + HDR_SIZE + tot * itemsize);
        res.sha_checked = opt.verify_sha;
        res.sha_ok = (!opt.verify_sha) || (sha256_bytes(res.raw_bytes.data(), res.raw_bytes.size()) == sha_exp);
        if (opt.verify_sha && !res.sha_ok) throw std::runtime_error("SHA-256 mismatch on raw HND3!");
        return res;
    }

    if (spec_len > len - HDR_SIZE) throw std::runtime_error("Truncated HND3 spectral header");
    uint32_t dims[3] = {d0, d1, d2};
    int B = static_cast<int>(dims[spec_axis]);
    int H = static_cast<int>(dims[(spec_axis == 0) ? 1 : 0]);
    int W = static_cast<int>(dims[(spec_axis == 2) ? 1 : 2]);
    if (n_stripes > B) throw std::runtime_error("n_stripes > B in HND3 header");

    auto sm = parse_spectral_model(data, HDR_SIZE, spec_len, spec_code, B);
#if defined(HND3_FORCE_SCALAR)
    bool use_i32_neon = false;
#else
    bool use_i32_neon = (dt_code != DT_I32);
#endif

    size_t dummy_crc = 0;
    auto work = decode_tile_layers(data + HDR_SIZE + spec_len, len - (HDR_SIZE + spec_len),
                                   B, H, W, n_lv, base_k, n_stripes, opt.stop_layer, opt.max_level,
                                   nullptr, nullptr, false, dummy_crc, n_threads);
    apply_spectral_inverse(work.data(), B, static_cast<size_t>(H) * W, sm, use_i32_neon, n_threads);

    std::vector<int64_t> out64(tot);
    move_axis_from_front(work.data(), out64.data(), d0, d1, d2, spec_axis);
    auto plan = build_dwt_plan(H, W, n_lv);
    int lv_cap = (opt.max_level < 0) ? static_cast<int>(plan.size()) : std::min(opt.max_level, static_cast<int>(plan.size()));
    int k_target = std::max(0, std::min(opt.stop_layer, static_cast<int>(base_k)));
    bool is_exact = !(flags & FLAG_LOSSY) && (q_base == 1) && (k_target == 0) && (lv_cap == static_cast<int>(plan.size()));

    res.raw_bytes.resize(tot * itemsize);
    for (size_t i = 0; i < tot; ++i) write_sample_le(res.raw_bytes.data() + i * itemsize, out64[i], dt_code, !is_exact);

    if (opt.verify_sha && is_exact) {
        res.sha_checked = true;
        res.sha_ok = (sha256_bytes(res.raw_bytes.data(), res.raw_bytes.size()) == sha_exp);
        if (!res.sha_ok) throw std::runtime_error("SHA-256 verification FAILED on lossless HND3 decode!");
    }
    return res;
}

// =============================================================================
// HND4 v4 Container: Spatial Tiling, Per-Layer CRC32C, Guaranteed-L_inf Near-Lossless
// =============================================================================
struct TileIndexEntry {
    uint64_t offset = 0;
    uint32_t tile_bytes = 0;
    std::array<uint32_t, 16> layer_bytes{};
    std::array<uint32_t, 16> layer_crc{};
};

static void pack_hnd4_header(std::vector<uint8_t>& hdr, uint8_t flags, uint8_t dt_code, uint8_t spec_code,
                             uint8_t n_lv, uint8_t spec_axis, uint8_t base_k, uint8_t n_stripes,
                             uint32_t d0, uint32_t d1, uint32_t d2,
                             uint32_t tile_h, uint32_t tile_w, uint32_t n_ty, uint32_t n_tx,
                             uint32_t max_error, const std::array<uint8_t, 32>& sha,
                             uint32_t spec_len, uint32_t header_crc) {
    hdr.clear();
    hdr.reserve(HDR4_SIZE);
    hdr.insert(hdr.end(), MAGIC4, MAGIC4 + 4);
    hdr.push_back(VERSION4);
    hdr.push_back(flags);
    hdr.push_back(dt_code);
    hdr.push_back(spec_code);
    hdr.push_back(n_lv);
    hdr.push_back(spec_axis);
    hdr.push_back(base_k);
    hdr.push_back(n_stripes);
    write_le<uint32_t>(hdr, d0);
    write_le<uint32_t>(hdr, d1);
    write_le<uint32_t>(hdr, d2);
    write_le<uint32_t>(hdr, tile_h);
    write_le<uint32_t>(hdr, tile_w);
    write_le<uint32_t>(hdr, n_ty);
    write_le<uint32_t>(hdr, n_tx);
    write_le<uint32_t>(hdr, max_error);
    hdr.insert(hdr.end(), sha.begin(), sha.end());
    write_le<uint32_t>(hdr, spec_len);
    write_le<uint32_t>(hdr, header_crc);
}

static void pack_tile_index(std::vector<uint8_t>& buf, const std::vector<TileIndexEntry>& idx) {
    buf.clear();
    buf.reserve(idx.size() * TILE_ENTRY_SIZE);
    for (const auto& e : idx) {
        write_le<uint64_t>(buf, e.offset);
        write_le<uint32_t>(buf, e.tile_bytes);
        for (int k = 0; k < 16; ++k) write_le<uint32_t>(buf, e.layer_bytes[k]);
        for (int k = 0; k < 16; ++k) write_le<uint32_t>(buf, e.layer_crc[k]);
    }
}

struct ParsedHnd4Meta {
    uint8_t flags = 0, dt_code = 0, spec_code = 0, n_lv = 0, spec_axis = 0, base_k = 0, n_stripes = 1;
    uint32_t d0 = 0, d1 = 0, d2 = 0;
    uint32_t tile_h = 0, tile_w = 0, n_ty = 1, n_tx = 1;
    uint32_t max_error = 0, spec_len = 0, header_crc = 0;
    std::array<uint8_t, 32> sha{};
    int B = 0, H = 0, W = 0;
    HistLut lut;
    SpectralModel sm;
    std::vector<TileIndexEntry> index;
    size_t payload_start = 0;
};

static ParsedHnd4Meta parse_hnd4_meta(const uint8_t* data, size_t len, uint64_t max_samples, bool verify_crc) {
    if (len < HDR4_SIZE || std::memcmp(data, MAGIC4, 4) != 0 || data[4] != VERSION4)
        throw std::runtime_error("Invalid HND4 header");
    ParsedHnd4Meta m;
    m.flags     = data[5];
    m.dt_code   = data[6];
    m.spec_code = data[7];
    m.n_lv      = data[8];
    m.spec_axis = data[9];
    m.base_k    = data[10];
    m.n_stripes = data[11];
    m.d0        = read_le<uint32_t>(data + 12);
    m.d1        = read_le<uint32_t>(data + 16);
    m.d2        = read_le<uint32_t>(data + 20);
    m.tile_h    = read_le<uint32_t>(data + 24);
    m.tile_w    = read_le<uint32_t>(data + 28);
    m.n_ty      = read_le<uint32_t>(data + 32);
    m.n_tx      = read_le<uint32_t>(data + 36);
    m.max_error = read_le<uint32_t>(data + 40);
    std::memcpy(m.sha.data(), data + 44, 32);
    m.spec_len  = read_le<uint32_t>(data + 76);
    m.header_crc = read_le<uint32_t>(data + 80);

    if (m.dt_code != DT_I16 && m.dt_code != DT_U16 && m.dt_code != DT_I32)
        throw std::runtime_error("Invalid HND4 dt_code");
    if (m.spec_code > SPEC_CLUSTERED_LDU || m.spec_axis > 2 || m.n_lv > 8 || m.base_k > 15 || m.n_stripes == 0)
        throw std::runtime_error("Invalid HND4 header field");
    if (m.d0 == 0 || m.d1 == 0 || m.d2 == 0 || m.tile_h == 0 || m.tile_w == 0)
        throw std::runtime_error("Zero dimension in HND4 header");
    uint64_t tot64 = static_cast<uint64_t>(m.d0) * m.d1;
    if (tot64 > max_samples || tot64 * m.d2 > max_samples)
        throw std::runtime_error("HND4 cube dimensions exceed sample safety cap");

    uint32_t dims[3] = {m.d0, m.d1, m.d2};
    m.B = static_cast<int>(dims[m.spec_axis]);
    m.H = static_cast<int>(dims[(m.spec_axis == 0) ? 1 : 0]);
    m.W = static_cast<int>(dims[(m.spec_axis == 2) ? 1 : 2]);
    if (m.n_stripes > m.B) throw std::runtime_error("n_stripes > B in HND4 header");

    uint32_t exp_ty = (static_cast<uint32_t>(m.H) + m.tile_h - 1) / m.tile_h;
    uint32_t exp_tx = (static_cast<uint32_t>(m.W) + m.tile_w - 1) / m.tile_w;
    if (m.n_ty != exp_ty || m.n_tx != exp_tx || m.n_ty == 0 || m.n_tx == 0)
        throw std::runtime_error("Invalid HND4 tile grid");
    uint64_t n_tiles = static_cast<uint64_t>(m.n_ty) * m.n_tx;
    if (n_tiles > 65536ULL) throw std::runtime_error("Excessive HND4 tile count");

    if (m.spec_len > len - HDR4_SIZE) throw std::runtime_error("Truncated HND4 spectral header");
    size_t idx_bytes = static_cast<size_t>(n_tiles) * TILE_ENTRY_SIZE;
    if (idx_bytes > len - HDR4_SIZE - m.spec_len) throw std::runtime_error("Truncated HND4 tile index");
    m.payload_start = HDR4_SIZE + m.spec_len + idx_bytes;

    if (verify_crc) {
        uint32_t c = crc32c_impl(data, HDR4_SIZE - 4, 0u);
        c = crc32c_impl(data + HDR4_SIZE, m.spec_len + idx_bytes, c);
        if (c != m.header_crc) throw std::runtime_error("HND4 header/index CRC32C mismatch!");
    }

    size_t sp_pos = HDR4_SIZE;
    size_t sp_end = HDR4_SIZE + m.spec_len;
    if (m.flags & FLAG_HIST_LUT) {
        if (m.max_error != 0) throw std::runtime_error("FLAG_HIST_LUT cannot combine with near-lossless max_error");
        m.lut = parse_hist_lut(data, sp_pos, sp_end, m.dt_code);
    }
    m.sm = parse_spectral_model(data, sp_pos, sp_end - sp_pos, m.spec_code, m.B);
    m.index.resize(static_cast<size_t>(n_tiles));
    const uint8_t* ip = data + HDR4_SIZE + m.spec_len;
    for (size_t t = 0; t < static_cast<size_t>(n_tiles); ++t) {
        auto& e = m.index[t];
        e.offset = read_le<uint64_t>(ip); ip += 8;
        e.tile_bytes = read_le<uint32_t>(ip); ip += 4;
        uint64_t lsum = 0;
        for (int k = 0; k < 16; ++k) {
            e.layer_bytes[k] = read_le<uint32_t>(ip); ip += 4;
            if (k <= static_cast<int>(m.base_k)) lsum += e.layer_bytes[k];
        }
        if (lsum > e.tile_bytes) throw std::runtime_error("Corrupt HND4 tile layer_bytes sum > tile_bytes");
        for (int k = 0; k < 16; ++k) { e.layer_crc[k] = read_le<uint32_t>(ip); ip += 4; }
    }
    return m;
}

// Extract spatial tile (ty, tx) from in-memory raw bytes into (B, th, tw) int64
static void extract_tile_from_mem(const uint8_t* raw, uint32_t /*d0*/, uint32_t d1, uint32_t d2,
                                  uint8_t dt_code, uint8_t spec_axis,
                                  int B, int y0, int th, int x0, int tw, int64_t* dst) {
    size_t itemsize = dtype_size(dt_code);
    size_t tHW = static_cast<size_t>(th) * tw;
    for (int b = 0; b < B; ++b) {
        for (int y = 0; y < th; ++y) {
            for (int x = 0; x < tw; ++x) {
                uint32_t i0, i1, i2;
                if (spec_axis == 0)      { i0 = b; i1 = y0 + y; i2 = x0 + x; }
                else if (spec_axis == 1) { i0 = y0 + y; i1 = b; i2 = x0 + x; }
                else                     { i0 = y0 + y; i1 = x0 + x; i2 = b; }
                size_t idx = (static_cast<size_t>(i0) * d1 + i1) * d2 + i2;
                dst[static_cast<size_t>(b) * tHW + y * tw + x] = read_sample_le(raw + idx * itemsize, dt_code);
            }
        }
    }
}

static void scatter_tile_to_mem(const int64_t* tile, uint8_t* raw, uint32_t /*d0*/, uint32_t d1, uint32_t d2,
                                uint8_t dt_code, uint8_t spec_axis,
                                int B, int y0, int th, int x0, int tw, bool clamp_val) {
    size_t itemsize = dtype_size(dt_code);
    size_t tHW = static_cast<size_t>(th) * tw;
    for (int b = 0; b < B; ++b) {
        for (int y = 0; y < th; ++y) {
            for (int x = 0; x < tw; ++x) {
                uint32_t i0, i1, i2;
                if (spec_axis == 0)      { i0 = b; i1 = y0 + y; i2 = x0 + x; }
                else if (spec_axis == 1) { i0 = y0 + y; i1 = b; i2 = x0 + x; }
                else                     { i0 = y0 + y; i1 = x0 + x; i2 = b; }
                size_t idx = (static_cast<size_t>(i0) * d1 + i1) * d2 + i2;
                write_sample_le(raw + idx * itemsize, tile[static_cast<size_t>(b) * tHW + y * tw + x], dt_code, clamp_val);
            }
        }
    }
}

// Streaming file I/O helpers for O(B * tile_h * tile_w) bounded Peak RSS!
static void extract_tile_from_file(std::ifstream& f, uint32_t /*d0*/, uint32_t d1, uint32_t d2,
                                   uint8_t dt_code, uint8_t spec_axis,
                                   int B, int y0, int th, int x0, int tw, int64_t* dst) {
    size_t itemsize = dtype_size(dt_code);
    size_t tHW = static_cast<size_t>(th) * tw;
    if (spec_axis == 0) {
        // BSQ: read all th rows of band b in ONE contiguous strip read (<= 4 MB) to cut syscalls by ~100x!
        size_t strip_bytes = static_cast<size_t>(th) * d2 * itemsize;
        if (strip_bytes <= (4ULL << 20)) {
            std::vector<uint8_t> strip(strip_bytes);
            for (int b = 0; b < B; ++b) {
                size_t off = ((static_cast<size_t>(b) * d1 + y0) * d2) * itemsize;
                f.seekg(static_cast<std::streamoff>(off));
                f.read(reinterpret_cast<char*>(strip.data()), static_cast<std::streamsize>(strip_bytes));
                if (!f) throw std::runtime_error("Short read extracting BSQ tile strip");
                for (int y = 0; y < th; ++y) {
                    const uint8_t* sp = strip.data() + (static_cast<size_t>(y) * d2 + x0) * itemsize;
                    int64_t* dp = dst + static_cast<size_t>(b) * tHW + static_cast<size_t>(y) * tw;
                    for (int x = 0; x < tw; ++x) dp[x] = read_sample_le(sp + x * itemsize, dt_code);
                }
            }
        } else {
            std::vector<uint8_t> row_buf(static_cast<size_t>(tw) * itemsize);
            for (int b = 0; b < B; ++b) {
                for (int y = 0; y < th; ++y) {
                    size_t off = ((static_cast<size_t>(b) * d1 + (y0 + y)) * d2 + x0) * itemsize;
                    f.seekg(static_cast<std::streamoff>(off));
                    f.read(reinterpret_cast<char*>(row_buf.data()), static_cast<std::streamsize>(row_buf.size()));
                    if (!f) throw std::runtime_error("Short read extracting BSQ tile");
                    int64_t* dp = dst + static_cast<size_t>(b) * tHW + static_cast<size_t>(y) * tw;
                    for (int x = 0; x < tw; ++x) dp[x] = read_sample_le(row_buf.data() + x * itemsize, dt_code);
                }
            }
        }
    } else if (spec_axis == 1) {
        // BIL: each scanline y holds B contiguous rows of length d2
        std::vector<uint8_t> row_buf(static_cast<size_t>(tw) * itemsize);
        for (int y = 0; y < th; ++y) {
            for (int b = 0; b < B; ++b) {
                size_t off = ((static_cast<size_t>(y0 + y) * d1 + b) * d2 + x0) * itemsize;
                f.seekg(static_cast<std::streamoff>(off));
                f.read(reinterpret_cast<char*>(row_buf.data()), static_cast<std::streamsize>(row_buf.size()));
                if (!f) throw std::runtime_error("Short read extracting BIL tile");
                int64_t* dp = dst + static_cast<size_t>(b) * tHW + static_cast<size_t>(y) * tw;
                for (int x = 0; x < tw; ++x) dp[x] = read_sample_le(row_buf.data() + x * itemsize, dt_code);
            }
        }
    } else {
        // BIP: each spatial row y holds tw contiguous B-vectors!
        std::vector<uint8_t> row_buf(static_cast<size_t>(tw) * B * itemsize);
        for (int y = 0; y < th; ++y) {
            size_t off = ((static_cast<size_t>(y0 + y) * d1 + x0) * d2) * itemsize;
            f.seekg(static_cast<std::streamoff>(off));
            f.read(reinterpret_cast<char*>(row_buf.data()), static_cast<std::streamsize>(row_buf.size()));
            if (!f) throw std::runtime_error("Short read extracting BIP tile");
            for (int x = 0; x < tw; ++x) {
                const uint8_t* px = row_buf.data() + static_cast<size_t>(x) * B * itemsize;
                for (int b = 0; b < B; ++b)
                    dst[static_cast<size_t>(b) * tHW + static_cast<size_t>(y) * tw + x] =
                        read_sample_le(px + static_cast<size_t>(b) * itemsize, dt_code);
            }
        }
    }
}

static void scatter_tile_to_file(const int64_t* tile, std::fstream& f, uint32_t /*d0*/, uint32_t d1, uint32_t d2,
                                 uint8_t dt_code, uint8_t spec_axis,
                                 int B, int y0, int th, int x0, int tw, bool clamp_val) {
    size_t itemsize = dtype_size(dt_code);
    size_t tHW = static_cast<size_t>(th) * tw;
    if (spec_axis == 0) {
        if (x0 == 0 && static_cast<uint32_t>(tw) == d2) {
            // Contiguous full-width band strip: 1 write per band instead of th row seeks!
            std::vector<uint8_t> strip(tHW * itemsize);
            for (int b = 0; b < B; ++b) {
                const int64_t* sp = tile + static_cast<size_t>(b) * tHW;
                for (size_t i = 0; i < tHW; ++i) write_sample_le(strip.data() + i * itemsize, sp[i], dt_code, clamp_val);
                size_t off = ((static_cast<size_t>(b) * d1 + y0) * d2) * itemsize;
                f.seekp(static_cast<std::streamoff>(off));
                f.write(reinterpret_cast<const char*>(strip.data()), static_cast<std::streamsize>(strip.size()));
            }
        } else {
            std::vector<uint8_t> row_buf(static_cast<size_t>(tw) * itemsize);
            for (int b = 0; b < B; ++b) {
                for (int y = 0; y < th; ++y) {
                    const int64_t* sp = tile + static_cast<size_t>(b) * tHW + static_cast<size_t>(y) * tw;
                    for (int x = 0; x < tw; ++x) write_sample_le(row_buf.data() + x * itemsize, sp[x], dt_code, clamp_val);
                    size_t off = ((static_cast<size_t>(b) * d1 + (y0 + y)) * d2 + x0) * itemsize;
                    f.seekp(static_cast<std::streamoff>(off));
                    f.write(reinterpret_cast<const char*>(row_buf.data()), static_cast<std::streamsize>(row_buf.size()));
                }
            }
        }
    } else if (spec_axis == 1) {
        std::vector<uint8_t> row_buf(static_cast<size_t>(tw) * itemsize);
        for (int y = 0; y < th; ++y) {
            for (int b = 0; b < B; ++b) {
                const int64_t* sp = tile + static_cast<size_t>(b) * tHW + static_cast<size_t>(y) * tw;
                for (int x = 0; x < tw; ++x) write_sample_le(row_buf.data() + x * itemsize, sp[x], dt_code, clamp_val);
                size_t off = ((static_cast<size_t>(y0 + y) * d1 + b) * d2 + x0) * itemsize;
                f.seekp(static_cast<std::streamoff>(off));
                f.write(reinterpret_cast<const char*>(row_buf.data()), static_cast<std::streamsize>(row_buf.size()));
            }
        }
    } else {
        std::vector<uint8_t> row_buf(static_cast<size_t>(tw) * B * itemsize);
        for (int y = 0; y < th; ++y) {
            for (int x = 0; x < tw; ++x) {
                uint8_t* px = row_buf.data() + static_cast<size_t>(x) * B * itemsize;
                for (int b = 0; b < B; ++b)
                    write_sample_le(px + static_cast<size_t>(b) * itemsize,
                                    tile[static_cast<size_t>(b) * tHW + static_cast<size_t>(y) * tw + x],
                                    dt_code, clamp_val);
            }
            size_t off = ((static_cast<size_t>(y0 + y) * d1 + x0) * d2) * itemsize;
            f.seekp(static_cast<std::streamoff>(off));
            f.write(reinterpret_cast<const char*>(row_buf.data()), static_cast<std::streamsize>(row_buf.size()));
        }
    }
    if (!f) throw std::runtime_error("I/O error writing decoded HND4 tile");
}

static std::array<uint8_t, 32> sha256_file_stream(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for SHA-256: " + path);
    Sha256 sha;
    std::vector<uint8_t> buf(1 << 20);
    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        std::streamsize got = f.gcount();
        if (got > 0) sha.update(buf.data(), static_cast<size_t>(got));
    }
    return sha.digest();
}

static std::array<uint8_t, 32> sha256_and_hist_file_stream(const std::string& path, uint8_t dt_code,
                                                           std::array<uint8_t, 8192>& bits, bool scan_hist) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for SHA-256: " + path);
    Sha256 sha;
    bits.fill(0);
    bool do_hist = scan_hist && (dt_code == DT_I16 || dt_code == DT_U16);
    std::vector<uint8_t> buf(1 << 20); // 1 MiB (divisible by 2!)
    uint8_t carry_byte = 0;
    bool has_carry = false;
    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        std::streamsize got = f.gcount();
        if (got > 0) {
            size_t n = static_cast<size_t>(got);
            sha.update(buf.data(), n);
            if (do_hist) {
                size_t i = 0;
                if (has_carry) {
                    uint8_t pair[2] = {carry_byte, buf[0]};
                    uint16_t k = hist_key(read_sample_le(pair, dt_code), dt_code);
                    bits[k >> 3] |= static_cast<uint8_t>(1u << (k & 7));
                    i = 1;
                    has_carry = false;
                }
                for (; i + 1 < n; i += 2) {
                    uint16_t k = hist_key(read_sample_le(buf.data() + i, dt_code), dt_code);
                    bits[k >> 3] |= static_cast<uint8_t>(1u << (k & 7));
                }
                if (i < n) { carry_byte = buf[i]; has_carry = true; }
            }
        }
    }
    return sha.digest();
}

// Sample up to ~16,384 spatial pixels cleanly (bounded <16 MB RAM!) for shared global LDU-RKLT
static std::vector<int64_t> sample_cube_for_ldu_file(std::ifstream& f, uint32_t /*d0*/, uint32_t d1, uint32_t d2,
                                                     uint8_t dt_code, uint8_t spec_axis,
                                                     int B, int H, int W, uint32_t max_error,
                                                     const HistLut& lut, size_t& out_nsamp) {
    int step_y = 1, step_x = 1;
    while (static_cast<size_t>((H + step_y - 1) / step_y) * ((W + step_x - 1) / step_x) > 16384u) {
        if (H / step_y >= W / step_x) step_y++;
        else step_x++;
    }
    int sy = (H + step_y - 1) / step_y;
    int sx = (W + step_x - 1) / step_x;
    out_nsamp = static_cast<size_t>(sy) * sx;
    std::vector<int64_t> X(static_cast<size_t>(B) * out_nsamp);
    size_t itemsize = dtype_size(dt_code);
    if (spec_axis == 0) {
        std::vector<uint8_t> row(static_cast<size_t>(W) * itemsize);
        for (int b = 0; b < B; ++b) {
            size_t k = 0;
            for (int y = 0; y < H; y += step_y) {
                size_t off = ((static_cast<size_t>(b) * d1 + y) * d2) * itemsize;
                f.seekg(static_cast<std::streamoff>(off));
                f.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
                if (!f) throw std::runtime_error("Short read sampling BSQ cube for LDU");
                for (int x = 0; x < W; x += step_x) {
                    int64_t v = read_sample_le(row.data() + x * itemsize, dt_code);
                    X[static_cast<size_t>(b) * out_nsamp + k++] = lut.enabled ? apply_hist_lut_fwd(v, lut) : nl_quant(v, max_error);
                }
            }
        }
    } else if (spec_axis == 1) {
        std::vector<uint8_t> row(static_cast<size_t>(W) * itemsize);
        for (int y_i = 0, y = 0; y < H; y += step_y, ++y_i) {
            for (int b = 0; b < B; ++b) {
                size_t off = ((static_cast<size_t>(y) * d1 + b) * d2) * itemsize;
                f.seekg(static_cast<std::streamoff>(off));
                f.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
                if (!f) throw std::runtime_error("Short read sampling BIL cube for LDU");
                size_t base = static_cast<size_t>(b) * out_nsamp + static_cast<size_t>(y_i) * sx;
                for (int x_i = 0, x = 0; x < W; x += step_x, ++x_i) {
                    int64_t v = read_sample_le(row.data() + x * itemsize, dt_code);
                    X[base + x_i] = lut.enabled ? apply_hist_lut_fwd(v, lut) : nl_quant(v, max_error);
                }
            }
        }
    } else {
        std::vector<uint8_t> px(static_cast<size_t>(B) * itemsize);
        size_t k = 0;
        for (int y = 0; y < H; y += step_y) {
            for (int x = 0; x < W; x += step_x) {
                size_t off = ((static_cast<size_t>(y) * d1 + x) * d2) * itemsize;
                f.seekg(static_cast<std::streamoff>(off));
                f.read(reinterpret_cast<char*>(px.data()), static_cast<std::streamsize>(px.size()));
                if (!f) throw std::runtime_error("Short read sampling BIP cube for LDU");
                for (int b = 0; b < B; ++b) {
                    int64_t v = read_sample_le(px.data() + b * itemsize, dt_code);
                    X[static_cast<size_t>(b) * out_nsamp + k] = lut.enabled ? apply_hist_lut_fwd(v, lut) : nl_quant(v, max_error);
                }
                k++;
            }
        }
    }
    return X;
}

static std::vector<uint8_t> encode_hnd4_buffer(const uint8_t* raw_bytes, size_t n_bytes,
                                               const EncodeOptions& opt, EncodeStats* st_out,
                                               uint8_t extra_flags = 0) {
    uint8_t dt_code = static_cast<uint8_t>(opt.dtype);
    size_t itemsize = dtype_size(dt_code);
    uint32_t d0 = opt.d0, d1 = opt.d1, d2 = opt.d2;
    uint32_t dims[3] = {d0, d1, d2};
    int B = static_cast<int>(dims[opt.spec_axis]);
    int H = static_cast<int>(dims[(opt.spec_axis == 0) ? 1 : 0]);
    int W = static_cast<int>(dims[(opt.spec_axis == 2) ? 1 : 2]);
    size_t tot = static_cast<size_t>(d0) * d1 * d2;
    int n_threads = resolve_threads(opt.threads);
    uint8_t n_stripes = static_cast<uint8_t>(std::max(1, std::min<int>(opt.stripes, B)));
#if defined(HND3_FORCE_SCALAR)
    bool use_i32_neon = false;
#else
    bool use_i32_neon = (dt_code != DT_I32);
#endif

    uint32_t tile_h = (opt.tile_size > 0) ? std::min<uint32_t>(opt.tile_size, static_cast<uint32_t>(H)) : static_cast<uint32_t>(H);
    uint32_t tile_w = (opt.tile_size > 0) ? std::min<uint32_t>(opt.tile_size, static_cast<uint32_t>(W)) : static_cast<uint32_t>(W);
    uint32_t n_ty = (static_cast<uint32_t>(H) + tile_h - 1) / tile_h;
    uint32_t n_tx = (static_cast<uint32_t>(W) + tile_w - 1) / tile_w;
    uint64_t n_tiles64 = static_cast<uint64_t>(n_ty) * n_tx;
    if (n_tiles64 == 0 || n_tiles64 > 65536ULL)
        throw std::runtime_error("HND4 tile count exceeds 65536 limit (increase --tile-size)");
    size_t n_tiles = static_cast<size_t>(n_tiles64);

    // Check reversible monotonic histogram compaction LUT (16-bit lossless only)
    std::array<uint8_t, 8192> hist_bits{};
    if (opt.hist_lut != 0 && opt.max_error == 0 && (dt_code == DT_I16 || dt_code == DT_U16))
        scan_hist_bits_mem(raw_bytes, n_bytes, dt_code, hist_bits);
    auto lut = build_hist_lut_from_bits(hist_bits, dt_code, tot, opt.hist_lut, opt.max_error);

    // Sample up to 16K spatial pixels for shared global LDU-RKLT
    int step_y = 1, step_x = 1;
    while (static_cast<size_t>((H + step_y - 1) / step_y) * ((W + step_x - 1) / step_x) > 16384u) {
        if (H / step_y >= W / step_x) step_y++; else step_x++;
    }
    int sy = (H + step_y - 1) / step_y, sx = (W + step_x - 1) / step_x;
    size_t nsamp = static_cast<size_t>(sy) * sx;
    std::vector<int64_t> Xs(static_cast<size_t>(B) * nsamp);
    for (int b = 0; b < B; ++b) {
        size_t k = 0;
        for (int y = 0; y < H; y += step_y) {
            for (int x = 0; x < W; x += step_x) {
                uint32_t i0, i1, i2;
                if (opt.spec_axis == 0)      { i0 = b; i1 = y; i2 = x; }
                else if (opt.spec_axis == 1) { i0 = y; i1 = b; i2 = x; }
                else                         { i0 = y; i1 = x; i2 = b; }
                size_t idx = (static_cast<size_t>(i0) * d1 + i1) * d2 + i2;
                int64_t v = read_sample_le(raw_bytes + idx * itemsize, dt_code);
                Xs[static_cast<size_t>(b) * nsamp + k++] = lut.enabled ? apply_hist_lut_fwd(v, lut) : nl_quant(v, opt.max_error);
            }
        }
    }
    auto sm = fit_spectral_model(Xs.data(), B, nsamp, tot, opt.mode, use_i32_neon, n_threads);
    Xs.clear(); Xs.shrink_to_fit();

    std::vector<uint8_t> spec_blob;
    spec_blob.reserve(lut.bytes.size() + sm.bytes.size());
    spec_blob.insert(spec_blob.end(), lut.bytes.begin(), lut.bytes.end());
    spec_blob.insert(spec_blob.end(), sm.bytes.begin(), sm.bytes.end());

    auto sha = sha256_bytes(raw_bytes, n_bytes);
    uint8_t flags = FLAG_CRC32C | extra_flags;
    if (n_tiles > 1) flags |= FLAG_TILED;
    if (opt.max_error > 0) flags |= FLAG_NEAR_LOSSLESS | FLAG_LOSSY;
    if (lut.enabled) flags |= FLAG_HIST_LUT;

    std::vector<TileIndexEntry> index(n_tiles);
    std::vector<uint8_t> all_tile_payloads;
    size_t payload_base = HDR4_SIZE + spec_blob.size() + n_tiles * TILE_ENTRY_SIZE;
    int64_t max_verified_err = 0;

    for (uint32_t ty = 0; ty < n_ty; ++ty) {
        int y0 = static_cast<int>(ty * tile_h);
        int th = std::min<int>(static_cast<int>(tile_h), H - y0);
        for (uint32_t tx = 0; tx < n_tx; ++tx) {
            int x0 = static_cast<int>(tx * tile_w);
            int tw = std::min<int>(static_cast<int>(tile_w), W - x0);
            size_t tHW = static_cast<size_t>(th) * tw;
            size_t t_idx = static_cast<size_t>(ty) * n_tx + tx;

            std::vector<int64_t> orig_tile(static_cast<size_t>(B) * tHW);
            extract_tile_from_mem(raw_bytes, d0, d1, d2, dt_code, opt.spec_axis, B, y0, th, x0, tw, orig_tile.data());
            std::vector<int64_t> work = orig_tile;
            if (lut.enabled) {
                for (int64_t& v : work) v = apply_hist_lut_fwd(v, lut);
            } else if (opt.max_error > 0) {
                for (int64_t& v : work) v = nl_quant(v, opt.max_error);
            }
            apply_spectral_forward(work.data(), B, tHW, sm, use_i32_neon, n_threads);
            auto et = encode_tile_layers(work.data(), B, th, tw, opt.levels, opt.base_k, n_stripes, n_threads);

            if (opt.verify) {
                size_t lc = 0;
                auto dec_w = decode_tile_layers(et.payload.data(), et.payload.size(), B, th, tw,
                                                opt.levels, opt.base_k, n_stripes, 0, -1,
                                                et.layer_bytes.data(), et.layer_crc.data(), true, lc, n_threads);
                apply_spectral_inverse(dec_w.data(), B, tHW, sm, use_i32_neon, n_threads);
                for (size_t i = 0; i < orig_tile.size(); ++i) {
                    int64_t rec_v = lut.enabled ? apply_hist_lut_inv(dec_w[i], lut)
                                                : nl_dequant(dec_w[i], opt.max_error, dt_code);
                    int64_t e = std::abs(orig_tile[i] - rec_v);
                    if (e > max_verified_err) max_verified_err = e;
                    if (e > static_cast<int64_t>(opt.max_error))
                        throw std::runtime_error("HND4 tile verification FAILED (exceeded max_error)!");
                }
            }

            index[t_idx].offset = payload_base + all_tile_payloads.size();
            index[t_idx].tile_bytes = static_cast<uint32_t>(et.payload.size());
            index[t_idx].layer_bytes = et.layer_bytes;
            index[t_idx].layer_crc = et.layer_crc;
            all_tile_payloads.insert(all_tile_payloads.end(), et.payload.begin(), et.payload.end());
        }
    }

    std::vector<uint8_t> idx_buf, hdr;
    pack_tile_index(idx_buf, index);
    pack_hnd4_header(hdr, flags, dt_code, sm.spec_code, opt.levels, opt.spec_axis, opt.base_k, n_stripes,
                     d0, d1, d2, tile_h, tile_w, n_ty, n_tx, opt.max_error, sha,
                     static_cast<uint32_t>(spec_blob.size()), 0u);
    uint32_t hcrc = crc32c_impl(hdr.data(), HDR4_SIZE - 4, 0u);
    hcrc = crc32c_impl(spec_blob.data(), spec_blob.size(), hcrc);
    hcrc = crc32c_impl(idx_buf.data(), idx_buf.size(), hcrc);
    std::memcpy(hdr.data() + HDR4_SIZE - 4, &hcrc, 4);

    std::vector<uint8_t> out;
    out.reserve(HDR4_SIZE + spec_blob.size() + idx_buf.size() + all_tile_payloads.size());
    out.insert(out.end(), hdr.begin(), hdr.end());
    out.insert(out.end(), spec_blob.begin(), spec_blob.end());
    out.insert(out.end(), idx_buf.begin(), idx_buf.end());
    out.insert(out.end(), all_tile_payloads.begin(), all_tile_payloads.end());

    if (st_out) {
        st_out->version = 4; st_out->d0 = d0; st_out->d1 = d1; st_out->d2 = d2;
        st_out->tile_h = tile_h; st_out->tile_w = tile_w; st_out->n_tiles_y = n_ty; st_out->n_tiles_x = n_tx;
        st_out->max_error = opt.max_error; st_out->spec_axis = opt.spec_axis;
        st_out->raw_bytes = n_bytes; st_out->compressed_bytes = out.size();
        st_out->verified = opt.verify; st_out->verified_max_abs_err = max_verified_err;
        st_out->hist_lut_used = lut.enabled; st_out->hist_lut_unique = lut.K;
    }
    return out;
}

static DecodedCube decode_hnd4_buffer(const uint8_t* data, size_t len, const DecodeOptions& opt) {
    auto m = parse_hnd4_meta(data, len, opt.max_samples, opt.verify_crc);
    int n_threads = resolve_threads(opt.threads);
    size_t itemsize = dtype_size(m.dt_code);
#if defined(HND3_FORCE_SCALAR)
    bool use_i32_neon = false;
#else
    bool use_i32_neon = (m.dt_code != DT_I32);
#endif
    bool single_tile = (opt.tile_y >= 0 || opt.tile_x >= 0);
    if (single_tile) {
        if (opt.tile_y < 0 || opt.tile_x < 0 ||
            static_cast<uint32_t>(opt.tile_y) >= m.n_ty || static_cast<uint32_t>(opt.tile_x) >= m.n_tx)
            throw std::runtime_error("Requested HND4 tile out of range");
    }

    int k_target = std::max(0, std::min(opt.stop_layer, static_cast<int>(m.base_k)));
    bool is_lossless = (m.max_error == 0) && !(m.flags & FLAG_LOSSY) && (k_target == 0) && (opt.max_level < 0) && !single_tile;

    DecodedCube res;
    res.version = 4; res.dt_code = m.dt_code; res.max_error = m.max_error;
    res.crc_checked = opt.verify_crc;
    res.crc_ok = opt.verify_crc;

    uint32_t ty0 = single_tile ? static_cast<uint32_t>(opt.tile_y) : 0u;
    uint32_t ty1 = single_tile ? (ty0 + 1u) : m.n_ty;
    uint32_t tx0 = single_tile ? static_cast<uint32_t>(opt.tile_x) : 0u;
    uint32_t tx1 = single_tile ? (tx0 + 1u) : m.n_tx;

    if (single_tile) {
        int y0 = static_cast<int>(ty0 * m.tile_h), th = std::min<int>(static_cast<int>(m.tile_h), m.H - y0);
        int x0 = static_cast<int>(tx0 * m.tile_w), tw = std::min<int>(static_cast<int>(m.tile_w), m.W - x0);
        if (m.spec_axis == 0)      { res.d0 = m.B; res.d1 = th; res.d2 = tw; }
        else if (m.spec_axis == 1) { res.d0 = th; res.d1 = m.B; res.d2 = tw; }
        else                       { res.d0 = th; res.d1 = tw; res.d2 = m.B; }
        res.raw_bytes.resize(static_cast<size_t>(m.B) * th * tw * itemsize);
    } else {
        res.d0 = m.d0; res.d1 = m.d1; res.d2 = m.d2;
        res.raw_bytes.resize(static_cast<size_t>(m.d0) * m.d1 * m.d2 * itemsize);
    }

    for (uint32_t ty = ty0; ty < ty1; ++ty) {
        int y0 = static_cast<int>(ty * m.tile_h), th = std::min<int>(static_cast<int>(m.tile_h), m.H - y0);
        for (uint32_t tx = tx0; tx < tx1; ++tx) {
            int x0 = static_cast<int>(tx * m.tile_w), tw = std::min<int>(static_cast<int>(m.tile_w), m.W - x0);
            const auto& e = m.index[static_cast<size_t>(ty) * m.n_tx + tx];
            // Support truncated files up to stop_layer!
            size_t need_bytes = 0;
            for (int k = static_cast<int>(m.base_k); k >= k_target; --k) need_bytes += e.layer_bytes[k];
            if (e.offset < m.payload_start || e.offset > len || need_bytes > len - e.offset)
                throw std::runtime_error("Truncated HND4 tile payload");
            size_t avail = std::min<size_t>(e.tile_bytes, len - e.offset);
            auto work = decode_tile_layers(data + e.offset, avail, m.B, th, tw,
                                           m.n_lv, m.base_k, m.n_stripes, opt.stop_layer, opt.max_level,
                                           e.layer_bytes.data(), e.layer_crc.data(),
                                           opt.verify_crc, res.crc_layers_verified, n_threads);
            apply_spectral_inverse(work.data(), m.B, static_cast<size_t>(th) * tw, m.sm, use_i32_neon, n_threads);
            if (m.lut.enabled) {
                for (int64_t& v : work) v = apply_hist_lut_inv(v, m.lut);
            } else if (m.max_error > 0) {
                for (int64_t& v : work) v = nl_dequant(v, m.max_error, m.dt_code);
            }
            if (single_tile) {
                scatter_tile_to_mem(work.data(), res.raw_bytes.data(), res.d0, res.d1, res.d2,
                                    m.dt_code, m.spec_axis, m.B, 0, th, 0, tw, !is_lossless);
            } else {
                scatter_tile_to_mem(work.data(), res.raw_bytes.data(), m.d0, m.d1, m.d2,
                                    m.dt_code, m.spec_axis, m.B, y0, th, x0, tw, !is_lossless);
            }
        }
    }

    if (opt.verify_sha && is_lossless) {
        res.sha_checked = true;
        res.sha_ok = (sha256_bytes(res.raw_bytes.data(), res.raw_bytes.size()) == m.sha);
        if (!res.sha_ok) throw std::runtime_error("SHA-256 verification FAILED on lossless HND4 decode!");
    }
    return res;
}

// =============================================================================
// Public C++ API: encode(), decode(), encode_file(), decode_file(), inspect()
// =============================================================================
std::vector<uint8_t> encode(const uint8_t* raw_bytes, size_t n_bytes,
                            const EncodeOptions& opt, EncodeStats* stats) {
    validate_encode_opts(opt);
    uint8_t dt_code = static_cast<uint8_t>(opt.dtype);
    size_t itemsize = dtype_size(dt_code);
    int n_threads = resolve_threads(opt.threads);

    auto t0 = std::chrono::steady_clock::now();
    EncodeStats local_st;
    std::vector<uint8_t> blob;

    if (opt.blind) {
        if (n_bytes == 0 || (n_bytes % itemsize != 0))
            throw std::runtime_error("Input byte size must be non-empty and divisible by dtype itemsize");
        size_t n = n_bytes / itemsize;
        size_t w_win = std::min<size_t>(n, 4194304ULL);
        std::vector<double> xf(w_win);
        for (size_t i = 0; i < w_win; ++i) xf[i] = static_cast<double>(read_sample_le(raw_bytes + i * itemsize, dt_code));
        auto rec = recover_strides_cpp(xf, n, 5);
        xf.clear(); xf.shrink_to_fit();
        local_st.blind_strides = rec;

        bool matched_3d = false;
        if (rec.size() >= 3 && rec[1] > 1 && rec[2] > rec[1] && (rec[2] % rec[1] == 0)) {
            size_t s1 = rec[1], s2 = rec[2];
            size_t d0 = n / s2, d1 = s2 / s1, d2 = s1;
            if (d0 * d1 * d2 == n && d0 >= 2 && d1 >= 4 && d2 >= 4) {
                // Check sparse-histogram compaction LUT before spectral-axis scoring!
                std::array<uint8_t, 8192> hb{};
                if (opt.hist_lut != 0 && opt.max_error == 0 && (dt_code == DT_I16 || dt_code == DT_U16))
                    scan_hist_bits_mem(raw_bytes, n_bytes, dt_code, hb);
                auto lut_chk = build_hist_lut_from_bits(hb, dt_code, n, opt.hist_lut, opt.max_error);

                std::vector<int64_t> xi(n);
                for (size_t i = 0; i < n; ++i) {
                    int64_t v = read_sample_le(raw_bytes + i * itemsize, dt_code);
                    xi[i] = lut_chk.enabled ? apply_hist_lut_fwd(v, lut_chk) : v;
                }
                int sp_ax = choose_spectral_axis_cpp(xi.data(), static_cast<uint32_t>(d0),
                                                     static_cast<uint32_t>(d1), static_cast<uint32_t>(d2), n_threads);
                xi.clear(); xi.shrink_to_fit();
                EncodeOptions resolved = opt;
                resolved.blind = false;
                resolved.d0 = static_cast<uint32_t>(d0);
                resolved.d1 = static_cast<uint32_t>(d1);
                resolved.d2 = static_cast<uint32_t>(d2);
                resolved.spec_axis = (sp_ax < 0) ? 0 : static_cast<uint8_t>(sp_ax);
                resolved.mode = (sp_ax < 0) ? "none" : opt.mode;
                local_st.d0 = resolved.d0; local_st.d1 = resolved.d1; local_st.d2 = resolved.d2;
                local_st.spec_axis = sp_ax;
                if (resolved.tile_size > 0 || resolved.max_error > 0 || resolved.per_layer_crc || lut_chk.enabled) {
                    blob = encode_hnd4_buffer(raw_bytes, n_bytes, resolved, &local_st, FLAG_BLIND);
                    local_st.spec_axis = sp_ax;
                } else {
                    blob = encode_hnd3_v3(raw_bytes, resolved.d0, resolved.d1, resolved.d2, dt_code,
                                          resolved.mode, resolved.levels, resolved.spec_axis,
                                          resolved.base_k, resolved.stripes, FLAG_BLIND, n_threads);
                    local_st.version = 3;
                }
                matched_3d = true;
            }
        }
        if (!matched_3d) {
            auto sha = sha256_bytes(raw_bytes, n_bytes);
            local_st.version = 3; local_st.d0 = 1; local_st.d1 = 1; local_st.d2 = static_cast<uint32_t>(n);
            local_st.spec_axis = -1;
            blob.reserve(HDR_SIZE + n_bytes);
            blob.insert(blob.end(), MAGIC, MAGIC + 4);
            blob.push_back(VERSION);
            blob.push_back(FLAG_BLIND | FLAG_RAW);
            blob.push_back(dt_code);
            blob.push_back(SPEC_NONE);
            blob.push_back(0); blob.push_back(0); blob.push_back(0); blob.push_back(1);
            write_le<uint16_t>(blob, 1);
            write_le<uint32_t>(blob, 1); write_le<uint32_t>(blob, 1); write_le<uint32_t>(blob, static_cast<uint32_t>(n));
            blob.insert(blob.end(), sha.begin(), sha.end());
            write_le<uint32_t>(blob, 0);
            blob.insert(blob.end(), raw_bytes, raw_bytes + n_bytes);
        }
    } else {
        if (opt.d0 == 0 || opt.d1 == 0 || opt.d2 == 0)
            throw std::runtime_error("Zero dimension in --shape");
        uint64_t tot64 = static_cast<uint64_t>(opt.d0) * opt.d1;
        if (tot64 > 512000000ULL || tot64 * opt.d2 > 512000000ULL)
            throw std::runtime_error("Shape exceeds 512M sample safety cap");
        if (tot64 * opt.d2 * itemsize != n_bytes)
            throw std::runtime_error("Input byte size (" + std::to_string(n_bytes) +
                                     " B) does not match shape*itemsize (" +
                                     std::to_string(tot64 * opt.d2 * itemsize) + " B)");
        local_st.d0 = opt.d0; local_st.d1 = opt.d1; local_st.d2 = opt.d2;
        local_st.spec_axis = opt.spec_axis;
        bool auto_lut = false;
        if (opt.tile_size == 0 && opt.max_error == 0 && !opt.per_layer_crc &&
            opt.hist_lut != 0 && (dt_code == DT_I16 || dt_code == DT_U16)) {
            std::array<uint8_t, 8192> hb{};
            scan_hist_bits_mem(raw_bytes, n_bytes, dt_code, hb);
            auto_lut = build_hist_lut_from_bits(hb, dt_code, static_cast<size_t>(tot64 * opt.d2), opt.hist_lut, 0).enabled;
        }
        if (opt.tile_size > 0 || opt.max_error > 0 || opt.per_layer_crc || auto_lut) {
            blob = encode_hnd4_buffer(raw_bytes, n_bytes, opt, &local_st);
        } else {
            blob = encode_hnd3_v3(raw_bytes, opt.d0, opt.d1, opt.d2, dt_code, opt.mode,
                                  opt.levels, opt.spec_axis, opt.base_k, opt.stripes, 0, n_threads);
            local_st.version = 3;
        }
    }
    local_st.encode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    local_st.raw_bytes = n_bytes;
    local_st.compressed_bytes = blob.size();

    if (opt.verify && local_st.version == 3) {
        DecodeOptions dopt; dopt.threads = n_threads; dopt.verify_sha = true;
        auto chk = decode_hnd3_v3(blob.data(), blob.size(), dopt);
        if (chk.raw_bytes.size() != n_bytes || std::memcmp(chk.raw_bytes.data(), raw_bytes, n_bytes) != 0)
            throw std::runtime_error("Encode round-trip verification FAILED!");
        local_st.verified = true;
        local_st.verified_max_abs_err = 0;
    }
    if (stats) *stats = local_st;
    return blob;
}

DecodedCube decode(const uint8_t* data, size_t len, const DecodeOptions& opt) {
    if (len >= 5 && std::memcmp(data, MAGIC4, 4) == 0 && data[4] == VERSION4)
        return decode_hnd4_buffer(data, len, opt);
    if (len >= 5 && std::memcmp(data, MAGIC, 4) == 0 && data[4] == VERSION)
        return decode_hnd3_v3(data, len, opt);
    throw std::runtime_error("Unrecognized HND3/HND4 magic or version");
}

FileInfo inspect(const uint8_t* data, size_t len) {
    FileInfo fi;
    fi.file_bytes = len;
    if (len >= HDR4_SIZE && std::memcmp(data, MAGIC4, 4) == 0 && data[4] == VERSION4) {
        fi.version = 4; fi.flags = data[5]; fi.dt_code = data[6]; fi.spec_code = data[7];
        fi.n_lv = data[8]; fi.spec_axis = data[9]; fi.base_k = data[10]; fi.n_stripes = data[11];
        fi.d0 = read_le<uint32_t>(data + 12); fi.d1 = read_le<uint32_t>(data + 16); fi.d2 = read_le<uint32_t>(data + 20);
        fi.tile_h = read_le<uint32_t>(data + 24); fi.tile_w = read_le<uint32_t>(data + 28);
        fi.n_tiles_y = read_le<uint32_t>(data + 32); fi.n_tiles_x = read_le<uint32_t>(data + 36);
        fi.max_error = read_le<uint32_t>(data + 40);
        uint32_t spec_len = read_le<uint32_t>(data + 76);
        uint32_t header_crc = read_le<uint32_t>(data + 80);
        uint64_t n_tiles = static_cast<uint64_t>(fi.n_tiles_y) * fi.n_tiles_x;
        if (n_tiles > 0 && n_tiles <= 65536ULL) {
            uint64_t meta_body = static_cast<uint64_t>(spec_len) + n_tiles * TILE_ENTRY_SIZE;
            if (meta_body <= len - HDR4_SIZE) {
                fi.header_crc_checked = true;
                uint32_t c = crc32c_impl(data, HDR4_SIZE - 4, 0u);
                c = crc32c_impl(data + HDR4_SIZE, static_cast<size_t>(meta_body), c);
                fi.header_crc_ok = (c == header_crc);
            }
        }
        return fi;
    }
    if (len >= HDR_SIZE && std::memcmp(data, MAGIC, 4) == 0 && data[4] == VERSION) {
        fi.version = 3; fi.flags = data[5]; fi.dt_code = data[6]; fi.spec_code = data[7];
        fi.n_lv = data[8]; fi.spec_axis = data[9]; fi.base_k = data[10]; fi.n_stripes = data[11];
        fi.d0 = read_le<uint32_t>(data + 14); fi.d1 = read_le<uint32_t>(data + 18); fi.d2 = read_le<uint32_t>(data + 22);
        return fi;
    }
    throw std::runtime_error("Invalid HND3/HND4 stream");
}

FileInfo inspect_file(const std::string& path) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) throw std::runtime_error("Cannot open input file: " + path);
    size_t file_sz = static_cast<size_t>(fin.tellg());
    if (file_sz < HDR_SIZE) throw std::runtime_error("Truncated HND3/HND4 file");
    size_t first_read = std::min<size_t>(file_sz, HDR4_SIZE);
    std::vector<uint8_t> hdr(first_read);
    fin.seekg(0);
    fin.read(reinterpret_cast<char*>(hdr.data()), static_cast<std::streamsize>(first_read));
    if (!fin) throw std::runtime_error("Short read inspecting HND header");

    if (first_read >= HDR4_SIZE && std::memcmp(hdr.data(), MAGIC4, 4) == 0 && hdr[4] == VERSION4) {
        uint32_t n_ty = read_le<uint32_t>(hdr.data() + 32);
        uint32_t n_tx = read_le<uint32_t>(hdr.data() + 36);
        uint32_t spec_len = read_le<uint32_t>(hdr.data() + 76);
        uint64_t n_tiles = static_cast<uint64_t>(n_ty) * n_tx;
        if (n_tiles > 0 && n_tiles <= 65536ULL) {
            uint64_t meta_sz = static_cast<uint64_t>(HDR4_SIZE) + spec_len + n_tiles * TILE_ENTRY_SIZE;
            if (meta_sz <= file_sz && meta_sz <= (64ULL << 20)) {
                std::vector<uint8_t> meta(static_cast<size_t>(meta_sz));
                fin.seekg(0);
                fin.read(reinterpret_cast<char*>(meta.data()), static_cast<std::streamsize>(meta_sz));
                if (!fin) throw std::runtime_error("Short read inspecting HND4 metadata");
                FileInfo fi = inspect(meta.data(), meta.size());
                fi.file_bytes = file_sz;
                return fi;
            }
        }
    }
    FileInfo fi = inspect(hdr.data(), hdr.size());
    fi.file_bytes = file_sz;
    return fi;
}

// =============================================================================
// True O(B * tile_h * tile_w) Streaming File Encoder & Decoder for Tiled HND4!
// =============================================================================
EncodeStats encode_file(const std::string& in_path, const std::string& out_path, const EncodeOptions& opt) {
    validate_encode_opts(opt);
    bool use_streaming_hnd4 = (!opt.blind) && (opt.tile_size > 0 || opt.max_error > 0 || opt.per_layer_crc || opt.hist_lut == 1);
    if (!use_streaming_hnd4) {
        std::ifstream fin(in_path, std::ios::binary | std::ios::ate);
        if (!fin) throw std::runtime_error("Cannot open input file: " + in_path);
        size_t sz = static_cast<size_t>(fin.tellg());
        fin.seekg(0);
        std::vector<uint8_t> raw(sz);
        if (sz > 0) fin.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(sz));
        EncodeStats st;
        auto blob = encode(raw.data(), raw.size(), opt, &st);
        std::ofstream fout(out_path, std::ios::binary);
        if (!fout) throw std::runtime_error("Cannot open output file: " + out_path);
        if (!blob.empty()) fout.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
        return st;
    }

    // Streaming O(tile) HND4 file encoder!
    uint8_t dt_code = static_cast<uint8_t>(opt.dtype);
    size_t itemsize = dtype_size(dt_code);
    if (opt.d0 == 0 || opt.d1 == 0 || opt.d2 == 0) throw std::runtime_error("Zero dimension in --shape");
    unsigned __int128 tot128 = static_cast<unsigned __int128>(opt.d0) * opt.d1 * opt.d2;
    if (tot128 > (std::numeric_limits<uint64_t>::max() / itemsize))
        throw std::runtime_error("Cube byte size overflows 64-bit limit");
    uint64_t tot64 = static_cast<uint64_t>(tot128);

    std::ifstream fin(in_path, std::ios::binary | std::ios::ate);
    if (!fin) throw std::runtime_error("Cannot open input file: " + in_path);
    size_t file_sz = static_cast<size_t>(fin.tellg());
    if (tot64 * itemsize != file_sz)
        throw std::runtime_error("Input file size (" + std::to_string(file_sz) +
                                 " B) does not match shape*itemsize (" + std::to_string(tot64 * itemsize) + " B)");

    uint32_t dims[3] = {opt.d0, opt.d1, opt.d2};
    int B = static_cast<int>(dims[opt.spec_axis]);
    int H = static_cast<int>(dims[(opt.spec_axis == 0) ? 1 : 0]);
    int W = static_cast<int>(dims[(opt.spec_axis == 2) ? 1 : 2]);
    int n_threads = resolve_threads(opt.threads);
    uint8_t n_stripes = static_cast<uint8_t>(std::max(1, std::min<int>(opt.stripes, B)));
#if defined(HND3_FORCE_SCALAR)
    bool use_i32_neon = false;
#else
    bool use_i32_neon = (dt_code != DT_I32);
#endif

    uint32_t tile_h = (opt.tile_size > 0) ? std::min<uint32_t>(opt.tile_size, static_cast<uint32_t>(H)) : static_cast<uint32_t>(H);
    uint32_t tile_w = (opt.tile_size > 0) ? std::min<uint32_t>(opt.tile_size, static_cast<uint32_t>(W)) : static_cast<uint32_t>(W);
    uint32_t n_ty = (static_cast<uint32_t>(H) + tile_h - 1) / tile_h;
    uint32_t n_tx = (static_cast<uint32_t>(W) + tile_w - 1) / tile_w;
    uint64_t n_tiles64 = static_cast<uint64_t>(n_ty) * n_tx;
    if (n_tiles64 == 0 || n_tiles64 > 65536ULL)
        throw std::runtime_error("HND4 tile count exceeds 65536 limit (increase --tile-size)");
    size_t n_tiles = static_cast<size_t>(n_tiles64);

    auto t0 = std::chrono::steady_clock::now();

    // Pass 1: Single streaming 1 MiB pass for SHA-256 + 8 KB sparse-histogram presence bitset!
    std::array<uint8_t, 8192> hist_bits{};
    bool scan_hist = (opt.hist_lut != 0 && opt.max_error == 0 && (dt_code == DT_I16 || dt_code == DT_U16));
    auto sha = sha256_and_hist_file_stream(in_path, dt_code, hist_bits, scan_hist);
    auto lut = build_hist_lut_from_bits(hist_bits, dt_code, static_cast<size_t>(tot64), opt.hist_lut, opt.max_error);

    // Pass 2: Strided <16 MB sample -> ONE shared global LDU-RKLT model!
    size_t nsamp = 0;
    auto Xs = sample_cube_for_ldu_file(fin, opt.d0, opt.d1, opt.d2, dt_code, opt.spec_axis, B, H, W, opt.max_error, lut, nsamp);
    auto sm = fit_spectral_model(Xs.data(), B, nsamp, static_cast<size_t>(tot64), opt.mode, use_i32_neon, n_threads);
    Xs.clear(); Xs.shrink_to_fit();

    std::vector<uint8_t> spec_blob;
    spec_blob.reserve(lut.bytes.size() + sm.bytes.size());
    spec_blob.insert(spec_blob.end(), lut.bytes.begin(), lut.bytes.end());
    spec_blob.insert(spec_blob.end(), sm.bytes.begin(), sm.bytes.end());

    // Reserve header + spec_blob + tile index table on disk, then stream tiles one-by-one!
    std::fstream fout(out_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!fout) throw std::runtime_error("Cannot open output file: " + out_path);
    std::vector<uint8_t> placeholder(HDR4_SIZE + spec_blob.size() + n_tiles * TILE_ENTRY_SIZE, 0);
    fout.write(reinterpret_cast<const char*>(placeholder.data()), static_cast<std::streamsize>(placeholder.size()));
    uint64_t cur_off = placeholder.size();
    placeholder.clear(); placeholder.shrink_to_fit();

    std::vector<TileIndexEntry> index(n_tiles);
    int64_t max_verified_err = 0;

    for (uint32_t ty = 0; ty < n_ty; ++ty) {
        int y0 = static_cast<int>(ty * tile_h);
        int th = std::min<int>(static_cast<int>(tile_h), H - y0);
        for (uint32_t tx = 0; tx < n_tx; ++tx) {
            int x0 = static_cast<int>(tx * tile_w);
            int tw = std::min<int>(static_cast<int>(tile_w), W - x0);
            size_t tHW = static_cast<size_t>(th) * tw;
            size_t t_idx = static_cast<size_t>(ty) * n_tx + tx;

            std::vector<int64_t> orig_tile(static_cast<size_t>(B) * tHW);
            extract_tile_from_file(fin, opt.d0, opt.d1, opt.d2, dt_code, opt.spec_axis, B, y0, th, x0, tw, orig_tile.data());

            std::vector<int64_t> work = orig_tile;
            if (!opt.verify) { orig_tile.clear(); orig_tile.shrink_to_fit(); }
            if (lut.enabled) {
                for (int64_t& v : work) v = apply_hist_lut_fwd(v, lut);
            } else if (opt.max_error > 0) {
                for (int64_t& v : work) v = nl_quant(v, opt.max_error);
            }
            apply_spectral_forward(work.data(), B, tHW, sm, use_i32_neon, n_threads);
            auto et = encode_tile_layers(work.data(), B, th, tw, opt.levels, opt.base_k, n_stripes, n_threads);
            work.clear(); work.shrink_to_fit();

            if (opt.verify) {
                size_t lc = 0;
                auto dec_w = decode_tile_layers(et.payload.data(), et.payload.size(), B, th, tw,
                                                opt.levels, opt.base_k, n_stripes, 0, -1,
                                                et.layer_bytes.data(), et.layer_crc.data(), true, lc, n_threads);
                apply_spectral_inverse(dec_w.data(), B, tHW, sm, use_i32_neon, n_threads);
                for (size_t i = 0; i < orig_tile.size(); ++i) {
                    int64_t rec_v = lut.enabled ? apply_hist_lut_inv(dec_w[i], lut)
                                                : nl_dequant(dec_w[i], opt.max_error, dt_code);
                    int64_t e = std::abs(orig_tile[i] - rec_v);
                    if (e > max_verified_err) max_verified_err = e;
                    if (e > static_cast<int64_t>(opt.max_error))
                        throw std::runtime_error("Streaming HND4 tile verification FAILED!");
                }
            }

            fout.write(reinterpret_cast<const char*>(et.payload.data()), static_cast<std::streamsize>(et.payload.size()));
            if (!fout) throw std::runtime_error("I/O error writing HND4 tile");
            index[t_idx].offset = cur_off;
            index[t_idx].tile_bytes = static_cast<uint32_t>(et.payload.size());
            index[t_idx].layer_bytes = et.layer_bytes;
            index[t_idx].layer_crc = et.layer_crc;
            cur_off += et.payload.size();
        }
    }

    uint8_t flags = FLAG_CRC32C;
    if (n_tiles > 1) flags |= FLAG_TILED;
    if (opt.max_error > 0) flags |= FLAG_NEAR_LOSSLESS | FLAG_LOSSY;
    if (lut.enabled) flags |= FLAG_HIST_LUT;

    std::vector<uint8_t> idx_buf, hdr;
    pack_tile_index(idx_buf, index);
    pack_hnd4_header(hdr, flags, dt_code, sm.spec_code, opt.levels, opt.spec_axis, opt.base_k, n_stripes,
                     opt.d0, opt.d1, opt.d2, tile_h, tile_w, n_ty, n_tx, opt.max_error, sha,
                     static_cast<uint32_t>(spec_blob.size()), 0u);
    uint32_t hcrc = crc32c_impl(hdr.data(), HDR4_SIZE - 4, 0u);
    hcrc = crc32c_impl(spec_blob.data(), spec_blob.size(), hcrc);
    hcrc = crc32c_impl(idx_buf.data(), idx_buf.size(), hcrc);
    std::memcpy(hdr.data() + HDR4_SIZE - 4, &hcrc, 4);

    fout.seekp(0);
    fout.write(reinterpret_cast<const char*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
    if (!spec_blob.empty())
        fout.write(reinterpret_cast<const char*>(spec_blob.data()), static_cast<std::streamsize>(spec_blob.size()));
    fout.write(reinterpret_cast<const char*>(idx_buf.data()), static_cast<std::streamsize>(idx_buf.size()));
    fout.flush();

    EncodeStats st;
    st.version = 4; st.d0 = opt.d0; st.d1 = opt.d1; st.d2 = opt.d2;
    st.tile_h = tile_h; st.tile_w = tile_w; st.n_tiles_y = n_ty; st.n_tiles_x = n_tx;
    st.max_error = opt.max_error; st.spec_axis = opt.spec_axis;
    st.raw_bytes = file_sz; st.compressed_bytes = static_cast<size_t>(cur_off);
    st.encode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    st.verified = opt.verify; st.verified_max_abs_err = max_verified_err;
    st.hist_lut_used = lut.enabled; st.hist_lut_unique = lut.K;
    return st;
}

DecodedCube decode_file(const std::string& in_path, const std::string& out_path, const DecodeOptions& opt) {
    std::ifstream fin(in_path, std::ios::binary | std::ios::ate);
    if (!fin) throw std::runtime_error("Cannot open input file: " + in_path);
    size_t file_sz = static_cast<size_t>(fin.tellg());
    if (file_sz < 5) throw std::runtime_error("Truncated HND file");
    fin.seekg(0);
    uint8_t magic5[5];
    fin.read(reinterpret_cast<char*>(magic5), 5);

    if (std::memcmp(magic5, MAGIC, 4) == 0 && magic5[4] == VERSION) {
        // HND3 v3 single-tile decode
        fin.seekg(0);
        std::vector<uint8_t> blob(file_sz);
        fin.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(file_sz));
        auto dec = decode_hnd3_v3(blob.data(), blob.size(), opt);
        if (!out_path.empty()) {
            std::ofstream fout(out_path, std::ios::binary);
            if (!fout) throw std::runtime_error("Cannot open output file: " + out_path);
            if (!dec.raw_bytes.empty())
                fout.write(reinterpret_cast<const char*>(dec.raw_bytes.data()), static_cast<std::streamsize>(dec.raw_bytes.size()));
        }
        return dec;
    }

    if (std::memcmp(magic5, MAGIC4, 4) != 0 || magic5[4] != VERSION4)
        throw std::runtime_error("Unrecognized HND3/HND4 magic");

    // Streaming O(tile) HND4 file decoder!
    if (file_sz < HDR4_SIZE) throw std::runtime_error("Truncated HND4 header");
    std::vector<uint8_t> hdr(HDR4_SIZE);
    fin.seekg(0);
    fin.read(reinterpret_cast<char*>(hdr.data()), HDR4_SIZE);
    uint32_t n_ty = read_le<uint32_t>(hdr.data() + 32);
    uint32_t n_tx = read_le<uint32_t>(hdr.data() + 36);
    uint32_t spec_len = read_le<uint32_t>(hdr.data() + 76);
    uint64_t n_tiles = static_cast<uint64_t>(n_ty) * n_tx;
    if (n_tiles == 0 || n_tiles > 65536ULL) throw std::runtime_error("Invalid HND4 tile grid");
    uint64_t idx_bytes64 = n_tiles * TILE_ENTRY_SIZE;
    if (static_cast<uint64_t>(spec_len) > file_sz - HDR4_SIZE ||
        idx_bytes64 > file_sz - HDR4_SIZE - spec_len)
        throw std::runtime_error("Truncated HND4 metadata/index");
    size_t meta_sz = static_cast<size_t>(HDR4_SIZE + static_cast<uint64_t>(spec_len) + idx_bytes64);

    std::vector<uint8_t> meta_buf(meta_sz);
    fin.seekg(0);
    fin.read(reinterpret_cast<char*>(meta_buf.data()), static_cast<std::streamsize>(meta_sz));
    if (!fin) throw std::runtime_error("Short read loading HND4 metadata/index");
    auto m = parse_hnd4_meta(meta_buf.data(), meta_buf.size(), opt.max_samples, opt.verify_crc);
    meta_buf.clear(); meta_buf.shrink_to_fit();

    int n_threads = resolve_threads(opt.threads);
    size_t itemsize = dtype_size(m.dt_code);
#if defined(HND3_FORCE_SCALAR)
    bool use_i32_neon = false;
#else
    bool use_i32_neon = (m.dt_code != DT_I32);
#endif
    bool single_tile = (opt.tile_y >= 0 || opt.tile_x >= 0);
    if (single_tile && (opt.tile_y < 0 || opt.tile_x < 0 ||
                        static_cast<uint32_t>(opt.tile_y) >= m.n_ty || static_cast<uint32_t>(opt.tile_x) >= m.n_tx))
        throw std::runtime_error("Requested HND4 tile out of range");

    int k_target = std::max(0, std::min(opt.stop_layer, static_cast<int>(m.base_k)));
    bool is_lossless = (m.max_error == 0) && !(m.flags & FLAG_LOSSY) && (k_target == 0) && (opt.max_level < 0) && !single_tile;

    DecodedCube res;
    res.version = 4; res.dt_code = m.dt_code; res.max_error = m.max_error;
    res.crc_checked = opt.verify_crc; res.crc_ok = opt.verify_crc;

    uint32_t ty0 = single_tile ? static_cast<uint32_t>(opt.tile_y) : 0u;
    uint32_t ty1 = single_tile ? (ty0 + 1u) : m.n_ty;
    uint32_t tx0 = single_tile ? static_cast<uint32_t>(opt.tile_x) : 0u;
    uint32_t tx1 = single_tile ? (tx0 + 1u) : m.n_tx;

    size_t out_total_bytes = 0;
    if (single_tile) {
        int y0 = static_cast<int>(ty0 * m.tile_h), th = std::min<int>(static_cast<int>(m.tile_h), m.H - y0);
        int x0 = static_cast<int>(tx0 * m.tile_w), tw = std::min<int>(static_cast<int>(m.tile_w), m.W - x0);
        if (m.spec_axis == 0)      { res.d0 = m.B; res.d1 = th; res.d2 = tw; }
        else if (m.spec_axis == 1) { res.d0 = th; res.d1 = m.B; res.d2 = tw; }
        else                       { res.d0 = th; res.d1 = tw; res.d2 = m.B; }
        out_total_bytes = static_cast<size_t>(m.B) * th * tw * itemsize;
    } else {
        res.d0 = m.d0; res.d1 = m.d1; res.d2 = m.d2;
        out_total_bytes = static_cast<size_t>(m.d0) * m.d1 * m.d2 * itemsize;
    }

    std::fstream fout;
    if (!out_path.empty()) {
        fout.open(out_path, std::ios::binary | std::ios::out | std::ios::in | std::ios::trunc);
        if (!fout) throw std::runtime_error("Cannot open output file: " + out_path);
        // Pre-allocate sparse output file cleanly
        if (out_total_bytes > 0) {
            fout.seekp(static_cast<std::streamoff>(out_total_bytes - 1));
            char zero = 0;
            fout.write(&zero, 1);
        }
    }

    // Multi-column tile-row strip write buffer (<= 64 MiB cap):
    // Converts O(B * H * n_tx) fragmented row seeks into B contiguous band writes per tile-row (BSQ)
    // or 1 contiguous write per tile-row (BIL/BIP) — e.g. 512x fewer syscalls on AVIRIS tile=128!
    unsigned __int128 max_strip_bytes128 =
        static_cast<unsigned __int128>(m.B) * m.tile_h * m.W * itemsize;
    bool use_row_strip = (!single_tile && !out_path.empty() && m.n_tx > 1 &&
                          max_strip_bytes128 > 0 && max_strip_bytes128 <= (64ULL << 20));
    std::vector<uint8_t> row_strip;
    if (use_row_strip) row_strip.resize(static_cast<size_t>(max_strip_bytes128));

    for (uint32_t ty = ty0; ty < ty1; ++ty) {
        int y0 = static_cast<int>(ty * m.tile_h), th = std::min<int>(static_cast<int>(m.tile_h), m.H - y0);
        for (uint32_t tx = tx0; tx < tx1; ++tx) {
            int x0 = static_cast<int>(tx * m.tile_w), tw = std::min<int>(static_cast<int>(m.tile_w), m.W - x0);
            const auto& e = m.index[static_cast<size_t>(ty) * m.n_tx + tx];
            size_t need_bytes = 0;
            for (int k = static_cast<int>(m.base_k); k >= k_target; --k) need_bytes += e.layer_bytes[k];
            if (e.offset < m.payload_start || e.offset > file_sz || need_bytes > file_sz - e.offset)
                throw std::runtime_error("Truncated HND4 tile payload");

            // Seek directly to tile and read ONLY layers down to stop_layer!
            std::vector<uint8_t> tbuf(need_bytes);
            fin.seekg(static_cast<std::streamoff>(e.offset));
            if (need_bytes > 0) {
                fin.read(reinterpret_cast<char*>(tbuf.data()), static_cast<std::streamsize>(need_bytes));
                if (!fin) throw std::runtime_error("Short read loading HND4 tile payload");
            }
            auto work = decode_tile_layers(tbuf.data(), tbuf.size(), m.B, th, tw,
                                           m.n_lv, m.base_k, m.n_stripes, opt.stop_layer, opt.max_level,
                                           e.layer_bytes.data(), e.layer_crc.data(),
                                           opt.verify_crc, res.crc_layers_verified, n_threads);
            tbuf.clear(); tbuf.shrink_to_fit();
            apply_spectral_inverse(work.data(), m.B, static_cast<size_t>(th) * tw, m.sm, use_i32_neon, n_threads);
            if (m.lut.enabled) {
                for (int64_t& v : work) v = apply_hist_lut_inv(v, m.lut);
            } else if (m.max_error > 0) {
                for (int64_t& v : work) v = nl_dequant(v, m.max_error, m.dt_code);
            }
            if (use_row_strip) {
                const bool clamp_val = !is_lossless;
                size_t tHW = static_cast<size_t>(th) * tw;
                if (m.spec_axis == 0) {
                    parallel_for(static_cast<size_t>(m.B), n_threads, [&](size_t b0, size_t b1) {
                        for (size_t b = b0; b < b1; ++b) {
                            size_t band_base = b * static_cast<size_t>(th) * m.W;
                            const int64_t* src_b = work.data() + b * tHW;
                            for (int y = 0; y < th; ++y) {
                                uint8_t* dp = row_strip.data() + (band_base + static_cast<size_t>(y) * m.W + x0) * itemsize;
                                const int64_t* sp = src_b + static_cast<size_t>(y) * tw;
                                for (int x = 0; x < tw; ++x)
                                    write_sample_le(dp + x * itemsize, sp[x], m.dt_code, clamp_val);
                            }
                        }
                    });
                } else if (m.spec_axis == 1) {
                    parallel_for(static_cast<size_t>(th), n_threads, [&](size_t ys0, size_t ys1) {
                        for (size_t y = ys0; y < ys1; ++y) {
                            for (int b = 0; b < m.B; ++b) {
                                uint8_t* dp = row_strip.data() + ((y * m.B + b) * m.W + x0) * itemsize;
                                const int64_t* sp = work.data() + static_cast<size_t>(b) * tHW + y * tw;
                                for (int x = 0; x < tw; ++x)
                                    write_sample_le(dp + x * itemsize, sp[x], m.dt_code, clamp_val);
                            }
                        }
                    });
                } else {
                    parallel_for(static_cast<size_t>(th), n_threads, [&](size_t ys0, size_t ys1) {
                        for (size_t y = ys0; y < ys1; ++y) {
                            for (int x = 0; x < tw; ++x) {
                                uint8_t* dp = row_strip.data() + ((y * m.W + (x0 + x)) * m.B) * itemsize;
                                for (int b = 0; b < m.B; ++b)
                                    write_sample_le(dp + static_cast<size_t>(b) * itemsize,
                                                    work[static_cast<size_t>(b) * tHW + y * tw + x],
                                                    m.dt_code, clamp_val);
                            }
                        }
                    });
                }
            } else if (!out_path.empty()) {
                if (single_tile) {
                    scatter_tile_to_file(work.data(), fout, res.d0, res.d1, res.d2,
                                         m.dt_code, m.spec_axis, m.B, 0, th, 0, tw, !is_lossless);
                } else {
                    scatter_tile_to_file(work.data(), fout, m.d0, m.d1, m.d2,
                                         m.dt_code, m.spec_axis, m.B, y0, th, x0, tw, !is_lossless);
                }
            } else if (opt.verify_sha && is_lossless && m.n_ty == 1 && m.n_tx == 1) {
                // Single-tile lossless decode without out_path (e.g. --bench / verify-only): verify SHA-256 in memory!
                std::vector<uint8_t> tmp(out_total_bytes);
                scatter_tile_to_mem(work.data(), tmp.data(), m.d0, m.d1, m.d2,
                                    m.dt_code, m.spec_axis, m.B, 0, th, 0, tw, false);
                res.sha_checked = true;
                res.sha_ok = (sha256_bytes(tmp.data(), tmp.size()) == m.sha);
                if (!res.sha_ok) throw std::runtime_error("SHA-256 verification FAILED on single-tile HND4 decode!");
            }
        }
        if (use_row_strip) {
            if (m.spec_axis == 0) {
                size_t band_strip_bytes = static_cast<size_t>(th) * m.W * itemsize;
                for (int b = 0; b < m.B; ++b) {
                    size_t off = ((static_cast<size_t>(b) * m.d1 + y0) * m.d2) * itemsize;
                    fout.seekp(static_cast<std::streamoff>(off));
                    fout.write(reinterpret_cast<const char*>(row_strip.data() + static_cast<size_t>(b) * band_strip_bytes),
                               static_cast<std::streamsize>(band_strip_bytes));
                }
            } else {
                size_t strip_bytes = static_cast<size_t>(th) * m.B * m.W * itemsize;
                size_t off = (static_cast<size_t>(y0) * m.d1 * m.d2) * itemsize;
                fout.seekp(static_cast<std::streamoff>(off));
                fout.write(reinterpret_cast<const char*>(row_strip.data()), static_cast<std::streamsize>(strip_bytes));
            }
            if (!fout) throw std::runtime_error("I/O error writing decoded HND4 tile-row strip");
        }
    }
    if (fout.is_open()) { fout.flush(); fout.close(); }

    if (opt.verify_sha && is_lossless && !out_path.empty()) {
        res.sha_checked = true;
        res.sha_ok = (sha256_file_stream(out_path) == m.sha);
        if (!res.sha_ok) throw std::runtime_error("SHA-256 verification FAILED on streaming HND4 decode!");
    }
    return res;
}

} // namespace hnd3

// =============================================================================
// extern "C" ABI Wrappers
// =============================================================================
static void write_err(char* buf, size_t len, const char* msg) {
    if (!buf || len == 0) return;
    std::snprintf(buf, len, "%s", msg ? msg : "unknown error");
}

extern "C" {

int hnd3_encode_buffer(const uint8_t* raw, size_t raw_len, const hnd3_c_encode_opts* opts,
                       uint8_t** out_buf, size_t* out_len, char* err_buf, size_t err_buf_len) {
    if (!raw || !opts || !out_buf || !out_len) { write_err(err_buf, err_buf_len, "Null argument"); return 1; }
    try {
        hnd3::EncodeOptions o;
        o.d0 = opts->d0; o.d1 = opts->d1; o.d2 = opts->d2;
        o.dtype = static_cast<hnd3::DType>(opts->dtype);
        o.levels = opts->levels; o.spec_axis = opts->spec_axis;
        o.base_k = opts->base_k; o.stripes = opts->stripes ? opts->stripes : 16;
        o.blind = (opts->blind != 0); o.per_layer_crc = (opts->per_layer_crc != 0);
        o.verify = (opts->verify != 0);
        o.tile_size = opts->tile_size; o.max_error = opts->max_error; o.threads = opts->threads;
        auto v = hnd3::encode(raw, raw_len, o);
        uint8_t* p = static_cast<uint8_t*>(std::malloc(std::max<size_t>(1, v.size())));
        if (!p) { write_err(err_buf, err_buf_len, "Out of memory"); return 2; }
        if (!v.empty()) std::memcpy(p, v.data(), v.size());
        *out_buf = p; *out_len = v.size();
        return 0;
    } catch (const std::exception& e) {
        write_err(err_buf, err_buf_len, e.what());
        return 1;
    }
}

int hnd3_decode_buffer(const uint8_t* stream, size_t stream_len, const hnd3_c_decode_opts* opts,
                       uint8_t** out_buf, size_t* out_len,
                       uint32_t* out_d0, uint32_t* out_d1, uint32_t* out_d2, uint8_t* out_dtype,
                       char* err_buf, size_t err_buf_len) {
    if (!stream || !out_buf || !out_len) { write_err(err_buf, err_buf_len, "Null argument"); return 1; }
    try {
        hnd3::DecodeOptions o;
        if (opts) {
            o.stop_layer = opts->stop_layer; o.max_level = opts->max_level;
            o.tile_y = opts->tile_y; o.tile_x = opts->tile_x;
            o.verify_sha = (opts->verify_sha != 0); o.verify_crc = (opts->verify_crc != 0);
            o.threads = opts->threads;
            if (opts->max_samples > 0) o.max_samples = opts->max_samples;
        }
        auto d = hnd3::decode(stream, stream_len, o);
        uint8_t* p = static_cast<uint8_t*>(std::malloc(std::max<size_t>(1, d.raw_bytes.size())));
        if (!p) { write_err(err_buf, err_buf_len, "Out of memory"); return 2; }
        if (!d.raw_bytes.empty()) std::memcpy(p, d.raw_bytes.data(), d.raw_bytes.size());
        *out_buf = p; *out_len = d.raw_bytes.size();
        if (out_d0) *out_d0 = d.d0;
        if (out_d1) *out_d1 = d.d1;
        if (out_d2) *out_d2 = d.d2;
        if (out_dtype) *out_dtype = d.dt_code;
        return 0;
    } catch (const std::exception& e) {
        write_err(err_buf, err_buf_len, e.what());
        return 1;
    }
}

void hnd3_free(uint8_t* ptr) { std::free(ptr); }

} // extern "C"
