// =============================================================================
// cpp/hnd3_cli.cpp — Thin CLI Driver for libhnd3 (HND3 v3 & HND4 v4)
// =============================================================================
#include "hnd3.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static int parse_int_strict(const char* s, const char* name) {
    if (!s || !*s) throw std::runtime_error(std::string("Missing integer value for ") + name);
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v < -2147483647L || v > 2147483647L)
        throw std::runtime_error(std::string("Invalid integer for ") + name + ": " + s);
    return static_cast<int>(v);
}

static uint32_t parse_u32_strict(const char* s, const char* name) {
    if (!s || !*s || s[0] == '-') throw std::runtime_error(std::string("Invalid unsigned integer for ") + name + ": " + (s ? s : ""));
    char* end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v > 0xFFFFFFFFULL)
        throw std::runtime_error(std::string("Invalid unsigned integer for ") + name + ": " + s);
    return static_cast<uint32_t>(v);
}

static void parse_shape_strict(const char* s, uint32_t& d0, uint32_t& d1, uint32_t& d2) {
    std::string str = s ? s : "";
    size_t c1 = str.find(',');
    size_t c2 = (c1 == std::string::npos) ? std::string::npos : str.find(',', c1 + 1);
    if (c1 == std::string::npos || c2 == std::string::npos || str.find(',', c2 + 1) != std::string::npos)
        throw std::runtime_error("Bad --shape (expected D0,D1,D2)");
    d0 = parse_u32_strict(str.substr(0, c1).c_str(), "--shape D0");
    d1 = parse_u32_strict(str.substr(c1 + 1, c2 - c1 - 1).c_str(), "--shape D1");
    d2 = parse_u32_strict(str.substr(c2 + 1).c_str(), "--shape D2");
}

static void parse_tile_strict(const char* s, int& ty, int& tx) {
    std::string str = s ? s : "";
    size_t c1 = str.find(',');
    if (c1 == std::string::npos || str.find(',', c1 + 1) != std::string::npos)
        throw std::runtime_error("Bad --tile (expected TY,TX)");
    ty = parse_int_strict(str.substr(0, c1).c_str(), "--tile TY");
    tx = parse_int_strict(str.substr(c1 + 1).c_str(), "--tile TX");
    if (ty < 0 || tx < 0) throw std::runtime_error("--tile coordinates must be >= 0");
}

#ifdef HND3_ENABLE_FUZZ
static std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open input file: " + path);
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    if (sz > 0) f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    return buf;
}
#endif

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  hnd3 enc <in.raw> <out.hnd3> (--shape D0,D1,D2 | --blind) --dtype u16|i16|i32\n"
                      << "           [--base-k 0..15] [--mode auto|full|clustered|none] [--levels 0..8]\n"
                      << "           [--stripes 1..64] [--spec-axis 0|1|2] [--tile-size T] [--max-error E]\n"
                      << "           [--hist-lut auto|on|off] [--crc] [--threads N] [--no-verify]\n"
                      << "  hnd3 dec <in.hnd3> [out.raw] [--stop-layer K] [--max-level L] [--tile TY,TX]\n"
                      << "           [--threads N] [--bench]\n"
                      << "  hnd3 info <in.hnd3>\n";
            return 1;
        }
        std::string cmd = argv[1];
        int def_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

        if (cmd == "enc") {
            if (argc < 4) throw std::runtime_error("enc requires <in.raw> <out.hnd3>");
            std::string in_path = argv[2], out_path = argv[3];
            hnd3::EncodeOptions opt;
            opt.threads = def_threads;
            bool has_shape = false, has_dtype = false;
            int base_k_i = 6, spec_axis_i = 0, n_lv_i = 3, n_stripes_i = 16;
            for (int i = 4; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--shape" && i + 1 < argc) {
                    parse_shape_strict(argv[++i], opt.d0, opt.d1, opt.d2);
                    has_shape = true;
                } else if (a == "--blind") {
                    opt.blind = true;
                } else if (a == "--dtype" && i + 1 < argc) {
                    std::string dt = argv[++i];
                    if (dt == "i16") opt.dtype = hnd3::DType::I16;
                    else if (dt == "u16") opt.dtype = hnd3::DType::U16;
                    else if (dt == "i32") opt.dtype = hnd3::DType::I32;
                    else throw std::runtime_error("Bad --dtype (expected u16|i16|i32)");
                    has_dtype = true;
                } else if (a == "--base-k" && i + 1 < argc) {
                    base_k_i = parse_int_strict(argv[++i], "--base-k");
                } else if (a == "--mode" && i + 1 < argc) {
                    opt.mode = argv[++i];
                } else if (a == "--stripes" && i + 1 < argc) {
                    n_stripes_i = parse_int_strict(argv[++i], "--stripes");
                } else if (a == "--spec-axis" && i + 1 < argc) {
                    spec_axis_i = parse_int_strict(argv[++i], "--spec-axis");
                } else if (a == "--levels" && i + 1 < argc) {
                    n_lv_i = parse_int_strict(argv[++i], "--levels");
                } else if (a == "--tile-size" && i + 1 < argc) {
                    opt.tile_size = parse_u32_strict(argv[++i], "--tile-size");
                } else if (a == "--max-error" && i + 1 < argc) {
                    opt.max_error = parse_u32_strict(argv[++i], "--max-error");
                } else if (a == "--hist-lut" && i + 1 < argc) {
                    std::string hl = argv[++i];
                    if (hl == "auto") opt.hist_lut = -1;
                    else if (hl == "on") opt.hist_lut = 1;
                    else if (hl == "off") opt.hist_lut = 0;
                    else throw std::runtime_error("Bad --hist-lut (expected auto|on|off)");
                } else if (a == "--crc") {
                    opt.per_layer_crc = true;
                } else if (a == "--threads" && i + 1 < argc) {
                    int nt = parse_int_strict(argv[++i], "--threads");
                    if (nt < 1) throw std::runtime_error("--threads must be >= 1");
                    opt.threads = nt;
                } else if (a == "--verify") {
                    opt.verify = true;
                } else if (a == "--no-verify") {
                    opt.verify = false;
                } else {
                    throw std::runtime_error("Unknown or incomplete option: " + a);
                }
            }
            if (!has_dtype) throw std::runtime_error("Missing required --dtype u16|i16|i32");
            if (!opt.blind && !has_shape) throw std::runtime_error("Must pass either --shape D0,D1,D2 or --blind");
            if (opt.blind && has_shape) throw std::runtime_error("Cannot pass both --shape and --blind");
            if (base_k_i < 0 || base_k_i > 15) throw std::runtime_error("--base-k must be in 0..15");
            if (spec_axis_i < 0 || spec_axis_i > 2) throw std::runtime_error("--spec-axis must be in 0..2");
            if (n_lv_i < 0 || n_lv_i > 8) throw std::runtime_error("--levels must be in 0..8");
            if (n_stripes_i < 1 || n_stripes_i > 64) throw std::runtime_error("--stripes must be in 1..64");
            opt.base_k = static_cast<uint8_t>(base_k_i);
            opt.spec_axis = static_cast<uint8_t>(spec_axis_i);
            opt.levels = static_cast<uint8_t>(n_lv_i);
            opt.stripes = static_cast<uint8_t>(n_stripes_i);

            auto st = hnd3::encode_file(in_path, out_path, opt);
            double itemsize = (opt.dtype == hnd3::DType::I32) ? 4.0 : 2.0;
            double n_samp = double(st.raw_bytes) / itemsize;
            if (opt.blind) {
                std::printf("BLIND-ENC strides=[");
                for (size_t i = 0; i < st.blind_strides.size(); ++i)
                    std::printf("%s%zu", i ? "," : "", st.blind_strides[i]);
                std::printf("] shape=(%u,%u,%u) spec_ax=%d base_k=%u lut=%u: %zu -> %zu B (%.3f b/s, %.2fx) %.1f MB/s (%.3fs) VERIFY=%s\n",
                            st.d0, st.d1, st.d2, st.spec_axis, opt.base_k,
                            st.hist_lut_used ? st.hist_lut_unique : 0u, st.raw_bytes, st.compressed_bytes,
                            st.compressed_bytes * 8.0 / n_samp, double(st.raw_bytes) / st.compressed_bytes,
                            st.raw_bytes / 1e6 / std::max(st.encode_seconds, 1e-6), st.encode_seconds,
                            st.verified ? "PASS" : "SKIP");
            } else if (st.version == 4) {
                std::printf("ENC4 (%u,%u,%u) tiles=%ux%u(%ux%u) base_k=%u max_err=%u lut=%u threads=%d: %zu -> %zu B (%.3f b/s, %.2fx) %.1f MB/s (%.3fs) VERIFY=%s(Linf=%lld)\n",
                            st.d0, st.d1, st.d2, st.n_tiles_y, st.n_tiles_x, st.tile_h, st.tile_w,
                            opt.base_k, st.max_error, st.hist_lut_used ? st.hist_lut_unique : 0u,
                            opt.threads, st.raw_bytes, st.compressed_bytes,
                            st.compressed_bytes * 8.0 / n_samp, double(st.raw_bytes) / st.compressed_bytes,
                            st.raw_bytes / 1e6 / std::max(st.encode_seconds, 1e-6), st.encode_seconds,
                            st.verified ? "PASS" : "SKIP", static_cast<long long>(st.verified_max_abs_err));
            } else {
                std::printf("ENC (%u,%u,%u) base_k=%u stripes=%u mode=%s threads=%d: %zu -> %zu B (%.3f b/s, %.2fx) %.1f MB/s (%.3fs) VERIFY=%s\n",
                            st.d0, st.d1, st.d2, opt.base_k, opt.stripes, opt.mode.c_str(), opt.threads,
                            st.raw_bytes, st.compressed_bytes,
                            st.compressed_bytes * 8.0 / n_samp, double(st.raw_bytes) / st.compressed_bytes,
                            st.raw_bytes / 1e6 / std::max(st.encode_seconds, 1e-6), st.encode_seconds,
                            st.verified ? "PASS" : "SKIP");
            }
            return 0;
        } else if (cmd == "dec") {
            std::string in_path = argv[2];
            std::string out_path;
            hnd3::DecodeOptions dopt;
            dopt.threads = def_threads;
            bool bench = false;
            for (int i = 3; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--stop-layer" && i + 1 < argc) {
                    dopt.stop_layer = parse_int_strict(argv[++i], "--stop-layer");
                    if (dopt.stop_layer < 0 || dopt.stop_layer > 15)
                        throw std::runtime_error("--stop-layer must be in 0..15");
                } else if (a == "--max-level" && i + 1 < argc) {
                    dopt.max_level = parse_int_strict(argv[++i], "--max-level");
                } else if (a == "--tile" && i + 1 < argc) {
                    parse_tile_strict(argv[++i], dopt.tile_y, dopt.tile_x);
                } else if (a == "--threads" && i + 1 < argc) {
                    int nt = parse_int_strict(argv[++i], "--threads");
                    if (nt < 1) throw std::runtime_error("--threads must be >= 1");
                    dopt.threads = nt;
                } else if (a == "--bench") bench = true;
                else if (out_path.empty() && a[0] != '-') out_path = a;
                else throw std::runtime_error("Unknown or incomplete option: " + a);
            }
            std::ifstream fi_sz(in_path, std::ios::binary | std::ios::ate);
            if (!fi_sz) throw std::runtime_error("Cannot open input file: " + in_path);
            size_t comp_sz = static_cast<size_t>(fi_sz.tellg());
            fi_sz.close();

            int reps = bench ? 3 : 1;
            double best_sec = 1e30;
            hnd3::DecodedCube dec;
            for (int r = 0; r < reps; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                dec = hnd3::decode_file(in_path, (r == reps - 1) ? out_path : "", dopt);
                double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (sec < best_sec) best_sec = sec;
            }
            size_t itemsize = (dec.dt_code == 3) ? 4u : 2u;
            size_t out_bytes = static_cast<size_t>(dec.d0) * dec.d1 * dec.d2 * itemsize;
            const char* sha_str = "N/A (preview)";
            if (dec.sha_checked) sha_str = dec.sha_ok ? "PASS" : "FAIL";
            else if (dopt.stop_layer > 0 || dopt.max_level >= 0) sha_str = "N/A (preview)";
            else if (dec.max_error > 0) sha_str = "N/A (near-lossless)";
            else if (dopt.tile_y >= 0 || dopt.tile_x >= 0) sha_str = "N/A (single-tile)";
            else if (out_path.empty()) sha_str = "N/A (no-out; CRC32C)";
            if (dec.version == 4) {
                std::printf("DEC4 (%u,%u,%u) stop_layer=%d threads=%d: %zu -> %zu B  %.1f MB/s (%.3fs)  CRC32C=%s(%zu layers)  SHA256=%s\n",
                            dec.d0, dec.d1, dec.d2, dopt.stop_layer, dopt.threads, comp_sz, out_bytes,
                            out_bytes / 1e6 / std::max(best_sec, 1e-6), best_sec,
                            dec.crc_checked ? (dec.crc_ok ? "PASS" : "FAIL") : "SKIP",
                            dec.crc_layers_verified, sha_str);
            } else {
                std::printf("DEC (%u,%u,%u) stop_layer=%d threads=%d: %zu -> %zu B  %.1f MB/s (%.3fs)  SHA256=%s\n",
                            dec.d0, dec.d1, dec.d2, dopt.stop_layer, dopt.threads, comp_sz, out_bytes,
                            out_bytes / 1e6 / std::max(best_sec, 1e-6), best_sec, sha_str);
            }
            return 0;
        } else if (cmd == "info") {
            auto fi = hnd3::inspect_file(argv[2]);
            if (fi.version == 4) {
                const char* hcrc_s = fi.header_crc_checked ? (fi.header_crc_ok ? "PASS" : "FAIL") : "SKIP";
                std::printf("HND4 shape=(%u,%u,%u) dtype=%u spec_ax=%u base_k=%u stripes=%u tiles=%ux%u(%ux%u) max_err=%u flags=0x%02x HeaderCRC32C=%s size=%zu B\n",
                            fi.d0, fi.d1, fi.d2, fi.dt_code, fi.spec_axis, fi.base_k, fi.n_stripes,
                            fi.n_tiles_y, fi.n_tiles_x, fi.tile_h, fi.tile_w, fi.max_error, fi.flags, hcrc_s, fi.file_bytes);
                if (fi.header_crc_checked && !fi.header_crc_ok) return 1;
            } else {
                std::printf("HND%u shape=(%u,%u,%u) dtype=%u spec_ax=%u base_k=%u stripes=%u flags=0x%02x size=%zu B\n",
                            fi.version, fi.d0, fi.d1, fi.d2, fi.dt_code, fi.spec_axis, fi.base_k, fi.n_stripes,
                            fi.flags, fi.file_bytes);
            }
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
                    size_t p = 4 + (nxt() % std::min<size_t>(mut.size() - 4, 80));
                    mut[p] = static_cast<uint8_t>(nxt() & 0xFF);
                } else if (mode == 3) {
                    size_t sz = nxt() % 1024;
                    mut.resize(sz);
                    for (size_t k = 0; k < sz; ++k) mut[k] = static_cast<uint8_t>(nxt() & 0xFF);
                    if (sz >= 5 && (nxt() & 1)) {
                        std::memcpy(mut.data(), (nxt() & 1) ? "HND4" : "HND3", 4);
                        mut[4] = (mut[3] == '4') ? 4 : 3;
                    }
                } else if (!mut.empty()) {
                    size_t p = nxt() % mut.size();
                    mut[p] = static_cast<uint8_t>(nxt() & 0xFF);
                }
                try {
                    hnd3::DecodeOptions dopt;
                    dopt.threads = 1; dopt.max_samples = 8000000ULL;
                    auto d = hnd3::decode(mut.data(), mut.size(), dopt);
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
