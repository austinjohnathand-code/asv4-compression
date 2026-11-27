// ASV3 Phase 2 Systems Implementation (C++20 + ARM64 NEON + native libzstd 1.5.5)
// Surviving stages only (Stage 3 permanently cut per Kill Gate 1B):
//   - Coupled Stage 1 & 2: NEON exact-match autocorrelation M(L), per-plane H0
//     pre-filter, Two-Tier zstd -1 scorer, 1D Viterbi DP segmentation, Gear
//     rolling-hash shift-invariant LDM duplication guard, stationary 2x256KB
//     zstd -3 verification gate, and Region-Anchored grid alignment (phase=0).
//   - Stage 4: Interleaved Flush-Group streaming container ("ASV3") with lazy
//     persistent libzstd --long=27 contexts, hardware-accelerated CommonCrypto
//     SHA-256 verification, zero backward seeks, and strict stdin/stdout pipe safety.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif

#include <zstd.h>

namespace asv3 {

// Portable SHA-256 wrapper: uses hardware-accelerated CommonCrypto on macOS,
// and a self-contained FIPS 180-4 implementation on Linux/other platforms.
class Sha256 {
public:
    Sha256() { reset(); }

#if defined(__APPLE__) && !defined(ASV3_FORCE_PORTABLE_SHA256)
    void reset() { CC_SHA256_Init(&ctx_); }
    void update(const uint8_t* data, size_t len) {
        while (len > 0) {
            CC_LONG chunk = static_cast<CC_LONG>(std::min<size_t>(len, 1ULL << 30));
            CC_SHA256_Update(&ctx_, data, chunk);
            data += chunk;
            len -= chunk;
        }
    }
    void final(uint8_t out[32]) { CC_SHA256_Final(out, &ctx_); }
private:
    CC_SHA256_CTX ctx_{};
#else
    void reset() {
        h_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
        bit_len_ = 0;
        buf_len_ = 0;
    }
    void update(const uint8_t* data, size_t len) {
        bit_len_ += static_cast<uint64_t>(len) * 8ULL;
        size_t i = 0;
        if (buf_len_ > 0) {
            size_t take = std::min(len, 64 - buf_len_);
            std::memcpy(buf_ + buf_len_, data, take);
            buf_len_ += take;
            i += take;
            if (buf_len_ == 64) {
                compress(buf_);
                buf_len_ = 0;
            }
        }
        while (i + 64 <= len) {
            compress(data + i);
            i += 64;
        }
        if (i < len) {
            buf_len_ = len - i;
            std::memcpy(buf_, data + i, buf_len_);
        }
    }
    void final(uint8_t out[32]) {
        buf_[buf_len_++] = 0x80;
        if (buf_len_ > 56) {
            std::memset(buf_ + buf_len_, 0, 64 - buf_len_);
            compress(buf_);
            buf_len_ = 0;
        }
        std::memset(buf_ + buf_len_, 0, 56 - buf_len_);
        for (int i = 0; i < 8; ++i) {
            buf_[56 + i] = static_cast<uint8_t>((bit_len_ >> (56 - i * 8)) & 0xFFU);
        }
        compress(buf_);
        for (int i = 0; i < 8; ++i) {
            out[i * 4 + 0] = static_cast<uint8_t>((h_[i] >> 24) & 0xFFU);
            out[i * 4 + 1] = static_cast<uint8_t>((h_[i] >> 16) & 0xFFU);
            out[i * 4 + 2] = static_cast<uint8_t>((h_[i] >> 8) & 0xFFU);
            out[i * 4 + 3] = static_cast<uint8_t>(h_[i] & 0xFFU);
        }
    }
private:
    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    void compress(const uint8_t block[64]) {
        static constexpr uint32_t K[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(block[i * 4 + 0]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
                   (uint32_t(block[i * 4 + 2]) << 8)  |  uint32_t(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
    }
    std::array<uint32_t, 8> h_{};
    uint64_t bit_len_ = 0;
    uint8_t buf_[64]{};
    size_t buf_len_ = 0;
#endif
};

constexpr char MAGIC[4] = {'A', 'S', 'V', '3'};
constexpr uint8_t VERSION = 3;
constexpr uint8_t DEFAULT_WLOG = 27;  // 128 MB window
constexpr uint8_t TAG_DATA = 0x01;
constexpr uint8_t TAG_EOF = 0xFF;

constexpr uint8_t T0_IDENTITY = 0;
constexpr uint8_t T1_TRANSPOSE = 1;
constexpr uint8_t T2_ELEM_DELTA = 2;
constexpr uint8_t T2_XOR_TRANSPOSE = 3;
constexpr uint8_t T2_BITSHUFFLE = 4;
constexpr uint8_t T3_2D_PAETH = 5;

constexpr uint8_t DELTA_NONE = 0;
constexpr size_t MB = 1024ULL * 1024ULL;

struct TransformSpec {
    uint8_t tid = T0_IDENTITY;
    uint16_t w = 1;
    uint16_t h = 0;

    bool operator==(const TransformSpec& o) const {
        return tid == o.tid && w == o.w && h == o.h;
    }
    bool operator!=(const TransformSpec& o) const { return !(*this == o); }
    bool operator<(const TransformSpec& o) const {
        if (tid != o.tid) return tid < o.tid;
        if (w != o.w) return w < o.w;
        return h < o.h;
    }

    size_t unit_stride() const {
        if (tid == T0_IDENTITY) return 1;
        if (tid == T2_BITSHUFFLE) return 8ULL * w;
        if (tid == T3_2D_PAETH && h > 0) return static_cast<size_t>(w) * h;
        return std::max<size_t>(1, w);
    }

    std::string name() const {
        switch (tid) {
            case T0_IDENTITY: return "T0";
            case T1_TRANSPOSE: return "T1";
            case T2_ELEM_DELTA: return "T2_ELEM";
            case T2_XOR_TRANSPOSE: return "T2_XOR";
            case T2_BITSHUFFLE: return "T2_BIT";
            case T3_2D_PAETH: return "T3_2D";
            default: return "T?";
        }
    }
};

struct __attribute__((packed)) FileHeader {
    char magic[4];
    uint8_t version;
    uint8_t wlog;
    uint16_t flags;
};
static_assert(sizeof(FileHeader) == 8);

struct __attribute__((packed)) GroupHeader {
    uint8_t tag;
    uint32_t group_uncomp_len;
    uint32_t len_meta;
    uint32_t len_raw;
    uint32_t len_xform;
    uint32_t len_resid;
};
static_assert(sizeof(GroupHeader) == 21);

struct __attribute__((packed)) TrailerHeader {
    uint8_t tag;
    uint32_t final_lens[4];
};
static_assert(sizeof(TrailerHeader) == 17);

struct __attribute__((packed)) TrailerTail {
    uint64_t total_orig_size;
    uint8_t sha256[32];
};
static_assert(sizeof(TrailerTail) == 40);

struct __attribute__((packed)) MetaRecord {
    uint32_t orig_len;
    uint8_t transform_id;
    uint8_t delta_op;
    uint16_t w;
    uint16_t h;
    uint16_t phase;
    uint32_t tail_len;
    uint64_t parent_rel_offset;
};
static_assert(sizeof(MetaRecord) == 24);

// ---------------------------------------------------------------------------
// ARM64 NEON SIMD + Optimized Bijective Transform Kernels
// ---------------------------------------------------------------------------

void t1_forward(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w) {
    const size_t rows = n / w;
    const size_t body = rows * w;
    if (rows == 0) {
        std::memcpy(dst, src, n);
        return;
    }

#if defined(__aarch64__)
    if (w == 4) {
        uint8_t* p0 = dst;
        uint8_t* p1 = dst + rows;
        uint8_t* p2 = dst + 2 * rows;
        uint8_t* p3 = dst + 3 * rows;
        size_t r = 0;
        for (; r + 16 <= rows; r += 16) {
            uint8x16x4_t v = vld4q_u8(src + r * 4);
            vst1q_u8(p0 + r, v.val[0]);
            vst1q_u8(p1 + r, v.val[1]);
            vst1q_u8(p2 + r, v.val[2]);
            vst1q_u8(p3 + r, v.val[3]);
        }
        for (; r < rows; ++r) {
            p0[r] = src[r * 4 + 0];
            p1[r] = src[r * 4 + 1];
            p2[r] = src[r * 4 + 2];
            p3[r] = src[r * 4 + 3];
        }
        if (body < n) std::memcpy(dst + body, src + body, n - body);
        return;
    }
    if (w == 2) {
        uint8_t* p0 = dst;
        uint8_t* p1 = dst + rows;
        size_t r = 0;
        for (; r + 16 <= rows; r += 16) {
            uint8x16x2_t v = vld2q_u8(src + r * 2);
            vst1q_u8(p0 + r, v.val[0]);
            vst1q_u8(p1 + r, v.val[1]);
        }
        for (; r < rows; ++r) {
            p0[r] = src[r * 2 + 0];
            p1[r] = src[r * 2 + 1];
        }
        if (body < n) std::memcpy(dst + body, src + body, n - body);
        return;
    }
#endif

    // Cache-friendly tiled transpose for general width w
    constexpr size_t RTILE = 64;
    for (size_t r0 = 0; r0 < rows; r0 += RTILE) {
        const size_t r1 = std::min(rows, r0 + RTILE);
        for (size_t c = 0; c < w; ++c) {
            uint8_t* out_c = dst + c * rows + r0;
            const uint8_t* in_c = src + r0 * w + c;
            for (size_t r = r0; r < r1; ++r) {
                *out_c++ = *in_c;
                in_c += w;
            }
        }
    }
    if (body < n) std::memcpy(dst + body, src + body, n - body);
}

void t1_inverse(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w) {
    const size_t rows = n / w;
    const size_t body = rows * w;
    if (rows == 0) {
        std::memcpy(dst, src, n);
        return;
    }

#if defined(__aarch64__)
    if (w == 4) {
        const uint8_t* p0 = src;
        const uint8_t* p1 = src + rows;
        const uint8_t* p2 = src + 2 * rows;
        const uint8_t* p3 = src + 3 * rows;
        size_t r = 0;
        for (; r + 16 <= rows; r += 16) {
            uint8x16x4_t v;
            v.val[0] = vld1q_u8(p0 + r);
            v.val[1] = vld1q_u8(p1 + r);
            v.val[2] = vld1q_u8(p2 + r);
            v.val[3] = vld1q_u8(p3 + r);
            vst4q_u8(dst + r * 4, v);
        }
        for (; r < rows; ++r) {
            dst[r * 4 + 0] = p0[r];
            dst[r * 4 + 1] = p1[r];
            dst[r * 4 + 2] = p2[r];
            dst[r * 4 + 3] = p3[r];
        }
        if (body < n) std::memcpy(dst + body, src + body, n - body);
        return;
    }
    if (w == 2) {
        const uint8_t* p0 = src;
        const uint8_t* p1 = src + rows;
        size_t r = 0;
        for (; r + 16 <= rows; r += 16) {
            uint8x16x2_t v;
            v.val[0] = vld1q_u8(p0 + r);
            v.val[1] = vld1q_u8(p1 + r);
            vst2q_u8(dst + r * 2, v);
        }
        for (; r < rows; ++r) {
            dst[r * 2 + 0] = p0[r];
            dst[r * 2 + 1] = p1[r];
        }
        if (body < n) std::memcpy(dst + body, src + body, n - body);
        return;
    }
#endif

    constexpr size_t RTILE = 64;
    for (size_t r0 = 0; r0 < rows; r0 += RTILE) {
        const size_t r1 = std::min(rows, r0 + RTILE);
        for (size_t c = 0; c < w; ++c) {
            const uint8_t* in_c = src + c * rows + r0;
            uint8_t* out_c = dst + r0 * w + c;
            for (size_t r = r0; r < r1; ++r) {
                *out_c = *in_c++;
                out_c += w;
            }
        }
    }
    if (body < n) std::memcpy(dst + body, src + body, n - body);
}

void t2_xor_forward(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w) {
    const size_t rows = n / w;
    const size_t body = rows * w;
    if (rows == 0) {
        std::memcpy(dst, src, n);
        return;
    }

#if defined(__aarch64__)
    if (w == 4 && rows >= 17) {
        // NEON vectorized XOR(src[r] ^ src[r-1]) fused with vld4q_u8 transpose!
        uint8_t* p0 = dst;
        uint8_t* p1 = dst + rows;
        uint8_t* p2 = dst + 2 * rows;
        uint8_t* p3 = dst + 3 * rows;
        p0[0] = src[0];
        p1[0] = src[1];
        p2[0] = src[2];
        p3[0] = src[3];
        size_t r = 1;
        for (; r + 16 <= rows; r += 16) {
            uint8x16x4_t cur = vld4q_u8(src + r * 4);
            uint8x16x4_t prv = vld4q_u8(src + (r - 1) * 4);
            vst1q_u8(p0 + r, veorq_u8(cur.val[0], prv.val[0]));
            vst1q_u8(p1 + r, veorq_u8(cur.val[1], prv.val[1]));
            vst1q_u8(p2 + r, veorq_u8(cur.val[2], prv.val[2]));
            vst1q_u8(p3 + r, veorq_u8(cur.val[3], prv.val[3]));
        }
        for (; r < rows; ++r) {
            p0[r] = src[r * 4 + 0] ^ src[(r - 1) * 4 + 0];
            p1[r] = src[r * 4 + 1] ^ src[(r - 1) * 4 + 1];
            p2[r] = src[r * 4 + 2] ^ src[(r - 1) * 4 + 2];
            p3[r] = src[r * 4 + 3] ^ src[(r - 1) * 4 + 3];
        }
        if (body < n) std::memcpy(dst + body, src + body, n - body);
        return;
    }
#endif

    // General fast two-pass: transpose into dst (cache-friendly), then NEON XOR each plane backward!
    t1_forward(src, dst, n, w);
    for (size_t c = 0; c < w; ++c) {
        uint8_t* p = dst + c * rows;
        // Backward XOR in-place so p[r] = p[r] ^ p[r-1] without temporary buffer!
        size_t r = rows - 1;
#if defined(__aarch64__)
        while (r >= 16) {
            size_t base = r - 15;
            uint8x16_t cur = vld1q_u8(p + base);
            uint8x16_t prv = vld1q_u8(p + base - 1);
            vst1q_u8(p + base, veorq_u8(cur, prv));
            r -= 16;
        }
#endif
        while (r >= 1) {
            p[r] ^= p[r - 1];
            --r;
        }
    }
}

void t2_xor_inverse(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w,
                    std::vector<uint8_t>& scratch) {
    const size_t rows = n / w;
    if (rows == 0) {
        std::memcpy(dst, src, n);
        return;
    }
    scratch.resize(n);
    const size_t body = rows * w;
    for (size_t c = 0; c < w; ++c) {
        const uint8_t* in_p = src + c * rows;
        uint8_t* out_p = scratch.data() + c * rows;
        uint8_t acc = in_p[0];
        out_p[0] = acc;
        for (size_t r = 1; r < rows; ++r) {
            acc ^= in_p[r];
            out_p[r] = acc;
        }
    }
    if (body < n) std::memcpy(scratch.data() + body, src + body, n - body);
    t1_inverse(scratch.data(), dst, n, w);
}

void t2_elem_forward(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w,
                     std::vector<uint8_t>& scratch) {
    const size_t rows = n / w;
    const size_t body = rows * w;
    if (rows < 2 || (w != 2 && w != 4 && w != 8)) {
        t1_forward(src, dst, n, w);
        return;
    }
    scratch.resize(n);
    if (w == 2) {
        uint16_t prev = 0;
        std::memcpy(&prev, src, 2);
        std::memcpy(scratch.data(), &prev, 2);
        for (size_t r = 1; r < rows; ++r) {
            uint16_t cur;
            std::memcpy(&cur, src + r * 2, 2);
            uint16_t d = static_cast<uint16_t>(cur - prev);
            std::memcpy(scratch.data() + r * 2, &d, 2);
            prev = cur;
        }
    } else if (w == 4) {
        uint32_t prev = 0;
        std::memcpy(&prev, src, 4);
        std::memcpy(scratch.data(), &prev, 4);
        for (size_t r = 1; r < rows; ++r) {
            uint32_t cur;
            std::memcpy(&cur, src + r * 4, 4);
            uint32_t d = cur - prev;
            std::memcpy(scratch.data() + r * 4, &d, 4);
            prev = cur;
        }
    } else {
        uint64_t prev = 0;
        std::memcpy(&prev, src, 8);
        std::memcpy(scratch.data(), &prev, 8);
        for (size_t r = 1; r < rows; ++r) {
            uint64_t cur;
            std::memcpy(&cur, src + r * 8, 8);
            uint64_t d = cur - prev;
            std::memcpy(scratch.data() + r * 8, &d, 8);
            prev = cur;
        }
    }
    if (body < n) std::memcpy(scratch.data() + body, src + body, n - body);
    t1_forward(scratch.data(), dst, n, w);
}

void t2_elem_inverse(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w,
                     std::vector<uint8_t>& scratch) {
    const size_t rows = n / w;
    const size_t body = rows * w;
    if (rows < 2 || (w != 2 && w != 4 && w != 8)) {
        t1_inverse(src, dst, n, w);
        return;
    }
    scratch.resize(n);
    t1_inverse(src, scratch.data(), n, w);
    if (w == 2) {
        uint16_t acc = 0;
        for (size_t r = 0; r < rows; ++r) {
            uint16_t d;
            std::memcpy(&d, scratch.data() + r * 2, 2);
            acc = static_cast<uint16_t>(acc + d);
            std::memcpy(dst + r * 2, &acc, 2);
        }
    } else if (w == 4) {
        uint32_t acc = 0;
        for (size_t r = 0; r < rows; ++r) {
            uint32_t d;
            std::memcpy(&d, scratch.data() + r * 4, 4);
            acc += d;
            std::memcpy(dst + r * 4, &acc, 4);
        }
    } else {
        uint64_t acc = 0;
        for (size_t r = 0; r < rows; ++r) {
            uint64_t d;
            std::memcpy(&d, scratch.data() + r * 8, 8);
            acc += d;
            std::memcpy(dst + r * 8, &acc, 8);
        }
    }
    if (body < n) std::memcpy(dst + body, scratch.data() + body, n - body);
}

// Fast 8x8 bit matrix transpose via SWAR (bit-plane shuffle for w in {4, 8})
static inline uint64_t transpose_8x8_bits(uint64_t x) {
    uint64_t t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    x = x ^ t ^ (t << 28);
    return x;
}

void t2_bitshuffle_forward(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w) {
    if (w == 0) {
        std::memcpy(dst, src, n);
        return;
    }
    const size_t unit = 8 * w;
    const size_t groups = n / unit;
    const size_t body = groups * unit;
    if (groups == 0) {
        std::memcpy(dst, src, n);
        return;
    }
    // L1-resident tiled scatter (eliminates 32/64-way TLB & L1 set thrashing on 4 MB blocks)
    constexpr size_t GTILE = 256;
    if (w == 4 || w == 8) {
        alignas(64) uint8_t tile[64][GTILE];
        const size_t nplanes = 8 * w;
        for (size_t g0 = 0; g0 < groups; g0 += GTILE) {
            const size_t glen = std::min(GTILE, groups - g0);
            for (size_t gi = 0; gi < glen; ++gi) {
                const uint8_t* gb = src + (g0 + gi) * unit;
                for (size_t c = 0; c < w; ++c) {
                    uint64_t mat =
                        (static_cast<uint64_t>(gb[0 * w + c]) << 0)  |
                        (static_cast<uint64_t>(gb[1 * w + c]) << 8)  |
                        (static_cast<uint64_t>(gb[2 * w + c]) << 16) |
                        (static_cast<uint64_t>(gb[3 * w + c]) << 24) |
                        (static_cast<uint64_t>(gb[4 * w + c]) << 32) |
                        (static_cast<uint64_t>(gb[5 * w + c]) << 40) |
                        (static_cast<uint64_t>(gb[6 * w + c]) << 48) |
                        (static_cast<uint64_t>(gb[7 * w + c]) << 56);
                    uint64_t tr = transpose_8x8_bits(mat);
                    const size_t pbase = c * 8;
                    tile[pbase + 0][gi] = static_cast<uint8_t>(tr >> 0);
                    tile[pbase + 1][gi] = static_cast<uint8_t>(tr >> 8);
                    tile[pbase + 2][gi] = static_cast<uint8_t>(tr >> 16);
                    tile[pbase + 3][gi] = static_cast<uint8_t>(tr >> 24);
                    tile[pbase + 4][gi] = static_cast<uint8_t>(tr >> 32);
                    tile[pbase + 5][gi] = static_cast<uint8_t>(tr >> 40);
                    tile[pbase + 6][gi] = static_cast<uint8_t>(tr >> 48);
                    tile[pbase + 7][gi] = static_cast<uint8_t>(tr >> 56);
                }
            }
            for (size_t p = 0; p < nplanes; ++p) {
                std::memcpy(dst + p * groups + g0, tile[p], glen);
            }
        }
    } else {
        for (size_t g = 0; g < groups; ++g) {
            const uint8_t* gb = src + g * unit;
            for (size_t c = 0; c < w; ++c) {
                uint64_t mat = 0;
                for (size_t e = 0; e < 8; ++e) {
                    mat |= static_cast<uint64_t>(gb[e * w + c]) << (e * 8);
                }
                uint64_t tr = transpose_8x8_bits(mat);
                for (size_t bit = 0; bit < 8; ++bit) {
                    dst[(c * 8 + bit) * groups + g] = static_cast<uint8_t>((tr >> (bit * 8)) & 0xFF);
                }
            }
        }
    }
    if (body < n) std::memcpy(dst + body, src + body, n - body);
}

void t2_bitshuffle_inverse(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w) {
    if (w == 0) {
        std::memcpy(dst, src, n);
        return;
    }
    const size_t unit = 8 * w;
    const size_t groups = n / unit;
    const size_t body = groups * unit;
    if (groups == 0) {
        std::memcpy(dst, src, n);
        return;
    }
    // L1-resident tiled gather (eliminates 32/64-way TLB & L1 set thrashing on 4 MB blocks)
    constexpr size_t GTILE = 256;
    if (w == 4 || w == 8) {
        alignas(64) uint8_t tile[64][GTILE];
        const size_t nplanes = 8 * w;
        for (size_t g0 = 0; g0 < groups; g0 += GTILE) {
            const size_t glen = std::min(GTILE, groups - g0);
            for (size_t p = 0; p < nplanes; ++p) {
                std::memcpy(tile[p], src + p * groups + g0, glen);
            }
            for (size_t gi = 0; gi < glen; ++gi) {
                uint8_t* gb = dst + (g0 + gi) * unit;
                for (size_t c = 0; c < w; ++c) {
                    const size_t pbase = c * 8;
                    uint64_t mat =
                        (static_cast<uint64_t>(tile[pbase + 0][gi]) << 0)  |
                        (static_cast<uint64_t>(tile[pbase + 1][gi]) << 8)  |
                        (static_cast<uint64_t>(tile[pbase + 2][gi]) << 16) |
                        (static_cast<uint64_t>(tile[pbase + 3][gi]) << 24) |
                        (static_cast<uint64_t>(tile[pbase + 4][gi]) << 32) |
                        (static_cast<uint64_t>(tile[pbase + 5][gi]) << 40) |
                        (static_cast<uint64_t>(tile[pbase + 6][gi]) << 48) |
                        (static_cast<uint64_t>(tile[pbase + 7][gi]) << 56);
                    uint64_t tr = transpose_8x8_bits(mat);
                    gb[0 * w + c] = static_cast<uint8_t>(tr >> 0);
                    gb[1 * w + c] = static_cast<uint8_t>(tr >> 8);
                    gb[2 * w + c] = static_cast<uint8_t>(tr >> 16);
                    gb[3 * w + c] = static_cast<uint8_t>(tr >> 24);
                    gb[4 * w + c] = static_cast<uint8_t>(tr >> 32);
                    gb[5 * w + c] = static_cast<uint8_t>(tr >> 40);
                    gb[6 * w + c] = static_cast<uint8_t>(tr >> 48);
                    gb[7 * w + c] = static_cast<uint8_t>(tr >> 56);
                }
            }
        }
    } else {
        for (size_t g = 0; g < groups; ++g) {
            uint8_t* gb = dst + g * unit;
            for (size_t c = 0; c < w; ++c) {
                uint64_t mat = 0;
                for (size_t bit = 0; bit < 8; ++bit) {
                    mat |= static_cast<uint64_t>(src[(c * 8 + bit) * groups + g]) << (bit * 8);
                }
                uint64_t tr = transpose_8x8_bits(mat);
                for (size_t e = 0; e < 8; ++e) {
                    gb[e * w + c] = static_cast<uint8_t>((tr >> (e * 8)) & 0xFF);
                }
            }
        }
    }
    if (body < n) std::memcpy(dst + body, src + body, n - body);
}

static inline int paeth_pred(int a, int b, int c) {
    int p = a + b - c;
    int pa = std::abs(p - a);
    int pb = std::abs(p - b);
    int pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

void t3_2d_paeth_forward(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w, size_t h,
                         std::vector<uint8_t>& scratch) {
    const size_t rows = n / w;
    if (rows <= h || h <= 1) {
        t1_forward(src, dst, n, w);
        return;
    }
    scratch.resize(n);
    t1_forward(src, scratch.data(), n, w);
    const size_t body = rows * w;
    for (size_t c = 0; c < w; ++c) {
        const uint8_t* tr = scratch.data() + c * rows;
        uint8_t* out = dst + c * rows;
        out[0] = tr[0];
        for (size_t i = 1; i < h; ++i) {
            out[i] = static_cast<uint8_t>((int(tr[i]) - int(tr[i - 1])) & 0xFF);
        }
        // Division-free 2D row/col loop (eliminates i % h modulo on every byte)
        for (size_t r0 = h; r0 < rows; r0 += h) {
            const size_t rlen = std::min(h, rows - r0);
            const uint8_t* cur = tr + r0;
            const uint8_t* up = tr + r0 - h;
            uint8_t* dst_r = out + r0;
            dst_r[0] = static_cast<uint8_t>((int(cur[0]) - int(up[0])) & 0xFF);
            for (size_t col = 1; col < rlen; ++col) {
                int pred = paeth_pred(cur[col - 1], up[col], up[col - 1]);
                dst_r[col] = static_cast<uint8_t>((int(cur[col]) - pred) & 0xFF);
            }
        }
    }
    if (body < n) std::memcpy(dst + body, src + body, n - body);
}

void t3_2d_paeth_inverse(const uint8_t* __restrict src, uint8_t* __restrict dst, size_t n, size_t w, size_t h,
                         std::vector<uint8_t>& scratch) {
    const size_t rows = n / w;
    if (rows <= h || h <= 1) {
        t1_inverse(src, dst, n, w);
        return;
    }
    scratch.resize(n);
    const size_t body = rows * w;
    for (size_t c = 0; c < w; ++c) {
        const uint8_t* in = src + c * rows;
        uint8_t* tr = scratch.data() + c * rows;
        int acc = in[0];
        tr[0] = static_cast<uint8_t>(acc);
        for (size_t i = 1; i < h; ++i) {
            acc = (acc + in[i]) & 0xFF;
            tr[i] = static_cast<uint8_t>(acc);
        }
        // Division-free 2D row/col loop with register-carried left neighbor
        for (size_t r0 = h; r0 < rows; r0 += h) {
            const size_t rlen = std::min(h, rows - r0);
            const uint8_t* in_r = in + r0;
            const uint8_t* up = tr + r0 - h;
            uint8_t* cur = tr + r0;
            int up_left = up[0];
            int left = (int(in_r[0]) + up_left) & 0xFF;
            cur[0] = static_cast<uint8_t>(left);
            for (size_t col = 1; col < rlen; ++col) {
                int up_cur = up[col];
                int pred = paeth_pred(left, up_cur, up_left);
                left = (int(in_r[col]) + pred) & 0xFF;
                cur[col] = static_cast<uint8_t>(left);
                up_left = up_cur;
            }
        }
    }
    if (body < n) std::memcpy(scratch.data() + body, src + body, n - body);
    t1_inverse(scratch.data(), dst, n, w);
}

void apply_transform_forward(const uint8_t* src, uint8_t* dst, size_t n, const TransformSpec& sp,
                             std::vector<uint8_t>& scratch) {
    switch (sp.tid) {
        case T0_IDENTITY: std::memcpy(dst, src, n); break;
        case T1_TRANSPOSE: t1_forward(src, dst, n, std::max<size_t>(2, sp.w)); break;
        case T2_ELEM_DELTA: t2_elem_forward(src, dst, n, sp.w, scratch); break;
        case T2_XOR_TRANSPOSE: t2_xor_forward(src, dst, n, std::max<size_t>(2, sp.w)); break;
        case T2_BITSHUFFLE: t2_bitshuffle_forward(src, dst, n, sp.w); break;
        case T3_2D_PAETH: t3_2d_paeth_forward(src, dst, n, std::max<size_t>(1, sp.w), std::max<size_t>(2, sp.h), scratch); break;
        default: std::memcpy(dst, src, n); break;
    }
}

void apply_transform_inverse(const uint8_t* src, uint8_t* dst, size_t n, const TransformSpec& sp,
                             std::vector<uint8_t>& scratch) {
    switch (sp.tid) {
        case T0_IDENTITY: std::memcpy(dst, src, n); break;
        case T1_TRANSPOSE: t1_inverse(src, dst, n, std::max<size_t>(2, sp.w)); break;
        case T2_ELEM_DELTA: t2_elem_inverse(src, dst, n, sp.w, scratch); break;
        case T2_XOR_TRANSPOSE: t2_xor_inverse(src, dst, n, std::max<size_t>(2, sp.w), scratch); break;
        case T2_BITSHUFFLE: t2_bitshuffle_inverse(src, dst, n, sp.w); break;
        case T3_2D_PAETH: t3_2d_paeth_inverse(src, dst, n, std::max<size_t>(1, sp.w), std::max<size_t>(2, sp.h), scratch); break;
        default: std::memcpy(dst, src, n); break;
    }
}

// ---------------------------------------------------------------------------
// Fast NEON Exact-Match Autocorrelation M(L), Per-Plane H0, Gear LDM Guard
// ---------------------------------------------------------------------------

static inline size_t count_eq_bytes(const uint8_t* a, const uint8_t* b, size_t n) {
    size_t matches = 0;
    size_t i = 0;
#if defined(__aarch64__)
    // Accumulate 16 equality flags per cycle in NEON
    while (i + 16 <= n) {
        uint8x16_t acc = vdupq_n_u8(0);
        size_t block = std::min<size_t>(n - i, 16 * 255);
        size_t end = i + (block & ~size_t(15));
        for (; i < end; i += 16) {
            uint8x16_t eq = vceqq_u8(vld1q_u8(a + i), vld1q_u8(b + i));
            acc = vsubq_u8(acc, eq);  // eq is 0xFF (-1), subtracting adds +1!
        }
        matches += vaddlvq_u8(acc);
    }
#endif
    for (; i < n; ++i) {
        matches += (a[i] == b[i]);
    }
    return matches;
}

std::pair<uint16_t, uint16_t> discover_lag_peaks(const uint8_t* p16, size_t n16,
                                                 const uint8_t* p64, size_t n64) {
    std::array<float, 257> scores{};
    const size_t scan_n = std::min<size_t>(n16, 8192);
    if (scan_n <= 320) return {0, 0};

    for (size_t lag = 2; lag <= 256; ++lag) {
        size_t cmp = scan_n - lag;
        scores[lag] = static_cast<float>(count_eq_bytes(p16, p16 + lag, cmp)) / static_cast<float>(cmp);
    }

    std::vector<float> tmp(scores.begin() + 2, scores.end());
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    float med = tmp[tmp.size() / 2];
    float thresh = med + 0.025f;

    std::vector<std::pair<float, uint16_t>> peaks;
    for (uint16_t lag = 2; lag <= 256; ++lag) {
        float sc = scores[lag];
        if (sc >= thresh) {
            float l = (lag > 2) ? scores[lag - 1] : 0.0f;
            float r = (lag < 256) ? scores[lag + 1] : 0.0f;
            if (sc >= l && sc >= r) peaks.push_back({sc, lag});
        }
    }

    if (p64 && n64 >= 8192) {
        constexpr uint16_t wide_lags[] = {384, 512, 768, 1024, 2048, 4096};
        uint16_t best_wlag = 0;
        float best_wsc = 0.0f;
        for (uint16_t lag : wide_lags) {
            if (n64 <= lag + 256) continue;
            size_t cmp = std::min<size_t>(n64 - lag, 16384);
            float sc = static_cast<float>(count_eq_bytes(p64, p64 + lag, cmp)) / static_cast<float>(cmp);
            if (sc > best_wsc) { best_wsc = sc; best_wlag = lag; }
        }
        if (best_wlag > 0 && best_wsc >= thresh) peaks.push_back({best_wsc, best_wlag});
    }

    std::sort(peaks.begin(), peaks.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    uint16_t l1 = 0, l2 = 0;
    for (const auto& [sc, lag] : peaks) {
        if (l1 == 0) {
            l1 = lag;
        } else if (lag % l1 != 0 && l1 % lag != 0) {
            l2 = lag;
            break;
        }
    }
    return {l1, l2};
}

double plane_h0_sum(const uint8_t* data, size_t n, size_t w) {
    const size_t rows = n / w;
    if (rows < 8) return 8.0;
    uint32_t hist[256];
    const double inv_r = 1.0 / static_cast<double>(rows);
    constexpr double INV_LN2 = 1.4426950408889634;
    double total_h = 0.0;
    for (size_t c = 0; c < w; ++c) {
        std::memset(hist, 0, sizeof(hist));
        const uint8_t* ptr = data + c;
        for (size_t r = 0; r < rows; ++r) {
            hist[*ptr]++;
            ptr += w;
        }
        double h = 0.0;
        int nonzero = 0;
        for (int b = 0; b < 256; ++b) {
            if (hist[b] > 0) {
                nonzero++;
                double p = hist[b] * inv_r;
                h -= p * std::log(p) * INV_LN2;
            }
        }
        // Miller-Madow finite-sample bias correction prevents large widths from evicting true small strides
        if (nonzero > 1) {
            h += (static_cast<double>(nonzero - 1) * INV_LN2) / (2.0 * static_cast<double>(rows));
        }
        total_h += h;
    }
    return total_h / static_cast<double>(w);
}

uint16_t discover_row_stride_in_plane(const uint8_t* plane, size_t n) {
    if (n < 2048) return 0;
    constexpr uint16_t cands[] = {
        32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768,
        800, 1000, 1024, 1280, 1536, 1600, 1920, 2048
    };
    const size_t rlim = std::min<size_t>(n - 137, 4096);
    int64_t rand_diff = 0;
    for (size_t i = 0; i < rlim; ++i) rand_diff += std::abs(int(plane[i + 137]) - int(plane[i]));
    double rand_avg = (rlim > 64) ? (static_cast<double>(rand_diff) / rlim) : 64.0;

    uint16_t best_h = 0;
    double best_avg = rand_avg * 0.55;
    for (uint16_t h : cands) {
        if (static_cast<size_t>(h) * 3 >= n) continue;
        size_t lim = std::min<size_t>(n - h, 4096);
        int64_t s = 0;
        for (size_t i = 0; i < lim; ++i) s += std::abs(int(plane[i + h]) - int(plane[i]));
        double avg = static_cast<double>(s) / lim;
        if (avg < best_avg) {
            best_avg = avg;
            best_h = h;
        }
    }
    return best_h;
}

static inline void compute_hist256(const uint8_t* data, size_t n, std::array<float, 256>& h) {
    uint32_t c[256] = {};
    for (size_t i = 0; i < n; ++i) c[data[i]]++;
    float inv = n ? (1.0f / static_cast<float>(n)) : 0.0f;
    for (int i = 0; i < 256; ++i) h[i] = c[i] * inv;
}

static inline double js_divergence(const std::array<float, 256>& p, const std::array<float, 256>& q) {
    double d = 0.0;
    constexpr double INV_LN2 = 1.4426950408889634;
    for (int i = 0; i < 256; ++i) {
        double pi = p[i], qi = q[i];
        double m = 0.5 * (pi + qi);
        if (pi > 1e-12) d += 0.5 * pi * std::log(pi / m) * INV_LN2;
        if (qi > 1e-12) d += 0.5 * qi * std::log(qi / m) * INV_LN2;
    }
    return d;
}

// Shift-invariant Gear rolling-hash LDM duplication guard (~256 KB RAM)
struct TrailingRawIndex {
    static constexpr size_t MAX_ENTRIES = 32768;
    std::array<uint64_t, 256> gear{};
    std::vector<uint64_t> ring;
    size_t pos = 0;
    std::unordered_map<uint64_t, uint32_t> counts;

    TrailingRawIndex() {
        uint64_t s = 0x415356339E3779B9ULL;
        for (int i = 0; i < 256; ++i) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            gear[i] = s * 0x2545F4914F6CDD1DULL;
        }
        ring.reserve(MAX_ENTRIES);
        counts.reserve(MAX_ENTRIES * 2);
    }

    size_t extract_anchors(const uint8_t* data, size_t n, uint64_t mask,
                           uint64_t* out_hashes, size_t max_anchors) const {
        if (n < 64) return 0;
        uint64_t h = 0;
        for (size_t i = 0; i < 32; ++i) h = (h << 1) + gear[data[i]];
        size_t cnt = 0;
        size_t i = 32;
        const size_t lim = n - 32;
        while (i < lim) {
            h = (h << 1) + gear[data[i]];
            if ((h & mask) == 0) {
                uint8_t b0 = data[i];
                if (!(data[i + 1] == b0 && data[i + 4] == b0 && data[i + 12] == b0 && data[i + 24] == b0)) {
                    uint64_t fh = 14695981039346656037ULL;
                    for (size_t j = 0; j < 32; ++j) fh = (fh ^ data[i + j]) * 1099511628211ULL;
                    out_hashes[cnt++] = fh;
                    if (cnt >= max_anchors) break;
                    i += 1024;
                    continue;
                }
            }
            ++i;
        }
        return cnt;
    }

    double overlap_fraction(const uint8_t* data, size_t n) const {
        if (counts.empty() || n < 4096) return 0.0;
        uint64_t hs[512];
        size_t cnt = extract_anchors(data, n, 0x7FFULL, hs, 512);
        if (cnt < 4) return 0.0;
        size_t hits = 0;
        for (size_t i = 0; i < cnt; ++i) {
            if (counts.find(hs[i]) != counts.end()) hits++;
        }
        return static_cast<double>(hits) / static_cast<double>(cnt);
    }

    void add_raw(const uint8_t* data, size_t n) {
        if (n < 4096) return;
        uint64_t hs[4096];
        size_t cnt = extract_anchors(data, n, 0x7FFULL, hs, 4096);
        for (size_t i = 0; i < cnt; ++i) {
            uint64_t h = hs[i];
            if (ring.size() < MAX_ENTRIES) {
                ring.push_back(h);
                counts[h]++;
            } else {
                uint64_t old = ring[pos];
                auto it = counts.find(old);
                if (it != counts.end()) {
                    if (--it->second == 0) counts.erase(it);
                }
                ring[pos] = h;
                counts[h]++;
                pos = (pos + 1) % MAX_ENTRIES;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Two-Tier Scorer (Reusable Zero-Alloc ZSTD_CCtx)
// ---------------------------------------------------------------------------

struct ProbeResult {
    TransformSpec best_spec;
    size_t t0_bytes = 0;
    size_t best_bytes = 0;
    std::map<TransformSpec, size_t> scores;
    size_t bytes_scored = 0;
    std::array<float, 256> hist{};
};

class TwoTierScorer {
public:
    explicit TwoTierScorer(double gain_gate = 0.96, bool allow_t3_2d = true)
        : gain_gate_(gain_gate), allow_t3_2d_(allow_t3_2d) {
        cctx1_ = ZSTD_createCCtx();
        cctx3_ = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(cctx1_, ZSTD_c_compressionLevel, 1);
        ZSTD_CCtx_setParameter(cctx3_, ZSTD_c_compressionLevel, 3);
        tmp_dst_.resize(ZSTD_compressBound(262144));
        xform_buf_.resize(262144);
    }
    ~TwoTierScorer() {
        ZSTD_freeCCtx(cctx1_);
        ZSTD_freeCCtx(cctx3_);
    }

    size_t zstd1_size(const uint8_t* data, size_t n) {
        size_t bnd = ZSTD_compressBound(n);
        if (tmp_dst_.size() < bnd) tmp_dst_.resize(bnd);
        return ZSTD_compress2(cctx1_, tmp_dst_.data(), tmp_dst_.size(), data, n);
    }

    size_t zstd3_size(const uint8_t* data, size_t n) {
        size_t bnd = ZSTD_compressBound(n);
        if (tmp_dst_.size() < bnd) tmp_dst_.resize(bnd);
        return ZSTD_compress2(cctx3_, tmp_dst_.data(), tmp_dst_.size(), data, n);
    }

    size_t score_spec_l1(const uint8_t* data, size_t n, const TransformSpec& sp) {
        if (sp.tid == T0_IDENTITY) return zstd1_size(data, n);
        if (xform_buf_.size() < n) xform_buf_.resize(n);
        apply_transform_forward(data, xform_buf_.data(), n, sp, scratch_);
        return zstd1_size(xform_buf_.data(), n);
    }

    size_t score_spec_l3(const uint8_t* data, size_t n, const TransformSpec& sp) {
        if (sp.tid == T0_IDENTITY) return zstd3_size(data, n);
        if (xform_buf_.size() < n) xform_buf_.resize(n);
        apply_transform_forward(data, xform_buf_.data(), n, sp, scratch_);
        return zstd3_size(xform_buf_.data(), n);
    }

    ProbeResult score_probe(const uint8_t* p16, size_t n16, const uint8_t* p64, size_t n64) {
        ProbeResult pr;
        compute_hist256(p16, n16, pr.hist);
        TransformSpec t0{T0_IDENTITY, 1, 0};
        pr.t0_bytes = zstd1_size(p16, n16);
        pr.best_bytes = pr.t0_bytes;
        pr.best_spec = t0;
        pr.scores[t0] = pr.t0_bytes;
        pr.bytes_scored = n16;

        // Early-out for clearly incompressible high-entropy probes
        if (plane_h0_sum(p16, n16, 1) > 7.95 && pr.t0_bytes >= static_cast<size_t>(n16 * 0.995)) {
            return pr;
        }

        auto [l1, l2] = discover_lag_peaks(p16, n16, p64, n64);
        std::vector<uint16_t> cand_w = {2, 4, 8, 16, 32, 64};
        auto add_w = [&](uint16_t lx) {
            if (lx >= 2 && lx <= 4096 && static_cast<size_t>(lx) * 4 <= n16) {
                if (std::find(cand_w.begin(), cand_w.end(), lx) == cand_w.end()) cand_w.push_back(lx);
            }
        };
        add_w(l1);
        add_w(l2);

        // Pre-filter Tier 1 by per-plane order-0 H0
        std::vector<std::pair<double, uint16_t>> h0_rank;
        for (uint16_t w : cand_w) {
            if (static_cast<size_t>(w) * 4 <= n16) h0_rank.push_back({plane_h0_sum(p16, n16, w), w});
        }
        std::sort(h0_rank.begin(), h0_rank.end());

        // Always evaluate fundamental word widths {4, 8} (prevents H0 refinement bias
        // where multiples 16/32/64 always have lower empirical H0 from evicting w=4/8!),
        // plus top remaining widths by Miller-Madow H0 and lag peak L1.
        std::vector<uint16_t> eval_w;
        for (uint16_t fw : {uint16_t(4), uint16_t(8)}) {
            if (static_cast<size_t>(fw) * 4 <= n16) eval_w.push_back(fw);
        }
        for (const auto& [hval, w] : h0_rank) {
            if (std::find(eval_w.begin(), eval_w.end(), w) == eval_w.end()) {
                eval_w.push_back(w);
                if (eval_w.size() >= 4) break;
            }
        }
        if (l1 >= 2 && static_cast<size_t>(l1) * 4 <= n16 &&
            std::find(eval_w.begin(), eval_w.end(), l1) == eval_w.end() && eval_w.size() < 5) {
            eval_w.push_back(l1);
        }

        // Tier 1
        std::vector<std::pair<size_t, uint16_t>> t1_res;
        for (uint16_t w : eval_w) {
            TransformSpec sp{T1_TRANSPOSE, w, 0};
            size_t sz = score_spec_l1(p16, n16, sp);
            pr.bytes_scored += n16;
            pr.scores[sp] = sz;
            t1_res.push_back({sz, w});
        }

        std::sort(t1_res.begin(), t1_res.end());
        size_t best_t1_sz = pr.t0_bytes;
        uint16_t best_t1_w = 1;
        if (!t1_res.empty()) {
            best_t1_sz = t1_res[0].first;
            best_t1_w = t1_res[0].second;
            // Harmonic tie-breaking (prefer smaller width unless larger beats by >= 2%)
            auto by_w = t1_res;
            std::sort(by_w.begin(), by_w.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
            for (const auto& [sz_s, w_s] : by_w) {
                if (best_t1_w > w_s && (best_t1_w % w_s == 0)) {
                    if (best_t1_sz > static_cast<size_t>(sz_s * 0.98)) {
                        best_t1_sz = sz_s;
                        best_t1_w = w_s;
                        break;
                    }
                }
            }
        }
        if (best_t1_w > 1 && best_t1_sz < pr.best_bytes) {
            pr.best_bytes = best_t1_sz;
            pr.best_spec = TransformSpec{T1_TRANSPOSE, best_t1_w, 0};
        }

        // Tier 2 (conditional: ONLY if best Tier 1 beats T0 by >= 2%)
        if (best_t1_sz <= static_cast<size_t>(pr.t0_bytes * 0.98)) {
            std::vector<uint16_t> top2;
            top2.push_back(best_t1_w);
            for (const auto& [sz, w] : t1_res) {
                if (w != best_t1_w) { top2.push_back(w); break; }
            }
            for (uint16_t w : top2) {
                // T2_XOR
                {
                    TransformSpec sp{T2_XOR_TRANSPOSE, w, 0};
                    size_t sz = score_spec_l1(p16, n16, sp);
                    pr.bytes_scored += n16;
                    pr.scores[sp] = sz;
                    if (sz < pr.best_bytes) { pr.best_bytes = sz; pr.best_spec = sp; }
                }
                // T2_ELEM
                if (w == 2 || w == 4 || w == 8) {
                    TransformSpec sp{T2_ELEM_DELTA, w, 0};
                    size_t sz = score_spec_l1(p16, n16, sp);
                    pr.bytes_scored += n16;
                    pr.scores[sp] = sz;
                    if (sz < pr.best_bytes) { pr.best_bytes = sz; pr.best_spec = sp; }
                }
                // T2_BIT
                if ((w == 4 || w == 8) && n16 >= static_cast<size_t>(w) * 32) {
                    TransformSpec sp{T2_BITSHUFFLE, w, 0};
                    size_t sz = score_spec_l1(p16, n16, sp);
                    pr.bytes_scored += n16;
                    pr.scores[sp] = sz;
                    if (sz < pr.best_bytes) { pr.best_bytes = sz; pr.best_spec = sp; }
                }
                // T3_2D (archival mode only: disabled at --level 3 to preserve 700-830 MB/s NEON decode)
                if (allow_t3_2d_ && w <= 8) {
                    const uint8_t* cb = (p64 && n64 >= 32768) ? p64 : p16;
                    size_t cn = (p64 && n64 >= 32768) ? n64 : n16;
                    size_t rows_c = cn / w;
                    msb_buf_.resize(rows_c);
                    uint16_t h_peak = 0;
                    // Check MSB lanes from both ends (w-1, w-2 for LE; 0, 1 for BE)
                    int check_lanes[4] = {static_cast<int>(w) - 1, static_cast<int>(w) - 2, 0, 1};
                    for (int li = 0; li < 4; ++li) {
                        int lane = check_lanes[li];
                        if (lane < 0 || lane >= static_cast<int>(w)) continue;
                        uint8_t mn = 255, mx = 0;
                        for (size_t r = 0; r < rows_c; ++r) {
                            uint8_t v = cb[r * w + lane];
                            msb_buf_[r] = v;
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                        }
                        if (mn != mx) {
                            h_peak = discover_row_stride_in_plane(msb_buf_.data(), rows_c);
                            if (h_peak >= 32) break;
                        }
                    }
                    if (h_peak >= 32 && static_cast<size_t>(h_peak) * w * 3 <= cn) {
                        TransformSpec sp{T3_2D_PAETH, w, h_peak};
                        size_t sz_2d;
                        if (cn == n16) {
                            sz_2d = score_spec_l1(cb, cn, sp);
                            pr.bytes_scored += cn;
                        } else {
                            size_t sz_w = score_spec_l1(cb, cn, sp);
                            size_t sz_0 = zstd1_size(cb, cn);
                            pr.bytes_scored += 2 * cn;
                            sz_2d = static_cast<size_t>(pr.t0_bytes * (double(sz_w) / std::max<size_t>(1, sz_0)));
                        }
                        pr.scores[sp] = sz_2d;
                        if (sz_2d < pr.best_bytes) { pr.best_bytes = sz_2d; pr.best_spec = sp; }
                    }
                }
            }
        }

        // Apply 2% harmonic tie-breaking to final winner as well (e.g. prefer T2_BIT(4) over T2_BIT(8)
        // unless w=8 beats w=4 by >= 2%!)
        if (pr.best_spec.tid != T0_IDENTITY && pr.best_spec.w > 2) {
            for (const auto& [cand_sp, cand_sz] : pr.scores) {
                if (cand_sp.tid == pr.best_spec.tid && cand_sp.h == pr.best_spec.h &&
                    cand_sp.w < pr.best_spec.w && (pr.best_spec.w % cand_sp.w == 0)) {
                    if (pr.best_bytes > static_cast<size_t>(cand_sz * 0.98)) {
                        pr.best_spec = cand_sp;
                        pr.best_bytes = cand_sz;
                        break;
                    }
                }
            }
        }

        if (pr.best_spec.tid != T0_IDENTITY && pr.best_bytes > static_cast<size_t>(pr.t0_bytes * gain_gate_)) {
            pr.best_spec = t0;
            pr.best_bytes = pr.t0_bytes;
        }
        return pr;
    }

    // Stationary fast-confirm probe: scores only T0 and active_spec (1 or 2 zstd -1 calls = 0.25x-0.50x work!)
    ProbeResult fast_confirm_probe(const uint8_t* p16, size_t n16, const TransformSpec& active_sp,
                                   const std::array<float, 256>& hist) {
        ProbeResult pr;
        pr.hist = hist;
        TransformSpec t0{T0_IDENTITY, 1, 0};
        pr.t0_bytes = zstd1_size(p16, n16);
        pr.scores[t0] = pr.t0_bytes;
        pr.bytes_scored = n16;
        pr.best_spec = t0;
        pr.best_bytes = pr.t0_bytes;
        if (active_sp.tid != T0_IDENTITY) {
            size_t sz = score_spec_l1(p16, n16, active_sp);
            pr.bytes_scored += n16;
            pr.scores[active_sp] = sz;
            if (sz <= static_cast<size_t>(pr.t0_bytes * gain_gate_)) {
                pr.best_spec = active_sp;
                pr.best_bytes = sz;
            }
        }
        return pr;
    }

private:
    double gain_gate_;
    bool allow_t3_2d_;
    ZSTD_CCtx* cctx1_ = nullptr;
    ZSTD_CCtx* cctx3_ = nullptr;
    std::vector<uint8_t> tmp_dst_;
    std::vector<uint8_t> xform_buf_;
    std::vector<uint8_t> scratch_;
    std::vector<uint8_t> msb_buf_;
};

// ---------------------------------------------------------------------------
// Coupled Stage 1 & 2: 1D DP Segmentation
// ---------------------------------------------------------------------------

struct RegionSpan {
    size_t start;
    size_t end;
    TransformSpec spec;
};

struct Config {
    int level = 19;
    uint8_t wlog = DEFAULT_WLOG;
    size_t flush_group_bytes = 16 * MB;
    size_t block_size_bytes = 4 * MB;
    size_t probe_step_bytes = 65536;
    size_t base_probe_bytes = 16384;
    double gain_gate = 0.96;
    double p_switch_bytes = 49152.0;
    size_t min_region_bytes = 65536;
    double js_merge_threshold = 0.15;
    double fast_confirm_js_xform = 0.12;      // Stationary fast-confirm JS threshold (non-T0)
    double fast_confirm_js_raw = 0.04;        // Stationary fast-confirm JS threshold (T0)
    double ldm_dup_threshold = 0.15;
    double region_z3_gate = 0.94;
    double min_region_saved_bytes = 262144.0; // Default 256 KB floor (protects binaries)
    double min_numeric_saved_bytes = 131072.0;// 128 KB warm-numeric floor once S_XFORM is active
    bool trace = false;
};

struct FastConfirmState {
    TransformSpec last_spec{T0_IDENTITY, 1, 0};
    int consec_same = 0;
    bool has_prev_hist = false;
    std::array<float, 256> prev_hist{};
};

std::pair<std::vector<RegionSpan>, size_t> segment_buffer_dp(
    const uint8_t* data,
    size_t n,
    TwoTierScorer& scorer,
    const Config& cfg,
    const TransformSpec& prev_active,
    const TrailingRawIndex& raw_index,
    FastConfirmState& fc_state,
    bool xform_active) {

    TransformSpec t0{T0_IDENTITY, 1, 0};
    if (n < cfg.base_probe_bytes) return {{{0, n, t0}}, 0};

    std::vector<std::pair<size_t, size_t>> slots;
    for (size_t s = 0; s < n; s += cfg.probe_step_bytes) {
        size_t e = std::min(n, s + cfg.probe_step_bytes);
        if (e - s < cfg.base_probe_bytes && !slots.empty()) {
            slots.back().second = e;
        } else {
            slots.push_back({s, e});
        }
    }
    const size_t K = slots.size();
    std::vector<ProbeResult> probes(K);
    std::vector<uint8_t> was_fast(K, 0);
    size_t bytes_scored = 0;
    std::map<TransformSpec, int> win_counts;

    for (size_t i = 0; i < K; ++i) {
        auto [s, e] = slots[i];
        size_t slen = e - s;
        size_t n16 = std::min(cfg.base_probe_bytes, slen);
        std::array<float, 256> cur_hist{};
        compute_hist256(data + s, n16, cur_hist);
        double js_prev = fc_state.has_prev_hist ? js_divergence(cur_hist, fc_state.prev_hist) : 1.0;

        // Stationary Fast-Confirm Path:
        //  - Non-T0 active winner: actively re-verify (T0 vs last_spec) on intermediate slots
        //    whenever D_JS < cfg.fast_confirm_js_xform, full sweep every 8th slot.
        //  - T0 active winner: require consec_same >= 3 and tight D_JS < cfg.fast_confirm_js_raw,
        //    full sweep every 4th slot so subtle ~4% float32 regimes (e.g. GPT-2) are caught promptly.
        bool try_fast = false;
        if (fc_state.last_spec.tid != T0_IDENTITY) {
            try_fast = (fc_state.consec_same >= 2 && js_prev < cfg.fast_confirm_js_xform && (i % 8 != 0));
        } else {
            try_fast = (fc_state.consec_same >= 3 && js_prev < cfg.fast_confirm_js_raw && (i % 4 != 0));
        }
        if (try_fast) {
            probes[i] = scorer.fast_confirm_probe(data + s, n16, fc_state.last_spec, cur_hist);
            bytes_scored += probes[i].bytes_scored;
            if (probes[i].best_spec != fc_state.last_spec) {
                bool do_wide = (slen >= 32768);
                probes[i] = scorer.score_probe(data + s, n16, do_wide ? (data + s) : nullptr, do_wide ? slen : 0);
                bytes_scored += probes[i].bytes_scored;
            } else {
                was_fast[i] = (fc_state.last_spec.tid != T0_IDENTITY) ? 2 : 1;
            }
        } else {
            bool do_wide = (i % 4 == 0 && slen >= 32768);
            probes[i] = scorer.score_probe(data + s, n16, do_wide ? (data + s) : nullptr, do_wide ? slen : 0);
            bytes_scored += probes[i].bytes_scored;
        }

        if (probes[i].best_spec == fc_state.last_spec) {
            fc_state.consec_same++;
        } else {
            fc_state.last_spec = probes[i].best_spec;
            fc_state.consec_same = 1;
        }
        fc_state.prev_hist = cur_hist;
        fc_state.has_prev_hist = true;

        if (probes[i].best_spec.tid != T0_IDENTITY) win_counts[probes[i].best_spec]++;
    }

    if (win_counts.empty() && prev_active.tid == T0_IDENTITY) {
        return {{{0, n, t0}}, bytes_scored};
    }

    std::vector<std::pair<int, TransformSpec>> ranked;
    for (const auto& [sp, cnt] : win_counts) ranked.push_back({cnt, sp});
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<TransformSpec> states;
    states.push_back(t0);
    for (size_t i = 0; i < std::min<size_t>(7, ranked.size()); ++i) states.push_back(ranked[i].second);
    if (prev_active.tid != T0_IDENTITY &&
        std::find(states.begin(), states.end(), prev_active) == states.end()) {
        states.push_back(prev_active);
    }
    const size_t M = states.size();

    std::vector<double> cost(K * M, 0.0);
    for (size_t i = 0; i < K; ++i) {
        auto [s, e] = slots[i];
        size_t slen = e - s;
        size_t n16 = std::min(cfg.base_probe_bytes, slen);
        double scale = static_cast<double>(slen) / std::max<size_t>(1, n16);
        auto& pr = probes[i];
        cost[i * M + 0] = pr.t0_bytes * scale;
        for (size_t j = 1; j < M; ++j) {
            const auto& sp = states[j];
            auto it = pr.scores.find(sp);
            size_t sz;
            if (it != pr.scores.end()) {
                sz = it->second;
            } else if (was_fast[i] == 2) {
                // Non-T0 fast-confirmed slot: do not waste zstd -1 calls on inactive states
                sz = pr.t0_bytes + 512;
            } else {
                bool nearby = (j == 1) ||
                              (was_fast[i] == 0 && j <= 2) ||
                              (i > 0 && probes[i - 1].best_spec == sp) ||
                              (i + 1 < K && probes[i + 1].best_spec == sp) ||
                              (i == 0 && prev_active == sp);
                if (nearby) {
                    sz = scorer.score_spec_l1(data + s, n16, sp);
                    bytes_scored += n16;
                    pr.scores[sp] = sz;
                } else {
                    sz = pr.t0_bytes + 512;
                }
            }
            cost[i * M + j] = sz * scale;
        }
    }

    // 1D Viterbi DP
    std::vector<double> dp(K * M, 1e18);
    std::vector<int> back(K * M, 0);
    size_t prev_idx = 0;
    for (size_t j = 0; j < M; ++j) {
        if (states[j] == prev_active) { prev_idx = j; break; }
    }
    for (size_t j = 0; j < M; ++j) {
        dp[j] = cost[j] + (j == prev_idx ? 0.0 : cfg.p_switch_bytes);
    }
    for (size_t i = 1; i < K; ++i) {
        double js = js_divergence(probes[i].hist, probes[i - 1].hist);
        double eff_sw = cfg.p_switch_bytes * (js < cfg.js_merge_threshold ? 1.15 : 0.85);
        for (size_t j = 0; j < M; ++j) {
            double best_v = 1e18;
            int best_k = 0;
            for (size_t k = 0; k < M; ++k) {
                double v = dp[(i - 1) * M + k] + (k == j ? 0.0 : eff_sw);
                if (v < best_v) { best_v = v; best_k = static_cast<int>(k); }
            }
            dp[i * M + j] = best_v + cost[i * M + j];
            back[i * M + j] = best_k;
        }
    }

    std::vector<int> path(K, 0);
    path[K - 1] = static_cast<int>(
        std::min_element(dp.begin() + (K - 1) * M, dp.begin() + K * M) - (dp.begin() + (K - 1) * M));
    for (size_t i = K - 1; i > 0; --i) path[i - 1] = back[i * M + path[i]];

    std::vector<RegionSpan> raw_regs;
    size_t cur_s = slots[0].first;
    TransformSpec cur_sp = states[path[0]];
    for (size_t i = 1; i < K; ++i) {
        if (states[path[i]] != cur_sp) {
            raw_regs.push_back({cur_s, slots[i - 1].second, cur_sp});
            cur_s = slots[i].first;
            cur_sp = states[path[i]];
        }
    }
    raw_regs.push_back({cur_s, slots[K - 1].second, cur_sp});

    // Stationary 2x256KB zstd -3 Region Verification + Gear LDM Duplication Guard
    std::vector<RegionSpan> verified;
    for (auto reg : raw_regs) {
        size_t rlen = reg.end - reg.start;
        if (reg.spec.tid != T0_IDENTITY) {
            double ov = raw_index.overlap_fraction(data + reg.start, rlen);
            if (rlen < cfg.min_region_bytes) {
                if (cfg.trace) std::fprintf(stderr, "  [CPP TRACE] [%zu..%zu] %s(%u,%u) REJECT min_len\n", reg.start, reg.end, reg.spec.name().c_str(), reg.spec.w, reg.spec.h);
                reg.spec = t0;
            } else if (ov >= cfg.ldm_dup_threshold) {
                if (cfg.trace) std::fprintf(stderr, "  [CPP TRACE] [%zu..%zu] %s(%u,%u) REJECT ldm_ov=%.3f\n", reg.start, reg.end, reg.spec.name().c_str(), reg.spec.w, reg.spec.h, ov);
                reg.spec = t0;
            } else {
                size_t vlen = std::min<size_t>(rlen, 262144);
                size_t z0_a = scorer.zstd3_size(data + reg.start, vlen);
                size_t zk_a = scorer.score_spec_l3(data + reg.start, vlen, reg.spec);
                size_t z0_b = z0_a, zk_b = zk_a;
                if (rlen >= 2 * vlen) {
                    size_t mid = reg.start + (rlen - vlen) / 2;
                    z0_b = scorer.zstd3_size(data + mid, vlen);
                    zk_b = scorer.score_spec_l3(data + mid, vlen, reg.spec);
                }
                double est_saved = (double(z0_a + z0_b) - double(zk_a + zk_b)) * (double(rlen) / std::max<size_t>(1, 2 * vlen));
                // Lower 48 KB floor for numeric w in {4, 8} ONLY once S_XFORM is already active
                // (protects adv_binaries.bin where S_XFORM never activates, while allowing smaller
                // recurring numeric regions once S_XFORM context is warm!)
                bool is_warm_numeric = xform_active && (reg.spec.w == 4 || reg.spec.w == 8) && ov == 0.0 &&
                                       zk_a <= static_cast<size_t>(z0_a * 0.90) &&
                                       zk_b <= static_cast<size_t>(z0_b * 0.90);
                double req_saved = is_warm_numeric ? cfg.min_numeric_saved_bytes : cfg.min_region_saved_bytes;
                bool reject = (zk_a > static_cast<size_t>(z0_a * cfg.region_z3_gate) ||
                               zk_b > static_cast<size_t>(z0_b * cfg.region_z3_gate) ||
                               est_saved < req_saved);
                if (cfg.trace) {
                    std::fprintf(stderr, "  [CPP TRACE] [%zu..%zu] (%.2fMB) %s(%u,%u) ov=%.3f z3_a=%.1f%% z3_b=%.1f%% saved=%.0f -> %s\n",
                                 reg.start, reg.end, double(rlen)/MB, reg.spec.name().c_str(), reg.spec.w, reg.spec.h,
                                 ov, 100.0*(double(zk_a)/z0_a - 1.0), 100.0*(double(zk_b)/z0_b - 1.0), est_saved,
                                 reject ? "REJECT_Z3" : "ACCEPT");
                }
                if (reject) reg.spec = t0;
            }
        }
        if (!verified.empty() && verified.back().spec == reg.spec) {
            verified.back().end = reg.end;
        } else {
            verified.push_back(reg);
        }
    }
    return {verified, bytes_scored};
}

// ---------------------------------------------------------------------------
// Lazy Persistent libzstd Streaming Context Wrapper (Bounded RSS!)
// ---------------------------------------------------------------------------

class LazyZstdEncoder {
public:
    LazyZstdEncoder(int level, uint8_t wlog) : level_(level), wlog_(wlog) {}
    ~LazyZstdEncoder() { if (cctx_) ZSTD_freeCCtx(cctx_); }

    bool is_initialized() const { return cctx_ != nullptr; }
    void cap_wlog_before_init(uint8_t max_wlog) {
        if (!cctx_ && wlog_ > max_wlog) wlog_ = max_wlog;
    }

    void flush_block(const uint8_t* data, size_t n, std::vector<uint8_t>& out) {
        out.clear();
        if (n == 0) return;
        ensure_init();
        ZSTD_inBuffer in_b{data, n, 0};
        size_t chunk_out = ZSTD_CStreamOutSize();
        while (true) {
            size_t prev = out.size();
            out.resize(prev + chunk_out);
            ZSTD_outBuffer out_b{out.data() + prev, chunk_out, 0};
            size_t ret = ZSTD_compressStream2(cctx_, &out_b, &in_b, ZSTD_e_flush);
            if (ZSTD_isError(ret)) {
                std::fprintf(stderr, "ZSTD_compressStream2 error: %s\n", ZSTD_getErrorName(ret));
                std::exit(1);
            }
            out.resize(prev + out_b.pos);
            if (ret == 0 && in_b.pos == in_b.size) break;
        }
    }

    void finish(std::vector<uint8_t>& out) {
        out.clear();
        if (!cctx_) return;
        ZSTD_inBuffer in_b{nullptr, 0, 0};
        size_t chunk_out = ZSTD_CStreamOutSize();
        while (true) {
            size_t prev = out.size();
            out.resize(prev + chunk_out);
            ZSTD_outBuffer out_b{out.data() + prev, chunk_out, 0};
            size_t ret = ZSTD_compressStream2(cctx_, &out_b, &in_b, ZSTD_e_end);
            if (ZSTD_isError(ret)) {
                std::fprintf(stderr, "ZSTD_e_end error: %s\n", ZSTD_getErrorName(ret));
                std::exit(1);
            }
            out.resize(prev + out_b.pos);
            if (ret == 0) break;
        }
    }

private:
    void ensure_init() {
        if (cctx_) return;
        cctx_ = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(cctx_, ZSTD_c_compressionLevel, level_);
        ZSTD_CCtx_setParameter(cctx_, ZSTD_c_enableLongDistanceMatching, 1);
        ZSTD_CCtx_setParameter(cctx_, ZSTD_c_windowLog, wlog_);
    }
    int level_;
    uint8_t wlog_;
    ZSTD_CCtx* cctx_ = nullptr;
};

class LazyZstdDecoder {
public:
    explicit LazyZstdDecoder(uint8_t wlog) : wlog_(wlog) {}
    ~LazyZstdDecoder() { if (dctx_) ZSTD_freeDCtx(dctx_); }

    bool decompress_append(const uint8_t* src, size_t n, std::vector<uint8_t>& dst, size_t max_dst_bytes) {
        if (n == 0) return true;
        if (!dctx_) {
            dctx_ = ZSTD_createDCtx();
            ZSTD_DCtx_setParameter(dctx_, ZSTD_d_windowLogMax, wlog_);
        }
        ZSTD_inBuffer in_b{src, n, 0};
        const size_t step = ZSTD_DStreamOutSize();
        while (true) {
            size_t prev = dst.size();
            if (prev + step > max_dst_bytes) {
                std::fprintf(stderr, "Decompressed buffer exceeds safety cap (%zu bytes)\n", max_dst_bytes);
                return false;
            }
            // Grow linearly in 4 MB chunks past 4 MB instead of geometric 2x doubling (saves ~40-60 MB RSS)
            if (dst.capacity() < prev + step) {
                size_t want = prev + step;
                size_t grown = (want <= 4 * MB) ? std::max(want, dst.capacity() * 2)
                                                : ((want + 4 * MB - 1) / (4 * MB)) * (4 * MB);
                dst.reserve(std::min(grown, max_dst_bytes));
            }
            dst.resize(prev + step);
            ZSTD_outBuffer out_b{dst.data() + prev, step, 0};
            size_t ret = ZSTD_decompressStream(dctx_, &out_b, &in_b);
            if (ZSTD_isError(ret)) {
                std::fprintf(stderr, "ZSTD_decompressStream error: %s\n", ZSTD_getErrorName(ret));
                return false;
            }
            dst.resize(prev + out_b.pos);
            if (in_b.pos == in_b.size && out_b.pos < step) break;
        }
        return true;
    }

private:
    uint8_t wlog_;
    ZSTD_DCtx* dctx_ = nullptr;
};

// ---------------------------------------------------------------------------
// Pipe-Safe I/O Helpers
// ---------------------------------------------------------------------------

static size_t read_exact(FILE* f, uint8_t* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        size_t r = std::fread(buf + got, 1, n - got, f);
        if (r == 0) break;
        got += r;
    }
    return got;
}

static void write_all(FILE* f, const void* buf, size_t n) {
    if (n == 0) return;
    if (std::fwrite(buf, 1, n, f) != n) {
        std::fprintf(stderr, "Write error\n");
        std::exit(1);
    }
}

static std::string sha256_hex(const uint8_t digest[32]) {
    static const char HEX[] = "0123456789abcdef";
    std::string s(64, '0');
    for (int i = 0; i < 32; ++i) {
        s[2 * i] = HEX[(digest[i] >> 4) & 0xF];
        s[2 * i + 1] = HEX[digest[i] & 0xF];
    }
    return s;
}

// ---------------------------------------------------------------------------
// Single-Pass Streaming Encoder & Decoder
// ---------------------------------------------------------------------------

int run_encode(FILE* fin, FILE* fout, const Config& cfg, const std::string& stats_path) {
    auto t_start = std::chrono::steady_clock::now();

    FileHeader fhdr{{'A', 'S', 'V', '3'}, VERSION, cfg.wlog, 0};
    write_all(fout, &fhdr, sizeof(fhdr));

    LazyZstdEncoder c_meta(3, 20);
    LazyZstdEncoder c_raw(cfg.level, cfg.wlog);
    LazyZstdEncoder c_xform(cfg.level, cfg.wlog);

    // Restrict T3_2D Paeth to archival mode (--level >= 15) so --level 3 stays fast-decode
    TwoTierScorer scorer(cfg.gain_gate, cfg.level >= 15);
    TrailingRawIndex raw_index;
    FastConfirmState fc_state;

    Sha256 sha;

    std::vector<uint8_t> in_buf(cfg.flush_group_bytes);
    std::vector<uint8_t> carry;
    std::vector<uint8_t> reg_buf;
    std::vector<uint8_t> xform_scratch;
    std::vector<uint8_t> xform_tmp;

    std::vector<uint8_t> meta_raw, raw_grp, xform_grp;
    std::vector<uint8_t> cm, cr, cx;

    TransformSpec active_spec{T0_IDENTITY, 1, 0};
    uint64_t total_orig = 0;
    uint64_t total_comp = sizeof(fhdr);
    uint64_t tot_meta = 0, tot_raw = 0, tot_xform = 0, tot_framing = sizeof(fhdr);
    uint64_t total_scored = 0;
    uint32_t regions_count = 0;
    uint32_t grp_idx = 0;
    std::map<std::string, uint64_t> bytes_by_t;

    auto flush_group = [&](uint32_t grp_uncomp) {
        if (grp_uncomp == 0) return;
        // Dual-stream RSS budget: when both S_RAW and S_XFORM are active, cap uninitialized
        // contexts to (wlog - 1) = 64 MB each so total decoder window stays within W_win = 128 MB!
        // Single-stream S_RAW files (enwik9, binaries, etc.) keep full wlog = 27 (128 MB).
        bool dual_active = (c_raw.is_initialized() || !raw_grp.empty()) &&
                           (c_xform.is_initialized() || !xform_grp.empty());
        if (dual_active && cfg.wlog > 20) {
            uint8_t dual_wlog = static_cast<uint8_t>(cfg.wlog - 1);
            c_raw.cap_wlog_before_init(dual_wlog);
            c_xform.cap_wlog_before_init(dual_wlog);
        }
        c_meta.flush_block(meta_raw.data(), meta_raw.size(), cm);
        c_raw.flush_block(raw_grp.data(), raw_grp.size(), cr);
        c_xform.flush_block(xform_grp.data(), xform_grp.size(), cx);

        GroupHeader ghdr{
            TAG_DATA,
            grp_uncomp,
            static_cast<uint32_t>(cm.size()),
            static_cast<uint32_t>(cr.size()),
            static_cast<uint32_t>(cx.size()),
            0,
        };
        write_all(fout, &ghdr, sizeof(ghdr));
        write_all(fout, cm.data(), cm.size());
        write_all(fout, cr.data(), cr.size());
        write_all(fout, cx.data(), cx.size());

        tot_meta += cm.size();
        tot_raw += cr.size();
        tot_xform += cx.size();
        tot_framing += sizeof(ghdr);
        total_comp += sizeof(ghdr) + cm.size() + cr.size() + cx.size();

        meta_raw.clear();
        raw_grp.clear();
        xform_grp.clear();
    };

    auto push_meta = [&](const MetaRecord& rec) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&rec);
        meta_raw.insert(meta_raw.end(), p, p + sizeof(rec));
    };

    auto emit_region = [&](const uint8_t* data, size_t n, const TransformSpec& sp, bool is_region_end) -> uint32_t {
        if (n == 0) return 0;
        if (sp.tid == T0_IDENTITY) {
            size_t pos = 0;
            while (pos < n) {
                size_t blen = std::min(cfg.block_size_bytes, n - pos);
                push_meta(MetaRecord{static_cast<uint32_t>(blen), T0_IDENTITY, DELTA_NONE, 1, 0, 0, 0, 0});
                raw_grp.insert(raw_grp.end(), data + pos, data + pos + blen);
                pos += blen;
            }
            bytes_by_t["T0"] += n;
            return static_cast<uint32_t>(n);
        }

        const size_t s_unit = sp.unit_stride();
        const size_t body = (n / s_unit) * s_unit;
        const size_t rem = n - body;
        uint32_t emitted = 0;

        if (body > 0) {
            const size_t blk_target = std::max(s_unit, (cfg.block_size_bytes / s_unit) * s_unit);
            size_t pos = 0;
            while (pos < body) {
                size_t end = std::min(body, pos + blk_target);
                size_t blen = end - pos;
                bool is_last = (end == body);
                size_t tail = (is_last && is_region_end) ? rem : 0;

                xform_tmp.resize(blen);
                apply_transform_forward(data + pos, xform_tmp.data(), blen, sp, xform_scratch);
                xform_grp.insert(xform_grp.end(), xform_tmp.begin(), xform_tmp.end());
                if (tail > 0) raw_grp.insert(raw_grp.end(), data + body, data + n);

                push_meta(MetaRecord{
                    static_cast<uint32_t>(blen + tail),
                    sp.tid,
                    DELTA_NONE,
                    sp.w,
                    sp.h,
                    0,
                    static_cast<uint32_t>(tail),
                    0,
                });
                emitted += static_cast<uint32_t>(blen + tail);
                pos = end;
            }
            bytes_by_t[sp.name()] += body;
            if (rem > 0 && is_region_end) bytes_by_t["T0"] += rem;
        } else if (rem > 0 && is_region_end) {
            push_meta(MetaRecord{static_cast<uint32_t>(rem), T0_IDENTITY, DELTA_NONE, 1, 0, 0, 0, 0});
            raw_grp.insert(raw_grp.end(), data, data + rem);
            bytes_by_t["T0"] += rem;
            emitted += static_cast<uint32_t>(rem);
        }

        if (rem > 0 && !is_region_end) {
            carry.insert(carry.end(), data + body, data + n);
        }
        return emitted;
    };

    while (true) {
        size_t got = read_exact(fin, in_buf.data(), cfg.flush_group_bytes);
        if (got == 0) break;
        sha.update(in_buf.data(), got);
        total_orig += got;

        if (cfg.trace) std::fprintf(stderr, "[CPP GROUP %u] orig_off=%.1fMB\n", grp_idx++, double(total_orig - got)/MB);
        auto [regs, scored] = segment_buffer_dp(in_buf.data(), got, scorer, cfg, active_spec, raw_index, fc_state, tot_xform > 0);
        total_scored += scored;
        // Only index bytes actually routed into S_RAW so S_RAW LDM guard reflects true S_RAW history!
        for (const auto& r : regs) {
            if (r.spec.tid == T0_IDENTITY) {
                raw_index.add_raw(in_buf.data() + r.start, r.end - r.start);
            }
        }

        uint32_t grp_emitted = 0;
        for (size_t ri = 0; ri < regs.size(); ++ri) {
            const auto& r = regs[ri];
            bool is_first = (ri == 0);
            bool is_last = (ri + 1 == regs.size());

            if (!carry.empty() && (!is_first || r.spec != active_spec)) {
                push_meta(MetaRecord{static_cast<uint32_t>(carry.size()), T0_IDENTITY, DELTA_NONE, 1, 0, 0, 0, 0});
                raw_grp.insert(raw_grp.end(), carry.begin(), carry.end());
                bytes_by_t["T0"] += carry.size();
                grp_emitted += static_cast<uint32_t>(carry.size());
                carry.clear();
            }

            if (r.spec != active_spec || regions_count == 0) {
                regions_count++;
                active_spec = r.spec;
            }

            const uint8_t* ptr = in_buf.data() + r.start;
            size_t rlen = r.end - r.start;
            if (is_first && !carry.empty()) {
                reg_buf.clear();
                reg_buf.reserve(carry.size() + rlen);
                reg_buf.insert(reg_buf.end(), carry.begin(), carry.end());
                reg_buf.insert(reg_buf.end(), ptr, ptr + rlen);
                carry.clear();
                grp_emitted += emit_region(reg_buf.data(), reg_buf.size(), r.spec, !is_last);
            } else {
                grp_emitted += emit_region(ptr, rlen, r.spec, !is_last);
            }
        }
        flush_group(grp_emitted);
    }

    if (!carry.empty()) {
        push_meta(MetaRecord{static_cast<uint32_t>(carry.size()), T0_IDENTITY, DELTA_NONE, 1, 0, 0, 0, 0});
        raw_grp.insert(raw_grp.end(), carry.begin(), carry.end());
        bytes_by_t["T0"] += carry.size();
        uint32_t clen = static_cast<uint32_t>(carry.size());
        carry.clear();
        flush_group(clen);
    }

    c_meta.finish(cm);
    c_raw.finish(cr);
    c_xform.finish(cx);

    TrailerHeader thdr{
        TAG_EOF,
        {
            static_cast<uint32_t>(cm.size()),
            static_cast<uint32_t>(cr.size()),
            static_cast<uint32_t>(cx.size()),
            0,
        },
    };
    uint8_t digest[32];
    sha.final(digest);
    TrailerTail ttail;
    ttail.total_orig_size = total_orig;
    std::memcpy(ttail.sha256, digest, 32);

    write_all(fout, &thdr, sizeof(thdr));
    write_all(fout, cm.data(), cm.size());
    write_all(fout, cr.data(), cr.size());
    write_all(fout, cx.data(), cx.size());
    write_all(fout, &ttail, sizeof(ttail));
    std::fflush(fout);

    tot_meta += cm.size();
    tot_raw += cr.size();
    tot_xform += cx.size();
    tot_framing += sizeof(thdr) + sizeof(ttail);
    total_comp += sizeof(thdr) + sizeof(ttail) + cm.size() + cr.size() + cx.size();

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    double work = total_orig ? (double(total_scored) / double(total_orig)) : 0.0;
    std::string sha_str = sha256_hex(digest);

    if (!stats_path.empty()) {
        std::ofstream sf(stats_path);
        sf << "{\n"
           << "  \"orig_bytes\": " << total_orig << ",\n"
           << "  \"comp_bytes\": " << total_comp << ",\n"
           << "  \"s_meta_bytes\": " << tot_meta << ",\n"
           << "  \"s_raw_bytes\": " << tot_raw << ",\n"
           << "  \"s_xform_bytes\": " << tot_xform << ",\n"
           << "  \"framing_bytes\": " << tot_framing << ",\n"
           << "  \"regions_count\": " << regions_count << ",\n"
           << "  \"scoring_work_mult\": " << work << ",\n"
           << "  \"elapsed_sec\": " << elapsed << ",\n"
           << "  \"sha256\": \"" << sha_str << "\"\n"
           << "}\n";
    }
    return 0;
}

int run_decode(FILE* fin, FILE* fout) {
    FileHeader fhdr;
    if (read_exact(fin, reinterpret_cast<uint8_t*>(&fhdr), sizeof(fhdr)) != sizeof(fhdr)) {
        std::fprintf(stderr, "Truncated FileHeader\n");
        return 1;
    }
    if (std::memcmp(fhdr.magic, MAGIC, 4) != 0 || fhdr.version != VERSION) {
        std::fprintf(stderr, "Invalid ASV3 header\n");
        return 1;
    }
    if (fhdr.wlog < 10 || fhdr.wlog > 27) {
        std::fprintf(stderr, "Invalid ASV3 wlog: %u\n", fhdr.wlog);
        return 1;
    }

    LazyZstdDecoder d_meta(20);
    LazyZstdDecoder d_raw(fhdr.wlog);
    LazyZstdDecoder d_xform(fhdr.wlog);

    Sha256 sha;
    uint64_t total_out = 0;

    constexpr size_t MAX_META_Q = 16 * MB;
    constexpr size_t MAX_DATA_Q = 64 * MB;
    std::vector<uint8_t> comp_buf;
    std::vector<uint8_t> meta_q, raw_q, xform_q;
    std::vector<uint8_t> rec_buf, scratch;

    auto consume_prefix = [](std::vector<uint8_t>& q, size_t n) {
        if (n == q.size()) {
            q.clear();
            if (q.capacity() > 24 * MB) {
                std::vector<uint8_t>().swap(q);
            }
        } else if (n > 0) {
            q.erase(q.begin(), q.begin() + static_cast<std::ptrdiff_t>(n));
        }
    };

    while (true) {
        uint8_t tag = 0;
        if (read_exact(fin, &tag, 1) != 1) {
            std::fprintf(stderr, "Unexpected EOF before FileTrailer\n");
            return 1;
        }
        if (tag == TAG_DATA) {
            uint32_t lens[5];
            if (read_exact(fin, reinterpret_cast<uint8_t*>(lens), sizeof(lens)) != sizeof(lens)) {
                std::fprintf(stderr, "Truncated GroupHeader\n");
                return 1;
            }
            uint32_t grp_uncomp = lens[0], lm = lens[1], lr = lens[2], lx = lens[3], ld = lens[4];
            if (ld != 0) {
                std::fprintf(stderr, "Unsupported non-zero S_RESID stream (len=%u)\n", ld);
                return 1;
            }
            auto feed = [&](uint32_t len, LazyZstdDecoder& dec, std::vector<uint8_t>& q, size_t cap) -> bool {
                if (len == 0) return true;
                if (len > cap) {
                    std::fprintf(stderr, "Compressed payload exceeds safety cap (%u > %zu)\n", len, cap);
                    return false;
                }
                comp_buf.resize(len);
                if (read_exact(fin, comp_buf.data(), len) != len) {
                    std::fprintf(stderr, "Truncated stream payload\n");
                    return false;
                }
                return dec.decompress_append(comp_buf.data(), len, q, cap);
            };
            if (!feed(lm, d_meta, meta_q, MAX_META_Q) ||
                !feed(lr, d_raw, raw_q, MAX_DATA_Q) ||
                !feed(lx, d_xform, xform_q, MAX_DATA_Q)) {
                return 1;
            }

            size_t m_pos = 0, r_pos = 0, x_pos = 0;
            uint64_t grp_dec = 0;
            while (m_pos + sizeof(MetaRecord) <= meta_q.size()) {
                MetaRecord rec;
                std::memcpy(&rec, meta_q.data() + m_pos, sizeof(rec));
                m_pos += sizeof(rec);

                if (rec.delta_op != DELTA_NONE) {
                    std::fprintf(stderr, "Unsupported delta_op: %u\n", rec.delta_op);
                    return 1;
                }

                if (rec.transform_id == T0_IDENTITY) {
                    if (rec.tail_len != 0) {
                        std::fprintf(stderr, "Corrupt T0 MetaRecord: non-zero tail_len %u\n", rec.tail_len);
                        return 1;
                    }
                    if (r_pos > raw_q.size() || rec.orig_len > raw_q.size() - r_pos) {
                        std::fprintf(stderr, "Corrupt stream: T0 read out of bounds (%zu + %u > %zu)\n",
                                     r_pos, rec.orig_len, raw_q.size());
                        return 1;
                    }
                    const uint8_t* ptr = raw_q.data() + r_pos;
                    r_pos += rec.orig_len;
                    write_all(fout, ptr, rec.orig_len);
                    sha.update(ptr, rec.orig_len);
                } else {
                    if (rec.transform_id > T3_2D_PAETH) {
                        std::fprintf(stderr, "Corrupt MetaRecord: unknown transform_id %u\n", rec.transform_id);
                        return 1;
                    }
                    if (rec.w == 0) {
                        std::fprintf(stderr, "Corrupt MetaRecord: zero width w for transform %u\n", rec.transform_id);
                        return 1;
                    }
                    if (rec.tail_len > rec.orig_len) {
                        std::fprintf(stderr, "Corrupt MetaRecord: tail_len %u > orig_len %u\n",
                                     rec.tail_len, rec.orig_len);
                        return 1;
                    }
                    size_t xlen = static_cast<size_t>(rec.orig_len) - static_cast<size_t>(rec.tail_len);
                    if (x_pos > xform_q.size() || xlen > xform_q.size() - x_pos) {
                        std::fprintf(stderr, "Corrupt stream: S_XFORM read out of bounds (%zu + %zu > %zu)\n",
                                     x_pos, xlen, xform_q.size());
                        return 1;
                    }
                    if (r_pos > raw_q.size() || rec.tail_len > raw_q.size() - r_pos) {
                        std::fprintf(stderr, "Corrupt stream: tail read out of bounds (%zu + %u > %zu)\n",
                                     r_pos, rec.tail_len, raw_q.size());
                        return 1;
                    }
                    rec_buf.resize(xlen);
                    TransformSpec sp{rec.transform_id, rec.w, rec.h};
                    apply_transform_inverse(xform_q.data() + x_pos, rec_buf.data(), xlen, sp, scratch);
                    x_pos += xlen;
                    write_all(fout, rec_buf.data(), xlen);
                    sha.update(rec_buf.data(), xlen);
                    if (rec.tail_len > 0) {
                        const uint8_t* tptr = raw_q.data() + r_pos;
                        r_pos += rec.tail_len;
                        write_all(fout, tptr, rec.tail_len);
                        sha.update(tptr, rec.tail_len);
                    }
                }
                total_out += rec.orig_len;
                grp_dec += rec.orig_len;
            }
            if (grp_dec != grp_uncomp) {
                std::fprintf(stderr, "FlushGroup length mismatch: %llu vs %u\n",
                             static_cast<unsigned long long>(grp_dec), grp_uncomp);
                return 1;
            }
            consume_prefix(meta_q, m_pos);
            consume_prefix(raw_q, r_pos);
            consume_prefix(xform_q, x_pos);
        } else if (tag == TAG_EOF) {
            uint32_t flens[4];
            if (read_exact(fin, reinterpret_cast<uint8_t*>(flens), sizeof(flens)) != sizeof(flens)) {
                std::fprintf(stderr, "Truncated TrailerHeader\n");
                return 1;
            }
            if (flens[3] != 0) {
                std::fprintf(stderr, "Unsupported non-zero S_RESID trailer (len=%u)\n", flens[3]);
                return 1;
            }
            auto drain = [&](uint32_t len, LazyZstdDecoder& dec, std::vector<uint8_t>& q, size_t cap) -> bool {
                if (len == 0) return true;
                if (len > cap) {
                    std::fprintf(stderr, "Trailer payload exceeds safety cap (%u > %zu)\n", len, cap);
                    return false;
                }
                comp_buf.resize(len);
                if (read_exact(fin, comp_buf.data(), len) != len) {
                    std::fprintf(stderr, "Truncated trailer stream payload\n");
                    return false;
                }
                return dec.decompress_append(comp_buf.data(), len, q, cap);
            };
            if (!drain(flens[0], d_meta, meta_q, MAX_META_Q) ||
                !drain(flens[1], d_raw, raw_q, MAX_DATA_Q) ||
                !drain(flens[2], d_xform, xform_q, MAX_DATA_Q)) {
                return 1;
            }
            if (!meta_q.empty() || !raw_q.empty() || !xform_q.empty()) {
                std::fprintf(stderr, "Trailing unconsumed stream bytes at EOF (meta=%zu raw=%zu xform=%zu)\n",
                             meta_q.size(), raw_q.size(), xform_q.size());
                return 1;
            }

            TrailerTail ttail;
            if (read_exact(fin, reinterpret_cast<uint8_t*>(&ttail), sizeof(ttail)) != sizeof(ttail)) {
                std::fprintf(stderr, "Truncated TrailerTail\n");
                return 1;
            }
            uint8_t actual[32];
            sha.final(actual);
            if (total_out != ttail.total_orig_size || std::memcmp(actual, ttail.sha256, 32) != 0) {
                std::fprintf(stderr, "SHA-256 verification FAILED!\n");
                return 1;
            }
            std::fflush(fout);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown tag: 0x%02x\n", tag);
            return 1;
        }
    }
}

}  // namespace asv3

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: asv3 <c|d> [in|-] [out|-] [--level 3|19] [--block-mb 4] [--stats file.json]\n");
        return 1;
    }
    std::string mode = argv[1];
    std::string in_path = "-";
    std::string out_path = "-";
    std::string stats_path;
    asv3::Config cfg;

    std::vector<std::string> pos_args;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--level" && i + 1 < argc) {
            cfg.level = std::atoi(argv[++i]);
        } else if (a == "--block-mb" && i + 1 < argc) {
            cfg.block_size_bytes = static_cast<size_t>(std::atoi(argv[++i])) * asv3::MB;
        } else if (a == "--flush-mb" && i + 1 < argc) {
            cfg.flush_group_bytes = static_cast<size_t>(std::atoi(argv[++i])) * asv3::MB;
        } else if (a == "--stats" && i + 1 < argc) {
            stats_path = argv[++i];
        } else if (a == "--trace") {
            cfg.trace = true;
        } else {
            pos_args.push_back(a);
        }
    }
    if (pos_args.size() >= 1) in_path = pos_args[0];
    if (pos_args.size() >= 2) out_path = pos_args[1];

    FILE* fin = (in_path == "-") ? stdin : std::fopen(in_path.c_str(), "rb");
    FILE* fout = (out_path == "-") ? stdout : std::fopen(out_path.c_str(), "wb");
    if (!fin || !fout) {
        std::fprintf(stderr, "Failed to open input/output file\n");
        return 1;
    }

    int rc = (mode == "c") ? asv3::run_encode(fin, fout, cfg, stats_path) : asv3::run_decode(fin, fout);
    if (in_path != "-") std::fclose(fin);
    if (out_path != "-") std::fclose(fout);
    return rc;
}
