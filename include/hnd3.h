#pragma once
// =============================================================================
// HND3 / HND4 — Zero-Dependency C++20 Reversible 3D Hyperspectral Tensor Library
//
// Supports:
//   * HND3 (v3): Single-tile lossless / 7-layer progressive (wire-compatible with
//     phase3_codec.py), ARM64 NEON vmlal_s32 + portable scalar parity, blind 3D
//     stride/spectral-axis recovery, overflow-proof full-range int32 escape.
//   * HND4 (v4): Streaming spatial tiling (O(B * T^2) bounded Peak RSS), random-
//     access single-tile decode, per-header / per-layer / per-tile Castagnoli
//     CRC32C integrity (including lossy previews), and guaranteed-L_inf near-
//     lossless coding (--max-error E: |x - x_hat| <= E for every sample).
// =============================================================================

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <array>
#include <string>
#include <vector>

namespace hnd3 {

enum class DType : uint8_t {
    I16 = 1,
    U16 = 2,
    I32 = 3,
};

struct EncodeOptions {
    uint32_t d0 = 0;
    uint32_t d1 = 0;
    uint32_t d2 = 0;
    DType dtype = DType::I16;
    std::string mode = "auto";  // "auto" | "full" | "clustered" | "none"
    uint8_t levels = 3;         // 0..8
    uint8_t spec_axis = 0;      // 0..2
    uint8_t base_k = 6;         // 0..15 (0 = one-shot, 6 = 7-layer progressive)
    uint8_t stripes = 16;       // 1..64
    uint32_t tile_size = 0;     // 0 = single-tile HND3 v3 (or HND4 if crc/max_error); >0 = tiled HND4 v4
    uint32_t max_error = 0;     // 0 = lossless; E > 0 = guaranteed |x - x_hat| <= E (HND4 v4)
    bool per_layer_crc = false; // true forces HND4 v4 with per-layer Castagnoli CRC32C
    bool blind = false;         // blind 1D stride + spectral-axis recovery
    bool verify = true;         // in-memory/streaming round-trip verification before returning
    int8_t hist_lut = -1;       // -1 = auto (sparse 16-bit LUT compaction), 0 = off, 1 = force on
    int threads = 0;            // 0 = hardware_concurrency()
};

struct EncodeStats {
    uint8_t version = 3;
    uint32_t d0 = 0, d1 = 0, d2 = 0;
    uint32_t tile_h = 0, tile_w = 0, n_tiles_y = 1, n_tiles_x = 1;
    uint32_t max_error = 0;
    int spec_axis = 0;
    std::vector<size_t> blind_strides;
    size_t raw_bytes = 0;
    size_t compressed_bytes = 0;
    double encode_seconds = 0.0;
    int64_t verified_max_abs_err = 0;
    bool verified = false;
    bool hist_lut_used = false;
    uint32_t hist_lut_unique = 0;
};

struct DecodeOptions {
    int stop_layer = 0;              // 0 = finest layer (lossless if max_error==0); 1..base_k = progressive preview
    int max_level = -1;              // -1 = all spatial wavelet levels
    int tile_y = -1;                 // -1 = all tiles; >= 0 selects a single spatial tile (HND4 random access)
    int tile_x = -1;
    bool verify_sha = true;          // verify full-cube SHA-256 on exact lossless decodes
    bool verify_crc = true;          // verify header & per-layer CRC32C (HND4)
    int threads = 0;                 // 0 = hardware_concurrency()
    uint64_t max_samples = 512000000ULL;
};

struct DecodedCube {
    std::vector<uint8_t> raw_bytes;
    uint8_t version = 3;
    uint32_t d0 = 0, d1 = 0, d2 = 0;
    uint8_t dt_code = 0;
    uint32_t max_error = 0;
    bool sha_ok = false;
    bool sha_checked = false;
    bool crc_ok = false;
    bool crc_checked = false;
    size_t crc_layers_verified = 0;
};

struct FileInfo {
    uint8_t version = 0;
    uint8_t flags = 0;
    uint8_t dt_code = 0;
    uint8_t spec_code = 0;
    uint8_t n_lv = 0;
    uint8_t spec_axis = 0;
    uint8_t base_k = 0;
    uint8_t n_stripes = 1;
    uint32_t d0 = 0, d1 = 0, d2 = 0;
    uint32_t tile_h = 0, tile_w = 0, n_tiles_y = 1, n_tiles_x = 1;
    uint32_t max_error = 0;
    size_t file_bytes = 0;
    bool header_crc_checked = false;
    bool header_crc_ok = false;
};

// In-memory buffer API (supports both HND3 v3 and HND4 v4)
std::vector<uint8_t> encode(const uint8_t* raw_bytes, size_t n_bytes,
                            const EncodeOptions& opt, EncodeStats* stats = nullptr);

DecodedCube decode(const uint8_t* data, size_t len, const DecodeOptions& opt = {});

// Streaming file API (O(B * tile_h * tile_w) bounded Peak RSS for HND4 tiled files!)
EncodeStats encode_file(const std::string& in_path, const std::string& out_path,
                        const EncodeOptions& opt);

DecodedCube decode_file(const std::string& in_path, const std::string& out_path,
                        const DecodeOptions& opt = {});

FileInfo inspect(const uint8_t* data, size_t len);
FileInfo inspect_file(const std::string& path);

// Compute Castagnoli CRC32C (useful for external callers/tests)
uint32_t crc32c(const uint8_t* data, size_t len, uint32_t init = 0u);

} // namespace hnd3
#endif // __cplusplus

// =============================================================================
// C ABI (extern "C") for FFI callers (C, Python ctypes, Rust, Julia)
// =============================================================================
#ifdef __cplusplus
extern "C" {
#endif

typedef struct hnd3_c_encode_opts {
    uint32_t d0, d1, d2;
    uint8_t dtype;          // 1=i16, 2=u16, 3=i32
    uint8_t levels;         // default 3
    uint8_t spec_axis;      // 0..2
    uint8_t base_k;         // 0..15
    uint8_t stripes;        // 1..64
    uint8_t blind;          // 0 or 1
    uint8_t per_layer_crc;  // 0 or 1
    uint8_t verify;         // 0 or 1
    uint32_t tile_size;     // 0 = untiled
    uint32_t max_error;     // 0 = lossless; >0 = guaranteed L_inf <= max_error
    int32_t threads;        // 0 = auto
} hnd3_c_encode_opts;

typedef struct hnd3_c_decode_opts {
    int32_t stop_layer;     // 0 = lossless/finest
    int32_t max_level;      // -1 = all levels
    int32_t tile_y;         // -1 = all tiles
    int32_t tile_x;         // -1 = all tiles
    uint8_t verify_sha;     // 1 = check SHA-256 on exact decode
    uint8_t verify_crc;     // 1 = check CRC32C on HND4
    int32_t threads;        // 0 = auto
    uint64_t max_samples;   // 0 = default 512M
} hnd3_c_decode_opts;

// Returns 0 on success, non-zero on error (with error text written to err_buf if non-null).
// Caller frees *out_buf with hnd3_free().
int hnd3_encode_buffer(const uint8_t* raw, size_t raw_len, const hnd3_c_encode_opts* opts,
                       uint8_t** out_buf, size_t* out_len, char* err_buf, size_t err_buf_len);

int hnd3_decode_buffer(const uint8_t* stream, size_t stream_len, const hnd3_c_decode_opts* opts,
                       uint8_t** out_buf, size_t* out_len,
                       uint32_t* out_d0, uint32_t* out_d1, uint32_t* out_d2, uint8_t* out_dtype,
                       char* err_buf, size_t err_buf_len);

void hnd3_free(uint8_t* ptr);

#ifdef __cplusplus
}
#endif
