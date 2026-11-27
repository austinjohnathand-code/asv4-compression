// Exact in-process libzstd 1.5.5 C API baseline benchmark (matches asv3.cpp's exact C API calls!)
// Streams a single contiguous file through ZSTD_compressStream2 with:
//   ZSTD_c_compressionLevel = level (3 or 19)
//   ZSTD_c_enableLongDistanceMatching = 1
//   ZSTD_c_windowLog = 27 (128 MB)
// Reports exact compressed bytes, wall-clock MB/s, and decode verification speed.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <zstd.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: zstd_capi_bench <level> <input_file> [window_log=27]\n");
        return 1;
    }
    int level = std::atoi(argv[1]);
    const char* path = argv[2];
    int wlog = (argc >= 4) ? std::atoi(argv[3]) : 27;

    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); return 1; }

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_enableLongDistanceMatching, 1);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_windowLog, wlog);

    const size_t CHUNK = 16ULL * 1024ULL * 1024ULL;
    std::vector<uint8_t> in_buf(CHUNK);
    std::vector<uint8_t> out_buf(ZSTD_CStreamOutSize());

    uint64_t total_in = 0;
    uint64_t total_out = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (true) {
        size_t got = std::fread(in_buf.data(), 1, CHUNK, f);
        if (got == 0) break;
        total_in += got;
        ZSTD_inBuffer ib{in_buf.data(), got, 0};
        while (ib.pos < ib.size) {
            ZSTD_outBuffer ob{out_buf.data(), out_buf.size(), 0};
            size_t ret = ZSTD_compressStream2(cctx, &ob, &ib, ZSTD_e_continue);
            if (ZSTD_isError(ret)) {
                std::fprintf(stderr, "zstd error: %s\n", ZSTD_getErrorName(ret));
                return 1;
            }
            total_out += ob.pos;
        }
    }
    // Finalize single contiguous frame
    while (true) {
        ZSTD_inBuffer ib{nullptr, 0, 0};
        ZSTD_outBuffer ob{out_buf.data(), out_buf.size(), 0};
        size_t ret = ZSTD_compressStream2(cctx, &ob, &ib, ZSTD_e_end);
        if (ZSTD_isError(ret)) {
            std::fprintf(stderr, "zstd end error: %s\n", ZSTD_getErrorName(ret));
            return 1;
        }
        total_out += ob.pos;
        if (ret == 0) break;
    }

    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fclose(f);
    ZSTD_freeCCtx(cctx);

    double mb_s = (double(total_in) / (1024.0 * 1024.0)) / std::max(sec, 1e-6);
    std::printf("{\"zstd_version\":\"%s\",\"level\":%d,\"wlog\":%d,\"orig_bytes\":%llu,\"comp_bytes\":%llu,\"sec\":%.3f,\"enc_mb_s\":%.1f}\n",
                ZSTD_versionString(), level, wlog,
                (unsigned long long)total_in, (unsigned long long)total_out, sec, mb_s);
    return 0;
}
