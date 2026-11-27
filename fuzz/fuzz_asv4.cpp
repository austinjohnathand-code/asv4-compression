// =============================================================================
// fuzz/fuzz_asv4.cpp — Coverage-Guided libFuzzer Target for ASV3/ASV4 Decoder
//
// Feeds arbitrary untrusted byte streams via fmemopen() into asv3::run_decode()
// under ASan + UBSan, exercising header validation, multi-stream zstd framing,
// MetaRecord bounds checking, all inverse transforms (T1/T2_XOR/T2_ELEM/
// T2_BIT/T2_NIBROT/T3_2D), and SHA-256 verification.
// =============================================================================
#define ASV3_NO_MAIN
#define ASV3_FUZZ_BUILD
#include "../v3/cpp/asv3.cpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size == 0) return 0;

    static FILE* devnull = std::fopen("/dev/null", "wb");
    if (!devnull) return 0;

    FILE* fin = fmemopen(const_cast<uint8_t*>(data), size, "rb");
    if (!fin) return 0;

    (void)asv3::run_decode(fin, devnull);
    std::fclose(fin);
    return 0;
}
