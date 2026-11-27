// =============================================================================
// fuzz/fuzz_hnd3.cpp — Coverage-Guided libFuzzer Target for HND3 (v3) & HND4 (v4)
//
// Exercises full decode, progressive prefix decode (--stop-layer K), single-tile
// random access (--tile ty,tx), and container inspection (hnd3::inspect) against
// arbitrary untrusted byte streams under ASan + UBSan.
// =============================================================================
#include "hnd3.h"
#include <cstddef>
#include <cstdint>
#include <exception>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size == 0) return 0;

    // 1) Container metadata & header CRC32C parser
    try {
        (void)hnd3::inspect(data, size);
    } catch (const std::exception&) {
    }

    // 2) Full decode (bounded max_samples to avoid OOM on crafted shape fields)
    try {
        hnd3::DecodeOptions opt;
        opt.stop_layer  = -1;
        opt.threads     = 1;
        opt.max_samples = 2000000; // <= 8 MB
        (void)hnd3::decode(data, size, opt);
    } catch (const std::exception&) {
    }

    // 3) Progressive preview, random-access tile, and checksum-bypassed deep parser fuzzing
    //    (verify_crc=false lets libFuzzer penetrate past CRC32C into Tile Index / rANS / 5/3 / LDU)
    if (size >= 4) {
        try {
            hnd3::DecodeOptions opt;
            opt.stop_layer  = static_cast<int>(data[size - 1] & 0x07u);
            opt.threads     = 1;
            opt.max_samples = 2000000;
            opt.verify_crc  = ((data[size - 1] & 0x10u) == 0);
            opt.verify_sha  = ((data[size - 1] & 0x20u) == 0);
            if ((data[size - 1] & 0x08u) != 0) {
                opt.tile_y = static_cast<int>((data[size - 2] >> 4) & 0x03u);
                opt.tile_x = static_cast<int>(data[size - 2] & 0x03u);
            }
            (void)hnd3::decode(data, size, opt);
        } catch (const std::exception&) {
        }
    }

    return 0;
}
