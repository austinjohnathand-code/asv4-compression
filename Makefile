CXX      ?= c++
AR       ?= ar
UNAME_S  := $(shell uname -s)
UNAME_M  := $(shell uname -m)

ifeq ($(UNAME_M),arm64)
  ARCH_FLAGS := -mcpu=native
else ifeq ($(UNAME_M),aarch64)
  ARCH_FLAGS := -mcpu=native
else
  ARCH_FLAGS := -march=native
endif

# Auto-detect libzstd include/lib prefix (Homebrew, Conda, or system)
ZSTD_PREFIX := $(shell \
  if [ -f /opt/homebrew/include/zstd.h ]; then echo /opt/homebrew; \
  elif [ -f /opt/anaconda3/include/zstd.h ]; then echo /opt/anaconda3; \
  elif [ -f /usr/local/include/zstd.h ]; then echo /usr/local; \
  else echo /usr; fi)

# Auto-detect LLVM clang++ for libFuzzer (Homebrew LLVM on macOS, clang++ on Linux)
FUZZ_CXX ?= $(shell \
  if [ -x /opt/homebrew/opt/llvm/bin/clang++ ]; then echo /opt/homebrew/opt/llvm/bin/clang++; \
  elif [ -x /usr/local/opt/llvm/bin/clang++ ]; then echo /usr/local/opt/llvm/bin/clang++; \
  else echo clang++; fi)

CXXFLAGS := -O3 -std=c++20 $(ARCH_FLAGS) -Wall -Wextra -I$(ZSTD_PREFIX)/include
LDFLAGS  := -L$(ZSTD_PREFIX)/lib -Wl,-rpath,$(ZSTD_PREFIX)/lib -lzstd
HND3_FLAGS := -O3 -std=c++20 $(ARCH_FLAGS) -Iinclude -Wall -Wextra -pthread

HND3_HDRS := include/hnd3.h cpp/hnd3_core.hpp cpp/hnd3_rans.hpp

.PHONY: all hnd2 hnd3 libhnd3 bench bench-hnd3 test fuzz clean

all: asv4 v3/cpp/asv3 hnd2 libhnd3.a hnd3

asv4: v3/cpp/asv3.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

hnd2: cpp/hnd2.cpp
	$(CXX) -O3 -std=c++20 $(ARCH_FLAGS) -Wall -Wextra -pthread $< -o $@

cpp/hnd3_lib.o: cpp/hnd3_lib.cpp $(HND3_HDRS)
	$(CXX) $(HND3_FLAGS) -c $< -o $@

libhnd3.a: cpp/hnd3_lib.o
	$(AR) rcs $@ $<

libhnd3: libhnd3.a

hnd3: cpp/hnd3_cli.cpp libhnd3.a $(HND3_HDRS)
	$(CXX) $(HND3_FLAGS) cpp/hnd3_cli.cpp libhnd3.a -o $@

v3/cpp/asv3: asv4
	cp asv4 v3/cpp/asv3

bench: v3/cpp/asv3 v3/cpp/zstd_capi_bench

bench-hnd3: hnd3
	python3 bench/fetch_hnd3_data.py --check
	python3 bench/bench_hnd3.py all

v3/cpp/zstd_capi_bench: v3/cpp/zstd_capi_bench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Quick UBSan + portable FIPS 180-4 SHA-256 + NEON/scalar parity + C ABI + fuzz + Phase 1-4 suites
test: hnd2 libhnd3.a hnd3
	$(CXX) -O2 -std=c++20 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all \
	  -DASV3_FORCE_PORTABLE_SHA256 -I$(ZSTD_PREFIX)/include v3/cpp/asv3.cpp -o /tmp/asv4_test $(LDFLAGS)
	head -c 262144 v3/cpp/asv3.cpp | /tmp/asv4_test c --level 3 | /tmp/asv4_test d > /dev/null
	rm -f /tmp/asv4_test
	$(CXX) -O2 -std=c++20 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all -pthread \
	  cpp/hnd2.cpp -o /tmp/hnd2_test
	$(CXX) -O2 -std=c++20 -Iinclude -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all \
	  -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG -D_GLIBCXX_ASSERTIONS -DHND3_ENABLE_FUZZ -pthread \
	  cpp/hnd3_lib.cpp cpp/hnd3_cli.cpp -o /tmp/hnd3_test
	$(CXX) -O3 -std=c++20 -Iinclude -Wall -Wextra -pthread -DHND3_FORCE_SCALAR \
	  cpp/hnd3_lib.cpp cpp/hnd3_cli.cpp -o /tmp/hnd3_scalar_test
	python3 -m unittest -v test_phase1.py test_phase2.py test_phase3.py test_phase4.py
	python3 -c "import subprocess,numpy as np; rng=np.random.default_rng(7); x=rng.integers(-2000,2000,(16,32,32),dtype='<i2'); x.tofile('/tmp/t_par.raw'); subprocess.run(['./hnd3','enc','/tmp/t_par.raw','/tmp/t_n.hnd3','--shape','16,32,32','--dtype','i16','--base-k','6'],check=True,stdout=subprocess.DEVNULL); subprocess.run(['/tmp/hnd3_scalar_test','enc','/tmp/t_par.raw','/tmp/t_s.hnd3','--shape','16,32,32','--dtype','i16','--base-k','6'],check=True,stdout=subprocess.DEVNULL); assert open('/tmp/t_n.hnd3','rb').read()==open('/tmp/t_s.hnd3','rb').read(), 'NEON/scalar mismatch'; subprocess.run(['/tmp/hnd3_test','fuzz','/tmp/t_n.hnd3','5000'],check=True)"
	rm -f /tmp/hnd2_test /tmp/hnd3_test /tmp/hnd3_scalar_test /tmp/t_par.raw /tmp/t_n.hnd3 /tmp/t_s.hnd3
	@echo "[PASS] UBSan + libc++ debug fuzz + NEON/scalar parity + C ABI + Phase 1-4 regression suites succeeded"

# Coverage-guided libFuzzer + ASan + UBSan (requires upstream LLVM clang++)
FUZZ_SAN ?= fuzzer,address,undefined
fuzz:
	$(FUZZ_CXX) -O2 -g -std=c++20 -Iinclude -Wall -Wextra \
	  -fsanitize=$(FUZZ_SAN) -fno-sanitize-recover=all -pthread \
	  fuzz/fuzz_hnd3.cpp cpp/hnd3_lib.cpp -o /tmp/fuzz_hnd3
	$(FUZZ_CXX) -O2 -g -std=c++20 -I$(ZSTD_PREFIX)/include -Wall -Wextra \
	  -fsanitize=$(FUZZ_SAN) -fno-sanitize-recover=all \
	  fuzz/fuzz_asv4.cpp -o /tmp/fuzz_asv4 $(LDFLAGS)
	@echo "[BUILT] /tmp/fuzz_hnd3 and /tmp/fuzz_asv4 (sanitizers=$(FUZZ_SAN))"

clean:
	rm -f asv4 hnd2 hnd3 libhnd3.a cpp/hnd3_lib.o v3/cpp/asv3 v3/cpp/zstd_capi_bench \
	      /tmp/asv4_test /tmp/hnd2_test /tmp/hnd3_test /tmp/fuzz_hnd3 /tmp/fuzz_asv4
