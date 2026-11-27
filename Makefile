CXX      ?= c++
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

CXXFLAGS := -O3 -std=c++20 $(ARCH_FLAGS) -Wall -Wextra -I$(ZSTD_PREFIX)/include
LDFLAGS  := -L$(ZSTD_PREFIX)/lib -Wl,-rpath,$(ZSTD_PREFIX)/lib -lzstd

.PHONY: all hnd2 hnd3 bench bench-hnd3 test clean

all: asv4 v3/cpp/asv3 hnd2 hnd3

asv4: v3/cpp/asv3.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

hnd2: cpp/hnd2.cpp
	$(CXX) -O3 -std=c++20 $(ARCH_FLAGS) -Wall -Wextra -pthread $< -o $@

hnd3: cpp/hnd3.cpp
	$(CXX) -O3 -std=c++20 $(ARCH_FLAGS) -Wall -Wextra -pthread $< -o $@

v3/cpp/asv3: asv4
	cp asv4 v3/cpp/asv3

bench: v3/cpp/asv3 v3/cpp/zstd_capi_bench

bench-hnd3: hnd3
	python3 bench/fetch_hnd3_data.py --check
	python3 bench/bench_hnd3.py all

v3/cpp/zstd_capi_bench: v3/cpp/zstd_capi_bench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Quick UBSan + portable FIPS 180-4 SHA-256 + NEON/scalar parity + fuzz + Python/C++ cross-verification
test: hnd2 hnd3
	$(CXX) -O2 -std=c++20 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all \
	  -DASV3_FORCE_PORTABLE_SHA256 -I$(ZSTD_PREFIX)/include v3/cpp/asv3.cpp -o /tmp/asv4_test $(LDFLAGS)
	head -c 262144 v3/cpp/asv3.cpp | /tmp/asv4_test c --level 3 | /tmp/asv4_test d > /dev/null
	rm -f /tmp/asv4_test
	$(CXX) -O2 -std=c++20 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all -pthread \
	  cpp/hnd2.cpp -o /tmp/hnd2_test
	$(CXX) -O2 -std=c++20 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all \
	  -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG -DHND3_ENABLE_FUZZ -pthread cpp/hnd3.cpp -o /tmp/hnd3_test
	$(CXX) -O3 -std=c++20 -Wall -Wextra -pthread -DHND3_FORCE_SCALAR cpp/hnd3.cpp -o /tmp/hnd3_scalar_test
	python3 -m unittest -v test_phase1.py test_phase2.py test_phase3.py
	python3 -c "import subprocess,numpy as np; rng=np.random.default_rng(7); x=rng.integers(-2000,2000,(16,32,32),dtype='<i2'); x.tofile('/tmp/t_par.raw'); subprocess.run(['./hnd3','enc','/tmp/t_par.raw','/tmp/t_n.hnd3','--shape','16,32,32','--dtype','i16','--base-k','6'],check=True,stdout=subprocess.DEVNULL); subprocess.run(['/tmp/hnd3_scalar_test','enc','/tmp/t_par.raw','/tmp/t_s.hnd3','--shape','16,32,32','--dtype','i16','--base-k','6'],check=True,stdout=subprocess.DEVNULL); assert open('/tmp/t_n.hnd3','rb').read()==open('/tmp/t_s.hnd3','rb').read(), 'NEON/scalar mismatch'; subprocess.run(['/tmp/hnd3_test','fuzz','/tmp/t_n.hnd3','5000'],check=True)"
	rm -f /tmp/hnd2_test /tmp/hnd3_test /tmp/hnd3_scalar_test /tmp/t_par.raw /tmp/t_n.hnd3 /tmp/t_s.hnd3
	@echo "[PASS] UBSan + libc++ debug fuzz + NEON/scalar parity + Phase 1-3 regression suites succeeded"

clean:
	rm -f asv4 hnd2 hnd3 v3/cpp/asv3 v3/cpp/zstd_capi_bench /tmp/asv4_test /tmp/hnd2_test /tmp/hnd3_test
