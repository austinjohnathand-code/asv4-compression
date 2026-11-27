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

.PHONY: all bench test clean

all: asv4 v3/cpp/asv3

asv4: v3/cpp/asv3.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

v3/cpp/asv3: asv4
	cp asv4 v3/cpp/asv3

bench: v3/cpp/asv3 v3/cpp/zstd_capi_bench

v3/cpp/zstd_capi_bench: v3/cpp/zstd_capi_bench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Quick UBSan + portable FIPS 180-4 SHA-256 + Unix pipe smoke test
test:
	$(CXX) -O2 -std=c++20 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all \
	  -DASV3_FORCE_PORTABLE_SHA256 -I$(ZSTD_PREFIX)/include v3/cpp/asv3.cpp -o /tmp/asv4_test $(LDFLAGS)
	head -c 262144 v3/cpp/asv3.cpp | /tmp/asv4_test c --level 3 | /tmp/asv4_test d > /dev/null
	rm -f /tmp/asv4_test
	@echo "[PASS] UBSan + portable SHA-256 + pipe smoke test succeeded"

clean:
	rm -f asv4 v3/cpp/asv3 v3/cpp/zstd_capi_bench /tmp/asv4_test
