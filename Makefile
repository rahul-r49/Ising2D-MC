# Build the C++ simulation.
#
#   make                                  L = 64, 8 seeds
#   make L=128 SEED_START=1 N_SEEDS=200   other lattice size / seeds
#   make L=16 QUICK=1                     very short schedule (tests, quick check)
#
# The source is chosen automatically: cpp/ising.cpp when L is a power of 2 (faster),
# cpp/ising_any_L.cpp otherwise (e.g. L=40 or 80). Force one with SRC=cpp/ising_any_L.cpp.
#
# The program is written to build/ising_L<L> (or build/ising_L<L>_quick).
# On macOS the default compiler has no OpenMP: install GCC (brew install gcc)
# and run, for example,  make CXX=g++-14

L          ?= 64
SEED_START ?= 1
N_SEEDS    ?= 8
CXX        ?= g++
CXXFLAGS   ?= -O3 -march=native -ffast-math -fopenmp -std=c++17

SRC ?= $(shell if [ $$(( $(L) & ($(L) - 1) )) -eq 0 ]; then echo cpp/ising.cpp; else echo cpp/ising_any_L.cpp; fi)

DEFS = -DL_SIZE=$(L) -DSEED_START=$(SEED_START) -DN_SEEDS=$(N_SEEDS)
SUFFIX =
ifeq ($(QUICK),1)
  DEFS += -DQUICK
  SUFFIX = _quick
endif

BIN = build/ising_L$(L)$(SUFFIX)

$(BIN): $(SRC)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ $<
	@echo "built $@"

clean:
	rm -rf build

.PHONY: clean
