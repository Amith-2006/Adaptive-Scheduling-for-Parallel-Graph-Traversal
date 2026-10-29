CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp
LDFLAGS ?= -fopenmp
ifeq ($(TBB),1)
CXXFLAGS += -DUSE_TBB -I/usr/include
LDFLAGS += -ltbb
endif

all: build/bfs_bench

build/bfs_bench: src/bfs_bench.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf build
.PHONY: all clean
