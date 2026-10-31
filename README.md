# Adaptive Scheduling for Parallel Breadth-First Search (BFS)

An experimental research project investigating whether graph-frontier characteristics can guide scheduling decisions in parallel Breadth-First Search (BFS).

The project evaluates multiple OpenMP scheduling strategies and explores the potential for an adaptive scheduler that selects a strategy based on frontier-level graph statistics.

## Research Motivation

Parallel BFS performance depends on the structure of the current frontier. Different scheduling strategies may perform differently depending on the distribution of vertex degrees and the amount of work associated with each frontier vertex.

This project investigates the following hypothesis:

**Can per-level frontier statistics predict which parallel BFS scheduling strategy will perform best, and can adaptive scheduler selection improve end-to-end execution time?**

The hypothesis is being evaluated experimentally. Its generality and research novelty have not yet been established.

## Implemented Scheduling Strategies

The current benchmark evaluates four strategies:

- **Static:** OpenMP static scheduling.
- **Dynamic:** OpenMP dynamic scheduling.
- **Guided:** OpenMP guided scheduling.
- **Edge-Balanced:** Partitions frontier vertices using cumulative degree information to distribute estimated edge-processing work across threads.

The edge-balanced strategy partitions whole vertices; it does not split individual high-degree vertices.

## System Overview

The benchmark pipeline consists of:

1. Synthetic graph generation.
2. Graph loading and BFS execution.
3. Parallel scheduling using four strategies.
4. Frontier-level feature extraction.
5. Per-level runtime logging.
6. Correctness validation against sequential BFS.
7. Experimental analysis.

## Frontier-Level Features

The benchmark records:

| Feature | Description |
|---|---|
| Frontier size | Number of vertices in the current frontier |
| Frontier edges | Total adjacency edges associated with frontier vertices |
| Degree mean | Mean frontier vertex degree |
| Degree variance | Degree dispersion within the frontier |
| Degree skewness | Asymmetry of the degree distribution |
| Degree CV | Coefficient of variation of frontier degrees |

These features are being investigated as potential predictors of scheduler performance.

## Graph Workloads

The graph generator supports:

- Grid graphs
- Erdős–Rényi random graphs
- R-MAT graphs

Graph sizes and instance counts can be configured through command-line arguments.

Note: Generated graph instances should not automatically be treated as independent samples. In particular, the current grid generator produces identical topology for a given graph size.

## Experimental Setup

Current experimental environment:

- Hardware: Apple MacBook Air, Apple Silicon
- CPU: 10 cores
- RAM: 16 GB
- Compiler: Apple Clang
- Parallel runtime: OpenMP (`libomp`)
- Language: C++17
- Python: 3.9.6

### Completed Benchmark Experiments

A 50,000-vertex benchmark was conducted using:

- 5 Erdős–Rényi instances
- 5 Grid files (one unique topology)
- 5 R-MAT instances
- 4 scheduling strategies
- 30 repetitions per graph

This resulted in **1,800 scheduler executions**.

All tested scheduler outputs passed comparison against the sequential BFS distance-vector oracle.

### Preliminary Findings

Median speedup relative to static scheduling:

| Graph family | Dynamic | Guided | Edge-Balanced |
|---|---:|---:|---:|
| Erdős–Rényi | -23.81% | +4.49% | +8.06% |
| Grid | -7.26% | -0.15% | +0.46% |
| R-MAT | +4.56% | +15.75% | +8.73% |

Positive values indicate faster execution than static scheduling.

These results are preliminary measurements from one hardware platform. The Grid files have identical topology, and repeated timing trials are not independent graph instances. Further experiments and statistical validation are required before drawing general conclusions.

## Repository Structure

```text
project29/
├── src/
│   ├── bfs_bench.cpp
│   ├── partition.h
│   └── partition_test.cpp
├── scripts/
│   ├── generate_graphs.py
│   ├── run_suite.py
│   └── analyze.py
├── graphs/
├── results/
├── Makefile
├── requirements.txt
└── README.md
```

Generated graph datasets, build artifacts, and experimental output files are excluded from version control.

## Build and Run

### Requirements

- C++17 compiler
- OpenMP runtime
- Python 3.9+
- Dependencies from `requirements.txt`

### macOS Build

With Homebrew `libomp` installed, compile using:

```bash
mkdir -p build

clang++ -O3 -std=c++17 -Wall -Wextra \
  -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  src/bfs_bench.cpp \
  -L/opt/homebrew/opt/libomp/lib \
  -lomp \
  -o build/bfs_bench
```

### Install Python Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Generate Graphs

```bash
python3 scripts/generate_graphs.py \
  --out graphs \
  --sizes 1000 3000 \
  --instances 3 \
  --seed 29
```

### Run the Benchmark

```bash
python3 scripts/run_suite.py \
  --binary ./build/bfs_bench \
  --graphs graphs \
  --out results \
  --threads 4 \
  --repeats 10
```

### Analyze Results

```bash
python3 scripts/analyze.py \
  --input results/all_levels.csv \
  --out results/analysis
```

## Experimental Considerations

- Keep all levels belonging to the same graph in the same train/test partition.
- Compare simpler frontier features against extended degree-distribution features.
- Report medians and variability across repeated trials.
- Validate every parallel scheduler against sequential BFS.
- Include feature-extraction and switching overhead when evaluating an adaptive scheduler.
- Distinguish repeated timing trials from independent graph instances.
- Avoid generalizing results from a single hardware platform.

## Current Limitations

- Adaptive scheduler selection has not yet been implemented or validated.
- Feature-based scheduler prediction remains under investigation.
- The edge-balanced strategy cannot split a single high-degree vertex.
- Current benchmark results are from one hardware platform.
- More independent graph instances and larger workloads are required.
- Feature extraction and scheduler switching costs must be included in end-to-end evaluation.

## Future Work

- Investigate relationships between frontier statistics and scheduler performance.
- Improve graph workload diversity.
- Evaluate feature-based adaptive scheduling.
- Measure prediction regret against an oracle scheduler.
- Compare adaptive execution against fixed scheduling strategies.
- Evaluate end-to-end performance, including decision overhead.
- Validate results across additional hardware platforms.

## Project Status

**Research in progress.**

This repository contains the experimental implementation and preliminary benchmark infrastructure for investigating adaptive parallel BFS scheduling. Results and conclusions will be updated as further experiments are completed.