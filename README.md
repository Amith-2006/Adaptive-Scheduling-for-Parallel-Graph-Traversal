# Project #29 — Adaptive Scheduling for Parallel BFS

Reproducible research scaffold for testing whether per-frontier graph statistics help choose a parallel BFS strategy.

## Status
- Implemented: synthetic graph generation, OpenMP BFS variants (static, dynamic, guided, degree-prefix edge-balanced), per-level logging, correctness comparison, analysis/classifier script.
- Optional TBB `parallel_for` scheduler is included behind `USE_TBB`; it is enabled with `make TBB=1` when oneTBB is installed.
- Results from this repository are experimental outputs, not evidence of novelty by themselves. Run repeated trials on the target VM before making paper claims.

## Requirements
- Linux, g++ with OpenMP
- Python 3.10+; `pip install -r requirements.txt`
- Optional: oneTBB development package for the TBB variant (`make TBB=1`)

## Quick start
```bash
make
python3 scripts/generate_graphs.py --out graphs --sizes 1000 3000 10000 --seed 29
./build/bfs_bench graphs/grid_n1000.edgelist 4 results/grid.csv 3
python3 scripts/run_suite.py --binary ./build/bfs_bench --graphs graphs --out results --threads 4 --repeats 5
python3 scripts/analyze.py --input results/all_levels.csv --out results/analysis
```

Edge list format: first line `n m`, then `u v` pairs, zero-based, undirected. Duplicate edges are removed by generator.

## Important experimental controls
- Keep graph-level splits: all levels from one graph must remain in the same train/test partition.
- Compare mean-only features against mean + variance/skew/CV; use graph-held-out evaluation.
- Report median and dispersion across repeated trials; avoid interpreting a 2-core sandbox's absolute times as general performance.
- Include the cost of feature extraction and prefix sums in end-to-end adaptive runtime.
- Ensure every scheduler produces the same BFS distances before accepting its timings.
