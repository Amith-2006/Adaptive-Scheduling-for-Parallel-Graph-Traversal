# Adaptive Scheduling for Parallel Breadth-First Search

An experimental study of scheduling strategies for parallel Breadth-First Search (BFS), implemented in C++17 and OpenMP.

The project investigates how graph structure and BFS frontier characteristics influence scheduler performance. It implements four scheduling strategies and evaluates a graph-level threshold policy that selects a scheduler before BFS execution.

## 1. Project Overview

Parallel BFS performance depends on the distribution of work across frontier vertices. A frontier containing vertices with highly varying degrees can produce load imbalance when work is distributed uniformly.

This project evaluates:
- Static OpenMP scheduling
- Dynamic OpenMP scheduling
- Guided OpenMP scheduling
- Custom Edge-Balanced scheduling
- A graph-level threshold policy using root degree and vertex count
- Exploratory machine-learning approaches for scheduler prediction

The final threshold policy makes one scheduler selection per graph. It does not switch schedulers during BFS execution.

## 2. Scheduling Strategies

| Strategy | Description |
|---|---|
| Static | Uses OpenMP static scheduling |
| Dynamic | Uses OpenMP dynamic scheduling with chunk size 1 |
| Guided | Uses OpenMP guided scheduling with chunk size 1 |
| Edge-Balanced | Partitions frontier vertices using cumulative degree information to distribute estimated edge-processing work |

The Edge-Balanced implementation partitions whole vertices. It does not split an individual high-degree vertex across threads. Its implementation uses per-thread candidate storage followed by serial deduplication and distance updates.

## 3. Final Graph-Level Threshold Policy

The final policy uses two graph-level inputs:
- Root degree: the degree of the BFS source vertex.
- Vertex count: the number of vertices in the graph.

The policy selects one scheduler before BFS begins:

```cpp
if (root_degree <= 3) {
    scheduler = (n == 50000) ? "static" : "guided";
} else if (root_degree <= 1594) {
    scheduler = "edge_balanced";
} else {
    scheduler = "guided";
}
```

The policy does not use graph-family labels as direct inputs, does not use a machine-learning model, and does not adapt the scheduler at each BFS level. Dynamic scheduling is not selected by this policy.

## 4. Experimental Evaluation

### Workloads

The final held-out evaluation contains 120 synthetic graphs:

| Graph family | 50,000 vertices | 500,000 vertices |
|---|---:|---:|
| Erdős–Rényi (ER) | 20 | 20 |
| Grid | 20 | 20 |
| R-MAT | 20 | 20 |
| **Total** | **60** | **60** |

The held-out graph files have unique content hashes and no exact hash overlap with the original graph dataset.

### Benchmark configuration

- Four baseline scheduling strategies
- 30 baseline repetitions per graph and scheduler
- Two policy evaluation sessions
- 30 Guided and 30 threshold-policy repetitions per graph in each session
- Four configured OpenMP threads
- Correctness comparison against sequential BFS distance vectors

The combined policy evaluation therefore contains 60 repetitions per graph for Guided and 60 for the threshold policy.

All evaluated policy executions passed the sequential BFS correctness check.

### Main results

The reported runtime effect compares the threshold policy with the fixed Guided baseline. Negative values indicate that the threshold policy is faster.

| Measure | Result |
|---|---:|
| Mean graph-level runtime effect | -2.2102% |
| Pooled runtime effect | -1.3193% |
| Graphs where policy was faster | 85 / 120 |
| Graphs where policy was slower | 35 / 120 |
| Agreement with baseline oracle scheduler | 95.83% |

The mean graph-level effect weights each graph equally. The pooled effect compares summed per-graph median runtimes and therefore weights graphs by runtime.

The policy does not improve every graph or every workload group. Its measured effects vary by graph family and scale. In particular, the R-MAT groups show small regressions on average, while the 500,000-vertex Grid group is approximately neutral.

Graph-level bootstrap intervals and additional robustness checks are included in the reproducibility package. These intervals are conditional on the two evaluation sessions and the tested machine; they do not quantify variability across machines or hardware environments.

## 5. Exploratory Machine-Learning Experiments

The repository also contains exploratory scripts that investigate whether frontier features can predict scheduler performance.

These experiments are separate from the final threshold policy.

| Script | Purpose |
|---|---|
| `scripts/analyze.py` | Grouped cross-validation for predicting per-level scheduler oracle labels |
| `scripts/evaluate_early_policy.py` | Decision-tree experiments using early BFS observations |
| `scripts/evaluate_graph_policy.py` | Cross-scale classification using root frontier degree |
| `scripts/leave_one_family_out.py` | Leave-One-Family-Out evaluation across ER, Grid and R-MAT |

These scripts use retrospective scheduler timing measurements to construct oracle labels. Their predictions and regret measurements should not be interpreted as proof of an online adaptive scheduler or as validation of the final threshold policy.

Some exploratory scripts expect earlier 15-graph datasets and machine-specific input paths. Check their command-line arguments and input requirements before running them on another machine.

## 6. Repository Structure

```text
project29/
├── src/
│   ├── bfs_bench.cpp
│   ├── bfs_bench_family_policy.cpp
│   ├── partition.h
│   └── partition_test.cpp
├── scripts/
│   ├── generate_graphs.py
│   ├── run_suite.py
│   ├── analyze.py
│   ├── plot_results.py
│   ├── evaluate_early_policy.py
│   ├── evaluate_graph_policy.py
│   └── leave_one_family_out.py
├── figures/
├── ARCHITECTURE.md
├── Makefile
├── requirements.txt
└── README.md
```

Generated graphs, build outputs and local experiment results are excluded from ordinary Git version control. The complete final reproducibility dataset and experiment outputs are distributed separately as a versioned GitHub Release archive.

## 7. Build Instructions

### Requirements

- C++17 compiler
- OpenMP runtime
- Python 3.9 or later
- Python packages listed in `requirements.txt`

### macOS with Homebrew OpenMP

Install the OpenMP runtime if required:

```bash
brew install libomp
```

Create the build directory:

```bash
mkdir -p build
```

Build the original four-scheduler benchmark:

```bash
clang++ -O3 -std=c++17 -Wall -Wextra \
  -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  src/bfs_bench.cpp \
  -L/opt/homebrew/opt/libomp/lib \
  -lomp \
  -o build/bfs_bench
```

Build the final threshold-policy benchmark:

```bash
clang++ -O3 -std=c++17 -Wall -Wextra \
  -Xpreprocessor -fopenmp \
  -Isrc \
  -I/opt/homebrew/opt/libomp/include \
  src/bfs_bench_family_policy.cpp \
  -L/opt/homebrew/opt/libomp/lib \
  -lomp \
  -o build/bfs_bench_family_policy
```

## 8. Python Environment

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate example synthetic graphs:

```bash
python3 scripts/generate_graphs.py \
  --out graphs \
  --sizes 1000 3000 \
  --instances 3 \
  --seed 29
```

Run the original benchmark suite:

```bash
python3 scripts/run_suite.py \
  --binary ./build/bfs_bench \
  --graphs graphs \
  --out results \
  --threads 4 \
  --repeats 10
```

Analyze its output:

```bash
python3 scripts/analyze.py \
  --input results/all_levels.csv \
  --out results/analysis
```

The example commands above demonstrate the original benchmark workflow. They do not reproduce the final 120-graph threshold-policy experiment; use the reproducibility package for that experiment.

## 9. Reproducibility

The full reproducibility archive contains:
- The 120 held-out graph files
- Baseline per-level measurements
- Both policy evaluation sessions
- Combined analysis CSV files
- Analysis and graph-generation scripts
- Plotting scripts and generated figures
- Source code for the final threshold policy
- Independent analysis reproduction outputs
- SHA-256 checksums and verification records
- Reproduction instructions

The archive is distributed as a GitHub Release asset so that large datasets do not need to be stored in ordinary Git history.

## 10. Limitations

- The evaluation uses three synthetic graph families.
- Held-out graphs are new instances within those families; this is not validation on an unseen graph family.
- Experiments were conducted on one Apple Silicon machine.
- Cross-machine and cross-architecture performance validation has not been performed.
- The threshold policy is not guaranteed to outperform Guided on every graph.
- Edge-Balanced scheduling cannot split the work of an individual high-degree vertex.
- Serial deduplication and distance updates may limit Edge-Balanced parallel scalability.
- The bootstrap intervals do not capture between-machine or between-environment variability.
- Exploratory ML results are distinct from the final threshold-policy evaluation.

## 11. Project Status

The four-scheduler benchmark, graph-level threshold policy, held-out evaluation, correctness checks and reproducibility package have been implemented.

Further work is needed to evaluate additional graph families, larger workloads and different hardware platforms.
