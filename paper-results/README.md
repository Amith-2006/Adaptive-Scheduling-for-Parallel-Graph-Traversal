# Project 29 — Final Experimental Results

This directory contains the consolidated results and publication figures for the adaptive parallel BFS scheduler study.

## Experimental scope

- 120 original graphs: ER, Grid, and R-MAT.
- Two graph scales: 50,000 and 500,000 vertices.
- 30 repetitions per graph and scheduler in the baseline.
- Two independent policy sessions, each covering all 120 graphs.
- Four OpenMP threads in the principal experiments.
- Four benchmark schedulers: static, dynamic (chunk size 1), guided (chunk size 1), and custom edge-balanced.

The adaptive policy selects one scheduler before BFS execution using source-vertex degree and graph vertex count. It does not switch schedulers during BFS.

## Main result

The policy achieved a mean graph-level total-runtime effect of -2.2102% relative to Guided, with a 95% confidence interval of [-2.4029%, -2.0028%]. It selected the oracle-best scheduler on 115 of 120 graphs (95.83%).

A negative runtime effect indicates that the policy was faster than Guided. A positive effect indicates that it was slower.

## Analysis files

| File | Contents |
|---|---|
| `combined_summary.csv` | Aggregate and family-scale performance, confidence intervals, oracle accuracy, regret, and win/loss counts |
| `combined_per_graph_results.csv` | Per-graph scheduler runtimes, policy selections, oracle matches, regret, and runtime effects |
| `combined_scheduler_counts.csv` | Number of graphs assigned to each policy scheduler by family and scale |
| `combined_oracle_agreement.csv` | Cross-tabulation of policy-selected scheduler against oracle scheduler |
| `session_consistency_summary.csv` | Aggregate comparison between the two independent policy sessions |
| `session_consistency_per_graph.csv` | Per-graph session effects and direction consistency |

## Important metric definitions

- `runtime_effect_pct`: percentage change in total runtime relative to Guided. Negative means faster.
- `wall_effect_pct`: percentage change in BFS wall time relative to Guided. Negative means faster.
- `policy_regret_pct`: percentage gap between the policy-selected scheduler and the oracle-best scheduler.
- `guided_regret_pct`: percentage gap between Guided and the oracle-best scheduler.
- `oracle_match`: whether the policy selected the oracle-best scheduler for that graph.
- `oracle_accuracy_pct`: percentage of graphs for which the policy matched the oracle.

The total-runtime measurement includes scheduler selection overhead plus BFS execution. BFS wall time is measured after selection. Per-level timing excludes frontier-statistics overhead; do not treat these timing scopes as interchangeable.

The confidence intervals in the summary are graph-level intervals. They do not represent variation across different machines.

## Figures

| Figure | Description |
|---|---|
| `figure1_family_runtime_effect` | Runtime effect by graph family and scale |
| `figure2_scheduler_distribution` | Distribution of policy scheduler selections |
| `figure3_per_graph_distribution` | Distribution of per-graph runtime effects |

Each figure is provided as PDF and PNG.

## Scope and reproducibility note

These are consolidated analysis outputs, not the complete raw experimental archive. The full archive contains per-run logs, baseline per-level measurements, graph inputs, and additional reproducibility material. The large archive is maintained separately because of its size.

Do not infer machine-to-machine generality from these results. The principal sessions were conducted on one Apple Silicon Mac using macOS, Apple Clang, C++17, and OpenMP with four threads. The exact Mac model, memory configuration, and OS version should be recorded before final paper submission.

## Interpretation cautions

- The oracle is an empirical best-scheduler reference derived from the measured baseline; it is not a scheduler available to the policy in advance.
- The policy uses graph-level features and selects once before BFS. It is not a machine-learning model.
- Dynamic scheduling was benchmarked but was not selected by the final policy.
- The held-out set contains 120 graphs and was hash-disjoint from the original graph set.
- One duplicate pair existed in the original generated set: R-MAT 500K instances 18 and 19. There were 119 unique hashes among the 120 original graph files.
