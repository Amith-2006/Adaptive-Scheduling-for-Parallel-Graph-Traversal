# Project Architecture

## Adaptive Scheduling for Parallel Graph Traversal

The project follows an experimental pipeline to evaluate parallel BFS scheduling strategies and investigate the potential for adaptive scheduling.

```mermaid
flowchart TD
    A["Synthetic Graph Generation<br/>generate_graphs.py"]
    B["Graph Input<br/>.edgelist files"]
    C["Benchmark Orchestration<br/>run_suite.py"]
    D["BFS Benchmark<br/>bfs_bench.cpp"]
    E["Parallel Scheduling Strategies"]
    E1["Static"]
    E2["Dynamic"]
    E3["Guided"]
    E4["Edge-Balanced<br/>partition.h"]
    F["Correctness Validation<br/>Sequential BFS Oracle"]
    G["Frontier Statistics<br/>Size, Edges, Mean, Variance,<br/>Skewness, CV"]
    H["Experimental Results<br/>CSV Files"]
    I["Performance Analysis<br/>analyze.py"]
    J["Adaptive Scheduling Research"]

    A --> B --> C --> D --> E
    E --> E1
    E --> E2
    E --> E3
    E --> E4
    E1 --> F
    E2 --> F
    E3 --> F
    E4 --> F
    F --> G --> H --> I --> J
```

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `generate_graphs.py` | Generates synthetic graph workloads |
| `run_suite.py` | Executes benchmarks across graph inputs |
| `bfs_bench.cpp` | Implements BFS and scheduling strategies |
| `partition.h` | Provides shared edge-balanced partitioning logic |
| `partition_test.cpp` | Tests partitioning correctness |
| `analyze.py` | Analyzes benchmark results |

## Scheduling Strategies

- **Static:** Assigns loop iterations using static scheduling.
- **Dynamic:** Assigns iterations dynamically to threads.
- **Guided:** Uses guided scheduling with decreasing chunk sizes.
- **Edge-Balanced:** Partitions frontier vertices using cumulative degree information.

## Research Direction

The long-term objective is to investigate whether frontier characteristics can guide scheduler selection and improve end-to-end BFS performance.

Adaptive scheduling is currently a research objective, not an implemented feature.