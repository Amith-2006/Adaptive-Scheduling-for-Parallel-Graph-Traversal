
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCHEDULERS = ["static", "dynamic", "guided", "edge_balanced"]
COLORS = {
    "static": "gray",
    "dynamic": "tab:blue",
    "guided": "tab:orange",
    "edge_balanced": "tab:green",
}


def load_data(path):
    df = pd.read_csv(path)

    required = {
        "graph",
        "trial",
        "scheduler",
        "level_seconds",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["level_seconds"].isna().any():
        raise ValueError("Timing data contains missing values")

    if (df["level_seconds"] <= 0).any():
        raise ValueError("Timing data contains non-positive values")

    df["family"] = df["graph"].str.split("_").str[0]

    return df


def calculate_run_times(df):
    runs = (
        df.groupby(["family", "graph", "trial", "scheduler"])
        ["level_seconds"]
        .sum()
        .reset_index(name="run_seconds")
    )

    return runs


def calculate_speedups(runs):
    baseline = runs[runs["scheduler"] == "static"][
        ["family", "graph", "trial", "run_seconds"]
    ].rename(columns={"run_seconds": "static_seconds"})

    merged = runs.merge(
        baseline,
        on=["family", "graph", "trial"],
        how="inner",
        validate="many_to_one",
    )

    merged["speedup_pct"] = (
        (merged["static_seconds"] / merged["run_seconds"]) - 1
    ) * 100

    return merged


def plot_family_performance(speedups, out):
    data = (
        speedups[speedups["scheduler"] != "static"]
        .groupby(["family", "scheduler"])["speedup_pct"]
        .median()
        .unstack()
        .reindex(columns=["dynamic", "guided", "edge_balanced"])
    )

    ax = data.plot(
        kind="bar",
        figsize=(10, 6),
        color=[
            COLORS["dynamic"],
            COLORS["guided"],
            COLORS["edge_balanced"],
        ],
    )

    ax.axhline(0, linewidth=1, color="black")
    ax.set_title("Scheduler Speedup by Graph Family")
    ax.set_xlabel("Graph Family")
    ax.set_ylabel("Median Speedup vs Static (%)")
    ax.legend(title="Scheduler")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "family_speedup.png", dpi=300)
    plt.close()


def plot_instance_performance(speedups, out):
    data = (
        speedups[speedups["scheduler"] != "static"]
        .groupby(["graph", "scheduler"])["speedup_pct"]
        .median()
        .reset_index()
    )

    families = ["er", "grid", "rmat"]

    fig, axes = plt.subplots(
        len(families),
        1,
        figsize=(12, 12),
        sharex=False,
    )

    for ax, family in zip(axes, families):
        subset = data[data["graph"].str.startswith(family + "_")]

        for scheduler in ["dynamic", "guided", "edge_balanced"]:
            values = subset[subset["scheduler"] == scheduler]

            ax.plot(
                values["graph"],
                values["speedup_pct"],
                marker="o",
                label=scheduler,
                color=COLORS[scheduler],
            )

        ax.axhline(0, linewidth=1, color="black")
        ax.set_title(f"{family.upper()} Graph Instances")
        ax.set_ylabel("Speedup vs Static (%)")
        ax.grid(alpha=0.3)
        ax.legend()

        ax.tick_params(axis="x", rotation=25)

    axes[-1].set_xlabel("Graph Instance")

    fig.suptitle("Scheduler Performance Across Graph Instances")
    fig.tight_layout()
    fig.savefig(out / "instance_speedup.png", dpi=300)
    plt.close(fig)


def plot_win_frequency(speedups, out):
    data = (
        speedups.groupby(["family", "graph", "scheduler"])
        ["run_seconds"]
        .median()
        .reset_index()
    )

    winners = (
        data.loc[data.groupby("graph")["run_seconds"].idxmin()]
        .groupby(["family", "scheduler"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SCHEDULERS, fill_value=0)
    )

    ax = winners.plot(
        kind="bar",
        figsize=(10, 6),
        color=[COLORS[s] for s in SCHEDULERS],
    )

    ax.set_title("Winning Scheduler by Graph File")
    ax.set_ylabel("Number of Graph Files Won")
    ax.set_xlabel("Graph Family")
    ax.legend(title="Scheduler")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "scheduler_wins.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    runs = calculate_run_times(df)
    speedups = calculate_speedups(runs)

    plot_family_performance(speedups, output)
    plot_instance_performance(speedups, output)
    plot_win_frequency(speedups, output)

    print(f"Plots saved in: {output}")
    print(f"Benchmark runs analyzed: {len(runs)}")
    print(f"Graph instances: {runs['graph'].nunique()}")


if __name__ == "__main__":
    main()
