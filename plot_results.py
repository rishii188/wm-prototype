import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_tables(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = results_dir / "tables"
    by_condition = pd.read_csv(tables_dir / "by_condition.csv")
    by_buffer = pd.read_csv(tables_dir / "by_buffer.csv")
    by_policy = pd.read_csv(tables_dir / "by_policy.csv")
    return by_condition, by_buffer, by_policy


def ensure_plots_dir(results_dir: Path) -> Path:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_accuracy_by_buffer(by_buffer: pd.DataFrame, plots_dir: Path) -> None:
    frame = by_buffer.sort_values("buffer_size")
    yerr_low = frame["accuracy"] - frame["ci_low"]
    yerr_high = frame["ci_high"] - frame["accuracy"]

    plt.figure(figsize=(8, 5))
    plt.bar(frame["buffer_size"].astype(str), frame["accuracy"], color="#4c78a8")
    plt.errorbar(
        x=range(len(frame)),
        y=frame["accuracy"],
        yerr=[yerr_low, yerr_high],
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    plt.ylim(0, 1)
    plt.title("Accuracy by Buffer Size")
    plt.xlabel("Buffer Size")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_by_buffer.png", dpi=200)
    plt.close()


def plot_accuracy_by_policy(by_policy: pd.DataFrame, plots_dir: Path) -> None:
    frame = by_policy.sort_values("accuracy", ascending=False)
    yerr_low = frame["accuracy"] - frame["ci_low"]
    yerr_high = frame["ci_high"] - frame["accuracy"]

    plt.figure(figsize=(9, 5))
    plt.bar(frame["policy"], frame["accuracy"], color="#f58518")
    plt.errorbar(
        x=range(len(frame)),
        y=frame["accuracy"],
        yerr=[yerr_low, yerr_high],
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    plt.ylim(0, 1)
    plt.title("Accuracy by Policy")
    plt.xlabel("Policy")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_by_policy.png", dpi=200)
    plt.close()


def plot_heatmap(by_condition: pd.DataFrame, plots_dir: Path) -> None:
    filtered = by_condition.copy()
    filtered = filtered[filtered["policy"] != "full_context"]
    pivot = filtered.pivot_table(
        index="policy",
        columns="buffer_size",
        values="accuracy",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Policy × Buffer Accuracy Heatmap")
    plt.xlabel("Buffer Size")
    plt.ylabel("Policy")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_heatmap_policy_buffer.png", dpi=200)
    plt.close()


def plot_trim_trends(by_condition: pd.DataFrame, plots_dir: Path) -> None:
    if "avg_trim_count" not in by_condition.columns:
        if "avg_steps" not in by_condition.columns:
            return

        frame = (
            by_condition.groupby(["buffer_size", "policy"], dropna=False)["avg_steps"]
            .mean()
            .reset_index()
            .sort_values("buffer_size")
        )

        plt.figure(figsize=(9, 5))
        sns.lineplot(data=frame, x="buffer_size", y="avg_steps", hue="policy", marker="o")
        plt.title("Average Reasoning Steps by Buffer and Policy")
        plt.xlabel("Buffer Size")
        plt.ylabel("Average Steps")
        plt.tight_layout()
        plt.savefig(plots_dir / "trim_event_trends.png", dpi=200)
        plt.close()
        return

    frame = (
        by_condition.groupby(["buffer_size", "policy"], dropna=False)["avg_trim_count"]
        .mean()
        .reset_index()
        .sort_values("buffer_size")
    )

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=frame, x="buffer_size", y="avg_trim_count", hue="policy", marker="o")
    plt.title("Average Trim Count by Buffer and Policy")
    plt.xlabel("Buffer Size")
    plt.ylabel("Average Trim Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "trim_event_trends.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style plots from results/tables CSVs.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory containing tables/ and plots/ (default: results)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    by_condition, by_buffer, by_policy = load_tables(results_dir)
    plots_dir = ensure_plots_dir(results_dir)

    sns.set_theme(style="whitegrid")

    plot_accuracy_by_buffer(by_buffer, plots_dir)
    plot_accuracy_by_policy(by_policy, plots_dir)
    plot_heatmap(by_condition, plots_dir)
    plot_trim_trends(by_condition, plots_dir)

    print(f"Wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()
