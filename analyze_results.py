import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + (z**2 / n)
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (centre - margin, centre + margin)


def load_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def build_agg_spec(frame: pd.DataFrame) -> Dict[str, tuple[str, str]]:
    spec: Dict[str, tuple[str, str]] = {
        "n": ("correct", "size"),
        "successes": ("correct", "sum"),
        "accuracy": ("correct", "mean"),
    }
    optional_metrics = {
        "avg_trim_count": "trim_count",
        "avg_usage_total": "usage_total",
        "avg_steps": "steps",
    }
    for out_name, source_col in optional_metrics.items():
        if source_col in frame.columns:
            spec[out_name] = (source_col, "mean")
    return spec


def build_summary_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    work = df.copy()

    if "correct" in work.columns:
        work["correct"] = work["correct"].astype(bool)
    for col in ["trim_count", "usage_total", "steps"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    agg_spec = build_agg_spec(work)

    by_condition = (
        work.groupby(["buffer_size", "policy", "summarisation_level"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )

    ci_pairs = [wilson_ci(int(row.successes), int(row.n)) for row in by_condition.itertuples(index=False)]
    by_condition["ci_low"] = [pair[0] for pair in ci_pairs]
    by_condition["ci_high"] = [pair[1] for pair in ci_pairs]

    by_buffer = (
        work.groupby("buffer_size", dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values("buffer_size")
    )

    ci_pairs_buffer = [wilson_ci(int(row.successes), int(row.n)) for row in by_buffer.itertuples(index=False)]
    by_buffer["ci_low"] = [pair[0] for pair in ci_pairs_buffer]
    by_buffer["ci_high"] = [pair[1] for pair in ci_pairs_buffer]

    by_policy = (
        work.groupby("policy", dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )

    ci_pairs_policy = [wilson_ci(int(row.successes), int(row.n)) for row in by_policy.itertuples(index=False)]
    by_policy["ci_low"] = [pair[0] for pair in ci_pairs_policy]
    by_policy["ci_high"] = [pair[1] for pair in ci_pairs_policy]

    return {
        "by_condition": by_condition,
        "by_buffer": by_buffer,
        "by_policy": by_policy,
    }


def write_outputs(
    df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    out_dir: Path,
    input_jsonl: Path,
) -> None:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    run_meta = {
        "input_file": str(input_jsonl),
        "rows": int(len(df)),
        "tasks": int(df["task_id"].nunique()) if "task_id" in df.columns else 0,
        "conditions": int(df["config_key"].nunique()) if "config_key" in df.columns else 0,
        "accuracy_overall": safe_mean(df["correct"].astype(float)) if "correct" in df.columns else 0.0,
        "buffers": sorted([int(value) for value in df["buffer_size"].dropna().unique().tolist()]) if "buffer_size" in df.columns else [],
        "policies": sorted(df["policy"].dropna().astype(str).unique().tolist()) if "policy" in df.columns else [],
    }

    with (tables_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(run_meta, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute reproducible pilot/full-run summary tables from experiment JSONL outputs.")
    parser.add_argument(
        "--input",
        default="data/full_run_core_matrix_results.jsonl",
        help="Path to results JSONL file (default: data/full_run_core_matrix_results.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for generated tables (default: results)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in input JSONL: {input_path}")

    frame = pd.DataFrame(rows)
    required_columns = {"task_id", "policy", "buffer_size", "correct"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Input JSONL missing required columns: {sorted(missing)}")

    summary_tables = build_summary_tables(frame)
    write_outputs(frame, summary_tables, Path(args.out_dir), input_path)

    print(f"Loaded rows: {len(frame)}")
    print(f"Tasks: {frame['task_id'].nunique()}")
    print(f"Conditions: {frame['config_key'].nunique() if 'config_key' in frame.columns else 'N/A'}")
    print(f"Wrote tables to: {Path(args.out_dir) / 'tables'}")


if __name__ == "__main__":
    main()
