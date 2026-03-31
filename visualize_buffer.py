import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def pick_row(rows: List[Dict[str, Any]], task_id: str | None, config_key: str | None) -> Dict[str, Any]:
    if config_key:
        for row in rows:
            if str(row.get("config_key")) == config_key:
                return row
        raise ValueError(f"No row found for config_key={config_key}")

    if task_id is not None:
        matches = [row for row in rows if str(row.get("task_id")) == str(task_id)]
        if not matches:
            raise ValueError(f"No row found for task_id={task_id}")
        return matches[0]

    return rows[0]


def print_trace_summary(row: Dict[str, Any]) -> None:
    print("=== Run Summary ===")
    print(f"task_id: {row.get('task_id')}")
    print(f"policy: {row.get('policy')}")
    print(f"buffer_size: {row.get('buffer_size')}")
    print(f"summarisation_level: {row.get('summarisation_level')}")
    print(f"expected: {row.get('expected')}")
    print(f"predicted: {row.get('predicted')}")
    print(f"correct: {row.get('correct')}")
    print(f"steps: {row.get('steps')}")
    print(f"trim_count: {row.get('trim_count', 'N/A')}")

    trim_events = row.get("trim_events") or []
    trace = row.get("trace") or []

    if not trace:
        print("\nNo `trace` field found in this row.")
        print("Tip: rerun experiments with include_verbose=True to inspect reasoning traces.")
        return

    print("\n=== Step Trace ===")
    for index, step in enumerate(trace, start=1):
        thought = (step.get("thought") or "").strip().replace("\n", " ")
        answer = step.get("answer")
        done = step.get("done")
        usage = step.get("step_usage")
        duration = step.get("step_duration")
        print(f"Step {index}: done={done} answer={answer} usage={usage} duration={duration}")
        if thought:
            print(f"  thought: {thought[:200]}{'...' if len(thought) > 200 else ''}")

    print("\n=== Trim Events ===")
    if not trim_events:
        print("No trim events recorded.")
        return

    for event in trim_events:
        print(
            f"step={event.get('step')} tokens {event.get('tokens_before')} -> {event.get('tokens_after')} "
            f"dropped={len(event.get('dropped_chunk_details', []) or [])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a single experiment row and print buffer/trace diagnostics.")
    parser.add_argument("--input", default="data/full_run_core_matrix_results.jsonl", help="Path to results JSONL")
    parser.add_argument("--task-id", default=None, help="Select first row for this task_id")
    parser.add_argument("--config-key", default=None, help="Select exact row by config_key")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"No rows in input JSONL: {input_path}")

    chosen = pick_row(rows, args.task_id, args.config_key)
    print_trace_summary(chosen)


if __name__ == "__main__":
    main()
