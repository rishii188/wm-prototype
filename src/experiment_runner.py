import json
import re
from typing import Any, Dict, Optional, List
import time
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from src.engine import ReasoningEngine
from src.buffer import Buffer
from src.tokenizer import TokenCounter
from src.summariser import Summariser
from src.policies import (
    RecencyPolicy,
    FullContextPolicy,
    SummarisingRecencyPolicy,
    RandomPolicy,
    ImportancePolicy,
)
from src.dataset import answers_match, load_multiple_datasets


class ExperimentRunner:
    # The mastermind behind the experiments!

    def __init__(self, llm, seed: int = 42):
        self.llm = llm
        self.tc = TokenCounter()
        self.seed = seed
        self.run_id = str(uuid4())  # Unique ID for this experimental run.

    @staticmethod
    def _extract_gsm8k_final(answer_text: str) -> Optional[str]:
        # Extracts the final answer.
        if answer_text is None:
            return None
        s = str(answer_text)
        match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", s)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_number_fallback(text: str) -> Optional[str]:
        # Extracts a number from any given string (fallback).
        if text is None:
            return None
        s = str(text).replace(",", " ")
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)  # Find all integers or floats.
        if not nums:
            return None
        return nums[-1]  # Return the last number found.

    @classmethod
    def norm_answer(cls, s: Any) -> Optional[str]:
        # Normalises an answer into a consistent string format for fair comparison.
        if s is None:
            return None

        string = str(s).strip()
        if not string:
            return None

        # Attempt GSM8K extraction.
        gsm_extracted = cls._extract_gsm8k_final(string)
        if gsm_extracted is not None:
            return gsm_extracted.strip()

        # Fallback to extracting the last number from the first line: where LLMs might output a sentence ending with the answer.
        first_line = string.splitlines()[0].strip()
        number_fallback = cls._extract_number_fallback(first_line)
        if number_fallback is not None:
            return number_fallback.strip()

        return first_line

    def _make_policy(self, policy_name: str, summarisation_level: str, summarisation_window_size: int):
        valid_levels = ("light", "heavy", "cognitive")

        if policy_name == "recency":
            return RecencyPolicy()
        if policy_name == "full_context":
            return FullContextPolicy()
        if policy_name == "random":
            return RandomPolicy(seed=self.seed)
        if policy_name == "importance":
            return ImportancePolicy()
        if policy_name == "summarising_recency":
            if summarisation_level not in valid_levels:
                summarisation_level = "light"
            
            # Summariser is deterministic/rule-based (including cognitive mode proxies).
            summariser = Summariser()
            return SummarisingRecencyPolicy(
                summariser=summariser,
                mode=summarisation_level,
                summarisation_window=summarisation_window_size
            )
        
        raise ValueError(f"Unknown policy: {policy_name}")

    def _build_error_row(
        self,
        task: Dict[str, Any],
        buffer_size: int,
        policy_name: str,
        summarisation_level: str,
        summarisation_window_size: int,
        error: Exception,
    ) -> Dict[str, Any]:
        expected_answer = self.norm_answer(task.get("answer"))
        return {
            "task_id": task.get("id"),
            "question": task.get("question"),
            "expected": expected_answer,
            "predicted": None,
            "correct": False,
            "buffer_size": buffer_size,
            "policy": policy_name,
            "summarisation_level": summarisation_level,
            "summarisation_window_size": summarisation_window_size,
            "steps": 0,
            "usage_total": None,
            "response_time": None,
            "duration": None,
            "model": getattr(self.llm, "model", "mock"),
            "run_id": self.run_id,
            "error": f"{type(error).__name__}: {error}",
            "config_key": f"id={task.get('id')}|bs={buffer_size}|pol={policy_name}|sum={summarisation_level}|win={summarisation_window_size}",
        }

    def _build_config_key(
        self,
        task_id: Any,
        buffer_size: int,
        policy_name: str,
        summarisation_level: str,
        summarisation_window_size: int,
    ) -> str:
        return (
            f"id={task_id}|bs={buffer_size}|pol={policy_name}|"
            f"sum={summarisation_level}|win={summarisation_window_size}"
        )

    def _load_existing_config_keys(self, out_path: str) -> set[str]:
        existing_keys: set[str] = set()
        output_file = Path(out_path)
        if not output_file.exists():
            return existing_keys

        with output_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    key = row.get("config_key")
                    if key:
                        existing_keys.add(str(key))
        return existing_keys

    def run_single_task(
        self,
        task: Dict[str, Any],
        buffer_size: int,
        policy_name: str,
        summarisation_level: str,
        summarisation_window_size: int,
        max_steps: int = 12,
        min_steps_before_done: int = 3,
        include_verbose: bool = False,
    ) -> Dict[str, Any]:
        
        if policy_name != "summarising_recency":
            current_summarisation_level = "none"
        elif summarisation_level not in ("light", "heavy", "cognitive"):
            current_summarisation_level = "light"
        else:
            current_summarisation_level = summarisation_level

        # Initialise the working memory buffer and the selected forgetting policy.
        buffer = Buffer(max_tokens=buffer_size, token_counter=self.tc)
        policy = self._make_policy(policy_name, current_summarisation_level, summarisation_window_size)

        # Instantiate the ReasoningEngine.
        engine = ReasoningEngine(
            self.llm,
            buffer,
            policy,
            max_steps=max_steps,
            min_steps_before_done=min_steps_before_done,
        )

        start_timestamp = datetime.utcnow().isoformat() + "Z"
        start_perf_counter = time.perf_counter()

        # Execute the reasoning process for the given problem.
        result_from_engine = engine.run(task["question"])

        end_perf_counter = time.perf_counter()
        total_duration = end_perf_counter - start_perf_counter

        trace = result_from_engine.get("trace") or []

        # Aggregate usage and response time across all executed reasoning steps.
        step_usages = [step.get("step_usage") for step in trace if isinstance(step.get("step_usage"), (int, float))]
        total_llm_tokens_used: Optional[int] = int(sum(step_usages)) if step_usages else None

        step_durations = [step.get("step_duration") for step in trace if isinstance(step.get("step_duration"), (int, float))]
        total_llm_response_time: Optional[float] = float(sum(step_durations)) if step_durations else None

        # Normalise expected and predicted answers for a strong comparison.
        expected_answer = self.norm_answer(task.get("answer"))
        predicted_answer = self.norm_answer(result_from_engine.get("answer"))

        # Numeric comparison.
        is_correct: bool
        try:
            is_correct = answers_match(task.get("answer"), result_from_engine.get("answer"))
        except Exception:
            is_correct = (predicted_answer == expected_answer)

        # Put the results together in one row.
        result_row = {
            "task_id": task.get("id"),
            "question": task.get("question"),
            "expected": expected_answer,
            "predicted": predicted_answer,
            "correct": is_correct,
            "buffer_size": buffer_size,
            "policy": policy_name,
            "summarisation_level": current_summarisation_level,
            "summarisation_window_size": summarisation_window_size,
            "steps": result_from_engine.get("steps"),
            "usage_total": total_llm_tokens_used,
            "response_time": total_llm_response_time,
            "duration": total_duration,
            "model": getattr(self.llm, "model", "mock"), # Record the LLM model used
            "run_id": self.run_id,
            # A unique key.
            "config_key": self._build_config_key(
                task_id=task.get("id"),
                buffer_size=buffer_size,
                policy_name=policy_name,
                summarisation_level=current_summarisation_level,
                summarisation_window_size=summarisation_window_size,
            ),
        }

        if include_verbose:
            result_row.update({
                "timestamp": start_timestamp,
                "trace": result_from_engine.get("trace"), # Full reasoning trace
                "trim_events": result_from_engine.get("trim_events", []),
                "trim_count": result_from_engine.get("trim_count", 0),
                "completed_all_phases": result_from_engine.get("completed_all_phases", False),
                "seed": self.seed,
            })

        return result_row

    def run_multiple(
        self,
        dataset_paths: List[str],
        buffer_sizes: List[int],
        policies: List[str],
        summarisation_levels: List[str],
        summarisation_window_sizes: List[int],
        out_path: str,
        max_steps: int = 12,
        min_steps_before_done: int = 3,
        include_verbose: bool = False,
    ):
        
        all_tasks = load_multiple_datasets(dataset_paths)

        valid_sum_levels = [level for level in summarisation_levels if level in ("light", "heavy", "cognitive")]
        if not valid_sum_levels:
            valid_sum_levels = ["light"]
        
        # Calculate number of runs.
        num_tasks = len(all_tasks)
        num_buffer_sizes = len(buffer_sizes)
        num_policies = len(policies)
        num_summarisation_levels = len(valid_sum_levels)
        num_summarisation_window_sizes = len(summarisation_window_sizes)

        num_summarising_policies = sum(1 for policy in policies if policy == "summarising_recency")
        num_full_context_policies = sum(1 for policy in policies if policy == "full_context")
        num_buffered_non_summarising_policies = sum(
            1 for policy in policies if policy not in ("summarising_recency", "full_context")
        )

        runs_per_task = (
            num_full_context_policies +
            num_buffer_sizes * (
                num_buffered_non_summarising_policies +
            (num_summarising_policies * num_summarisation_levels * num_summarisation_window_sizes)
            )
        )
        total_runs = num_tasks * runs_per_task
        print(f"Starting {total_runs} experimental runs...")

        existing_keys = self._load_existing_config_keys(out_path)
        if existing_keys:
            print(f"Resume mode: found {len(existing_keys)} completed rows in {out_path}; duplicates will be skipped.")

        # Iterate through all combinations of experimental parameters.
        for task in all_tasks:
            # Run full-context baseline once per task (not repeated across buffer sizes).
            if "full_context" in policies:
                baseline_buffer = max(buffer_sizes) if buffer_sizes else 256
                baseline_key = self._build_config_key(
                    task_id=task.get("id"),
                    buffer_size=baseline_buffer,
                    policy_name="full_context",
                    summarisation_level="none",
                    summarisation_window_size=1,
                )
                if baseline_key in existing_keys:
                    print(f"Skipping completed run: {baseline_key}")
                else:
                    print(f"Running task {task.get('id')} | Buffer: baseline | Policy: full_context | Sum Level: none (N/A)")
                    try:
                        result_row = self.run_single_task(
                            task=task,
                            buffer_size=baseline_buffer,
                            policy_name="full_context",
                            summarisation_level="none",
                            summarisation_window_size=1,
                            max_steps=max_steps,
                            min_steps_before_done=min_steps_before_done,
                            include_verbose=include_verbose,
                        )
                    except Exception as error:
                        print(f"[ERROR] task={task.get('id')} policy=full_context: {error}")
                        result_row = self._build_error_row(
                            task=task,
                            buffer_size=baseline_buffer,
                            policy_name="full_context",
                            summarisation_level="none",
                            summarisation_window_size=1,
                            error=error,
                        )
                    self.append_jsonl(out_path, result_row)
                    existing_keys.add(baseline_key)

            for buffer_size in buffer_sizes:
                for policy_name in policies:
                    if policy_name == "full_context":
                        continue

                    if policy_name == "summarising_recency":
                        for summarisation_level in valid_sum_levels:
                            for summarisation_window_size in summarisation_window_sizes:
                                config_key = self._build_config_key(
                                    task_id=task.get("id"),
                                    buffer_size=buffer_size,
                                    policy_name=policy_name,
                                    summarisation_level=summarisation_level,
                                    summarisation_window_size=summarisation_window_size,
                                )
                                if config_key in existing_keys:
                                    print(f"Skipping completed run: {config_key}")
                                    continue
                                print(f"Running task {task.get('id')} | Buffer: {buffer_size} | Policy: {policy_name} | Sum Level: {summarisation_level} | Sum Window: {summarisation_window_size}")
                                try:
                                    result_row = self.run_single_task(
                                        task=task,
                                        buffer_size=buffer_size,
                                        policy_name=policy_name,
                                        summarisation_level=summarisation_level,
                                        summarisation_window_size=summarisation_window_size,
                                        max_steps=max_steps,
                                        min_steps_before_done=min_steps_before_done,
                                        include_verbose=include_verbose,
                                    )
                                except Exception as error:
                                    print(f"[ERROR] task={task.get('id')} bs={buffer_size} policy={policy_name} sum={summarisation_level} win={summarisation_window_size}: {error}")
                                    result_row = self._build_error_row(
                                        task=task,
                                        buffer_size=buffer_size,
                                        policy_name=policy_name,
                                        summarisation_level=summarisation_level,
                                        summarisation_window_size=summarisation_window_size,
                                        error=error,
                                    )
                                self.append_jsonl(out_path, result_row)
                                existing_keys.add(config_key)
                    else:
                        # For non-summarising policies, run exactly once per task and buffer.
                        config_key = self._build_config_key(
                            task_id=task.get("id"),
                            buffer_size=buffer_size,
                            policy_name=policy_name,
                            summarisation_level="none",
                            summarisation_window_size=1,
                        )
                        if config_key in existing_keys:
                            print(f"Skipping completed run: {config_key}")
                            continue
                        print(f"Running task {task.get('id')} | Buffer: {buffer_size} | Policy: {policy_name} | Sum Level: none (N/A)")
                        try:
                            result_row = self.run_single_task(
                                task=task,
                                buffer_size=buffer_size,
                                policy_name=policy_name,
                                summarisation_level="none",
                                summarisation_window_size=1, # Dummy value.
                                max_steps=max_steps,
                                min_steps_before_done=min_steps_before_done,
                                include_verbose=include_verbose,
                            )
                        except Exception as error:
                            print(f"[ERROR] task={task.get('id')} bs={buffer_size} policy={policy_name}: {error}")
                            result_row = self._build_error_row(
                                task=task,
                                buffer_size=buffer_size,
                                policy_name=policy_name,
                                summarisation_level="none",
                                summarisation_window_size=1,
                                error=error,
                            )
                        self.append_jsonl(out_path, result_row)
                        existing_keys.add(config_key)
        print("All experimental runs completed.")

    def append_jsonl(self, out_path: str, row: Dict[str, Any]):
        # Appends a single result row (as a JSON object) to a JSONL file.
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def analyze_failure(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Structured analysis of a single failed experiment run.

        # Extract the reasoning trace from the row.
        trace = row.get("trace", []) or []

        # Find out at which steps the LLM produced an answer.
        completed_phases: List[int] = []
        for i, step_data in enumerate(trace):
            if step_data.get("answer") not in (None, "NONE"):
                completed_phases.append(i + 1)

        return {
            "expected": row.get("expected"),
            "predicted": row.get("predicted"),
            "trim_count": row.get("trim_count", 0),
            "completed_phases": completed_phases,
            "values_dropped": self._extract_dropped_values_from_trim_events(row.get("trim_events", [])),
        }

    def _extract_dropped_values_from_trim_events(self, trim_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Method to analyse trime events and get information about dropped numeric values that may have led to reasoning failures.
        dropped_values_summary: List[Dict[str, Any]] = []
        for trim_event in trim_events:
            for chunk_detail in trim_event.get("dropped_chunk_details", []):
                for value_info in chunk_detail.get("dropped_introduced_values", []):
                    dropped_values_summary.append({
                        "value": value_info["value"],
                        "introduced_step": value_info["introduced_step"],
                        "dropped_at_step": trim_event["step"],
                        "chunk_content_start": chunk_detail["content"][:80]
                    })
        return dropped_values_summary
