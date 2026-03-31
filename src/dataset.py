import json
import math
import re
from typing import List, Dict, Any, Optional


def load_multiple_datasets(paths: List[str]) -> List[Dict[str, Any]]:
# Reads multiple JSONL files and combines their contents into a single list of tasks.
# Each line in the files is expected to be a JSON object representing a task.
    tasks = []
    
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        task = json.loads(line)
                    except json.JSONDecodeError as error:
                        print(f"[WARN] Skipping malformed task JSONL line in {path}:{line_number} ({error})")
                        continue
                    if not isinstance(task, dict):
                        print(f"[WARN] Skipping non-object task JSONL line in {path}:{line_number}")
                        continue
                    tasks.append(task)
    
    return tasks


def _extract_gsm8k_final(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    return None


def _extract_last_number(text: str) -> Optional[str]:
    cleaned = text.replace(",", " ")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not numbers:
        return None
    return numbers[-1]


def _normalise_scalar(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    gsm_final = _extract_gsm8k_final(text)
    if gsm_final is not None:
        text = gsm_final
    else:
        first_line = text.splitlines()[0].strip()
        maybe_number = _extract_last_number(first_line)
        text = maybe_number if maybe_number is not None else first_line

    text = text.strip().replace("$", "")

    try:
        number = float(text)
        if math.isfinite(number):
            if number.is_integer():
                return str(int(number))
            return ("{:.10f}".format(number)).rstrip("0").rstrip(".")
    except Exception:
        pass

    return text.lower()


def answers_match(predicted: Optional[Any], actual_answer: Optional[Any]) -> bool:
    # Checks if a predicted answer matches the actual answer from the dataset.

    predicted_norm = _normalise_scalar(predicted)
    actual_norm = _normalise_scalar(actual_answer)

    if predicted_norm is None or actual_norm is None:
        return predicted_norm == actual_norm

    try:
        pred_num = float(predicted_norm)
        actual_num = float(actual_norm)
        return abs(pred_num - actual_num) <= 1e-6
    except Exception:
        return predicted_norm == actual_norm
    
