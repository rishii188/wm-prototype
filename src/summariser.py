from typing import Literal, Optional, List, Dict, Any
import re


Mode = Literal["light", "heavy", "cognitive"]


class Summariser:
# Summarises text to condense information and store it more compactly.
# Preserves important information for logical and mathematical reasoning.

    KEYWORDS = [
        "total", "difference", "sum", "final", "remaining", "compute", "calculate", "result", "answer", "therefore",
    ]

    def summarise(
        self,
        text: str,
        mode: Mode = "light",
        pinned_goal: Optional[str] = None,
        max_tokens: Optional[int] = None,
        token_counter=None,
    ) -> str:
        if mode == "cognitive":
            return self._cognitive_summarise(
                text,
                pinned_goal=pinned_goal,
                max_tokens=max_tokens,
                token_counter=token_counter,
            )
        return self._rule_based_summarise(text, mode=mode)

    def _rule_based_summarise(self, text: str, mode: Literal["light", "heavy"]) -> str:
    # Assigns scores to each line based on numbers, common variables and operators.
    # Two modes: light (keep more important lines up to 4) and heavy (keep fewer up to 2).

        split_lines = re.split(r"[.\n]+", text)
        lines = [line.strip() for line in split_lines if line.strip()]

        key_lines: List[str] = []

        for line in lines:
            contains_number = bool(re.search(r"\d", line))
            contains_variable = bool(re.search(r"\b[xynab]\b", line))
            contains_operators = any(sym in line for sym in ["+", "-", "*", "/", "=", "<", ">"])

            score = 0
            if contains_number:
                score += 2
            if contains_variable:
                score += 1
            if contains_operators:
                score += 1

            if score > 0:
                key_lines.append(line)

        if not key_lines:
            key_lines = lines[:2]

        if mode == "light":
            max_lines_to_keep = min(4, len(key_lines))
        else:
            max_lines_to_keep = min(2, len(key_lines))

        chosen_lines = key_lines[:max_lines_to_keep]
        summary = ". ".join(chosen_lines)
        if summary and not summary.endswith("."):
            summary += "."
        return summary if summary else (lines[0] + "." if lines else "")

    def _split_units(self, text: str) -> List[str]:
        raw = re.split(r"\n+|(?<!\d)\.(?!\d)", text)
        return [unit.strip() for unit in raw if unit and unit.strip()]

    def _estimate_rst_role(self, unit: str) -> str:
        has_number = bool(re.search(r"\d", unit))
        has_operator = any(sym in unit for sym in ["+", "-", "*", "/", "=", "%"])
        has_keyword = any(keyword in unit.lower() for keyword in self.KEYWORDS)
        return "nucleus" if (has_number or has_operator or has_keyword) else "satellite"

    def _segment_events(self, units: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        events: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []

        for unit in units:
            text = unit["text"].lower()
            boundary = text.startswith("step ") or text.startswith("answer:") or "therefore" in text
            if boundary and current:
                events.append(current)
                current = []
            current.append(unit)

        if current:
            events.append(current)
        return events

    def _apply_macro_rules(self, units: List[Dict[str, Any]]) -> List[str]:
        compressed: List[str] = []
        seen = set()

        for unit in units:
            text = re.sub(r"\b(first|next|then|now|so|therefore|thus|let's)\b", "", unit["text"], flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip(" :,-")
            if not text:
                continue

            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
            if numbers and not any(sym in text for sym in ["+", "-", "*", "/", "=", "%"]):
                text = f"values: {' '.join(numbers)}"

            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            compressed.append(text)

        return compressed

    def _relevance(self, unit: str, goal: str) -> float:
        unit_terms = set(re.findall(r"[a-zA-Z]+", unit.lower()))
        goal_terms = set(re.findall(r"[a-zA-Z]+", goal.lower()))
        overlap = len(unit_terms.intersection(goal_terms))
        number_bonus = 1.0 if re.search(r"\d", unit) else 0.0
        operator_bonus = 0.5 if any(sym in unit for sym in ["+", "-", "*", "/", "=", "%"]) else 0.0
        return overlap + number_bonus + operator_bonus

    def _redundancy(self, unit: str, selected: List[str]) -> float:
        if not selected:
            return 0.0
        unit_terms = set(re.findall(r"[a-zA-Z0-9]+", unit.lower()))
        if not unit_terms:
            return 0.0
        max_overlap = 0.0
        for other in selected:
            other_terms = set(re.findall(r"[a-zA-Z0-9]+", other.lower()))
            if not other_terms:
                continue
            jaccard = len(unit_terms.intersection(other_terms)) / len(unit_terms.union(other_terms))
            max_overlap = max(max_overlap, jaccard)
        return max_overlap

    def _mmr_select(self, candidates: List[str], goal: str, max_items: int) -> List[str]:
        remaining = list(candidates)
        selected: List[str] = []

        while remaining and len(selected) < max_items:
            best_idx = 0
            best_score = float("-inf")
            for index, candidate in enumerate(remaining):
                rel = self._relevance(candidate, goal)
                red = self._redundancy(candidate, selected)
                score = 0.7 * rel - 0.3 * red
                if score > best_score:
                    best_score = score
                    best_idx = index
            selected.append(remaining.pop(best_idx))

        return selected

    def _fit_to_budget(
        self,
        lines: List[str],
        max_tokens: Optional[int],
        token_counter,
    ) -> List[str]:
        if max_tokens is None or max_tokens <= 0:
            return lines

        fitted: List[str] = []
        for line in lines:
            candidate = fitted + [line]
            text = ". ".join(candidate)
            if token_counter is not None:
                length = token_counter.count(text)
            else:
                length = len(text.split())
            if length <= max_tokens:
                fitted.append(line)
            else:
                break

        return fitted if fitted else lines[:1]

    def _cognitive_summarise(
        self,
        text: str,
        pinned_goal: Optional[str],
        max_tokens: Optional[int],
        token_counter,
    ) -> str:
        units_text = self._split_units(text)
        if not units_text:
            return ""

        units = [{"text": unit, "role": self._estimate_rst_role(unit)} for unit in units_text]
        events = self._segment_events(units)

        macro_props: List[str] = []
        for event in events:
            core = [unit for unit in event if unit["role"] == "nucleus"] or event
            macro_props.extend(self._apply_macro_rules(core))

        goal_text = pinned_goal or ""
        max_items = 4
        selected = self._mmr_select(macro_props, goal_text, max_items=max_items)
        fitted = self._fit_to_budget(selected, max_tokens=max_tokens, token_counter=token_counter)

        summary = ". ".join(fitted).strip()
        if summary and not summary.endswith("."):
            summary += "."
        return summary if summary else (units_text[0] + ".")