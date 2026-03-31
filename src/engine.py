from __future__ import annotations

from typing import Dict, Any, List, Optional
import re
import time

from src.buffer import Buffer


class ReasoningEngine:
# This is where the multi-step reasoning process happens with the LLM under constraints.
# Prompts LLM iteratively, processes responses, updates buffer and applies forgetting policy.

    def __init__(self, llm, buffer: Buffer, policy, max_steps: int = 8, min_steps_before_done: int = 3):
        self.llm = llm
        self.buffer = buffer
        self.policy = policy
        self.max_steps = max_steps
        self.min_steps_before_done = min_steps_before_done

    def build_prompt(self) -> str:
        instructions = (
            "Solve the problem step-by-step.\n"
            "At each step, output EXACTLY in this format:\n"
            "THOUGHT: <your reasoning>\n"
            "ANSWER: <final answer or NONE>\n"
            "DONE: yes/no\n"
            "Rules:\n"
            "- Put the full reasoning after THOUGHT: (it can be multi-line)\n"
            "- ANSWER must be a single short string/number\n"
            "- The 'DONE: yes' signal should only be given when you are confident in your final answer.\n"
            f"- Do not output 'DONE: yes' before step {self.min_steps_before_done}."
        )
        return instructions + "\n\n" + self.buffer.get_view()

    def parse_step(self, output: str) -> Dict[str, Any]:
        lines = output.splitlines()

        thought_lines: List[str] = []
        answer: Optional[str] = None
        done: bool = False

        # Parsing logic to get thought, answer and done.
        current: Optional[str] = None  # "thought" | "answer" | "done" | None

        for original_line in lines:
            line = original_line.strip()
            lower_line = line.lower()
            # Check for though section.
            if lower_line.startswith("thought:"):
                current = "thought"
                thought_lines = [line[len("thought:"):].strip()]
                continue
            # Check for answer section.
            if lower_line.startswith("answer:"):
                current = "answer"
                og_answer = line[len("answer:"):].strip()
                answer = None if og_answer.upper() == "NONE" else og_answer
                continue
            # Check for done section.
            if lower_line.startswith("done:"):
                current = "done"
                done_value = line[len("done:"):].strip().lower().strip(".")
                done = (done_value == "yes" or done_value == "true")
                continue

            # If it's still thinking and the line isn't empty, keep adding it to thought.
            if current == "thought" and line:
                thought_lines.append(line)

        # Join thought lines, filtering out any empty strings that might come from parsing.
        parsed_thought = "\n".join([t for t in thought_lines if t]).strip()

        # Make sure DONE is yes to finalise answer.
        return {"thought": parsed_thought, "answer": answer, "done": done, "raw_output": output}

    def run(self, problem: str) -> Dict[str, Any]:
    # This executes the multi-step reasoning process for 1 problem.
    # Logs answer, trace of all steps, no. of steps, every memory trim event, number of times buffer was trimmed and whether it is done.
 
        # Initialise the buffer.
        self.buffer.chunks = []
        self.buffer.add("PROBLEM: " + problem)

        # Initialis logging.
        trace: List[Dict[str, Any]] = []
        trim_events: List[Dict[str, Any]] = []
        trim_count: int = 0
        completed_all_phases: bool = False

        seen_values: Dict[str, int] = {}

        for step in range(1, self.max_steps + 1):
            # Building the prompt using current buffer content.
            prompt = self.build_prompt()

            # Calling the LLM and measuring response time.
            start_time = time.perf_counter()
            og_llm_output = self.llm.complete(prompt)
            end_time = time.perf_counter()

            # Parse the LLM's output for thought, answer, and done status.
            step_data = self.parse_step(og_llm_output)

            step_data.setdefault("expected_phase", None) # Multi-step problems
            step_data.setdefault("values", None)
            step_data.setdefault("ready", False)

            # Record the time taken for the step.
            try:
                response_time = getattr(self.llm, "get_last_response_time", lambda: None)()
            except Exception:
                response_time = None
            step_data["step_duration"] = response_time if response_time is not None else (end_time - start_time)

            # Record the token usage for this step.
            try:
                token_usage = getattr(self.llm, "get_last_usage", lambda: None)()
            except Exception:
                token_usage = None

            step_usage: Optional[int] = None
            if token_usage is not None:
                if isinstance(token_usage, dict):
                    step_usage = int(token_usage.get("total_tokens", token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0)))
                else:
                    try:
                        step_usage = int(getattr(token_usage, "total_tokens", None) or getattr(token_usage, "prompt_tokens", 0) + getattr(token_usage, "completion_tokens", 0))
                    except Exception:
                        step_usage = None
            step_data["step_usage"] = step_usage

            # Find numbers
            current_step_numbers: List[str] = []
            for source_text in (step_data.get("thought"), step_data.get("answer")):
                if source_text:
                    current_step_numbers.extend(re.findall(r"[-+]?\d+(?:\.\d+)?", str(source_text)))

            new_values: List[str] = []
            for value in current_step_numbers:
                if value not in seen_values:
                    seen_values[value] = step
                    new_values.append(value)

            step_data["introduced_values"] = new_values

            trace.append(step_data) # Add the detailed step data to the trace.

            # Update the buffer (thought + answer).
            thought_content = step_data["thought"] if step_data["thought"] else "(no thought parsed)"
            answer_content = step_data["answer"] if step_data["answer"] is not None else "NONE"
            self.buffer.add(f"STEP {step}:\nTHOUGHT: {thought_content}\nANSWER: {answer_content}")

            # Save buffer state
            tokens_before = self.buffer.token_length()
            buffer_chunks_before_policy = list(self.buffer.chunks)

            # Forgetting policy time!
            self.policy.apply(self.buffer)

            # Save state again.
            tokens_after = self.buffer.token_length()
            buffer_chunks_after_policy = list(self.buffer.chunks)

            # Memory trimming events.
            # Trims happen if tokens decreased or chunks were removed.
            if tokens_after < tokens_before or \
               len(buffer_chunks_after_policy) < len(buffer_chunks_before_policy):
                
                chunks_dropped_content = [c for c in buffer_chunks_before_policy if c not in buffer_chunks_after_policy]
                
                dropped_chunk_details: List[Dict[str, Any]] = []
                for i, chunk_content in enumerate(buffer_chunks_before_policy):
                    if chunk_content not in buffer_chunks_after_policy:
                        trimmed_values_info: List[Dict[str, Any]] = []
                        numbers_in_chunk = re.findall(r"[-+]?\d+(?:\.\d+)?", str(chunk_content))
                        for num_value in numbers_in_chunk:
                            if num_value in seen_values:
                                trimmed_values_info.append({
                                    "value": num_value,
                                    "introduced_step": seen_values[num_value]
                                })

                        dropped_chunk_details.append({
                            "original_index": i, # The index of the chunk before dropping
                            "content": chunk_content,
                            "dropped_introduced_values": trimmed_values_info
                        })

                # Full trim event.
                trim_event = {
                    "step": step,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "chunks_dropped_content": chunks_dropped_content, # For backward compatibility/simplicity
                    "dropped_chunk_details": dropped_chunk_details, # Detailed info on what was dropped
                    "buffer_state_before": buffer_chunks_before_policy,
                    "buffer_state_after": buffer_chunks_after_policy,
                }
                trim_events.append(trim_event)
                trim_count += 1

            # Final check for completion.
            # The LLM must say DONE:yes.
            if step_data["done"] and step >= self.min_steps_before_done:
                completed_all_phases = True
                return {
                    "answer": step_data["answer"],
                    "trace": trace,
                    "steps": step,
                    "trim_events": trim_events,
                    "trim_count": trim_count,
                    "completed_all_phases": completed_all_phases,
                }

        # If max_steps is reached without DONE:yes.
        return {
            "answer": None, # As it never finished.
            "trace": trace,
            "steps": self.max_steps,
            "trim_events": trim_events,
            "trim_count": trim_count,
            "completed_all_phases": completed_all_phases,
        }
