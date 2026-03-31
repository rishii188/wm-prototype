from __future__ import annotations

import random
import re
from typing import List

from src.buffer import Buffer
from src.summariser import Summariser


class RecencyPolicy:
# This forgetting policy works by dropping the oldest droppable chunks first when bufffer is full.
    def apply(self, buffer: Buffer):
        if not buffer.is_buffer_full():
            return  # Nothing happens if buffer is not full.
        buffer.trim_until_fits()


class FullContextPolicy:
# Baseline policy: keep all chunks (no active forgetting/compression).
    def apply(self, buffer: Buffer):
        return


class SummarisingRecencyPolicy:
# This forgetting policy works by summarising the oldest droppable chunks when buffer is full and replacing the original chunks with their summary.
# If the buffer is still full after summarisation, recency policy is applied to drop the oldest chunks.
    def __init__(self, summariser: Summariser, mode: str = "light", summarisation_window: int = 3):
        self.summariser = summariser
        self.mode = mode
        self.summarisation_window = summarisation_window

    def apply(self, buffer: Buffer):
        if not buffer.is_buffer_full():
            return
        
        if len(buffer.chunks) <= 1:
            buffer.trim_until_fits()
            return
        
        start_index = 1 if buffer.has_pinned_problem() else 0 # Grab the start index for droppable chunks.
        
        if len(buffer.chunks) <= start_index: # There are no droppable chunks.
            buffer.trim_until_fits()
            return

        # Stating the bounds of the summarisation window (the window of chunks to be summarised).
        # Targetting the oldest droppable chunks.
        end_index = min(start_index + self.summarisation_window, len(buffer.chunks) - 1)

        # Edge case: if there is only one or no chunks to summarise, skip to recency trimming.
        if end_index <= start_index:
            buffer.trim_until_fits()
            return

        # Grabbing the text from the selected window of chunks so we can summarise it.
        text_to_summarise = "\n".join(buffer.chunks[start_index:end_index])

        pinned_goal = buffer.chunks[0] if buffer.has_pinned_problem() else ""
        window_tokens = buffer.token_counter.count(text_to_summarise)
        if self.mode == "cognitive":
            summary_budget = max(12, window_tokens // 2)
        elif self.mode == "heavy":
            summary_budget = max(10, window_tokens // 3)
        else:
            summary_budget = max(14, (window_tokens * 2) // 3)

        summary = self.summariser.summarise(
            text_to_summarise,
            mode=self.mode,
            pinned_goal=pinned_goal,
            max_tokens=summary_budget,
            token_counter=buffer.token_counter,
        )

        # Reconstructing the buffer 
        new_chunks: List[str] = []
        if buffer.has_pinned_problem():
            new_chunks.append(buffer.chunks[0]) # Adding summary chunk.
        new_chunks.append(summary)
        new_chunks.extend(buffer.chunks[end_index:]) # Add the rest.
        buffer.chunks = new_chunks

        if buffer.is_buffer_full():
            buffer.trim_until_fits()


class RandomPolicy:
# Random forgetting policy using RNG to drop chunnks.

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def apply(self, buffer: Buffer):
        if not buffer.is_buffer_full():
            return

        while buffer.is_buffer_full() and len(buffer.chunks) > 0:
            # Finding the range of droppable chunks.
            start_index = 1 if buffer.has_pinned_problem() else 0
            
            if len(buffer.chunks) <= start_index:
                break
            
            # RNG time!
            index = self.rng.randint(start_index, len(buffer.chunks) - 1)
            buffer.drop_at(index)


class ImportancePolicy:
    # Importance-based forgetting policy.
    # Uses heuristics to score a score of importance for each chunk and drops least scored chunk when buffer is full.
    # Points for a chunk:
    # +2 points if a chunk has numbers as this is important for maths problems.
    # +2 points if a chunk has maths operators (+, -, *, /, =).
    # +1 point if a chunk has keywords to do with solving problems.

    KEYWORDS = [
        "total", "difference", "sum", "final", "remaining",
        "compute", "calculate", "result"
    ]

    def score(self, text: str) -> int:
        score = 0
        if re.search(r"\d", text): # Check for numbers.
            score += 2
        if any(operator in text for operator in ["+", "-", "*", "/", "="]):
            score += 2
        if any(keyword in text.lower() for keyword in self.KEYWORDS):
            score += 1
        return score

    def apply(self, buffer: Buffer):
        if not buffer.is_buffer_full():
            return

        while buffer.is_buffer_full() and len(buffer.chunks) > 0:
            start_index = 1 if buffer.has_pinned_problem() else 0
            
            if len(buffer.chunks) <= start_index:
                break

            # SCORE TIME!!!
            scored_chunks = [(i, self.score(buffer.chunks[i])) for i in range(start_index, len(buffer.chunks))]
            
            # Find the loser chunk.
            # If two losers, drop the first one found.
            index, _ = min(scored_chunks, key=lambda x: x[1])
            buffer.drop_at(index)
