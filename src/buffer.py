from typing import List
from src.tokenizer import TokenCounter


class Buffer:
# This is a working-memory buffer which is limited by token count. 
# Stores information in chunks of text and pins important information.
    def __init__(self, max_tokens: int, token_counter: TokenCounter):
        self.max_tokens = max_tokens
        self.token_counter = token_counter
        self.chunks: List[str] = []

    def add(self, text: str):
        # Adds a new text chunk to the end of the buffer.
        self.chunks.append(text)

    def get_view(self) -> str:
        # Joins all chunks in the buffer, returns it as a single string separated by newlines.
        return "\n".join(self.chunks)

    def token_length(self) -> int:
        return self.token_counter.count(self.get_view())

    def is_buffer_full(self) -> bool:
        return self.token_length() > self.max_tokens

    def has_pinned_problem(self) -> bool:
        # A pinned problem is always the first chunk if it is there.
        return len(self.chunks) > 0 and self.chunks[0].startswith("PROBLEM:")

    def drop_at(self, index: int):
        # Removes the chunk at the index specified.
        # The first chunk is never removed if it is a pinned problem statement.
        if not self.chunks or index == 0 and self.has_pinned_problem():
            return
        if 0 <= index < len(self.chunks):
            self.chunks.pop(index)

    def drop_oldest(self, n: int = 1):
        # Removes the n oldest chunks from the buffer. 
        # But it will never drop a pinned problem statement.
        for _ in range(n):
            if not self.chunks or self.has_pinned_problem() and len(self.chunks) < 2:
                return
            drop_index = 1 if self.has_pinned_problem() else 0
            self.chunks.pop(drop_index)

    def trim_until_fits(self):
        # Keep removing the oldest chunks that can be dropped until buffer is within its max_tokens capacity.
        # This is the mechanism for recency-based forgetting.
        # There is an edge case where the pinned problem chunk itself can be more than max_tokens, the overflow would not be resolved.
        # But most GSM8K problems have short problem statements, so this should not be an issue.
        while self.is_buffer_full() and len(self.chunks) > 0:
            current = len(self.chunks) # Stores current number of chunks before trying to drop.
            self.drop_oldest(1)
            if len(self.chunks) == current: # Nothing was dropped, exit to avoid infinite loop.
                break
