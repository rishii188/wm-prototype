from typing import List

import tiktoken

class TokenCounter:
# Counts and manages tokens using tiktoken.
    def __init__(self, gptModel: str = "gpt-4o-mini"):
        self.gptModel = gptModel
        try:
            self.encoding = tiktoken.encoding_for_model(gptModel)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base") # A fallback tockenizer encoding used by OpenAI models. 

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def token_limiter(self, text: str, max: int) -> str:
        # Trims text to be under max by removing tokens from the start.
        tokens = self.encode(text)
        if len(tokens) <= max:
            return text
        trimmed_tokens = tokens[-max:]
        return self.decode(trimmed_tokens)
