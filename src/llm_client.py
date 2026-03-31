import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, Any
import time

class LLMClient:

    def __init__(self, model: str = "gpt-4o-mini"):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError()

        request_timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

        self.client = OpenAI(api_key=api_key, timeout=request_timeout, max_retries=max_retries)
        self.model = model
        self.last_api_usage: Optional[Dict[str, Any]] = None
        self.last_response_time: Optional[float] = None

    def complete(self, prompt: str) -> str:
 
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            end_time = time.perf_counter()
            self.last_response_time = end_time - start_time

            usage = getattr(response, "usage", None)
            if usage is None and isinstance(response, dict):
                usage = response.get("usage")
            self.last_api_usage = usage

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during LLM completion: {e}")
            raise

    def get_last_usage(self) -> Optional[Dict[str, Any]]:
        return self.last_api_usage

    def get_last_response_time(self) -> Optional[float]:
        return self.last_response_time
