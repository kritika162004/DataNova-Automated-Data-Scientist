"""
llm_client.py - the single place that talks to the LLM (Groq through the OpenAI SDK).

The client is created on the first call, so importing this file never fails when
GROQ_API_KEY is missing (tests and the data/PSO/modeling stages do not need a key).
"""
import os
import time

MODEL_NAME = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
_client = None


class LLMUnavailable(RuntimeError):
    """Raised when the LLM cannot be used (no key, or network failure after retries)."""


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMUnavailable("GROQ_API_KEY is not set. Put it in your .env file.")
        from openai import OpenAI
        _client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return _client


def call_llm(prompt: str, model: str = MODEL_NAME, max_retries: int = 4) -> str:
    """Call the LLM. Retries network problems and rate limits instead of crashing a live demo."""
    from openai import APIConnectionError, APITimeoutError, RateLimitError
    client = _get_client()
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
            return response.choices[0].message.content or ""
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_error = e
            wait = attempt * 3
            print(f"  [LLM] {type(e).__name__}, attempt {attempt}/{max_retries}, retrying in {wait}s...")
            time.sleep(wait)
    raise LLMUnavailable(f"LLM unreachable after {max_retries} attempts. Last error: {last_error}")
