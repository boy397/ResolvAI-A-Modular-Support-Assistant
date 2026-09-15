"""Thin, swappable LLM client. Three providers are wired up: "groq" and "gemini"
have usable free tiers but rate-limit aggressively, and "local" runs a small
instruct model on-device via transformers with no external calls and no rate
limit at all -- this is the default now, precisely to avoid the free-tier
429s that used to stall stage 4 (hundreds of sequential calls in a loop).
Add a new provider by adding one more branch to `make_llm_call`.

For the network providers, every call goes through `_with_retry`, which backs
off and retries on rate-limit errors, and `MIN_SECONDS_BETWEEN_CALLS` throttles
the loop itself so you don't even hit the limit in the first place. The local
provider skips both -- there's nothing to rate-limit and the extra sleeps
would blow the "reproduce in under 15 minutes" budget for no reason.
"""
import os
import time
import random
from typing import Callable

from resolvai import logger

MIN_SECONDS_BETWEEN_CALLS = 1.2   # raise this if you're still getting rate-limited
MAX_RETRIES = 10
BASE_BACKOFF_SECONDS = 5.0

_last_call_time = [0.0]


def _throttle():
    elapsed = time.time() - _last_call_time[0]
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time[0] = time.time()


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(s in msg for s in ("429", "rate limit", "rate_limit", "quota", "too many requests"))


def _with_retry(fn: Callable[[], str]) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        _throttle()
        try:
            return fn()
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    f"rate limited (attempt {attempt}/{MAX_RETRIES}), backing off {backoff:.1f}s: {e}"
                )
                time.sleep(backoff)
                continue
            raise
    raise RuntimeError("exhausted retries on LLM call")


def make_llm_call(provider: str, model: str) -> Callable[[str, int], str]:
    provider = provider.lower()

    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])

        def call(prompt: str, max_tokens: int = 200) -> str:
            def _do():
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()
            return _with_retry(_do)

        return call

    if provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel(model)

        def call(prompt: str, max_tokens: int = 200) -> str:
            def _do():
                resp = gemini_model.generate_content(prompt)
                return resp.text.strip()
            return _with_retry(_do)

        return call

    if provider == "local":
        return _make_local_call(model)

    raise ValueError(f"Unknown LLM provider: {provider}")


def _make_local_call(model_name: str) -> Callable[[str, int], str]:
    """CPU-friendly local instruct model via transformers. No network calls, no
    rate limit, no API key -- runs entirely on this machine. Loaded once and
    reused (see LLMClient below); the model is small on purpose (default
    Qwen2.5-0.5B-Instruct, ~1GB) so hundreds of calls on a CPU-only box still
    finish in minutes, not hours."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, os.cpu_count() or 4))

    logger.info(f"loading local model '{model_name}' (first run downloads it from Hugging Face, then caches)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

    def call(prompt: str, max_tokens: int = 200) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return call


class LLMClient:
    """Lazily-initialized singleton-ish wrapper so components don't each
    re-read env vars / re-init SDK clients."""

    _call = None
    _provider = None
    _model = None

    @classmethod
    def get(cls, provider: str, model: str) -> Callable[[str, int], str]:
        if cls._call is None or cls._provider != provider or cls._model != model:
            logger.info(f"initializing LLM client: provider={provider} model={model}")
            cls._call = make_llm_call(provider, model)
            cls._provider, cls._model = provider, model
        return cls._call
