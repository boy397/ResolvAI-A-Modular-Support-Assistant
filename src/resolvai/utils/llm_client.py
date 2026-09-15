"""Thin, swappable LLM client. Only two providers are wired up on purpose:
both have usable free tiers, which matters for a project with no API budget.
Add a new provider by adding one more branch to `make_llm_call`.
"""
import os
from typing import Callable
from dotenv import load_dotenv

load_dotenv()  # load .env file so API keys are available without manual sourcing

from resolvai import logger


def make_llm_call(provider: str, model: str) -> Callable[[str, int], str]:
    provider = provider.lower()

    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)

        def call(prompt: str, max_tokens: int = 200) -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()

        return call

    if provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel(model)

        def call(prompt: str, max_tokens: int = 200) -> str:
            resp = gemini_model.generate_content(prompt)
            return resp.text.strip()

        return call

    raise ValueError(f"Unknown LLM provider: {provider}")


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
