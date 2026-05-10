"""
Centralised AutoGen / LLM configuration helpers.

Reads model settings from environment variables so they can be overridden
without touching code.
"""
from __future__ import annotations

import os

from autogen_ext.models.openai import OpenAIChatCompletionClient


def resolve_base_model_name() -> str:
    """Resolve the model name for AutoGen / mini-swe-agent in priority order.

    1. ``MAS_EVAL_BASE_MODEL`` — the value injected by the evaluation pipeline
       from ``study.toml``'s ``base_model`` field. This is the source of truth
       under the harness, so editing the toml takes effect without touching
       every ``.env``. Only ever set when running through the pipeline.
    2. ``AUTOGEN_MODEL`` — legacy per-process / ``.env`` override. Still honored
       when running outside the pipeline.
    3. ``"gpt-4o"`` — fallback default.
    """
    return (
        os.environ.get("MAS_EVAL_BASE_MODEL")
        or os.environ.get("AUTOGEN_MODEL")
        or "gpt-4o"
    )


def get_model_client() -> OpenAIChatCompletionClient:
    """Return a configured OpenAI model client for AutoGen agents."""
    return OpenAIChatCompletionClient(
        model=resolve_base_model_name(),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
