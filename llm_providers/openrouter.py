"""
OpenRouter LLM client implementation.

Uses the OpenAI-compatible API provided by OpenRouter.

Environment variables:
- OPENROUTER_API_KEY: API key for OpenRouter
- OPENROUTER_API_KEY_FILE: Path to file containing API key (fallback)
- OPENROUTER_SITE_URL: Optional site URL for attribution
- OPENROUTER_APP_TITLE: Optional app title for attribution
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from . import BaseLLMClient

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_api_key() -> str:
    """Retrieve OpenRouter API key from environment or file."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    key_file = os.getenv("OPENROUTER_API_KEY_FILE")
    if key_file and Path(key_file).exists():
        return Path(key_file).read_text(encoding="utf-8").strip()

    raise RuntimeError("OPENROUTER_API_KEY is not set.")


class OpenRouterClient(BaseLLMClient):
    """OpenRouter LLM client using OpenAI-compatible API."""
    
    def __init__(self):
        """Initialize the OpenRouter client."""
        default_headers = {}
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_title = os.getenv("OPENROUTER_APP_TITLE")

        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_title:
            default_headers["X-Title"] = app_title

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=_get_api_key(),
            default_headers=default_headers or None,
        )
    
    @property
    def provider_name(self) -> str:
        return "OpenRouter"
    
    def get_model_name(self) -> str | None:
        """Return the model name from OPENROUTER_MODEL environment variable."""
        return os.getenv("OPENROUTER_MODEL")
    
    def call(
        self,
        prompt: str,
        model_name: str,
        max_retries: int = 3,
        retry_delay: int = 10,
        seed: int = 42,
        timeout: int = 60,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        json_mode: bool = True,
    ) -> str:
        """Call the LLM with retries and return the response text."""
        for attempt in range(max_retries):
            try:
                response = self._make_request(
                    prompt=prompt,
                    model_name=model_name,
                    seed=seed,
                    timeout=timeout,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    json_mode=json_mode,
                )
                
                content = response.choices[0].message.content
                if not content:
                    finish_reason = response.choices[0].finish_reason if response.choices else "no_choices"
                    refusal = getattr(response.choices[0].message, 'refusal', None) if response.choices else None
                    logger.info(
                        f"Empty content debug - finish_reason: {finish_reason}, "
                        f"refusal: {refusal}, "
                        f"model: {response.model if hasattr(response, 'model') else 'unknown'}"
                    )
                    raise ValueError(f"Empty response from LLM (finish_reason={finish_reason})")
                return content.strip()
                
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
    
    def _make_request(
        self,
        prompt: str,
        model_name: str,
        seed: int,
        timeout: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
        json_mode: bool,
    ) -> Any:
        """Make the actual API request, handling JSON mode fallback."""
        base_kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }
        
        if json_mode:
            try:
                return self._client.chat.completions.create(
                    **base_kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                msg = str(e).lower()
                if "response_format" in msg or "json" in msg or "unsupported" in msg:
                    logger.warning(
                        "Model/provider rejected JSON mode; retrying without response_format. Error: %s",
                        e,
                    )
                    return self._client.chat.completions.create(**base_kwargs)
                raise
        else:
            return self._client.chat.completions.create(**base_kwargs)

