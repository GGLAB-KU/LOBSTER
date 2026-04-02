"""
LLM Provider abstraction for review bias detection.

This module provides a unified interface for different LLM providers:
- OpenRouter (via OpenAI-compatible API)
- Google Cloud Vertex AI (Gemini models)

Usage:
    from llm_providers import create_llm_client, LLMProvider
    
    # Auto-detect provider from environment
    client = create_llm_client()
    response = client.call(prompt, model_name)
    
    # Or specify provider explicitly
    client = create_llm_client(provider=LLMProvider.GOOGLE_CLOUD)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENROUTER = "openrouter"
    GOOGLE_CLOUD = "google_cloud"


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
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
    ) -> str:
        """
        Call the LLM and return the response text.
        
        Args:
            prompt: The prompt to send to the LLM
            model_name: The model identifier
            max_retries: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            seed: Random seed for reproducibility
            timeout: Request timeout in seconds
            temperature: Sampling temperature (0.0 for deterministic)
            top_p: Top-p sampling parameter
            
        Returns:
            The response text from the LLM
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for logging."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str | None:
        """Return the model name from provider-specific environment variable."""
        pass


def create_llm_client(provider: LLMProvider | None = None) -> BaseLLMClient:
    """
    Create an LLM client for the specified provider.
    
    If provider is None, auto-detect from environment variables:
    - If GOOGLE_APPLICATION_CREDENTIALS is set, use Google Cloud (Vertex AI)
    - Otherwise, use OpenRouter
    
    Args:
        provider: The provider to use, or None for auto-detection
        
    Returns:
        An LLM client instance
    """
    if provider is None:
        # Auto-detect based on environment
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            provider = LLMProvider.GOOGLE_CLOUD
        else:
            provider = LLMProvider.OPENROUTER
    
    if provider == LLMProvider.OPENROUTER:
        from .openrouter import OpenRouterClient
        return OpenRouterClient()
    elif provider == LLMProvider.GOOGLE_CLOUD:
        from .google_cloud import GoogleCloudClient
        return GoogleCloudClient()
    else:
        raise ValueError(f"Unsupported provider: {provider}")


__all__ = ["LLMProvider", "BaseLLMClient", "create_llm_client"]

