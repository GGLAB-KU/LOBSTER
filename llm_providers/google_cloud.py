"""
Google Cloud LLM API client implementation.

Supports Gemini, Llama 4, Mistral, and Claude models via Vertex AI.

Environment variables:
- GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON key file (preferred)
- GOOGLE_CLOUD_PROJECT: GCP project ID (optional, auto-detected from service account)
- GOOGLE_GENAI_USE_VERTEXAI: Set to "True" to use Vertex AI (auto-set by this module)

Llama 4 models:
- llama-4-maverick-17b-128e-instruct-maas
- llama-4-scout-17b-16e-instruct-maas

Mistral models:
- mistral-medium-3

Claude models:
- claude-opus-4-5
"""

from __future__ import annotations

import json
import logging
import os
import time

import google.auth
import google.auth.transport.requests
import requests

from . import BaseLLMClient

logger = logging.getLogger(__name__)

# Llama 4 configuration
LLAMA_REGION = "us-east5"
LLAMA_ENDPOINT = f"{LLAMA_REGION}-aiplatform.googleapis.com"
LLAMA_MODEL_PREFIX = "meta/"

# Mistral configuration
MISTRAL_REGION = "us-central1"
MISTRAL_ENDPOINT = f"{MISTRAL_REGION}-aiplatform.googleapis.com"
MISTRAL_PUBLISHER = "mistralai"

# Claude configuration
CLAUDE_REGION = "us-east5"
CLAUDE_ENDPOINT = f"{CLAUDE_REGION}-aiplatform.googleapis.com"
CLAUDE_PUBLISHER = "anthropic"
CLAUDE_ANTHROPIC_VERSION = "vertex-2023-10-16"


class GoogleCloudClient(BaseLLMClient):
    """Google Cloud LLM client supporting Gemini, Llama 4, Mistral, and Claude models via Vertex AI."""
    
    def __init__(self):
        """Initialize the Google Cloud LLM client."""
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        
        try:
            from google import genai
            from google.genai.types import HttpOptions
        except ImportError:
            raise ImportError(
                "google-genai is required for Google Cloud provider. "
                "Install with: pip install google-genai"
            )
        
        self._genai = genai
        self._HttpOptions = HttpOptions
        
        # Check for service account credentials
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
                "Set it to the path of your service account JSON key file."
            )
        
        if not os.path.exists(creds_path):
            raise RuntimeError(f"Service account key file not found: {creds_path}")
        
        # Extract project ID from service account file if not set
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            try:
                with open(creds_path, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get("project_id")
                    if project_id:
                        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                        logger.info(f"Auto-detected project ID: {project_id}")
            except Exception as e:
                logger.warning(f"Could not extract project ID: {e}")
        
        self._project_id = project_id
        self._client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))
        
        # Initialize Google Cloud credentials for Llama API with proper scopes
        self._credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        logger.info(f"Initialized Google Cloud LLM client (project: {project_id})")
    
    def _is_llama_model(self, model_name: str) -> bool:
        """Check if the model is a Llama model."""
        return "llama" in model_name.lower()
    
    def _is_mistral_model(self, model_name: str) -> bool:
        """Check if the model is a Mistral model."""
        return "mistral" in model_name.lower()
    
    def _is_claude_model(self, model_name: str) -> bool:
        """Check if the model is a Claude model."""
        return "claude" in model_name.lower()
    
    def _get_access_token(self) -> str:
        """Get a valid access token for API calls."""
        # Refresh credentials if needed
        auth_req = google.auth.transport.requests.Request()
        self._credentials.refresh(auth_req)
        return self._credentials.token
    
    def _call_llama(
        self,
        prompt: str,
        model_name: str,
        max_retries: int = 3,
        retry_delay: int = 10,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> str:
        """Call Llama model via OpenAI-compatible API."""
        # Ensure model name has the meta/ prefix
        if not model_name.startswith(LLAMA_MODEL_PREFIX):
            model_name = f"{LLAMA_MODEL_PREFIX}{model_name}"
        
        url = (
            f"https://{LLAMA_ENDPOINT}/v1/projects/{self._project_id}"
            f"/locations/{LLAMA_REGION}/endpoints/openapi/chat/completions"
        )
        
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                
                result = response.json()
                
                if "choices" not in result or not result["choices"]:
                    raise ValueError("No choices in response")
                
                content = result["choices"][0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError("Empty content in response")
                
                return content.strip()
                
            except requests.exceptions.HTTPError as e:
                is_last_attempt = attempt >= max_retries - 1
                
                if response.status_code == 429:
                    # Rate limit - retry
                    if not is_last_attempt:
                        time.sleep(retry_delay)
                        continue
                
                logger.error(f"Llama API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_last_attempt:
                    raise
                
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                
            except Exception as e:
                is_last_attempt = attempt >= max_retries - 1
                logger.error(f"Llama API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_last_attempt:
                    raise
                
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        raise RuntimeError(f"Failed to call Llama model after {max_retries} attempts")
    
    def _call_mistral(
        self,
        prompt: str,
        model_name: str,
        max_retries: int = 3,
        retry_delay: int = 10,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> str:
        """Call Mistral model via rawPredict API."""
        url = (
            f"https://{MISTRAL_ENDPOINT}/v1/projects/{self._project_id}"
            f"/locations/{MISTRAL_REGION}/publishers/{MISTRAL_PUBLISHER}"
            f"/models/{model_name}:rawPredict"
        )
        
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }
        
        # Mistral API expects content as a simple string
        # Note: Mistral requires top_p=1 when temperature=0 (greedy sampling)
        effective_top_p = 0.95 if temperature == 0.0 else top_p
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": 4096,
            "temperature": temperature,
            "top_p": effective_top_p,
            "stream": False,
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                
                result = response.json()
                
                if "choices" not in result or not result["choices"]:
                    raise ValueError("No choices in response")
                
                content = result["choices"][0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError("Empty content in response")
                
                return content.strip()
                
            except requests.exceptions.HTTPError as e:
                is_last_attempt = attempt >= max_retries - 1
                
                # Log detailed error response for debugging
                try:
                    error_detail = response.text
                    logger.error(f"Mistral API error response: {error_detail}")
                except Exception as exc:
                    logger.debug(f"Could not read error response body: {exc}")

                if response.status_code == 429:
                    # Rate limit - retry
                    if not is_last_attempt:
                        time.sleep(retry_delay)
                        continue
                
                logger.error(f"Mistral API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_last_attempt:
                    raise
                
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                
            except Exception as e:
                is_last_attempt = attempt >= max_retries - 1
                logger.error(f"Mistral API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_last_attempt:
                    raise
                
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        raise RuntimeError(f"Failed to call Mistral model after {max_retries} attempts")
    
    def _call_claude(
        self,
        prompt: str,
        model_name: str,
        max_retries: int = 3,
        retry_delay: int = 10,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> str:
        """Call Claude model via Vertex AI rawPredict API."""
        # Normalize model name - add version suffix if not present
        if "@" not in model_name:
            # Map common names to versioned names
            model_versions = {
                "claude-opus-4-5": "claude-opus-4-5@20251101",
            }
            model_name = model_versions.get(model_name, f"{model_name}@20251101")
        
        url = (
            f"https://{CLAUDE_ENDPOINT}/v1/projects/{self._project_id}"
            f"/locations/{CLAUDE_REGION}/publishers/{CLAUDE_PUBLISHER}"
            f"/models/{model_name}:rawPredict"
        )
        
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }
        
        # Claude API requires anthropic_version and different payload structure
        payload = {
            "anthropic_version": CLAUDE_ANTHROPIC_VERSION,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": 4096,
            "stream": False,
        }
        
        # Only add temperature if non-zero (Claude doesn't require top_p adjustment)
        if temperature > 0:
            payload["temperature"] = temperature
            payload["top_p"] = top_p
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                
                result = response.json()
                
                # Claude returns content as array of objects
                if "content" not in result or not result["content"]:
                    raise ValueError("No content in response")
                
                # Extract text from content array
                text_parts = []
                for content_item in result["content"]:
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                
                content = "".join(text_parts)
                if not content:
                    raise ValueError("Empty text in response")
                
                return content.strip()
                
            except requests.exceptions.HTTPError as e:
                is_last_attempt = attempt >= max_retries - 1
                
                # Log detailed error response for debugging
                try:
                    error_detail = response.text
                    logger.error(f"Claude API error response: {error_detail}")
                except Exception as exc:
                    logger.debug(f"Could not read error response body: {exc}")

                if response.status_code == 429:
                    # Rate limit - retry
                    if not is_last_attempt:
                        time.sleep(retry_delay)
                        continue
                
                logger.error(f"Claude API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_last_attempt:
                    raise
                
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                
            except Exception as e:
                is_last_attempt = attempt >= max_retries - 1
                logger.error(f"Claude API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_last_attempt:
                    raise
                
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        raise RuntimeError(f"Failed to call Claude model after {max_retries} attempts")
    
    @property
    def provider_name(self) -> str:
        return "Google Cloud (Gemini/Llama/Mistral/Claude)"
    
    def get_model_name(self) -> str | None:
        """Return the model name from GOOGLE_CLOUD_MODEL environment variable."""
        return os.getenv("GOOGLE_CLOUD_MODEL")
    
    def list_available_models(self) -> list[str]:
        """List available Gemini models."""
        try:
            models = list(self._client.models.list())
            model_names = [m.name for m in models if hasattr(m, 'name')]
            logger.info(f"Found {len(model_names)} available models")
            return model_names
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def test_connection(self, model_name: str) -> bool:
        """Test if the model is accessible with a simple request."""
        logger.info(f"Testing model connection: {model_name}")
        
        # Handle Llama models
        if self._is_llama_model(model_name):
            return self._test_llama_connection(model_name)
        
        # Handle Mistral models
        if self._is_mistral_model(model_name):
            return self._test_mistral_connection(model_name)
        
        # Handle Claude models
        if self._is_claude_model(model_name):
            return self._test_claude_connection(model_name)
        
        # Handle Gemini models
        return self._test_gemini_connection(model_name)
    
    def _test_llama_connection(self, model_name: str) -> bool:
        """Test Llama model connection."""
        try:
            response = self._call_llama(
                prompt="Say hello",
                model_name=model_name,
                max_retries=1,
                temperature=0.0,
            )
            logger.info("Llama model test successful!")
            return True
        except Exception as e:
            logger.error(f"Llama model test failed: {e}")
            return False
    
    def _test_mistral_connection(self, model_name: str) -> bool:
        """Test Mistral model connection."""
        try:
            response = self._call_mistral(
                prompt="Say hello",
                model_name=model_name,
                max_retries=1,
                temperature=0.0,
            )
            logger.info("Mistral model test successful!")
            return True
        except Exception as e:
            logger.error(f"Mistral model test failed: {e}")
            return False
    
    def _test_claude_connection(self, model_name: str) -> bool:
        """Test Claude model connection."""
        try:
            response = self._call_claude(
                prompt="Say hello",
                model_name=model_name,
                max_retries=1,
                temperature=0.0,
            )
            logger.info("Claude model test successful!")
            return True
        except Exception as e:
            logger.error(f"Claude model test failed: {e}")
            return False
    
    def _test_gemini_connection(self, model_name: str) -> bool:
        """Test Gemini model connection."""
        from google.genai.types import GenerateContentConfig
        
        test_prompt = "Say 'hello' in JSON format like: {\"response\": \"hello\"}"
        
        try:
            config = GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100,
                response_mime_type="application/json",
            )
            response = self._client.models.generate_content(
                model=model_name,
                contents=test_prompt,
                config=config,
            )
            logger.info("Gemini model test successful!")
            return True
            
        except Exception as e:
            logger.warning(f"JSON mode test failed: {e}")
            
            # Try without JSON mode
            try:
                config = GenerateContentConfig(temperature=0.0, max_output_tokens=100)
                response = self._client.models.generate_content(
                    model=model_name,
                    contents="Say hello",
                    config=config,
                )
                logger.warning("Model works without JSON mode only")
                return True
            except Exception as e2:
                logger.error(f"Model test failed: {e2}")
                return False
    
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
        """Call the LLM with retries and return the response text."""
        # Route to appropriate implementation
        if self._is_llama_model(model_name):
            return self._call_llama(
                prompt=prompt,
                model_name=model_name,
                max_retries=max_retries,
                retry_delay=retry_delay,
                temperature=temperature,
                top_p=top_p,
            )
        
        if self._is_mistral_model(model_name):
            return self._call_mistral(
                prompt=prompt,
                model_name=model_name,
                max_retries=max_retries,
                retry_delay=retry_delay,
                temperature=temperature,
                top_p=top_p,
            )
        
        if self._is_claude_model(model_name):
            return self._call_claude(
                prompt=prompt,
                model_name=model_name,
                max_retries=max_retries,
                retry_delay=retry_delay,
                temperature=temperature,
                top_p=top_p,
            )
        
        # Gemini implementation
        return self._call_gemini(
            prompt=prompt,
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
        )
    
    def _call_gemini(
        self,
        prompt: str,
        model_name: str,
        max_retries: int = 3,
        retry_delay: int = 10,
        seed: int = 42,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> str:
        """Call Gemini model via Gen AI SDK."""
        from google.genai.types import (
            GenerateContentConfig,
            GoogleSearch,
            ThinkingConfig,
            Tool,
        )
        
        config_kwargs = {
            "temperature": temperature,
            "seed": seed,
            "top_p": top_p,
        }

        # Enable thinking mode with high level (always enabled for Gemini 3)
        if "gemini-3" in model_name.lower():
            config_kwargs["thinking_config"] = ThinkingConfig(thinking_level="high")
        
        generation_config = GenerateContentConfig(**config_kwargs)
        
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=generation_config,
                )
                
                if not response.candidates:
                    raise ValueError("No candidates in response")
                
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    if "SAFETY" in str(candidate.finish_reason).upper():
                        raise ValueError("Response blocked by safety filters")
                
                text = response.text
                if not text:
                    raise ValueError("Empty text in response")
                
                return text.strip()
                
            except Exception as e:
                is_last_attempt = attempt >= max_retries - 1
                error_code = getattr(e, 'code', None)
                
                if error_code == 429:
                    # Rate limit - just retry quietly
                    if not is_last_attempt:
                        time.sleep(retry_delay)
                        continue
                
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_last_attempt:
                    raise
                
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        raise RuntimeError(f"Failed to call Gemini model after {max_retries} attempts")
