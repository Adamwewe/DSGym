"""
LiteLLM backend for DSGym agents.

Supports various API providers through LiteLLM.
"""

import os
import time
from typing import List, Dict, Any, Optional

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None

from .base import BaseBackend


class LiteLLMBackend(BaseBackend):
    """LiteLLM backend for calling various API providers."""
    
    def __init__(
        self, 
        model_name: str, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        temperature: float = 0.0, 
        top_p: float = 1.0, 
        max_tokens: int = 1524, 
        timeout: int = 180, 
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize LiteLLM backend.
        
        Args:
            model_name: Model name (e.g., 'gpt-4', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')
            api_key: API key (uses environment variable if None)
            base_url: Base URL for API
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            **kwargs: Additional configuration
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for LiteLLMBackend. Please install: pip install litellm")
        
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.is_reasoning_model = self._is_reasoning_model(model_name)
        
        # Set generation parameters based on model type
        if self.is_reasoning_model:
            # Reasoning models don't support temperature/top_p and use max_completion_tokens
            # For reasoning models, we need much higher token limits since they use tokens for internal reasoning
            # reasoning_max_tokens = max(max_tokens * 4, 4096)  # At least 4x the requested tokens, minimum 4096
            # self.generation_params = {"max_completion_tokens": reasoning_max_tokens}
            self.generation_params = {}
        else:
            # Only include top_p if explicitly set to non-default value or when temperature is 0
            # This avoids parameter conflicts with models that don't support both
            self.generation_params = {
                "temperature": temperature, 
                "max_tokens": max_tokens
            }
            if self.model_name.startswith("gpt-5"):
                self.generation_params["reasoning_effort"] = "medium"
            if top_p != 1.0:
                self.generation_params["top_p"] = top_p
        
        self._setup_litellm()
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if model is a reasoning model (o1, o3, o4, gpt-5 series)."""
        reasoning_model_prefixes = ["o1-", "o3-", "o4-", "gpt-5"]
        return any(model_name.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _normalize_api_key(self, key: Optional[str]) -> Optional[str]:
        """Normalize API keys from env/CLI to avoid malformed auth headers."""
        if key is None:
            return None

        normalized = key.strip().strip('"').strip("'")
        if normalized.lower().startswith("bearer "):
            normalized = normalized[7:].strip()

        return normalized or None
    
    def _setup_litellm(self):
        """Resolve and validate the API key; do NOT write it to os.environ.

        Writing provider keys into os.environ at constructor time pollutes the
        process environment globally.  In multi-threaded evaluation runs this
        causes different LiteLLMBackend instances (e.g. OpenAI + Together) to
        overwrite each other’s keys.  We keep the key on self.api_key and pass
        it per-call instead.
        """
        self.api_key = self._normalize_api_key(self.api_key)

        if not self.api_key:
            # Fall back to the provider’s conventional env var.
            if self.model_name.startswith("together_ai/"):
                self.api_key = self._normalize_api_key(os.environ.get("TOGETHER_API_KEY"))
                if not self.api_key:
                    raise ValueError("TOGETHER_API_KEY environment variable is required for Together AI models")
            elif self.model_name.startswith("gpt-") or self._is_reasoning_model(self.model_name):
                self.api_key = self._normalize_api_key(os.environ.get("OPENAI_API_KEY"))
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
            elif self.model_name.startswith("claude-") or self.model_name.startswith("anthropic/"):
                self.api_key = self._normalize_api_key(os.environ.get("ANTHROPIC_API_KEY"))
                if not self.api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic Claude models")

        # Configure LiteLLM settings (these are safe module-level flags).
        litellm.drop_params = True   # Drop unsupported parameters silently
        litellm.set_verbose = False  # Suppress verbose logging
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using LiteLLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if messages is None:
            raise RuntimeError("Messages is None in LiteLLM generate - this indicates an upstream error")
        elif not isinstance(messages, list):
            raise TypeError(f"Messages should be a list, got {type(messages)}")
        
        # Merge generation parameters
        params = self.generation_params.copy()
        params.update(kwargs)
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Pass api_key and api_base per-call so each LiteLLMBackend
                # instance is fully isolated — no global os.environ mutation.
                call_kwargs: dict = dict(
                    model=self.model_name,
                    messages=messages,
                    timeout=self.timeout,
                )
                if self.api_key:
                    call_kwargs["api_key"] = self.api_key
                if self.base_url:
                    call_kwargs["api_base"] = self.base_url
                call_kwargs.update(params)

                response = litellm.completion(**call_kwargs)
                content = response.choices[0].message.content
                return content if content is not None else ""

            except litellm.ContextWindowExceededError:
                # Retrying won't help — the payload is too large. Raise immediately.
                raise
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    raise e
                else:
                    # Wait and retry
                    wait_time = 2 ** attempt
                    print(f"LiteLLM API call failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
        
        raise RuntimeError(f"LiteLLM API call failed after {self.max_retries} attempts")
    
    def is_api_based(self) -> bool:
        """LiteLLM backend is always API-based."""
        return True
    
    def apply_chat_template(self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False) -> Optional[str]:
        """
        LiteLLM handles chat templating internally.
        
        Returns:
            None (LiteLLM handles templating automatically)
        """
        return None
