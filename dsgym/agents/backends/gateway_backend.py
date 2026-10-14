"""Backend for the Invisible AI Coding Gateway's OpenAI-compatible API."""

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import BaseBackend


GATEWAY_HOST = "api.ai-coding-gateway.inv.systems"
GATEWAY_BASE_URL = f"https://{GATEWAY_HOST}/v1"
USER_AGENT = "dsgym-gateway/1.0"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


class GatewayError(RuntimeError):
    """Raised when a gateway request cannot be completed or decoded."""


def _route_model(model_name: str) -> str:
    """Normalize a bare or LiteLLM-prefixed model name for the gateway."""
    model = model_name if "/" in model_name else f"anthropic/{model_name}"
    model = model.removeprefix("openai/")
    if not _MODEL_ID_RE.fullmatch(model):
        raise ValueError(f"Invalid gateway model name: {model_name!r}")
    return model


class GatewayBackend(BaseBackend):
    """Send chat-completion requests through the AI Coding Gateway.

    Set ``AI_CODING_GATEWAY_API_KEY`` and ``AI_CODING_GATEWAY_PROJECT`` in
    the process environment. The project value is sent as the required
    ``X-Project`` header on every request.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1524,
        timeout: int = 180,
        max_retries: int = 3,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get("AI_CODING_GATEWAY_API_KEY")
        self.project = project or os.environ.get("AI_CODING_GATEWAY_PROJECT")
        if not self.api_key:
            raise ValueError("AI_CODING_GATEWAY_API_KEY environment variable is required")
        if not self.project:
            raise ValueError("AI_CODING_GATEWAY_PROJECT environment variable is required")
        if re.search(r"[\x00-\x1f\x7f]", self.project):
            raise ValueError("AI_CODING_GATEWAY_PROJECT must not contain control characters")
        if re.search(r"[\x00-\x1f\x7f]", self.api_key):
            raise ValueError("AI_CODING_GATEWAY_API_KEY must not contain control characters")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self.routed_model = _route_model(model_name)
        self.timeout = timeout
        self.max_retries = max_retries
        self.generation_params: Dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        if top_p != 1.0:
            self.generation_params["top_p"] = top_p
        self.last_usage: Dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate a response using the gateway's OpenAI-compatible endpoint."""
        if not isinstance(messages, list):
            raise TypeError(f"Messages should be a list, got {type(messages)}")
        if {"model", "messages"} & kwargs.keys():
            raise GatewayError("model and messages cannot be overridden in request parameters")

        params = {**self.generation_params, **kwargs}
        payload = json.dumps({
            "model": self.routed_model,
            "messages": messages,
            **params,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{GATEWAY_BASE_URL}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Project": self.project,
                "User-Agent": USER_AGENT,
            },
        )

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return self._parse_response(result)
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                message = (
                    f"gateway HTTP {error.code}: "
                    f"{body.replace(self.api_key, '[REDACTED]')[:2000]}"
                )
                if error.code < 500 or attempt == self.max_retries - 1:
                    raise GatewayError(message) from error
            except urllib.error.URLError as error:
                if attempt == self.max_retries - 1:
                    raise GatewayError(f"gateway request failed: {error.reason}") from error
            except json.JSONDecodeError as error:
                raise GatewayError(f"gateway returned non-JSON response: {error}") from error

            time.sleep(2**attempt)

        raise GatewayError(f"gateway request failed after {self.max_retries} attempts")

    def _parse_response(self, response: Any) -> str:
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GatewayError(f"unexpected gateway response shape: {response}") from error
        if not isinstance(content, str):
            raise GatewayError("gateway response has no text content")

        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        self.last_usage = {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }
        return content

    def is_api_based(self) -> bool:
        """The gateway is a remote API backend."""
        return True

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> Optional[str]:
        """The gateway accepts chat messages and handles templating remotely."""
        return None
