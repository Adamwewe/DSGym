import io
import json
import os
import urllib.error
import unittest
from unittest.mock import Mock, patch

from dsgym.agents.backends import GatewayBackend, get_backend
from dsgym.agents.backends.gateway_backend import GatewayError


class _Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class GatewayBackendTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "AI_CODING_GATEWAY_API_KEY": "gateway-secret",
                "AI_CODING_GATEWAY_PROJECT": "dsgym-tests",
            },
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_routes_request_and_records_usage(self):
        response = _Response({
            "choices": [{"message": {"content": "generated"}}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
            },
        })
        urlopen = Mock(return_value=response)
        with patch(
            "dsgym.agents.backends.gateway_backend.urllib.request.urlopen",
            urlopen,
        ):
            backend = get_backend("gateway", "claude-opus-4-8", max_tokens=50)
            result = backend.generate([{"role": "user", "content": "hello"}])

        self.assertIsInstance(backend, GatewayBackend)
        self.assertEqual(result, "generated")
        self.assertEqual(
            backend.last_usage,
            {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.ai-coding-gateway.inv.systems/v1/chat/completions",
        )
        self.assertEqual(
            {key.lower(): value for key, value in request.header_items()},
            {
                "authorization": "Bearer gateway-secret",
                "content-type": "application/json",
                "x-project": "dsgym-tests",
                "user-agent": "dsgym-gateway/1.0",
            },
        )
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "anthropic/claude-opus-4-8")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(payload["max_tokens"], 50)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 180)

    def test_preserves_qualified_model_names(self):
        backend = GatewayBackend("openai/anthropic/claude-opus-4-8")

        self.assertEqual(backend.routed_model, "anthropic/claude-opus-4-8")

    def test_rejects_missing_credentials_and_invalid_project(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "AI_CODING_GATEWAY_API_KEY"):
                GatewayBackend("claude-opus-4-8")

        os.environ["AI_CODING_GATEWAY_API_KEY"] = "gateway-secret"
        os.environ["AI_CODING_GATEWAY_PROJECT"] = "bad\nproject"
        with self.assertRaisesRegex(ValueError, "control characters"):
            GatewayBackend("claude-opus-4-8")

    def test_redacts_key_from_http_errors(self):
        error = urllib.error.HTTPError(
            "https://gateway.invalid",
            400,
            "Bad Request",
            {},
            io.BytesIO(b"request contained gateway-secret"),
        )
        with patch(
            "dsgym.agents.backends.gateway_backend.urllib.request.urlopen",
            Mock(side_effect=error),
        ):
            backend = GatewayBackend("claude-opus-4-8")
            with self.assertRaises(GatewayError) as exc_info:
                backend.generate([{"role": "user", "content": "hello"}])

        self.assertIn("gateway HTTP 400", str(exc_info.exception))
        self.assertNotIn("gateway-secret", str(exc_info.exception))
        self.assertIn("[REDACTED]", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
