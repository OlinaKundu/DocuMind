"""Unit tests for DocuMind's Groq LLM module (Phase 5.1).

All default tests use in-memory mocked Groq responses and require no real API key
or internet access. An optional real API smoke test is marked with @pytest.mark.real_api.
"""

import os
from typing import Any, List, Optional
from unittest.mock import patch
import httpx
import pytest
import groq

from src.llm import (
    DEFAULT_MODEL,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    generate_chat_response,
    generate_response,
    get_groq_client,
    get_model_name,
)


class MockMessage:
    """Mock message container mirroring Groq ChatCompletionMessage."""

    def __init__(self, content: Optional[str]):
        self.content = content


class MockChoice:
    """Mock choice container mirroring Groq Choice."""

    def __init__(self, message: Optional[MockMessage]):
        self.message = message


class MockChatCompletion:
    """Mock completion response mirroring Groq ChatCompletion."""

    def __init__(self, choices: List[MockChoice]):
        self.choices = choices


class MockCompletionsResource:
    """Mock completions resource with a create() method."""

    def __init__(self, response: Any = None, side_effect: Optional[Exception] = None):
        self.response = response
        self.side_effect = side_effect
        self.last_call_params: dict[str, Any] = {}

    def create(self, **kwargs) -> Any:
        self.last_call_params = kwargs
        if self.side_effect is not None:
            raise self.side_effect
        return self.response


class MockGroqClient:
    """Mock Groq client adhering to groq.Groq interface."""

    def __init__(self, response: Any = None, side_effect: Optional[Exception] = None):
        self.completions = MockCompletionsResource(response=response, side_effect=side_effect)
        self.chat = self  # allows client.chat.completions.create(...)


def _make_http_response(status_code: int) -> httpx.Response:
    """Helper to build a dummy httpx.Response for Groq exception constructors."""
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status_code, request=req)


# ==============================================================================
# 1. Model Resolution & Client Creation Tests
# ==============================================================================

def test_get_model_name_default():
    """Verify get_model_name returns DEFAULT_MODEL when no override or env var is set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_model_name() == DEFAULT_MODEL
        assert get_model_name(None) == DEFAULT_MODEL
        assert get_model_name("") == DEFAULT_MODEL


def test_get_model_name_env_override():
    """Verify get_model_name honors GROQ_MODEL environment variable."""
    with patch.dict(os.environ, {"GROQ_MODEL": "custom-env-model"}):
        assert get_model_name() == "custom-env-model"


def test_get_model_name_explicit_argument():
    """Verify explicit argument takes precedence over environment variable."""
    with patch.dict(os.environ, {"GROQ_MODEL": "env-model"}):
        assert get_model_name("explicit-model") == "explicit-model"


def test_get_groq_client_missing_key_raises_configuration_error():
    """Verify get_groq_client raises LLMConfigurationError when no API key is found."""
    with patch("src.llm.load_dotenv"):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY is not set"):
                get_groq_client()

            with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY is not set"):
                get_groq_client(api_key="")


def test_get_groq_client_with_valid_key():
    """Verify get_groq_client initializes when an API key is provided."""
    client = get_groq_client(api_key="gsk_mock_valid_key_12345")
    assert isinstance(client, groq.Groq)
    assert client.api_key == "gsk_mock_valid_key_12345"


# ==============================================================================
# 2. Response Generation Tests (Success Paths)
# ==============================================================================

def test_generate_chat_response_successful_mock():
    """Verify generate_chat_response correctly sends structured chat messages."""
    mock_response = MockChatCompletion(
        choices=[MockChoice(message=MockMessage(content="Grounded chat response."))]
    )
    mock_client = MockGroqClient(response=mock_response)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    result = generate_chat_response(messages, client=mock_client)

    assert result == "Grounded chat response."
    assert mock_client.completions.last_call_params["model"] == DEFAULT_MODEL
    assert mock_client.completions.last_call_params["messages"] == messages
    assert mock_client.completions.last_call_params["temperature"] == 0.0


def test_generate_chat_response_validation():
    """Verify generate_chat_response strictly validates message structures and roles."""
    mock_client = MockGroqClient()

    # None input
    with pytest.raises(TypeError, match="Expected a sequence"):
        generate_chat_response(None, client=mock_client)

    # String input
    with pytest.raises(TypeError, match="Expected a sequence"):
        generate_chat_response("single string", client=mock_client)

    # Empty messages list
    with pytest.raises(ValueError, match="messages cannot be empty"):
        generate_chat_response([], client=mock_client)

    # Element not a dictionary
    with pytest.raises(TypeError, match="must be a dictionary"):
        generate_chat_response(["not a dict"], client=mock_client)

    # Missing role or content
    with pytest.raises(ValueError, match="must contain both 'role' and 'content'"):
        generate_chat_response([{"role": "user"}], client=mock_client)

    with pytest.raises(ValueError, match="must contain both 'role' and 'content'"):
        generate_chat_response([{"content": "hello"}], client=mock_client)

    # Invalid role
    with pytest.raises(ValueError, match="Invalid role 'admin'"):
        generate_chat_response([{"role": "admin", "content": "hi"}], client=mock_client)

    # Non-string role
    with pytest.raises(TypeError, match="Message 'role' .* must be a string"):
        generate_chat_response([{"role": 123, "content": "hi"}], client=mock_client)

    # Non-string content
    with pytest.raises(TypeError, match="Message 'content' .* must be a string"):
        generate_chat_response([{"role": "user", "content": 456}], client=mock_client)


def test_generate_response_successful_mock():
    """Verify generate_response parses valid completion text and uses default model."""
    mock_response = MockChatCompletion(
        choices=[MockChoice(message=MockMessage(content="DocuMind retrieval answer."))]
    )
    mock_client = MockGroqClient(response=mock_response)

    prompt = "Summarize the key findings."
    result = generate_response(prompt, client=mock_client)

    assert result == "DocuMind retrieval answer."
    assert mock_client.completions.last_call_params["model"] == DEFAULT_MODEL
    assert mock_client.completions.last_call_params["messages"] == [
        {"role": "user", "content": prompt}
    ]
    assert mock_client.completions.last_call_params["temperature"] == 0.0


def test_generate_response_custom_model_and_tokens():
    """Verify generate_response passes custom model and max_tokens parameters."""
    mock_response = MockChatCompletion(
        choices=[MockChoice(message=MockMessage(content="Response with custom parameters."))]
    )
    mock_client = MockGroqClient(response=mock_response)

    result = generate_response(
        prompt="Explain attention mechanism",
        model="llama-3.3-70b-versatile",
        client=mock_client,
        temperature=0.7,
        max_tokens=256,
    )

    assert result == "Response with custom parameters."
    assert mock_client.completions.last_call_params["model"] == "llama-3.3-70b-versatile"
    assert mock_client.completions.last_call_params["temperature"] == 0.7
    assert mock_client.completions.last_call_params["max_tokens"] == 256


# ==============================================================================
# 3. Error Handling Tests
# ==============================================================================

def test_generate_response_authentication_error():
    """Verify groq.AuthenticationError maps to LLMAuthenticationError."""
    http_resp = _make_http_response(401)
    groq_exc = groq.AuthenticationError("Invalid API key provided", response=http_resp, body=None)
    mock_client = MockGroqClient(side_effect=groq_exc)

    with pytest.raises(LLMAuthenticationError, match="authentication failed"):
        generate_response("Test prompt", client=mock_client)


def test_generate_response_rate_limit_error():
    """Verify groq.RateLimitError maps to LLMRateLimitError."""
    http_resp = _make_http_response(429)
    groq_exc = groq.RateLimitError("Rate limit reached for requests per minute", response=http_resp, body=None)
    mock_client = MockGroqClient(side_effect=groq_exc)

    with pytest.raises(LLMRateLimitError, match="rate limit exceeded"):
        generate_response("Test prompt", client=mock_client)


def test_generate_response_network_connection_error():
    """Verify groq.APIConnectionError maps to LLMAPIError."""
    groq_exc = groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
    mock_client = MockGroqClient(side_effect=groq_exc)

    with pytest.raises(LLMAPIError, match="Network error"):
        generate_response("Test prompt", client=mock_client)


def test_generate_response_network_timeout_error():
    """Verify groq.APITimeoutError maps to LLMAPIError."""
    groq_exc = groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))
    mock_client = MockGroqClient(side_effect=groq_exc)

    with pytest.raises(LLMAPIError, match="Network error"):
        generate_response("Test prompt", client=mock_client)


def test_generate_response_api_status_error():
    """Verify groq.APIStatusError (500 internal server error) maps to LLMAPIError."""
    http_resp = _make_http_response(500)
    groq_exc = groq.InternalServerError("Internal server error", response=http_resp, body=None)
    mock_client = MockGroqClient(side_effect=groq_exc)

    with pytest.raises(LLMAPIError, match="Groq API error"):
        generate_response("Test prompt", client=mock_client)


def test_generate_response_empty_choices():
    """Verify empty choices list raises LLMEmptyResponseError."""
    mock_response = MockChatCompletion(choices=[])
    mock_client = MockGroqClient(response=mock_response)

    with pytest.raises(LLMEmptyResponseError, match="returned no choices"):
        generate_response("Test prompt", client=mock_client)


def test_generate_response_none_content():
    """Verify None message content raises LLMEmptyResponseError."""
    mock_response = MockChatCompletion(choices=[MockChoice(message=MockMessage(content=None))])
    mock_client = MockGroqClient(response=mock_response)

    with pytest.raises(LLMEmptyResponseError, match="empty or whitespace-only"):
        generate_response("Test prompt", client=mock_client)


def test_generate_response_empty_string_content():
    """Verify empty or whitespace-only message content raises LLMEmptyResponseError."""
    mock_response = MockChatCompletion(choices=[MockChoice(message=MockMessage(content="   \n\t  "))])
    mock_client = MockGroqClient(response=mock_response)

    with pytest.raises(LLMEmptyResponseError, match="empty or whitespace-only"):
        generate_response("Test prompt", client=mock_client)


# ==============================================================================
# 4. Prompt Validation & API Key Isolation Tests
# ==============================================================================

def test_generate_response_invalid_prompt_inputs():
    """Verify invalid prompt inputs (None, non-string, whitespace) raise TypeError or ValueError."""
    mock_client = MockGroqClient()

    # None
    with pytest.raises(TypeError, match="prompt must be a string"):
        generate_response(None, client=mock_client)

    # Non-string integer
    with pytest.raises(TypeError, match="prompt must be a string"):
        generate_response(12345, client=mock_client)

    # Empty string
    with pytest.raises(ValueError, match="cannot be empty or contain only whitespace"):
        generate_response("", client=mock_client)

    # Whitespace-only string
    with pytest.raises(ValueError, match="cannot be empty or contain only whitespace"):
        generate_response("   \t\n  ", client=mock_client)


def test_api_key_never_exposed_in_exceptions():
    """Verify secret API keys are never leaked into error messages or exception strings."""
    secret_key = "gsk_super_confidential_secret_key_998877"
    http_resp = _make_http_response(401)
    groq_exc = groq.AuthenticationError("Invalid API key", response=http_resp, body=None)

    with patch.dict(os.environ, {"GROQ_API_KEY": secret_key}):
        mock_client = MockGroqClient(side_effect=groq_exc)
        try:
            generate_response("Test prompt", client=mock_client)
            pytest.fail("Expected LLMAuthenticationError was not raised.")
        except LLMAuthenticationError as err:
            err_message = str(err)
            assert secret_key not in err_message, "Secret API key was exposed in exception string!"


# ==============================================================================
# 5. Optional Real API Smoke Test
# ==============================================================================

@pytest.mark.real_api
def test_real_groq_api_smoke_test():
    """Smoke test making a live call to Groq API.

    Requires an active GROQ_API_KEY in .env or environment.
    Runs only when explicitly invoked: pytest -m real_api
    """
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key or not key.strip():
        pytest.skip("GROQ_API_KEY not configured. Skipping real API smoke test.")

    prompt = "Respond with the single word: PONG"
    response = generate_response(prompt)

    assert isinstance(response, str)
    assert len(response.strip()) > 0
    assert "PONG" in response.upper()
