"""Groq LLM connectivity module for DocuMind.

Phase 5.1: Minimal, reusable Groq client interface using environment variables.
Provides structured error handling, strict API key isolation, and response validation.
"""

import os
from typing import Any, Optional, Sequence
from dotenv import load_dotenv
import groq

# Default model identifier
DEFAULT_MODEL: str = "openai/gpt-oss-120b"


class LLMError(Exception):
    """Base exception for all LLM and Groq API errors."""
    pass


class LLMConfigurationError(LLMError, ValueError):
    """Raised when configuration is missing or invalid (e.g., missing API key)."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when Groq API authentication fails (e.g., invalid API key)."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when Groq API rate limits are exceeded (HTTP 429)."""
    pass


class LLMAPIError(LLMError):
    """Raised for network failures, timeouts, connection errors, or general API errors."""
    pass


class LLMEmptyResponseError(LLMError):
    """Raised when the model returns an empty, whitespace-only, or missing response."""
    pass


def get_model_name(model: Optional[str] = None) -> str:
    """Returns the configured model name or the default.

    Resolution order:
    1. Explicit `model` parameter (if non-empty)
    2. GROQ_MODEL environment variable (if non-empty)
    3. DEFAULT_MODEL ('openai/gpt-oss-120b')

    Args:
        model: Optional model name override.

    Returns:
        str: Active model name.
    """
    load_dotenv()
    if model and model.strip():
        return model.strip()
    env_model = os.getenv("GROQ_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    return DEFAULT_MODEL


def get_groq_client(api_key: Optional[str] = None) -> groq.Groq:
    """Instantiates and returns a reusable Groq client using environment variables.

    Guarantees API keys are never printed, logged, or exposed in exceptions.

    Args:
        api_key: Optional API key override. If None, GROQ_API_KEY from environment is used.

    Returns:
        groq.Groq: Initialized Groq client.

    Raises:
        LLMConfigurationError: If GROQ_API_KEY is not set or empty.
    """
    load_dotenv()
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or not key.strip():
        raise LLMConfigurationError(
            "GROQ_API_KEY is not set. Please set the GROQ_API_KEY environment variable "
            "or configure it in a local .env file."
        )
    return groq.Groq(api_key=key.strip())


VALID_ROLES: set[str] = {"system", "user", "assistant"}


def generate_chat_response(
    messages: Sequence[dict[str, str]],
    model: Optional[str] = None,
    client: Optional[Any] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict[str, str]] = None,
) -> str:
    """Sends a sequence of chat messages to the configured Groq model and returns the generated text.

    Args:
        messages: A sequence of message dictionaries, each containing 'role' and 'content'.
            Supported roles are 'system', 'user', and 'assistant'.
        model: Optional model identifier. Defaults to GROQ_MODEL env var or DEFAULT_MODEL.
        client: Optional Groq client or mock. If None, `get_groq_client()` is used.
        temperature: Sampling temperature (default: 0.0 for deterministic answers).
        max_tokens: Optional maximum completion tokens.

    Returns:
        str: Generated text response from the model.

    Raises:
        TypeError: If messages is not a sequence, or any message is not a valid dictionary.
        ValueError: If messages is empty, or any message has invalid role or missing keys.
        LLMConfigurationError: If API key is missing.
        LLMAuthenticationError: If API authentication fails (401).
        LLMRateLimitError: If rate limit is exceeded (429).
        LLMAPIError: If network, connection, timeout, or internal server errors occur.
        LLMEmptyResponseError: If the model returns an empty or whitespace response.
    """
    if messages is None or isinstance(messages, (str, bytes)):
        raise TypeError(
            f"Expected a sequence of message dictionaries, got {type(messages).__name__}."
        )

    try:
        msg_list = list(messages)
    except TypeError as exc:
        raise TypeError(f"messages must be an iterable sequence: {exc}") from exc

    if not msg_list:
        raise ValueError("messages cannot be empty. At least one message is required.")

    # Validate each message dictionary
    for idx, msg in enumerate(msg_list):
        if not isinstance(msg, dict):
            raise TypeError(
                f"Each message must be a dictionary, got {type(msg).__name__} at index {idx}."
            )
        if "role" not in msg or "content" not in msg:
            raise ValueError(
                f"Message at index {idx} must contain both 'role' and 'content' keys."
            )
        role = msg["role"]
        if not isinstance(role, str):
            raise TypeError(
                f"Message 'role' at index {idx} must be a string, got {type(role).__name__}."
            )
        if role not in VALID_ROLES:
            raise ValueError(
                f"Invalid role '{role}' at index {idx}. Must be one of {sorted(VALID_ROLES)}."
            )
        content = msg["content"]
        if not isinstance(content, str):
            raise TypeError(
                f"Message 'content' at index {idx} must be a string, got {type(content).__name__}."
            )

    active_client = client if client is not None else get_groq_client()
    active_model = get_model_name(model)

    try:
        params: dict[str, Any] = {
            "model": active_model,
            "messages": msg_list,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format is not None:
            params["response_format"] = response_format

        chat_completion = active_client.chat.completions.create(**params)

    except groq.AuthenticationError as exc:
        raise LLMAuthenticationError(
            f"Groq API authentication failed for model '{active_model}'. "
            "Please check that your GROQ_API_KEY is valid."
        ) from exc
    except groq.RateLimitError as exc:
        raise LLMRateLimitError(
            f"Groq API rate limit exceeded: {exc}"
        ) from exc
    except (groq.APIConnectionError, groq.APITimeoutError) as exc:
        raise LLMAPIError(
            f"Network error communicating with Groq API: {exc}"
        ) from exc
    except groq.APIStatusError as exc:
        raise LLMAPIError(
            f"Groq API error ({exc.status_code}): {exc.message}"
        ) from exc
    except groq.GroqError as exc:
        raise LLMAPIError(
            f"Groq API encountered an error: {exc}"
        ) from exc
    except Exception as exc:
        if isinstance(exc, (LLMError, ValueError, TypeError)):
            raise
        raise LLMAPIError(
            f"Unexpected error communicating with Groq API: {exc}"
        ) from exc

    # Validate response structure and content
    if not chat_completion or not getattr(chat_completion, "choices", None):
        raise LLMEmptyResponseError("Groq model returned no choices or an empty response object.")

    choice = chat_completion.choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None

    if content is None or not str(content).strip():
        raise LLMEmptyResponseError("Groq model returned an empty or whitespace-only response.")

    return str(content).strip()


def generate_response(
    prompt: str,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    """Sends a user prompt to the configured Groq model and returns the generated text.

    Maintains backward compatibility by converting the single prompt into a user message
    and delegating to `generate_chat_response`.

    Args:
        prompt: User input prompt string.
        model: Optional model identifier. Defaults to GROQ_MODEL env var or DEFAULT_MODEL.
        client: Optional Groq client or mock. If None, `get_groq_client()` is used.
        temperature: Sampling temperature (default: 0.0 for deterministic answers).
        max_tokens: Optional maximum completion tokens.

    Returns:
        str: Generated text response from the model.

    Raises:
        TypeError: If prompt is not a string.
        ValueError: If prompt is empty or only whitespace.
        LLMConfigurationError: If API key is missing.
        LLMAuthenticationError: If API authentication fails (401).
        LLMRateLimitError: If rate limit is exceeded (429).
        LLMAPIError: If network, connection, timeout, or internal server errors occur.
        LLMEmptyResponseError: If the model returns an empty or whitespace response.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}.")

    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("prompt cannot be empty or contain only whitespace.")

    messages = [{"role": "user", "content": cleaned_prompt}]
    return generate_chat_response(
        messages=messages,
        model=model,
        client=client,
        temperature=temperature,
        max_tokens=max_tokens,
    )


