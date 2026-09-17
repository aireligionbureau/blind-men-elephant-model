from typing import Any, Protocol

from .deepseek import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekTimeoutError,
    LLMResponse,
    OpenAICompatibleClient,
    ProviderAPIError,
    ProviderTimeoutError,
    bounded_request_timeout_seconds,
    retry_delay_seconds,
    synthesis_request_timeout_seconds,
)


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        ...


OpenAICompatibleAPIError = DeepSeekAPIError
OpenAICompatibleTimeoutError = DeepSeekTimeoutError


def build_chat_client(**kwargs: Any) -> ChatClient:
    return OpenAICompatibleClient(**kwargs)

__all__ = [
    "DeepSeekAPIError",
    "DeepSeekClient",
    "DeepSeekTimeoutError",
    "ChatClient",
    "LLMResponse",
    "OpenAICompatibleAPIError",
    "OpenAICompatibleClient",
    "OpenAICompatibleTimeoutError",
    "ProviderAPIError",
    "ProviderTimeoutError",
    "bounded_request_timeout_seconds",
    "build_chat_client",
    "retry_delay_seconds",
    "synthesis_request_timeout_seconds",
]
