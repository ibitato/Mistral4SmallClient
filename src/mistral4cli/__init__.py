"""mistral4cli package."""

from .cli import main
from .conversation_registry import ConversationRegistry
from .local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    BackendKind,
    ContextConfig,
    ConversationConfig,
    LocalGenerationConfig,
    LocalMistralConfig,
    RemoteMistralConfig,
    build_client,
    get_health,
    list_models,
    remote_api_key_available,
)
from .session import MistralCodingSession, MistralSession, PendingConversationSettings

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_SERVER_URL",
    "BackendKind",
    "ContextConfig",
    "ConversationConfig",
    "ConversationRegistry",
    "LocalGenerationConfig",
    "LocalMistralConfig",
    "MistralCodingSession",
    "MistralSession",
    "PendingConversationSettings",
    "RemoteMistralConfig",
    "build_client",
    "get_health",
    "list_models",
    "main",
    "remote_api_key_available",
]
