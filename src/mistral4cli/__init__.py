"""mistral4cli package."""

from .cli import main
from .local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    BackendKind,
    ConversationConfig,
    LocalGenerationConfig,
    LocalMistralConfig,
    RemoteMistralConfig,
    build_client,
    get_health,
    list_models,
    remote_api_key_available,
)
from .session import MistralCodingSession, MistralSession

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_SERVER_URL",
    "BackendKind",
    "ConversationConfig",
    "LocalGenerationConfig",
    "LocalMistralConfig",
    "MistralCodingSession",
    "MistralSession",
    "RemoteMistralConfig",
    "build_client",
    "get_health",
    "list_models",
    "main",
    "remote_api_key_available",
]
