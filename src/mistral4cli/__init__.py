"""mistral4cli package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mistral4cli")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .cli import main
from .conversation_registry import ConversationRegistry
from .local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    REMOTE_MEDIUM_MODEL_ID,
    REMOTE_MODEL_ID,
    BackendKind,
    ContextConfig,
    ConversationConfig,
    LocalGenerationConfig,
    LocalMistralConfig,
    RemoteMistralConfig,
    build_client,
    get_health,
    list_models,
    normalize_remote_model_id,
    remote_api_key_available,
)
from .session import MistralCodingSession, MistralSession, PendingConversationSettings

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_SERVER_URL",
    "REMOTE_MEDIUM_MODEL_ID",
    "REMOTE_MODEL_ID",
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
    "__version__",
    "build_client",
    "get_health",
    "list_models",
    "main",
    "normalize_remote_model_id",
    "remote_api_key_available",
]
