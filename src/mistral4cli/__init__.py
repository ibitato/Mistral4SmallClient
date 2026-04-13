"""mistral4cli package."""

from .cli import main
from .local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    BackendKind,
    LocalGenerationConfig,
    LocalMistralConfig,
    RemoteMistralConfig,
    build_client,
    get_health,
    list_models,
    remote_api_key_available,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_SERVER_URL",
    "BackendKind",
    "LocalGenerationConfig",
    "LocalMistralConfig",
    "RemoteMistralConfig",
    "build_client",
    "get_health",
    "list_models",
    "main",
    "remote_api_key_available",
]
