"""mistral4cli package."""

from .cli import main
from .local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    LocalMistralConfig,
    build_client,
    get_health,
    list_models,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_SERVER_URL",
    "LocalMistralConfig",
    "build_client",
    "get_health",
    "list_models",
    "main",
]
