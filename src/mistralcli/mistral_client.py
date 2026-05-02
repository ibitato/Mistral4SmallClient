"""Structural typing helpers for the Mistral SDK client boundary."""

from __future__ import annotations

from typing import Any, Protocol


class MistralClientProtocol(Protocol):
    """Structural client contract used across the CLI and tests."""

    @property
    def chat(self) -> Any:
        """Return the SDK chat surface."""
        ...

    @property
    def beta(self) -> Any:
        """Return the SDK beta namespace used for Conversations."""
        ...
