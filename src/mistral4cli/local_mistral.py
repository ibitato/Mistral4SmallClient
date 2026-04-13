"""Helpers for the local Mistral Small 4 deployment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

from mistralai import Mistral

DEFAULT_API_KEY = "local-test"
DEFAULT_MODEL_ID = "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_MS = 120_000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_PROMPT_MODE = "reasoning"

ENV_API_KEY = "MISTRAL_LOCAL_API_KEY"
ENV_MODEL_ID = "MISTRAL_LOCAL_MODEL_ID"
ENV_SERVER_URL = "MISTRAL_LOCAL_SERVER_URL"
ENV_TIMEOUT_MS = "MISTRAL_LOCAL_TIMEOUT_MS"
ENV_TEMPERATURE = "MISTRAL_LOCAL_TEMPERATURE"
ENV_TOP_P = "MISTRAL_LOCAL_TOP_P"
ENV_PROMPT_MODE = "MISTRAL_LOCAL_PROMPT_MODE"
ENV_MAX_TOKENS = "MISTRAL_LOCAL_MAX_TOKENS"


@dataclass(frozen=True, slots=True)
class LocalMistralConfig:
    """Configuration for the local llama.cpp-backed Mistral endpoint."""

    api_key: str = DEFAULT_API_KEY
    model_id: str = DEFAULT_MODEL_ID
    server_url: str = DEFAULT_SERVER_URL
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    @classmethod
    def from_env(cls) -> LocalMistralConfig:
        """Build a config from environment variables with safe defaults."""

        return cls(
            api_key=os.environ.get(ENV_API_KEY, DEFAULT_API_KEY),
            model_id=os.environ.get(ENV_MODEL_ID, DEFAULT_MODEL_ID),
            server_url=os.environ.get(ENV_SERVER_URL, DEFAULT_SERVER_URL),
            timeout_ms=int(os.environ.get(ENV_TIMEOUT_MS, str(DEFAULT_TIMEOUT_MS))),
        )


@dataclass(frozen=True, slots=True)
class LocalGenerationConfig:
    """Sampling defaults for the local Mistral Small 4 deployment."""

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    prompt_mode: str | None = DEFAULT_PROMPT_MODE
    max_tokens: int | None = None

    @classmethod
    def from_env(cls) -> LocalGenerationConfig:
        """Build generation defaults from environment variables."""

        prompt_mode = os.environ.get(ENV_PROMPT_MODE)
        if prompt_mode is None:
            prompt_mode = DEFAULT_PROMPT_MODE
        else:
            prompt_mode = prompt_mode.strip()
            if prompt_mode.lower() in {"", "none", "null", "off"}:
                prompt_mode = None

        max_tokens_raw = os.environ.get(ENV_MAX_TOKENS)
        max_tokens = (
            int(max_tokens_raw)
            if max_tokens_raw is not None and max_tokens_raw.strip()
            else None
        )

        return cls(
            temperature=float(
                os.environ.get(ENV_TEMPERATURE, str(DEFAULT_TEMPERATURE))
            ),
            top_p=float(os.environ.get(ENV_TOP_P, str(DEFAULT_TOP_P))),
            prompt_mode=prompt_mode,
            max_tokens=max_tokens,
        )


def build_client(config: LocalMistralConfig | None = None) -> Mistral:
    """Construct an official `mistralai` client pointed at the local server."""

    active_config = config or LocalMistralConfig.from_env()
    return Mistral(
        api_key=active_config.api_key,
        server_url=active_config.server_url,
        timeout_ms=active_config.timeout_ms,
    )


def get_json(url: str, timeout_s: float = 2.0) -> dict[str, Any]:
    """Fetch and decode a JSON document from the local server."""

    try:
        with urlopen(url, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8")
    except URLError as exc:  # pragma: no cover - exercised by integration setup
        raise RuntimeError(f"Could not reach local Mistral server at {url}") from exc

    return cast(dict[str, Any], json.loads(payload))


def get_health(server_url: str | None = None) -> dict[str, Any]:
    """Return the `/health` payload from the local server."""

    base_url = (server_url or DEFAULT_SERVER_URL).rstrip("/")
    return get_json(f"{base_url}/health")


def list_models(server_url: str | None = None) -> dict[str, Any]:
    """Return the `/v1/models` payload from the local server."""

    base_url = (server_url or DEFAULT_SERVER_URL).rstrip("/")
    return get_json(f"{base_url}/v1/models")
