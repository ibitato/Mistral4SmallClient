"""Configuration and client helpers for local and remote Mistral backends."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

from mistralai.client import Mistral

from mistral4cli.mistral_client import MistralClientProtocol

DEFAULT_API_KEY = "local-test"
DEFAULT_MODEL_ID = "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_MS = 300_000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_PROMPT_MODE = "reasoning"

REMOTE_MODEL_ID = "mistral-small-latest"
REMOTE_SERVER_LABEL = "Mistral Cloud"

ENV_API_KEY = "MISTRAL_LOCAL_API_KEY"
ENV_MODEL_ID = "MISTRAL_LOCAL_MODEL_ID"
ENV_SERVER_URL = "MISTRAL_LOCAL_SERVER_URL"
ENV_TIMEOUT_MS = "MISTRAL_LOCAL_TIMEOUT_MS"
ENV_TEMPERATURE = "MISTRAL_LOCAL_TEMPERATURE"
ENV_TOP_P = "MISTRAL_LOCAL_TOP_P"
ENV_PROMPT_MODE = "MISTRAL_LOCAL_PROMPT_MODE"
ENV_MAX_TOKENS = "MISTRAL_LOCAL_MAX_TOKENS"
ENV_REMOTE_API_KEY = "MISTRAL_API_KEY"
ENV_CONVERSATIONS = "MISTRAL_CONVERSATIONS"
ENV_CONVERSATION_STORE = "MISTRAL_CONVERSATION_STORE"
ENV_CONVERSATION_RESUME = "MISTRAL_CONVERSATION_RESUME"
ENV_CONTEXT_AUTO_COMPACT = "MISTRAL_CONTEXT_AUTO_COMPACT"
ENV_CONTEXT_THRESHOLD = "MISTRAL_CONTEXT_THRESHOLD"
ENV_CONTEXT_RESERVE_TOKENS = "MISTRAL_CONTEXT_RESERVE_TOKENS"
ENV_CONTEXT_LOCAL_WINDOW_TOKENS = "MISTRAL_CONTEXT_LOCAL_WINDOW_TOKENS"
ENV_CONTEXT_REMOTE_WINDOW_TOKENS = "MISTRAL_CONTEXT_REMOTE_WINDOW_TOKENS"
ENV_CONTEXT_KEEP_RECENT_TURNS = "MISTRAL_CONTEXT_KEEP_RECENT_TURNS"
ENV_CONTEXT_SUMMARY_MAX_TOKENS = "MISTRAL_CONTEXT_SUMMARY_MAX_TOKENS"

DEFAULT_LOCAL_CONTEXT_WINDOW_TOKENS = 262_144
DEFAULT_REMOTE_CONTEXT_WINDOW_TOKENS = 256_000
DEFAULT_CONTEXT_THRESHOLD = 0.9
DEFAULT_CONTEXT_RESERVE_TOKENS = 8_192
DEFAULT_CONTEXT_KEEP_RECENT_TURNS = 6
DEFAULT_CONTEXT_SUMMARY_MAX_TOKENS = 2_048
DEFAULT_CONVERSATION_RESUME_POLICY = "last"


class BackendKind(str, Enum):
    """Runtime backend modes supported by the CLI."""

    LOCAL = "local"
    REMOTE = "remote"


class RemoteAPIKeyError(RuntimeError):
    """Raised when remote Mistral cloud mode is requested without an API key."""


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
class RemoteMistralConfig:
    """Configuration for the remote Mistral cloud endpoint."""

    api_key: str
    model_id: str = REMOTE_MODEL_ID
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    @classmethod
    def from_env(cls, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> RemoteMistralConfig:
        """Build a remote config from environment variables."""

        api_key = os.environ.get(ENV_REMOTE_API_KEY, "").strip()
        if not api_key:
            raise RemoteAPIKeyError(
                f"Set {ENV_REMOTE_API_KEY} in your environment to enable remote mode."
            )
        return cls(api_key=api_key, timeout_ms=timeout_ms)


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


@dataclass(frozen=True, slots=True)
class ConversationConfig:
    """Runtime defaults for Mistral Cloud Conversations mode."""

    enabled: bool = False
    store: bool = True
    resume_policy: str = DEFAULT_CONVERSATION_RESUME_POLICY

    @classmethod
    def from_env(cls) -> ConversationConfig:
        """Build conversation defaults from environment variables."""

        resume_policy = (
            os.environ.get(
                ENV_CONVERSATION_RESUME,
                DEFAULT_CONVERSATION_RESUME_POLICY,
            )
            .strip()
            .lower()
        )
        if resume_policy not in {"last", "new", "prompt"}:
            resume_policy = DEFAULT_CONVERSATION_RESUME_POLICY
        return cls(
            enabled=_env_bool(ENV_CONVERSATIONS, default=False),
            store=_env_bool(ENV_CONVERSATION_STORE, default=True),
            resume_policy=resume_policy,
        )


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Client-side context overflow and compaction policy."""

    auto_compact: bool = True
    threshold: float = DEFAULT_CONTEXT_THRESHOLD
    reserve_tokens: int = DEFAULT_CONTEXT_RESERVE_TOKENS
    local_window_tokens: int = DEFAULT_LOCAL_CONTEXT_WINDOW_TOKENS
    remote_window_tokens: int = DEFAULT_REMOTE_CONTEXT_WINDOW_TOKENS
    keep_recent_turns: int = DEFAULT_CONTEXT_KEEP_RECENT_TURNS
    summary_max_tokens: int = DEFAULT_CONTEXT_SUMMARY_MAX_TOKENS

    @classmethod
    def from_env(cls) -> ContextConfig:
        """Build context defaults from environment variables."""

        return cls(
            auto_compact=_env_bool(ENV_CONTEXT_AUTO_COMPACT, default=True),
            threshold=_env_float(
                ENV_CONTEXT_THRESHOLD,
                default=DEFAULT_CONTEXT_THRESHOLD,
            ),
            reserve_tokens=_env_int(
                ENV_CONTEXT_RESERVE_TOKENS,
                default=DEFAULT_CONTEXT_RESERVE_TOKENS,
            ),
            local_window_tokens=_env_int(
                ENV_CONTEXT_LOCAL_WINDOW_TOKENS,
                default=DEFAULT_LOCAL_CONTEXT_WINDOW_TOKENS,
            ),
            remote_window_tokens=_env_int(
                ENV_CONTEXT_REMOTE_WINDOW_TOKENS,
                default=DEFAULT_REMOTE_CONTEXT_WINDOW_TOKENS,
            ),
            keep_recent_turns=_env_int(
                ENV_CONTEXT_KEEP_RECENT_TURNS,
                default=DEFAULT_CONTEXT_KEEP_RECENT_TURNS,
            ),
            summary_max_tokens=_env_int(
                ENV_CONTEXT_SUMMARY_MAX_TOKENS,
                default=DEFAULT_CONTEXT_SUMMARY_MAX_TOKENS,
            ),
        ).normalized()

    def normalized(self) -> ContextConfig:
        """Return a sanitized copy with bounded operational values."""

        threshold = self.threshold
        if threshold > 1:
            threshold = threshold / 100
        threshold = min(max(threshold, 0.1), 0.99)
        return ContextConfig(
            auto_compact=self.auto_compact,
            threshold=threshold,
            reserve_tokens=max(0, self.reserve_tokens),
            local_window_tokens=max(1_024, self.local_window_tokens),
            remote_window_tokens=max(1_024, self.remote_window_tokens),
            keep_recent_turns=max(1, self.keep_recent_turns),
            summary_max_tokens=max(256, self.summary_max_tokens),
        )


MistralConfig = LocalMistralConfig | RemoteMistralConfig


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value.strip().rstrip("%"))
    except ValueError:
        return default


def get_client_timeout_ms(
    client: MistralClientProtocol | object,
    default: int = DEFAULT_TIMEOUT_MS,
) -> int:
    """Return the effective timeout configured on a Mistral client."""

    timeout_ms = getattr(client, "timeout_ms", None)
    if isinstance(timeout_ms, int) and timeout_ms > 0:
        return timeout_ms
    sdk_configuration = getattr(client, "sdk_configuration", None)
    sdk_timeout_ms = getattr(sdk_configuration, "timeout_ms", None)
    if isinstance(sdk_timeout_ms, int) and sdk_timeout_ms > 0:
        return sdk_timeout_ms
    return default


def set_client_timeout_ms(
    client: MistralClientProtocol | object,
    timeout_ms: int,
) -> None:
    """Update the effective timeout on a Mistral client in place."""

    cast(Any, client).timeout_ms = timeout_ms
    sdk_configuration = getattr(client, "sdk_configuration", None)
    if sdk_configuration is not None:
        sdk_configuration.timeout_ms = timeout_ms


def build_client(config: MistralConfig | None = None) -> Mistral:
    """Construct an official `mistralai` client for the selected backend."""

    active_config = config or LocalMistralConfig.from_env()
    if isinstance(active_config, LocalMistralConfig):
        client = Mistral(
            api_key=active_config.api_key,
            server_url=active_config.server_url,
            timeout_ms=active_config.timeout_ms,
        )
        set_client_timeout_ms(client, active_config.timeout_ms)
        return client
    client = Mistral(
        api_key=active_config.api_key,
        timeout_ms=active_config.timeout_ms,
    )
    set_client_timeout_ms(client, active_config.timeout_ms)
    return client


def remote_api_key_available() -> bool:
    """Return whether the remote Mistral cloud API key is available."""

    return bool(os.environ.get(ENV_REMOTE_API_KEY, "").strip())


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
