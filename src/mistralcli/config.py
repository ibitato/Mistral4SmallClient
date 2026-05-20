"""Central configuration management for the dual-model Mistral CLI.

This module provides a unified configuration system that supports:
- File-based configuration (YAML, JSON, TOML)
- Environment variables (MISTRAL_* prefix)
- Command-line arguments (via argparse)

Precedence (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Configuration file
4. Default values

Configuration file search order:
1. Explicit path via --config-path
2. $MISTRAL_CONFIG_PATH environment variable
3. ~/.config/mistralcli/config.yaml
4. ~/.mistralcli.{yaml,json,toml}
5. ./mistralcli.{yaml,json,toml} (current directory)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("mistralcli.config")

# Configuration environment variables
ENV_CONFIG_PATH = "MISTRAL_CONFIG_PATH"

# Default configuration file locations
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "mistralcli"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

# Supported configuration file names
CONFIG_FILE_NAMES = ("mistralcli.yaml", "mistralcli.json", "mistralcli.toml")


class BackendKind(str, Enum):
    """Runtime backend modes supported by the CLI."""

    LOCAL = "local"
    REMOTE = "remote"


class ResumePolicy(str, Enum):
    """Resume policy for Conversations mode."""

    LAST = "last"
    NEW = "new"
    PROMPT = "prompt"


@dataclass(frozen=True, slots=True)
class LocalBackendConfig:
    """Configuration for the local llama.cpp-backed Mistral endpoint."""

    api_key: str = "local-test"
    model_id: str = "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
    server_url: str = "http://127.0.0.1:8080"
    timeout_ms: int = 300_000

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.timeout_ms < 1:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")

    @classmethod
    def from_env(cls) -> LocalBackendConfig:
        """Build configuration from environment variables."""
        from mistralcli.local_mistral import (
            ENV_API_KEY,
            ENV_MODEL_ID,
            ENV_SERVER_URL,
            ENV_TIMEOUT_MS,
        )

        return cls(
            api_key=os.environ.get(ENV_API_KEY, "local-test"),
            model_id=os.environ.get(
                ENV_MODEL_ID, "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
            ),
            server_url=os.environ.get(ENV_SERVER_URL, "http://127.0.0.1:8080"),
            timeout_ms=int(os.environ.get(ENV_TIMEOUT_MS, "300000")),
        )


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Sampling defaults for the local llama.cpp deployment."""

    temperature: float = 0.3
    top_p: float = 0.95
    prompt_mode: str | None = "reasoning"
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0 <= self.temperature <= 1:
            raise ValueError(
                f"temperature must be between 0 and 1, got {self.temperature}"
            )
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")

    @classmethod
    def from_env(cls) -> GenerationConfig:
        """Build configuration from environment variables."""
        from mistralcli.local_mistral import (
            ENV_MAX_TOKENS,
            ENV_PROMPT_MODE,
            ENV_TEMPERATURE,
            ENV_TOP_P,
        )

        prompt_mode = os.environ.get(ENV_PROMPT_MODE)
        if prompt_mode is None:
            prompt_mode = "reasoning"
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
            temperature=float(os.environ.get(ENV_TEMPERATURE, "0.3")),
            top_p=float(os.environ.get(ENV_TOP_P, "0.95")),
            prompt_mode=prompt_mode,
            max_tokens=max_tokens,
        )


@dataclass(frozen=True, slots=True)
class RemoteBackendConfig:
    """Configuration for the remote Mistral cloud endpoint."""

    model_id: str = "mistral-small-latest"
    timeout_ms: int = 300_000

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.timeout_ms < 1:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")

    @classmethod
    def from_env(cls, *, model_id: str | None = None) -> RemoteBackendConfig:
        """Build configuration from environment variables."""
        from mistralcli.local_mistral import (
            ENV_REMOTE_API_KEY,
            ENV_REMOTE_MODEL_ID,
            RemoteAPIKeyError,
            normalize_remote_model_id,
        )

        api_key = os.environ.get(ENV_REMOTE_API_KEY, "").strip()
        if not api_key:
            raise RemoteAPIKeyError(
                f"Set {ENV_REMOTE_API_KEY} in your environment to enable remote mode."
            )

        selected_model = normalize_remote_model_id(
            model_id or os.environ.get(ENV_REMOTE_MODEL_ID)
        )

        return cls(
            model_id=selected_model,
            timeout_ms=300_000,  # Default timeout for remote
        )


@dataclass(frozen=True, slots=True)
class ConversationsConfig:
    """Runtime defaults for Mistral Cloud Conversations mode."""

    enabled: bool = False
    store: bool = True
    resume_policy: ResumePolicy = ResumePolicy.LAST

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.resume_policy, ResumePolicy):
            raise ValueError(
                f"resume_policy must be a ResumePolicy enum, got {self.resume_policy!r}"
            )

    @classmethod
    def from_env(cls) -> ConversationsConfig:
        """Build configuration from environment variables."""
        from mistralcli.local_mistral import (
            DEFAULT_CONVERSATION_RESUME_POLICY,
            ENV_CONVERSATION_RESUME,
            ENV_CONVERSATION_STORE,
            ENV_CONVERSATIONS,
        )

        def _env_bool(name: str, default: bool) -> bool:
            value = os.environ.get(name)
            if value is None:
                return default
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return default

        resume_policy = (
            os.environ.get(ENV_CONVERSATION_RESUME, DEFAULT_CONVERSATION_RESUME_POLICY)
            .strip()
            .lower()
        )
        if resume_policy not in {"last", "new", "prompt"}:
            resume_policy = DEFAULT_CONVERSATION_RESUME_POLICY

        return cls(
            enabled=_env_bool(ENV_CONVERSATIONS, default=False),
            store=_env_bool(ENV_CONVERSATION_STORE, default=True),
            resume_policy=ResumePolicy(resume_policy),
        )


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Client-side context overflow and compaction policy."""

    auto_compact: bool = True
    threshold: float = 0.9
    reserve_tokens: int = 8_192
    local_window_tokens: int = 262_144
    remote_window_tokens: int = 256_000
    keep_recent_turns: int = 6
    summary_max_tokens: int = 2_048

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        if self.reserve_tokens < 0:
            raise ValueError(
                f"reserve_tokens must be non-negative, got {self.reserve_tokens}"
            )
        if self.local_window_tokens < 1_024:
            raise ValueError(
                f"local_window_tokens must be at least 1024, "
                f"got {self.local_window_tokens}"
            )
        if self.remote_window_tokens < 1_024:
            raise ValueError(
                f"remote_window_tokens must be at least 1024, "
                f"got {self.remote_window_tokens}"
            )
        if self.keep_recent_turns < 1:
            raise ValueError(
                f"keep_recent_turns must be at least 1, got {self.keep_recent_turns}"
            )
        if self.summary_max_tokens < 256:
            raise ValueError(
                f"summary_max_tokens must be at least 256, "
                f"got {self.summary_max_tokens}"
            )

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

    @classmethod
    def from_env(cls) -> ContextConfig:
        """Build configuration from environment variables."""
        from mistralcli.local_mistral import (
            DEFAULT_CONTEXT_KEEP_RECENT_TURNS,
            DEFAULT_CONTEXT_RESERVE_TOKENS,
            DEFAULT_CONTEXT_SUMMARY_MAX_TOKENS,
            DEFAULT_CONTEXT_THRESHOLD,
            DEFAULT_LOCAL_CONTEXT_WINDOW_TOKENS,
            DEFAULT_REMOTE_CONTEXT_WINDOW_TOKENS,
            ENV_CONTEXT_AUTO_COMPACT,
            ENV_CONTEXT_KEEP_RECENT_TURNS,
            ENV_CONTEXT_LOCAL_WINDOW_TOKENS,
            ENV_CONTEXT_REMOTE_WINDOW_TOKENS,
            ENV_CONTEXT_RESERVE_TOKENS,
            ENV_CONTEXT_SUMMARY_MAX_TOKENS,
            ENV_CONTEXT_THRESHOLD,
        )

        def _env_bool(name: str, default: bool) -> bool:
            value = os.environ.get(name)
            if value is None:
                return default
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return default

        def _env_int(name: str, default: int) -> int:
            value = os.environ.get(name)
            if value is None or not value.strip():
                return default
            try:
                return int(value.strip())
            except ValueError:
                return default

        def _env_float(name: str, default: float) -> float:
            value = os.environ.get(name)
            if value is None or not value.strip():
                return default
            try:
                return float(value.strip().rstrip("%"))
            except ValueError:
                return default

        return cls(
            auto_compact=_env_bool(ENV_CONTEXT_AUTO_COMPACT, default=True),
            threshold=_env_float(
                ENV_CONTEXT_THRESHOLD, default=DEFAULT_CONTEXT_THRESHOLD
            ),
            reserve_tokens=_env_int(
                ENV_CONTEXT_RESERVE_TOKENS, default=DEFAULT_CONTEXT_RESERVE_TOKENS
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


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Runtime logging configuration for the CLI."""

    directory: Path = field(default_factory=lambda: _default_log_directory())
    debug_enabled: bool = True
    retention_days: int = 2
    file_name: str = "mistralcli.log"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.retention_days < 1:
            raise ValueError(
                f"retention_days must be at least 1, got {self.retention_days}"
            )

    @classmethod
    def from_env(cls) -> LoggingConfig:
        """Build configuration from environment variables."""
        # Import here to avoid circular imports
        import mistralcli.logging_config as _logging_config

        configured_dir = os.environ.get(_logging_config.ENV_LOG_DIR, "").strip()
        directory = (
            Path(configured_dir).expanduser()
            if configured_dir
            else _logging_config._default_log_directory()
        )
        retention_raw = os.environ.get(
            _logging_config.ENV_LOG_RETENTION_DAYS,
            str(_logging_config.DEFAULT_LOG_RETENTION_DAYS),
        )
        retention_days = max(1, int(retention_raw))
        debug_enabled = _logging_config._parse_bool(
            os.environ.get(_logging_config.ENV_DEBUG), default=True
        )

        return cls(
            directory=directory,
            debug_enabled=debug_enabled,
            retention_days=retention_days,
        )


@dataclass(frozen=True, slots=True)
class MCPConfig:
    """MCP (Model Context Protocol) configuration."""

    enabled: bool = True
    config_path: Path | str | None = None

    @classmethod
    def from_env(cls) -> MCPConfig:
        """Build configuration from environment variables."""
        # Import here to avoid circular imports
        import mistralcli.mcp_bridge as _mcp_bridge

        config_path = _mcp_bridge.discover_mcp_config_path(
            os.environ.get(_mcp_bridge.ENV_MCP_CONFIG)
        )
        return cls(
            enabled=config_path is not None,
            config_path=config_path,
        )


@dataclass(frozen=True, slots=True)
class UIConfig:
    """UI and REPL configuration."""

    stream_enabled: bool = True
    show_reasoning: bool = True
    show_thinking: bool = True
    system_prompt: str = (
        "You are a helpful assistant. Follow the user's instructions carefully. "
        "Use tools when available to get accurate information. "
        "If you don't know something, say so instead of making it up."
    )
    max_tool_rounds: int = 20

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_tool_rounds < 1:
            raise ValueError(
                f"max_tool_rounds must be at least 1, got {self.max_tool_rounds}"
            )


@dataclass(frozen=True, slots=True)
class RegistryConfig:
    """Conversation registry configuration."""

    conversation_index_path: Path | str | None = None

    @classmethod
    def from_env(cls) -> RegistryConfig:
        """Build configuration from environment variables."""
        # Use None to indicate default path should be used
        return cls(conversation_index_path=None)


def _default_log_directory() -> Path:
    """Return the default log directory."""
    xdg_state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg_state_home:
        return Path(xdg_state_home) / "mistralcli" / "logs"
    return Path.home() / ".local" / "state" / "mistralcli" / "logs"


@dataclass(slots=True)
class AppConfig:
    """Root configuration container for MistralCLI.

    This class holds all configuration options for the CLI and provides
    methods for loading, saving, and merging configurations from various sources.
    """

    version: str = "3.4"
    backend: BackendKind = BackendKind.LOCAL
    local: LocalBackendConfig = field(default_factory=LocalBackendConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    remote: RemoteBackendConfig = field(default_factory=RemoteBackendConfig)
    conversations: ConversationsConfig = field(default_factory=ConversationsConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)

    @classmethod
    def default(cls) -> AppConfig:
        """Return a new configuration with all default values."""
        return cls()

    @classmethod
    def load(cls, path: Path | str | None = None) -> AppConfig:
        """Load configuration from a file.

        Args:
            path: Optional explicit path to a configuration file.
                  If None, will search standard locations.

        Returns:
            AppConfig instance with loaded values.

        Raises:
            FileNotFoundError: If path is provided but doesn't exist.
            ValueError: If configuration file has invalid format or values.
        """
        # Determine the config file path
        config_path = cls._resolve_config_path(path)

        if config_path is None:
            logger.debug("No configuration file found, using defaults")
            return cls.default()

        logger.debug("Loading configuration from %s", config_path)

        # Load the raw configuration data
        raw_data = cls._load_raw_config(config_path)

        # Build AppConfig from raw data
        return cls._build_from_data(raw_data, config_path)

    @classmethod
    def _resolve_config_path(cls, explicit_path: Path | str | None) -> Path | None:
        """Resolve the configuration file path from various sources."""
        # 1. Explicit path provided
        if explicit_path is not None:
            path = Path(explicit_path).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # 2. Environment variable
        env_path = os.environ.get(ENV_CONFIG_PATH)
        if env_path:
            path = Path(env_path).expanduser()
            if path.exists():
                return path
            logger.warning(
                "Configuration file from %s not found: %s", ENV_CONFIG_PATH, path
            )

        # 3. Standard locations
        locations = [
            DEFAULT_CONFIG_FILE,
            Path.home() / ".mistralcli.yaml",
            Path.home() / ".mistralcli.json",
            Path.home() / ".mistralcli.toml",
            Path.cwd() / "mistralcli.yaml",
            Path.cwd() / "mistralcli.json",
            Path.cwd() / "mistralcli.toml",
        ]

        for location in locations:
            if location.exists():
                logger.debug("Found configuration file at %s", location)
                return location

        return None

    @classmethod
    def _load_raw_config(cls, path: Path) -> dict[str, Any]:
        """Load raw configuration data from a file."""
        path = path.expanduser()

        # Determine format from extension
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return cls._load_yaml(path)
        elif suffix == ".json":
            return cls._load_json(path)
        elif suffix == ".toml":
            return cls._load_toml(path)
        else:
            # Try to detect format from content
            return cls._load_auto(path)

    @classmethod
    def _load_yaml(cls, path: Path) -> dict[str, Any]:
        """Load YAML configuration file."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML configuration. "
                "Install it with: pip install pyyaml"
            ) from exc

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration file must contain a mapping, got {type(data)}"
            )
        return data

    @classmethod
    def _load_json(cls, path: Path) -> dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration file must contain an object, got {type(data)}"
            )

        return data

    @classmethod
    def _load_toml(cls, path: Path) -> dict[str, Any]:
        """Load TOML configuration file."""
        # Try Python 3.11+ tomllib first
        try:
            import tomllib  # type: ignore[import-not-found]

            with open(path, "rb") as f:
                data = tomllib.load(f)
        except ImportError:
            # Fall back to tomli for Python 3.10
            try:
                import tomli

                with open(path, "rb") as f:
                    data = tomli.load(f)
            except ImportError as exc:
                raise ImportError(
                    "tomllib (Python 3.11+) or tomli is required to load TOML "
                    "configuration. Install tomli with: pip install tomli"
                ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration file must contain a mapping, got {type(data)}"
            )

        return data

    @classmethod
    def _load_auto(cls, path: Path) -> dict[str, Any]:
        """Auto-detect configuration file format."""
        # Try YAML first (most common)
        try:
            return cls._load_yaml(path)
        except (ImportError, ValueError):
            pass

        # Try JSON
        try:
            return cls._load_json(path)
        except (ValueError, json.JSONDecodeError):
            pass

        # Try TOML
        try:
            return cls._load_toml(path)
        except (ImportError, ValueError):
            pass

        raise ValueError(f"Could not determine format of configuration file: {path}")

    @classmethod
    def _build_from_data(cls, data: dict[str, Any], config_path: Path) -> AppConfig:
        """Build AppConfig from raw configuration data."""
        # Extract version
        version = str(data.get("version", "3.4"))

        # Extract backend
        backend_str = str(data.get("backend", "local")).lower()
        try:
            backend = BackendKind(backend_str)
        except ValueError:
            logger.warning(
                "Invalid backend value '%s', defaulting to 'local'", backend_str
            )
            backend = BackendKind.LOCAL

        # Build sub-configurations
        local = cls._build_local_config(data.get("local", {}))
        generation = cls._build_generation_config(data.get("generation", {}))
        remote = cls._build_remote_config(data.get("remote", {}))
        conversations = cls._build_conversations_config(data.get("conversations", {}))
        context = cls._build_context_config(data.get("context", {}))
        logging = cls._build_logging_config(data.get("logging", {}))
        mcp = cls._build_mcp_config(data.get("mcp", {}), config_path)
        ui = cls._build_ui_config(data.get("ui", {}))
        registry = cls._build_registry_config(data.get("registry", {}))

        return cls(
            version=version,
            backend=backend,
            local=local,
            generation=generation,
            remote=remote,
            conversations=conversations,
            context=context,
            logging=logging,
            mcp=mcp,
            ui=ui,
            registry=registry,
        )

    @classmethod
    def _build_local_config(cls, data: dict[str, Any]) -> LocalBackendConfig:
        """Build LocalBackendConfig from data."""
        return LocalBackendConfig(
            api_key=str(data.get("api_key", "local-test")),
            model_id=str(
                data.get(
                    "model_id", "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
                )
            ),
            server_url=str(data.get("server_url", "http://127.0.0.1:8080")),
            timeout_ms=int(data.get("timeout_ms", 300_000)),
        )

    @classmethod
    def _build_generation_config(cls, data: dict[str, Any]) -> GenerationConfig:
        """Build GenerationConfig from data."""
        prompt_mode = data.get("prompt_mode")
        if prompt_mode is None:
            prompt_mode = "reasoning"
        elif isinstance(prompt_mode, str):
            prompt_mode = prompt_mode.strip()
            if prompt_mode.lower() in {"", "none", "null", "off"}:
                prompt_mode = None

        return GenerationConfig(
            temperature=float(data.get("temperature", 0.3)),
            top_p=float(data.get("top_p", 0.95)),
            prompt_mode=prompt_mode,
            max_tokens=(
                int(data["max_tokens"])
                if isinstance(data.get("max_tokens"), int)
                else None
            ),
        )

    @classmethod
    def _build_remote_config(cls, data: dict[str, Any]) -> RemoteBackendConfig:
        """Build RemoteBackendConfig from data."""
        model_id = str(data.get("model_id", "mistral-small-latest"))
        # Normalize model ID
        try:
            from mistralcli.local_mistral import normalize_remote_model_id

            model_id = normalize_remote_model_id(model_id)
        except ValueError:
            logger.warning("Invalid remote model ID '%s', using default", model_id)
            model_id = "mistral-small-latest"

        return RemoteBackendConfig(
            model_id=model_id,
            timeout_ms=int(data.get("timeout_ms", 300_000)),
        )

    @classmethod
    def _build_conversations_config(cls, data: dict[str, Any]) -> ConversationsConfig:
        """Build ConversationsConfig from data."""
        resume_policy = str(data.get("resume_policy", "last")).lower()
        try:
            resume_policy = ResumePolicy(resume_policy)
        except ValueError:
            logger.warning(
                "Invalid resume_policy '%s', defaulting to 'last'", resume_policy
            )
            resume_policy = ResumePolicy.LAST

        return ConversationsConfig(
            enabled=bool(data.get("enabled", False)),
            store=bool(data.get("store", True)),
            resume_policy=resume_policy,
        )

    @classmethod
    def _build_context_config(cls, data: dict[str, Any]) -> ContextConfig:
        """Build ContextConfig from data."""
        return ContextConfig(
            auto_compact=bool(data.get("auto_compact", True)),
            threshold=float(data.get("threshold", 0.9)),
            reserve_tokens=int(data.get("reserve_tokens", 8_192)),
            local_window_tokens=int(data.get("local_window_tokens", 262_144)),
            remote_window_tokens=int(data.get("remote_window_tokens", 256_000)),
            keep_recent_turns=int(data.get("keep_recent_turns", 6)),
            summary_max_tokens=int(data.get("summary_max_tokens", 2_048)),
        ).normalized()

    @classmethod
    def _build_logging_config(cls, data: dict[str, Any]) -> LoggingConfig:
        """Build LoggingConfig from data."""
        directory = data.get("directory")
        if directory is None:
            directory = _default_log_directory()
        elif isinstance(directory, str):
            directory = Path(directory).expanduser()

        return LoggingConfig(
            directory=directory,
            debug_enabled=bool(data.get("debug_enabled", True)),
            retention_days=int(data.get("retention_days", 2)),
            file_name=str(data.get("file_name", "mistralcli.log")),
        )

    @classmethod
    def _build_mcp_config(cls, data: dict[str, Any], config_path: Path) -> MCPConfig:
        """Build MCPConfig from data."""
        import mistralcli.mcp_bridge as _mcp_bridge

        enabled = bool(data.get("enabled", True))
        config_path_str = data.get("config_path")

        if config_path_str is None:
            # Use default mcp.json path
            mcp_config_path = _mcp_bridge._default_config_path()
        elif isinstance(config_path_str, str):
            mcp_config_path = Path(
                _mcp_bridge._expand_env_variables(config_path_str)
            ).expanduser()
        else:
            mcp_config_path = None

        return MCPConfig(
            enabled=enabled
            and (
                config_path_str is not None
                or (mcp_config_path is not None and mcp_config_path.exists())
            ),
            config_path=mcp_config_path,
        )

    @classmethod
    def _build_ui_config(cls, data: dict[str, Any]) -> UIConfig:
        """Build UIConfig from data."""
        return UIConfig(
            stream_enabled=bool(data.get("stream_enabled", True)),
            show_reasoning=bool(data.get("show_reasoning", True)),
            show_thinking=bool(data.get("show_thinking", True)),
            system_prompt=str(
                data.get(
                    "system_prompt",
                    "You are a helpful assistant. Follow the user's instructions "
                    "carefully. Use tools when available to get accurate "
                    "information. If you don't know something, say so instead of "
                    "making it up.",
                )
            ),
            max_tool_rounds=int(data.get("max_tool_rounds", 20)),
        )

    @classmethod
    def _build_registry_config(cls, data: dict[str, Any]) -> RegistryConfig:
        """Build RegistryConfig from data."""
        path = data.get("conversation_index_path")
        if path is None:
            return RegistryConfig(conversation_index_path=None)

        if isinstance(path, str):
            path = Path(path).expanduser()

        return RegistryConfig(conversation_index_path=path)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary for serialization."""
        result: dict[str, Any] = {}

        # Convert each sub-config to dict
        for field_name in dataclasses.fields(self):
            field_value = getattr(self, field_name.name)
            if dataclasses.is_dataclass(field_value) and not isinstance(
                field_value, type
            ):
                result[field_name.name] = self._dataclass_to_dict(field_value)
            elif isinstance(field_value, Path):
                result[field_name.name] = str(field_value)
            elif isinstance(field_value, Enum):
                result[field_name.name] = field_value.value
            else:
                result[field_name.name] = field_value

        return result

    def _dataclass_to_dict(self, obj: Any) -> dict[str, Any] | Any:
        """Recursively convert a dataclass to a dictionary."""
        if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
            return obj

        result: dict[str, Any] = {}
        for field_name in dataclasses.fields(obj):
            field_value = getattr(obj, field_name.name)
            if dataclasses.is_dataclass(field_value) and not isinstance(
                field_value, type
            ):
                result[field_name.name] = self._dataclass_to_dict(field_value)
            elif isinstance(field_value, Path):
                result[field_name.name] = str(field_value)
            elif isinstance(field_value, Enum):
                result[field_name.name] = field_value.value
            else:
                result[field_name.name] = field_value

        return result

    def save(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where to save the configuration.

        Raises:
            ImportError: If PyYAML is not installed.
            IOError: If file cannot be written.
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import yaml as yaml_module
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to save YAML configuration. "
                "Install it with: pip install pyyaml"
            ) from exc

        with open(path, "w", encoding="utf-8") as f:
            yaml_module.dump(self.to_dict(), f, sort_keys=False)

        logger.info("Configuration saved to %s", path)

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build configuration from environment variables only."""
        return cls(
            version="3.4",
            backend=BackendKind.LOCAL,
            local=LocalBackendConfig.from_env(),
            generation=GenerationConfig.from_env(),
            remote=RemoteBackendConfig(
                model_id="mistral-small-latest", timeout_ms=300_000
            ),
            conversations=ConversationsConfig.from_env(),
            context=ContextConfig.from_env(),
            logging=LoggingConfig.from_env(),
            mcp=MCPConfig.from_env(),
            ui=UIConfig(),
            registry=RegistryConfig.from_env(),
        )
