"""Argument parsing and session bootstrap helpers for the CLI."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from mistralcli.conversation_registry import ConversationRegistry

if TYPE_CHECKING:
    from mistralcli.config import AppConfig
from mistralcli.local_mistral import (
    DEFAULT_API_KEY,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_MODE,
    DEFAULT_SERVER_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TOP_P,
    ENV_API_KEY,
    ENV_CONTEXT_AUTO_COMPACT,
    ENV_CONTEXT_KEEP_RECENT_TURNS,
    ENV_CONTEXT_LOCAL_WINDOW_TOKENS,
    ENV_CONTEXT_REMOTE_WINDOW_TOKENS,
    ENV_CONTEXT_RESERVE_TOKENS,
    ENV_CONTEXT_SUMMARY_MAX_TOKENS,
    ENV_CONTEXT_THRESHOLD,
    ENV_CONVERSATION_RESUME,
    ENV_CONVERSATION_STORE,
    ENV_CONVERSATIONS,
    ENV_MAX_TOKENS,
    ENV_MODEL_ID,
    ENV_PROMPT_MODE,
    ENV_REMOTE_MODEL_ID,
    ENV_SERVER_URL,
    ENV_TEMPERATURE,
    ENV_TIMEOUT_MS,
    ENV_TOP_P,
    REMOTE_MEDIUM_MODEL_ID,
    REMOTE_MODEL_ID,
    BackendKind,
    ContextConfig,
    ConversationConfig,
    LocalGenerationConfig,
    LocalMistralConfig,
    MistralConfig,
    RemoteMistralConfig,
    normalize_remote_model_id,
)
from mistralcli.local_tools import LocalToolBridge
from mistralcli.logging_config import LoggingConfig
from mistralcli.mcp_bridge import MCPConfig, MCPToolBridge, discover_mcp_config_path
from mistralcli.mistral_client import MistralClientProtocol
from mistralcli.session import (
    DEFAULT_SYSTEM_PROMPT,
    MistralSession,
    PendingConversationSettings,
)
from mistralcli.tooling import CompositeToolBridge, ToolBridge

logger = logging.getLogger("mistralcli.cli")


def _optional_prompt_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower() in {"", "none", "null", "off"}:
        return None
    return normalized


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the CLI."""

    parser = argparse.ArgumentParser(
        prog="mistralcli",
        description=(
            "Interactive multimodal CLI for Mistral models across "
            "local llama.cpp and Mistral Cloud backends."
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Print the installed CLI version and exit.",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help=f"Local server URL (default: {DEFAULT_SERVER_URL}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model identifier (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--remote-model",
        default=None,
        help=(
            "Remote Mistral Cloud model identifier or alias "
            f"(default: {REMOTE_MODEL_ID}; supported: {REMOTE_MODEL_ID}, "
            f"{REMOTE_MEDIUM_MODEL_ID}; env: {ENV_REMOTE_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"API key placeholder for the local server (default: {DEFAULT_API_KEY}).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help=f"Request timeout in milliseconds (default: {DEFAULT_TIMEOUT_MS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=f"Nucleus sampling top-p (default: {DEFAULT_TOP_P}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional maximum generated tokens; unset by default.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=_optional_prompt_mode,
        default=None,
        help=(
            "Prompt mode for the local template; use 'reasoning' or 'none' "
            f"(default: {DEFAULT_PROMPT_MODE})."
        ),
    )
    parser.add_argument(
        "--conversations",
        dest="conversations",
        action="store_true",
        default=None,
        help="Start in Mistral Cloud Conversations mode.",
    )
    parser.add_argument(
        "--no-conversations",
        dest="conversations",
        action="store_false",
        help="Disable Conversations mode even if enabled by environment.",
    )
    parser.add_argument(
        "--conversation-store",
        choices=("on", "off"),
        default=None,
        help="Persist Mistral Conversations server-side (default: on).",
    )
    parser.add_argument(
        "--conversation-resume",
        choices=("last", "new", "prompt"),
        default=None,
        help="How Conversations mode resumes the last known remote conversation.",
    )
    parser.add_argument(
        "--conversation-name",
        default=None,
        help="Pending remote conversation name for the next start or restart.",
    )
    parser.add_argument(
        "--conversation-description",
        default=None,
        help="Pending remote conversation description for the next start or restart.",
    )
    parser.add_argument(
        "--conversation-meta",
        action="append",
        default=None,
        help="Pending remote conversation metadata pair in KEY=VALUE form.",
    )
    parser.add_argument(
        "--conversation-index",
        default=None,
        help="Path to the persistent local conversation registry JSON file.",
    )
    parser.add_argument(
        "--no-auto-compact",
        dest="auto_compact",
        action="store_false",
        default=None,
        help="Disable automatic context compaction before overflowing the window.",
    )
    parser.add_argument(
        "--auto-compact",
        dest="auto_compact",
        action="store_true",
        help="Enable automatic context compaction.",
    )
    parser.add_argument(
        "--compact-threshold",
        type=float,
        default=None,
        help="Context compaction threshold as 0-1 or percent (default: 90).",
    )
    parser.add_argument(
        "--context-reserve-tokens",
        type=int,
        default=None,
        help="Tokens reserved for the model response during context checks.",
    )
    parser.add_argument(
        "--context-local-window-tokens",
        type=int,
        default=None,
        help="Configured local context window tokens (default: 262144).",
    )
    parser.add_argument(
        "--context-remote-window-tokens",
        type=int,
        default=None,
        help="Fallback remote context window tokens (default: 256000).",
    )
    parser.add_argument(
        "--context-keep-turns",
        type=int,
        default=None,
        help="Recent user turns preserved when compacting context (default: 6).",
    )
    parser.add_argument(
        "--context-summary-max-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens for compact summaries (default: 2048).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override the default assistant system prompt.",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help=(
            "Path to mcp.json (default: ./mcp.json or "
            "${MISTRAL_LOCAL_MCP_CONFIG} if set)."
        ),
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable the FireCrawl MCP backend for this run.",
    )
    parser.add_argument(
        "--once",
        default=None,
        help="Send one prompt and exit instead of opening the REPL.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming and wait for the full response.",
    )
    parser.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_true",
        default=None,
        help="Request backend reasoning when the backend supports it.",
    )
    parser.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        help="Do not request reasoning from the backend.",
    )
    parser.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        default=None,
        help="Render returned thinking blocks in the terminal.",
    )
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Hide returned thinking blocks without disabling reasoning requests.",
    )
    parser.add_argument(
        "--print-defaults",
        action="store_true",
        help="Print the effective defaults and exit.",
    )
    parser.add_argument(
        "--log-dir", default=None, help="Directory for rotated log files."
    )
    parser.add_argument(
        "--log-retention-days",
        type=int,
        default=None,
        help="Keep rotated log files for this many days (default: 2).",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=None,
        help="Enable debug logging (default: on).",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Reduce logging verbosity to info.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help=(
            "Path to configuration file (YAML, JSON, or TOML). "
            "Search order: explicit path, $MISTRAL_CONFIG_PATH, "
            "~/.config/mistralcli/config.yaml, ~/.mistralcli.{yaml|json|toml}, "
            "./mistralcli.{yaml|json|toml}."
        ),
    )
    parser.add_argument(
        "--generate-config",
        default=None,
        nargs="?",
        const="",
        metavar="PATH",
        help=(
            "Generate a sample configuration file and exit. "
            "If PATH is '-', prints to stdout. "
            "Otherwise saves to PATH (default: ~/.config/mistralcli/config.yaml)."
        ),
    )
    return parser


def _resolve_local_configs(
    args: argparse.Namespace,
) -> tuple[LocalMistralConfig, LocalGenerationConfig, str]:
    base_config = LocalMistralConfig.from_env()
    generation = LocalGenerationConfig.from_env()

    resolved_config = replace(
        base_config,
        api_key=args.api_key if args.api_key is not None else base_config.api_key,
        model_id=args.model if args.model is not None else base_config.model_id,
        server_url=(
            args.server_url if args.server_url is not None else base_config.server_url
        ),
        timeout_ms=(
            args.timeout_ms if args.timeout_ms is not None else base_config.timeout_ms
        ),
    )
    resolved_generation = replace(
        generation,
        temperature=(
            args.temperature if args.temperature is not None else generation.temperature
        ),
        top_p=args.top_p if args.top_p is not None else generation.top_p,
        max_tokens=(
            args.max_tokens if args.max_tokens is not None else generation.max_tokens
        ),
        prompt_mode=(
            args.prompt_mode if args.prompt_mode is not None else generation.prompt_mode
        ),
    )
    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    return resolved_config, resolved_generation, system_prompt


def _resolve_conversation_config(args: argparse.Namespace) -> ConversationConfig:
    config = ConversationConfig.from_env()
    enabled = config.enabled if args.conversations is None else bool(args.conversations)
    store = config.store
    resume_policy = config.resume_policy
    if args.conversation_store is not None:
        store = args.conversation_store == "on"
    if args.conversation_resume is not None:
        resume_policy = args.conversation_resume
    return ConversationConfig(
        enabled=enabled,
        store=store,
        resume_policy=resume_policy,
    )


def _resolve_remote_model_id(args: argparse.Namespace) -> str:
    """Resolve the selected remote model id from CLI args and environment."""

    return normalize_remote_model_id(args.remote_model)


def _resolve_context_config(args: argparse.Namespace) -> ContextConfig:
    config = ContextConfig.from_env()
    return replace(
        config,
        auto_compact=(
            config.auto_compact
            if args.auto_compact is None
            else bool(args.auto_compact)
        ),
        threshold=(
            config.threshold
            if args.compact_threshold is None
            else args.compact_threshold
        ),
        reserve_tokens=(
            config.reserve_tokens
            if args.context_reserve_tokens is None
            else args.context_reserve_tokens
        ),
        local_window_tokens=(
            config.local_window_tokens
            if args.context_local_window_tokens is None
            else args.context_local_window_tokens
        ),
        remote_window_tokens=(
            config.remote_window_tokens
            if args.context_remote_window_tokens is None
            else args.context_remote_window_tokens
        ),
        keep_recent_turns=(
            config.keep_recent_turns
            if args.context_keep_turns is None
            else args.context_keep_turns
        ),
        summary_max_tokens=(
            config.summary_max_tokens
            if args.context_summary_max_tokens is None
            else args.context_summary_max_tokens
        ),
    ).normalized()


def _resolve_reasoning_visibility(args: argparse.Namespace) -> bool:
    return True if args.reasoning is None else bool(args.reasoning)


def _resolve_thinking_visibility(args: argparse.Namespace) -> bool:
    return True if args.thinking is None else bool(args.thinking)


def _parse_metadata_pairs(values: Sequence[str] | None) -> dict[str, str]:
    """Parse repeated KEY=VALUE CLI entries into a dictionary."""

    parsed: dict[str, str] = {}
    for raw_value in values or []:
        key, separator, value = raw_value.partition("=")
        if not separator or not key.strip():
            raise ValueError(
                f"Invalid metadata entry {raw_value!r}; use KEY=VALUE syntax."
            )
        parsed[key.strip()] = value.strip()
    return parsed


def _resolve_conversation_registry(args: argparse.Namespace) -> ConversationRegistry:
    """Load the persistent local conversation registry."""

    return ConversationRegistry.load(args.conversation_index)


def _resolve_remote_mcp_bridge(
    args: argparse.Namespace,
    stderr: TextIO,
) -> MCPToolBridge | None:
    if args.no_mcp:
        logger.info("MCP disabled via --no-mcp")
        return None

    config_path = discover_mcp_config_path(args.mcp_config)
    if config_path is None:
        logger.debug("No MCP configuration discovered")
        return None

    try:
        config = MCPConfig.load(config_path)
    except FileNotFoundError:
        logger.warning("MCP configuration not found path=%s", config_path)
        if args.mcp_config is not None or os.environ.get("MISTRAL_LOCAL_MCP_CONFIG"):
            stderr.write(f"[mcp] config not found: {config_path}\n")
            stderr.flush()
        return None
    except Exception as exc:
        logger.exception("Could not load MCP configuration path=%s", config_path)
        stderr.write(f"[mcp] could not load {config_path}: {exc}\n")
        stderr.flush()
        return None

    if not config.configured:
        logger.info("MCP configuration loaded but no servers are configured")
        return None

    logger.info("MCP bridge enabled path=%s", config_path)
    return MCPToolBridge(config)


def _build_tool_bridge(
    args: argparse.Namespace,
    stderr: TextIO,
) -> CompositeToolBridge:
    bridges: list[ToolBridge] = [LocalToolBridge()]
    remote_bridge = _resolve_remote_mcp_bridge(args, stderr)
    if remote_bridge is not None:
        bridges.append(remote_bridge)
    logger.debug("Built tool bridge count=%s", len(bridges))
    return CompositeToolBridge(bridges=bridges)


def _resolve_logging_config(args: argparse.Namespace) -> LoggingConfig:
    base_config = LoggingConfig.from_env()
    retention_days = (
        args.log_retention_days
        if args.log_retention_days is not None
        else base_config.retention_days
    )
    return replace(
        base_config,
        directory=(
            Path(args.log_dir).expanduser()
            if args.log_dir is not None
            else base_config.directory
        ),
        debug_enabled=(
            args.debug if args.debug is not None else base_config.debug_enabled
        ),
        retention_days=max(1, retention_days),
    )


def _build_session(
    *,
    client_factory: Callable[[MistralConfig], MistralClientProtocol],
    config: LocalMistralConfig,
    generation: LocalGenerationConfig,
    system_prompt: str,
    tool_bridge: ToolBridge,
    stdout: TextIO,
    stream: bool,
    reasoning_enabled: bool,
    thinking_visible: bool,
    logging_summary: str,
    conversations: ConversationConfig,
    context: ContextConfig,
    conversation_registry: ConversationRegistry,
    pending_conversation: PendingConversationSettings,
    remote_model_id: str,
) -> MistralSession:
    logger.debug(
        "Building session backend=%s model=%s stream=%s",
        BackendKind.LOCAL.value,
        config.model_id,
        stream,
    )
    session = MistralSession(
        client=client_factory(config),
        backend_kind=BackendKind.LOCAL,
        model_id=config.model_id,
        server_url=config.server_url,
        generation=generation,
        system_prompt=system_prompt,
        tool_bridge=tool_bridge,
        stdout=stdout,
        stream_enabled=stream,
        show_reasoning=reasoning_enabled,
        show_thinking=thinking_visible,
        logging_summary=logging_summary,
        context=context,
        conversation_registry=conversation_registry,
    )
    session.pending_conversation = pending_conversation
    if conversations.enabled:
        remote_config = RemoteMistralConfig.from_env(
            timeout_ms=config.timeout_ms,
            model_id=remote_model_id,
        )
        session.enable_conversations(
            client=client_factory(remote_config),
            model_id=remote_config.model_id,
            store=conversations.store,
        )
        session.apply_conversation_resume_policy(conversations.resume_policy)
    else:
        session.conversations = conversations
    return session


def _load_app_config(args: argparse.Namespace) -> tuple[AppConfig | None, str]:
    """Load application configuration from file.

    Returns a tuple of (AppConfig or None, source description string).
    """
    from mistralcli.config import AppConfig

    config_path = args.config_path
    try:
        app_config = AppConfig.load(config_path)
        if app_config is not None:
            resolved = AppConfig._resolve_config_path(config_path)
            source = str(resolved) if resolved else "defaults"
        else:
            source = "defaults"
        logger.debug(
            "Loaded app configuration from %s", config_path or "default locations"
        )
        return app_config, source
    except FileNotFoundError as e:
        if config_path is not None:
            logger.warning("Configuration file not found: %s", e)
        return None, "defaults"
    except (ValueError, ImportError) as e:
        logger.warning("Error loading configuration: %s", e)
        return None, "defaults"


def _generate_app_config(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Generate a sample configuration file and exit.

    Returns exit code (0 on success, 1 on error).
    """
    from mistralcli.config import AppConfig

    try:
        # Create default configuration
        config = AppConfig.default()

        # Convert to dictionary for serialization
        config_dict = config.to_dict()

        # Determine output path
        output_path = args.generate_config
        if output_path == "-":
            # Output to stdout
            try:
                import yaml  # type: ignore[import-untyped]

                stdout.write(yaml.dump(config_dict, sort_keys=False))
                stdout.write("\n")
            except ImportError:
                import json

                stdout.write(json.dumps(config_dict, indent=2))
                stdout.write("\n")
            stdout.flush()
            return 0

        # Save to file
        if not output_path:
            # Use default location
            from mistralcli.config import DEFAULT_CONFIG_DIR

            output_path = DEFAULT_CONFIG_DIR / "config.yaml"
        else:
            output_path = Path(output_path).expanduser()

        try:
            config.save(output_path)
            stdout.write(f"Configuration file generated: {output_path}\n")
            stdout.flush()
            return 0
        except ImportError as e:
            stderr.write(f"Error: {e}\n")
            stderr.write(
                "Install PyYAML to generate YAML configuration: pip install pyyaml\n"
            )
            stderr.flush()
            return 1

    except Exception as e:
        stderr.write(f"Error generating configuration: {e}\n")
        stderr.flush()
        return 1


def _merge_config_with_args(
    args: argparse.Namespace,
    app_config: AppConfig | None,
) -> tuple[
    LocalMistralConfig, LocalGenerationConfig, str, ConversationConfig, ContextConfig
]:
    """Merge configuration from file with command-line arguments.

    Precedence (highest to lowest):
    1. Command-line arguments
    2. Environment variables
    3. Configuration file
    4. Default values

    Returns a tuple of resolved configurations.
    """
    # Always start from env (which itself falls back to defaults).
    # Then overlay file values for fields whose env var is NOT explicitly set.
    env_config = LocalMistralConfig.from_env()
    env_generation = LocalGenerationConfig.from_env()
    env_conversations = ConversationConfig.from_env()
    env_context = ContextConfig.from_env()

    if app_config is not None:
        # For each field, use env value if its env var is set, else file value.
        base_config = LocalMistralConfig(
            api_key=(
                env_config.api_key
                if os.environ.get(ENV_API_KEY)
                else app_config.local.api_key
            ),
            model_id=(
                env_config.model_id
                if os.environ.get(ENV_MODEL_ID)
                else app_config.local.model_id
            ),
            server_url=(
                env_config.server_url
                if os.environ.get(ENV_SERVER_URL)
                else app_config.local.server_url
            ),
            timeout_ms=(
                env_config.timeout_ms
                if os.environ.get(ENV_TIMEOUT_MS)
                else app_config.local.timeout_ms
            ),
        )
        generation = LocalGenerationConfig(
            temperature=(
                env_generation.temperature
                if os.environ.get(ENV_TEMPERATURE)
                else app_config.generation.temperature
            ),
            top_p=(
                env_generation.top_p
                if os.environ.get(ENV_TOP_P)
                else app_config.generation.top_p
            ),
            prompt_mode=(
                env_generation.prompt_mode
                if os.environ.get(ENV_PROMPT_MODE)
                else app_config.generation.prompt_mode
            ),
            max_tokens=(
                env_generation.max_tokens
                if os.environ.get(ENV_MAX_TOKENS)
                else app_config.generation.max_tokens
            ),
        )
        conversations = ConversationConfig(
            enabled=(
                env_conversations.enabled
                if os.environ.get(ENV_CONVERSATIONS)
                else app_config.conversations.enabled
            ),
            store=(
                env_conversations.store
                if os.environ.get(ENV_CONVERSATION_STORE)
                else app_config.conversations.store
            ),
            resume_policy=(
                env_conversations.resume_policy
                if os.environ.get(ENV_CONVERSATION_RESUME)
                else (
                    app_config.conversations.resume_policy.value
                    if isinstance(app_config.conversations.resume_policy, Enum)
                    else app_config.conversations.resume_policy
                )
            ),
        )
        context = ContextConfig(
            auto_compact=(
                env_context.auto_compact
                if os.environ.get(ENV_CONTEXT_AUTO_COMPACT)
                else app_config.context.auto_compact
            ),
            threshold=(
                env_context.threshold
                if os.environ.get(ENV_CONTEXT_THRESHOLD)
                else app_config.context.threshold
            ),
            reserve_tokens=(
                env_context.reserve_tokens
                if os.environ.get(ENV_CONTEXT_RESERVE_TOKENS)
                else app_config.context.reserve_tokens
            ),
            local_window_tokens=(
                env_context.local_window_tokens
                if os.environ.get(ENV_CONTEXT_LOCAL_WINDOW_TOKENS)
                else app_config.context.local_window_tokens
            ),
            remote_window_tokens=(
                env_context.remote_window_tokens
                if os.environ.get(ENV_CONTEXT_REMOTE_WINDOW_TOKENS)
                else app_config.context.remote_window_tokens
            ),
            keep_recent_turns=(
                env_context.keep_recent_turns
                if os.environ.get(ENV_CONTEXT_KEEP_RECENT_TURNS)
                else app_config.context.keep_recent_turns
            ),
            summary_max_tokens=(
                env_context.summary_max_tokens
                if os.environ.get(ENV_CONTEXT_SUMMARY_MAX_TOKENS)
                else app_config.context.summary_max_tokens
            ),
        ).normalized()
    else:
        base_config = env_config
        generation = env_generation
        conversations = env_conversations
        context = env_context

    # Override with command-line arguments (highest precedence)
    resolved_config = replace(
        base_config,
        api_key=args.api_key if args.api_key is not None else base_config.api_key,
        model_id=args.model if args.model is not None else base_config.model_id,
        server_url=(
            args.server_url if args.server_url is not None else base_config.server_url
        ),
        timeout_ms=(
            args.timeout_ms if args.timeout_ms is not None else base_config.timeout_ms
        ),
    )
    resolved_generation = replace(
        generation,
        temperature=(
            args.temperature if args.temperature is not None else generation.temperature
        ),
        top_p=args.top_p if args.top_p is not None else generation.top_p,
        max_tokens=(
            args.max_tokens if args.max_tokens is not None else generation.max_tokens
        ),
        prompt_mode=(
            args.prompt_mode if args.prompt_mode is not None else generation.prompt_mode
        ),
    )
    resolved_conversations = replace(
        conversations,
        enabled=(
            conversations.enabled
            if args.conversations is None
            else bool(args.conversations)
        ),
        store=(
            conversations.store
            if args.conversation_store is None
            else args.conversation_store == "on"
        ),
        resume_policy=(
            conversations.resume_policy
            if args.conversation_resume is None
            else args.conversation_resume
        ),
    )
    resolved_context = replace(
        context,
        auto_compact=(
            context.auto_compact
            if args.auto_compact is None
            else bool(args.auto_compact)
        ),
        threshold=(
            context.threshold
            if args.compact_threshold is None
            else args.compact_threshold
        ),
        reserve_tokens=(
            context.reserve_tokens
            if args.context_reserve_tokens is None
            else args.context_reserve_tokens
        ),
        local_window_tokens=(
            context.local_window_tokens
            if args.context_local_window_tokens is None
            else args.context_local_window_tokens
        ),
        remote_window_tokens=(
            context.remote_window_tokens
            if args.context_remote_window_tokens is None
            else args.context_remote_window_tokens
        ),
        keep_recent_turns=(
            context.keep_recent_turns
            if args.context_keep_turns is None
            else args.context_keep_turns
        ),
        summary_max_tokens=(
            context.summary_max_tokens
            if args.context_summary_max_tokens is None
            else args.context_summary_max_tokens
        ),
    ).normalized()

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        resolved_config,
        resolved_generation,
        system_prompt,
        resolved_conversations,
        resolved_context,
    )
