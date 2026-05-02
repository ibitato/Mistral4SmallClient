"""Argument parsing and session bootstrap helpers for the CLI."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from mistral4cli.conversation_registry import ConversationRegistry
from mistral4cli.local_mistral import (
    DEFAULT_API_KEY,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_MODE,
    DEFAULT_SERVER_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TOP_P,
    ENV_REMOTE_MODEL_ID,
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
from mistral4cli.local_tools import LocalToolBridge
from mistral4cli.logging_config import LoggingConfig
from mistral4cli.mcp_bridge import MCPConfig, MCPToolBridge, discover_mcp_config_path
from mistral4cli.mistral_client import MistralClientProtocol
from mistral4cli.session import (
    DEFAULT_SYSTEM_PROMPT,
    MistralSession,
    PendingConversationSettings,
)
from mistral4cli.tooling import CompositeToolBridge, ToolBridge

logger = logging.getLogger("mistral4cli.cli")


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
        prog="mistral4cli",
        description=(
            "Interactive multimodal CLI for using and testing "
            "Mistral Small 4 and Mistral Medium 3.5."
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
