"""Linux-only command-line entrypoint for the general Mistral Small 4 CLI."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TextIO

from mistralai.client import Mistral

from mistral4cli import __version__
from mistral4cli.attachments import (
    DOCUMENT_FILETYPES,
    DOCUMENT_SUFFIXES,
    IMAGE_FILETYPES,
    IMAGE_SUFFIXES,
    PathPicker,
    build_document_message,
    build_image_message,
    build_remote_document_message,
    build_remote_image_message,
    choose_paths,
    format_selection_summary,
)
from mistral4cli.conversation_registry import ConversationRegistry
from mistral4cli.local_mistral import (
    DEFAULT_API_KEY,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_MODE,
    DEFAULT_SERVER_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TOP_P,
    REMOTE_MODEL_ID,
    BackendKind,
    ContextConfig,
    ConversationConfig,
    LocalGenerationConfig,
    LocalMistralConfig,
    MistralConfig,
    RemoteAPIKeyError,
    RemoteMistralConfig,
    build_client,
    get_client_timeout_ms,
    remote_api_key_available,
)
from mistral4cli.local_tools import LocalToolBridge
from mistral4cli.logging_config import (
    LoggingConfig,
    configure_logging,
    render_logging_summary,
)
from mistral4cli.mcp_bridge import (
    MCPConfig,
    MCPToolBridge,
    discover_mcp_config_path,
)
from mistral4cli.session import (
    DEFAULT_SYSTEM_PROMPT,
    ContextStatus,
    MistralSession,
    PendingConversationSettings,
    SessionStatusSnapshot,
    UsageSnapshot,
    render_defaults_summary,
)
from mistral4cli.tooling import CompositeToolBridge, ToolBridge
from mistral4cli.ui import (
    CLEAR_SCREEN,
    InteractiveTTYRenderer,
    render_help_screen,
    render_welcome_banner,
    supports_full_terminal_ui,
    terminal_recommendation,
)

logger = logging.getLogger("mistral4cli.cli")
LINUX_ONLY_MESSAGE = "This client is currently supported on Linux only."


@dataclass(slots=True)
class _InputHistory:
    """In-memory REPL input history with up/down navigation state."""

    entries: list[str] = field(default_factory=list)
    browse_index: int | None = None
    draft: str = ""

    def add(self, line: str) -> None:
        """Append a non-empty line unless it duplicates the last entry."""

        clean = line.strip()
        if not clean:
            self.reset_navigation()
            return
        if not self.entries or self.entries[-1] != clean:
            self.entries.append(clean)
        self.reset_navigation()

    def previous(self, current_buffer: str) -> str:
        """Move one step back in history."""

        if not self.entries:
            return current_buffer
        if self.browse_index is None:
            self.draft = current_buffer
            self.browse_index = len(self.entries) - 1
        elif self.browse_index > 0:
            self.browse_index -= 1
        return self.entries[self.browse_index]

    def next(self) -> str:
        """Move one step forward in history."""

        if self.browse_index is None:
            return self.draft
        if self.browse_index < len(self.entries) - 1:
            self.browse_index += 1
            return self.entries[self.browse_index]
        draft = self.draft
        self.reset_navigation()
        return draft

    def reset_navigation(self) -> None:
        """Reset history navigation state after a committed line."""

        self.browse_index = None
        self.draft = ""


@dataclass(slots=True)
class _PendingAttachment:
    """Attachment selection staged for the next user prompt."""

    kind: str
    summary: str
    paths: list[Path]


@dataclass(slots=True)
class _ReplState:
    """Mutable REPL state that survives between commands."""

    pending_attachment: _PendingAttachment | None = None
    active_images: list[Path] = field(default_factory=list)
    active_documents: list[Path] = field(default_factory=list)


def _set_active_attachment(
    repl_state: _ReplState, *, kind: str, paths: Sequence[Path]
) -> None:
    active_paths = [path.expanduser() for path in paths]
    if kind == "image":
        repl_state.active_images = active_paths
        return
    repl_state.active_documents = active_paths


def _clear_attachments(
    repl_state: _ReplState,
    *,
    clear_images: bool,
    clear_documents: bool,
    clear_pending: bool = True,
) -> None:
    if clear_pending and repl_state.pending_attachment is not None:
        pending_kind = repl_state.pending_attachment.kind
        if (pending_kind == "image" and clear_images) or (
            pending_kind == "document" and clear_documents
        ):
            repl_state.pending_attachment = None
    if clear_images:
        repl_state.active_images = []
    if clear_documents:
        repl_state.active_documents = []


def _repl_prompt(repl_state: _ReplState) -> str:
    tokens: list[str] = []
    if repl_state.pending_attachment is not None:
        tokens.append(f"stage:{repl_state.pending_attachment.kind}")
    if repl_state.active_images:
        tokens.append(f"img:{len(repl_state.active_images)}")
    if repl_state.active_documents:
        tokens.append(f"doc:{len(repl_state.active_documents)}")
    if not tokens:
        return "M4S> "
    return f"M4S[{','.join(tokens)}]> "


def _format_estimated_context_for_status(
    status: ContextStatus | None,
    *,
    conversations_enabled: bool,
) -> str:
    if conversations_enabled:
        return "est:backend"
    if status is None:
        return "est:-"
    return f"est:{status.estimated_tokens}/{status.window_tokens}"


def _format_usage_for_status(usage: UsageSnapshot | None) -> str:
    if usage is None or usage.total_tokens is None:
        return "last:-"
    if usage.max_context_tokens is None:
        return f"last:{usage.total_tokens}/?"
    return f"last:{usage.total_tokens}/{usage.max_context_tokens}"


def _format_session_total_for_status(usage: UsageSnapshot | None) -> str:
    if usage is None or usage.total_tokens is None:
        return "usage:-"
    return f"usage:{usage.total_tokens}"


def _status_phase_label(snapshot: SessionStatusSnapshot) -> str:
    if snapshot.phase == "tool" and snapshot.detail:
        return f"tool:{snapshot.detail}"
    if snapshot.phase == "thinking":
        return "thinking..."
    return snapshot.phase


def _repl_status_line(session: MistralSession, repl_state: _ReplState) -> str:
    snapshot = session.status_snapshot()
    parts = [
        _status_phase_label(snapshot),
        session.backend_kind.value,
        session.model_id,
        f"reasoning:{'on' if session.show_reasoning else 'off'}",
        f"thinking:{'on' if session.show_thinking else 'off'}",
        f"conv:{'on' if session.conversations.enabled else 'off'}",
    ]
    if session.conversations.enabled and session.conversation_id:
        parts.append(f"cid:{session.conversation_id[:8]}")
    if repl_state.pending_attachment is not None:
        parts.append(f"stage:{repl_state.pending_attachment.kind}")
    if repl_state.active_images:
        parts.append(f"img:{len(repl_state.active_images)}")
    if repl_state.active_documents:
        parts.append(f"doc:{len(repl_state.active_documents)}")
    parts.append(
        _format_estimated_context_for_status(
            snapshot.estimated_context,
            conversations_enabled=session.conversations.enabled,
        )
    )
    parts.append(_format_usage_for_status(snapshot.last_usage))
    parts.append(_format_session_total_for_status(snapshot.cumulative_usage))
    return " | ".join(parts)


def _render_session_status(
    session: MistralSession,
    repl_state: _ReplState,
) -> str:
    snapshot = session.status_snapshot()
    phase = _status_phase_label(snapshot)
    server = "Mistral Cloud"
    if session.backend_kind is BackendKind.LOCAL:
        server = session.server_url or "local server"

    lines = ["Session status:"]
    lines.append(f"Phase: {phase}")
    lines.append(
        "Runtime: "
        f"backend={session.backend_kind.value} "
        f"server={server} "
        f"model={session.model_id}"
    )
    lines.append(
        "Response: "
        f"stream={'on' if session.stream_enabled else 'off'} "
        f"reasoning={'on' if session.show_reasoning else 'off'} "
        f"thinking={'on' if session.show_thinking else 'off'} "
        f"timeout={session.timeout_ms}ms"
    )
    lines.append(
        "Conversations: "
        f"mode={'on' if session.conversations.enabled else 'off'} "
        f"store={'on' if session.conversations.store else 'off'} "
        f"resume={session.conversations.resume_policy} "
        f"id={session.conversation_id or 'not started'}"
    )
    estimated = _format_estimated_context_for_status(
        snapshot.estimated_context,
        conversations_enabled=session.conversations.enabled,
    )
    last_usage = _format_usage_for_status(snapshot.last_usage)
    total_usage = _format_session_total_for_status(snapshot.cumulative_usage)
    lines.append(f"Context: {estimated} {last_usage} {total_usage}")
    attachment_parts: list[str] = []
    if repl_state.pending_attachment is not None:
        attachment_parts.append(f"stage={repl_state.pending_attachment.kind}")
    if repl_state.active_images:
        attachment_parts.append(f"images={len(repl_state.active_images)}")
    if repl_state.active_documents:
        attachment_parts.append(f"documents={len(repl_state.active_documents)}")
    if attachment_parts:
        lines.append("Attachments: " + " ".join(attachment_parts))
    pending = session.pending_conversation_text()
    if pending:
        lines.append(pending)
    return "\n".join(lines)


def _build_active_attachment_message(
    session: MistralSession,
    *,
    prompt: str,
    image_paths: Sequence[Path],
    document_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    if not image_paths and not document_paths:
        raise ValueError("At least one active attachment is required")

    text_lines = [prompt.strip()]
    if image_paths:
        text_lines.extend(["", "Active images:"])
        text_lines.extend(f"- {path.name}" for path in image_paths)
    if document_paths:
        text_lines.extend(["", "Active documents:"])
        text_lines.extend(f"- {path.name}" for path in document_paths)
        if session.backend_kind is BackendKind.REMOTE:
            text_lines.append("")
            text_lines.append("The active documents are attached natively.")
        else:
            text_lines.append("")
            text_lines.append("The active documents are attached as OCR images.")

    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(text_lines)}]
    if image_paths:
        image_message = (
            build_remote_image_message(image_paths, prompt=prompt)
            if session.backend_kind is BackendKind.REMOTE
            else build_image_message(image_paths, prompt=prompt)
        )
        content.extend(image_message[1:])
    if document_paths:
        document_message = (
            build_remote_document_message(document_paths, prompt=prompt)
            if session.backend_kind is BackendKind.REMOTE
            else build_document_message(document_paths, prompt=prompt)
        )
        content.extend(document_message[1:])
    return content


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
            "Interactive multimodal CLI for Mistral Small 4 local and remote backends."
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
        "--log-dir",
        default=None,
        help="Directory for rotated log files.",
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
    args: argparse.Namespace, stderr: TextIO
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


def _build_tool_bridge(args: argparse.Namespace, stderr: TextIO) -> CompositeToolBridge:
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


def _split_shortcut_argument(argument: str) -> tuple[str, str | None]:
    head, separator, tail = argument.partition(" -- ")
    if separator:
        return head.strip(), tail.strip()
    return argument.strip(), None


def _parse_shortcut_options(
    parser: argparse.ArgumentParser, tokens: list[str], stdout: TextIO
) -> argparse.Namespace | None:
    try:
        return parser.parse_args(tokens)
    except (argparse.ArgumentError, SystemExit, ValueError) as exc:
        stdout.write(f"[error] invalid shortcut arguments: {exc}\n")
        stdout.flush()
        return None


def _print_tool_result(stdout: TextIO, text: str) -> None:
    stdout.write(text)
    if not text.endswith("\n"):
        stdout.write("\n")
    stdout.flush()


def _run_shell_shortcut(argument: str, session: MistralSession, stdout: TextIO) -> bool:
    option_text, command_text = _split_shortcut_argument(argument)
    if command_text is None:
        command_text = option_text
        option_text = ""

    parser = argparse.ArgumentParser(prog="/run", add_help=False, exit_on_error=False)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--lines", type=int, default=120)
    parser.add_argument("--offset", type=int, default=0)
    options = _parse_shortcut_options(parser, shlex.split(option_text), stdout)
    if options is None:
        return False

    command = command_text.strip()
    if not command:
        stdout.write(
            "Usage: /run [--cwd PATH] [--timeout SECONDS] [--lines N] -- COMMAND\n"
        )
        stdout.flush()
        return False

    result = session.call_tool(
        "shell",
        {
            "command": command,
            "cwd": options.cwd,
            "timeout_seconds": options.timeout,
            "max_lines": options.lines,
            "offset_lines": options.offset,
        },
    )
    _print_tool_result(stdout, result.text)
    return False


def _run_ls_shortcut(argument: str, session: MistralSession, stdout: TextIO) -> bool:
    tokens = shlex.split(argument)
    parser = argparse.ArgumentParser(prog="/ls", add_help=False, exit_on_error=False)
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--max-entries", type=int, default=200)
    options = _parse_shortcut_options(parser, tokens, stdout)
    if options is None:
        return False

    result = session.call_tool(
        "list_dir",
        {"path": options.path, "max_entries": options.max_entries},
    )
    _print_tool_result(stdout, result.text)
    return False


def _run_find_shortcut(argument: str, session: MistralSession, stdout: TextIO) -> bool:
    option_text, query_text = _split_shortcut_argument(argument)
    if query_text is None:
        query_text = option_text
        option_text = ""

    parser = argparse.ArgumentParser(prog="/find", add_help=False, exit_on_error=False)
    parser.add_argument("--path", default=".")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0)
    options = _parse_shortcut_options(parser, shlex.split(option_text), stdout)
    if options is None:
        return False

    query = query_text.strip()
    if not query:
        stdout.write("Usage: /find [--path PATH] [--limit N] [--offset N] -- QUERY\n")
        stdout.flush()
        return False

    result = session.call_tool(
        "search_text",
        {
            "query": query,
            "path": options.path,
            "max_results": options.limit,
            "offset": options.offset,
        },
    )
    _print_tool_result(stdout, result.text)
    return False


def _run_edit_shortcut(argument: str, session: MistralSession, stdout: TextIO) -> bool:
    option_text, content_text = _split_shortcut_argument(argument)
    if content_text is None:
        parts = shlex.split(option_text)
        if len(parts) < 2:
            stdout.write("Usage: /edit PATH -- CONTENT\n")
            stdout.flush()
            return False
        path = parts[0]
        content = " ".join(parts[1:])
    else:
        path_parts = shlex.split(option_text)
        if not path_parts:
            stdout.write("Usage: /edit PATH -- CONTENT\n")
            stdout.flush()
            return False
        path = path_parts[0]
        content = content_text

    if not content.strip():
        stdout.write("Usage: /edit PATH -- CONTENT\n")
        stdout.flush()
        return False

    result = session.call_tool(
        "write_file",
        {
            "path": path,
            "content": content,
        },
    )
    _print_tool_result(stdout, result.text)
    return False


def _run_image_shortcut(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState,
    input_func: Callable[[str], str],
    stdin: TextIO,
    path_picker: PathPicker | None,
) -> bool:
    prompt = _normalize_inline_prompt(argument)

    paths = choose_paths(
        kind="image",
        input_func=input_func,
        stdin=stdin,
        stdout=stdout,
        path_picker=path_picker,
        filetypes=IMAGE_FILETYPES,
        suffixes=IMAGE_SUFFIXES,
        multiple=False,
    )
    if not paths:
        stdout.write("[image] selection canceled.\n")
        stdout.flush()
        return False

    summary = format_selection_summary(paths)
    stdout.write(f"[image] selected: {summary}\n")
    stdout.flush()
    if any(path.suffix.lower() not in IMAGE_SUFFIXES for path in paths):
        stdout.write("[image] could not prepare attachment: unsupported image file\n")
        stdout.flush()
        return False
    if prompt is None:
        repl_state.pending_attachment = _PendingAttachment(
            kind="image",
            summary=summary,
            paths=list(paths),
        )
        stdout.write(
            "[image] attachment staged. Type your next prompt to analyze it.\n"
        )
        stdout.flush()
        return False

    try:
        if session.backend_kind is BackendKind.REMOTE:
            message = build_remote_image_message(paths, prompt=prompt)
        else:
            message = build_image_message(paths, prompt=prompt)
    except Exception as exc:
        stdout.write(f"[image] could not prepare attachment: {exc}\n")
        stdout.flush()
        return False

    repl_state.pending_attachment = None
    _set_active_attachment(repl_state, kind="image", paths=paths)
    stdout.write("[image] attachment active. Use /dropimage or /drop to release it.\n")
    stdout.flush()
    session.send_content(message, stream=session.stream_enabled)
    return False


def _run_doc_shortcut(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState,
    input_func: Callable[[str], str],
    stdin: TextIO,
    path_picker: PathPicker | None,
) -> bool:
    prompt = _normalize_inline_prompt(argument)

    paths = choose_paths(
        kind="document",
        input_func=input_func,
        stdin=stdin,
        stdout=stdout,
        path_picker=path_picker,
        filetypes=DOCUMENT_FILETYPES,
        suffixes=DOCUMENT_SUFFIXES,
        multiple=False,
    )
    if not paths:
        stdout.write("[doc] selection canceled.\n")
        stdout.flush()
        return False

    summary = format_selection_summary(paths)
    stdout.write(f"[doc] selected: {summary}\n")
    stdout.flush()
    if any(path.suffix.lower() not in DOCUMENT_SUFFIXES for path in paths):
        stdout.write("[doc] could not prepare attachment: unsupported document file\n")
        stdout.flush()
        return False
    if prompt is None:
        repl_state.pending_attachment = _PendingAttachment(
            kind="document",
            summary=summary,
            paths=list(paths),
        )
        stdout.write("[doc] attachment staged. Type your next prompt to analyze it.\n")
        stdout.flush()
        return False

    try:
        if session.backend_kind is BackendKind.REMOTE:
            message = build_remote_document_message(paths, prompt=prompt)
        else:
            message = build_document_message(paths, prompt=prompt)
    except Exception as exc:
        stdout.write(f"[doc] could not prepare attachment: {exc}\n")
        stdout.flush()
        return False

    repl_state.pending_attachment = None
    _set_active_attachment(repl_state, kind="document", paths=paths)
    stdout.write("[doc] attachment active. Use /dropdoc or /drop to release it.\n")
    stdout.flush()
    session.send_content(message, stream=session.stream_enabled)
    return False


def _normalize_inline_prompt(argument: str) -> str | None:
    prompt = argument.strip()
    if not prompt:
        return None

    for prefix in ("--prompt", "-p"):
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix) :].lstrip(" =")
            break

    prompt = prompt.strip()
    if not prompt:
        return None
    if len(prompt) >= 2 and prompt[0] == prompt[-1] and prompt[0] in {'"', "'"}:
        prompt = prompt[1:-1].strip()
    return prompt or None


def _linux_supported() -> bool:
    """Return whether the current runtime platform is supported."""

    return sys.platform.startswith("linux")


def _ensure_supported_platform(stderr: TextIO) -> bool:
    """Fail fast when the CLI is launched outside Linux."""

    if _linux_supported():
        return True
    # Fail before any session setup because local tool semantics and terminal
    # behavior are intentionally defined only for Linux in this client.
    stderr.write(LINUX_ONLY_MESSAGE + "\n")
    stderr.flush()
    return False


def _print_banner(stdout: TextIO, session: MistralSession) -> None:
    stdout.write(
        render_welcome_banner(session.describe_defaults(), stream=stdout) + "\n"
    )
    stdout.flush()


def _print_runtime_refresh(stdout: TextIO, session: MistralSession) -> None:
    stdout.write(session.describe_defaults() + "\n")
    stdout.flush()


def _clear_screen_if_supported(stdout: TextIO) -> None:
    """Clear the interactive terminal screen when supported."""

    if not supports_full_terminal_ui(stdout):
        return
    stdout.write(CLEAR_SCREEN)
    stdout.flush()


def _print_terminal_recommendation(stdout: TextIO) -> None:
    """Print a short terminal recommendation when the palette may degrade."""

    recommendation = terminal_recommendation(stream=stdout)
    if not recommendation:
        return
    stdout.write(recommendation + "\n")
    stdout.flush()


def _refresh_repl_screen(
    stdout: TextIO,
    session: MistralSession,
    *,
    startup: bool,
) -> None:
    """Redraw the REPL after clearing the terminal in interactive mode."""

    _clear_screen_if_supported(stdout)
    _print_terminal_recommendation(stdout)
    if startup:
        _print_banner(stdout, session)
    else:
        _print_runtime_refresh(stdout, session)


def _is_default_input_func(input_func: Callable[[str], str]) -> bool:
    """Return whether the REPL uses the standard input() function."""

    return input_func is input


def _redraw_prompt_line(stdout: TextIO, prompt: str, buffer: str) -> None:
    """Redraw the current prompt line after history navigation or editing."""

    stdout.write(f"\r{prompt}{buffer}\x1b[K")
    stdout.flush()


def _write_tty_newline(stdout: TextIO) -> None:
    """Emit a proper CRLF while the terminal is in raw mode."""

    stdout.write("\r\n")
    stdout.flush()


def _read_tty_line(
    prompt: str,
    stdin: TextIO,
    stdout: TextIO,
    history: _InputHistory,
    renderer: InteractiveTTYRenderer | None = None,
) -> str:
    """Read one REPL line with simple raw-mode editing and history navigation."""

    import termios
    import tty

    fileno = stdin.fileno()
    original_attrs = termios.tcgetattr(fileno)
    buffer = ""
    if renderer is not None:
        renderer.render_input(prompt, buffer)
    else:
        stdout.write(prompt)
        stdout.flush()
    try:
        tty.setraw(fileno)
        while True:
            char = stdin.read(1)
            if char == "":
                if renderer is not None:
                    renderer.clear_overlay()
                raise EOFError
            if char in {"\r", "\n"}:
                if renderer is not None:
                    renderer.commit_input(prompt, buffer)
                else:
                    _write_tty_newline(stdout)
                history.reset_navigation()
                return buffer
            if char == "\x03":
                if renderer is not None:
                    renderer.clear_overlay()
                raise KeyboardInterrupt
            if char == "\x04":
                if not buffer:
                    if renderer is not None:
                        renderer.clear_overlay()
                    raise EOFError
                continue
            if char in {"\x7f", "\b"}:
                if buffer:
                    buffer = buffer[:-1]
                    if renderer is not None:
                        renderer.render_input(prompt, buffer)
                    else:
                        _redraw_prompt_line(stdout, prompt, buffer)
                continue
            if char == "\x1b":
                next_char = stdin.read(1)
                if next_char != "[":
                    continue
                direction = stdin.read(1)
                if direction == "A":
                    buffer = history.previous(buffer)
                    if renderer is not None:
                        renderer.render_input(prompt, buffer)
                    else:
                        _redraw_prompt_line(stdout, prompt, buffer)
                elif direction == "B":
                    buffer = history.next()
                    if renderer is not None:
                        renderer.render_input(prompt, buffer)
                    else:
                        _redraw_prompt_line(stdout, prompt, buffer)
                continue
            if char.isprintable():
                buffer += char
                if renderer is not None:
                    renderer.render_input(prompt, buffer)
                else:
                    stdout.write(char)
                    stdout.flush()
    finally:
        termios.tcsetattr(fileno, termios.TCSADRAIN, original_attrs)


def _read_repl_line(
    *,
    prompt: str,
    input_func: Callable[[str], str],
    stdin: TextIO,
    stdout: TextIO,
    history: _InputHistory,
    renderer: InteractiveTTYRenderer | None = None,
) -> str:
    """Read one REPL line using raw TTY mode when available."""

    if stdin.isatty() and _is_default_input_func(input_func):
        return _read_tty_line(prompt, stdin, stdout, history, renderer)
    return input_func(prompt)


def _help_page_lines(stdout: TextIO) -> int:
    """Return the number of help lines to show per page in TTY mode."""

    try:
        terminal_lines = shutil.get_terminal_size().lines
    except OSError:
        terminal_lines = 24
    return max(terminal_lines - 2, 8)


def _print_paginated_text(
    *,
    text: str,
    stdout: TextIO,
    stdin: TextIO,
    prompt_label: str,
) -> None:
    """Print text, paging through it in interactive terminals."""

    if not (supports_full_terminal_ui(stdout) and stdin.isatty()):
        stdout.write(text + "\n")
        stdout.flush()
        return

    lines = text.splitlines()
    page_size = _help_page_lines(stdout)
    index = 0
    while index < len(lines):
        stdout.write("\n".join(lines[index : index + page_size]) + "\n")
        stdout.flush()
        index += page_size
        if index >= len(lines):
            return
        stdout.write(f"[{prompt_label}] Press Enter for more, q to quit: ")
        stdout.flush()
        answer = stdin.readline()
        if answer == "":
            stdout.write("\n")
            stdout.flush()
            return
        if answer.strip().lower() == "q":
            return


def _run_command(
    command: str,
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState | None = None,
    local_config: LocalMistralConfig | None = None,
    client_factory: Callable[[MistralConfig], Mistral] = build_client,
    input_func: Callable[[str], str] = input,
    stdin: TextIO = sys.stdin,
    path_picker: PathPicker | None = None,
) -> bool:
    if repl_state is None:
        repl_state = _ReplState()
    logger.debug("Running command name=%s has_argument=%s", command, bool(argument))
    if command in {"help", "h", "?"}:
        _print_paginated_text(
            text=render_help_screen(
                summary=session.describe_defaults(),
                tools=session.describe_tools().splitlines(),
                stream=stdout,
            ),
            stdout=stdout,
            stdin=stdin,
            prompt_label="help",
        )
        return False
    if command == "tools":
        _print_paginated_text(
            text=session.describe_tools(),
            stdout=stdout,
            stdin=stdin,
            prompt_label="tools",
        )
        return False
    if command == "status":
        stdout.write(_render_session_status(session, repl_state) + "\n")
        stdout.flush()
        return False
    if command in {"conversations", "conv"}:
        return _run_conversations_command(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            local_config=local_config,
            client_factory=client_factory,
            stdin=stdin,
            input_func=input_func,
        )
    if command == "compact":
        return _run_compact_command(argument, session, stdout)
    if command == "remote":
        return _run_remote_command(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            local_config=local_config,
            client_factory=client_factory,
        )
    if command == "timeout":
        return _run_timeout_command(argument, session, stdout)
    if command == "reasoning":
        normalized = argument.strip().lower()
        if not normalized:
            stdout.write(session.reasoning_status_text() + "\n")
        elif normalized in {"on", "true", "1"}:
            session.set_reasoning_visibility(True)
            stdout.write("Reasoning request enabled.\n")
        elif normalized in {"off", "false", "0"}:
            session.set_reasoning_visibility(False)
            stdout.write("Reasoning request disabled.\n")
        elif normalized == "toggle":
            session.toggle_reasoning_visibility()
            stdout.write(session.reasoning_status_text() + "\n")
        else:
            stdout.write("Usage: /reasoning [on|off|toggle]\n")
        stdout.flush()
        return False
    if command == "thinking":
        normalized = argument.strip().lower()
        if not normalized:
            stdout.write(session.thinking_status_text() + "\n")
        elif normalized in {"on", "true", "1"}:
            session.set_thinking_visibility(True)
            stdout.write("Thinking display enabled.\n")
        elif normalized in {"off", "false", "0"}:
            session.set_thinking_visibility(False)
            stdout.write("Thinking display disabled.\n")
        elif normalized == "toggle":
            session.toggle_thinking_visibility()
            stdout.write(session.thinking_status_text() + "\n")
        else:
            stdout.write("Usage: /thinking [on|off|toggle]\n")
        stdout.flush()
        return False
    if command == "run":
        return _run_shell_shortcut(argument, session, stdout)
    if command == "ls":
        return _run_ls_shortcut(argument, session, stdout)
    if command == "find":
        return _run_find_shortcut(argument, session, stdout)
    if command == "edit":
        return _run_edit_shortcut(argument, session, stdout)
    if command == "image":
        return _run_image_shortcut(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            input_func=input_func,
            stdin=stdin,
            path_picker=path_picker,
        )
    if command == "doc":
        return _run_doc_shortcut(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            input_func=input_func,
            stdin=stdin,
            path_picker=path_picker,
        )
    if command == "docs":
        return _run_doc_shortcut(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            input_func=input_func,
            stdin=stdin,
            path_picker=path_picker,
        )
    if command == "drop":
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        stdout.write("Active attachments cleared.\n")
        stdout.flush()
        return False
    if command == "dropimage":
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=False,
            clear_pending=True,
        )
        stdout.write("Active image attachments cleared.\n")
        stdout.flush()
        return False
    if command == "dropdoc":
        _clear_attachments(
            repl_state,
            clear_images=False,
            clear_documents=True,
            clear_pending=True,
        )
        stdout.write("Active document attachments cleared.\n")
        stdout.flush()
        return False
    if command in {"exit", "quit", "q"}:
        return True
    if command in {"reset", "new"}:
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        session.reset()
        _refresh_repl_screen(stdout, session, startup=False)
        stdout.write("Conversation reset.\n")
        stdout.flush()
        return False
    if command == "defaults":
        stdout.write(session.describe_defaults() + "\n")
        stdout.flush()
        return False
    if command == "system":
        if argument:
            _clear_attachments(
                repl_state,
                clear_images=True,
                clear_documents=True,
                clear_pending=True,
            )
            session.set_system_prompt(argument)
            _refresh_repl_screen(stdout, session, startup=False)
            stdout.write("System prompt updated and conversation reset.\n")
        else:
            stdout.write("Current system prompt:\n")
            stdout.write(session.system_prompt + "\n")
        stdout.flush()
        return False

    stdout.write(f"Unknown command: /{command}\n")
    stdout.flush()
    return False


def _parse_command(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped[0] not in {"/", ":"}:
        return None
    command_body = stripped[1:].strip()
    if not command_body:
        return None
    command, _, argument = command_body.partition(" ")
    return command.lower(), argument.strip()


def _run_conversations_command(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState,
    local_config: LocalMistralConfig | None,
    client_factory: Callable[[MistralConfig], Mistral],
    stdin: TextIO,
    input_func: Callable[[str], str],
) -> bool:
    def ensure_enabled(*, allow_resume: bool) -> bool:
        if session.conversations.enabled:
            return True
        try:
            remote_config = RemoteMistralConfig.from_env(
                timeout_ms=get_client_timeout_ms(session.client, DEFAULT_TIMEOUT_MS)
            )
        except RemoteAPIKeyError as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        session.enable_conversations(
            client=client_factory(remote_config),
            model_id=remote_config.model_id,
            store=session.conversations.store,
        )
        if allow_resume:
            _maybe_resume_conversation(
                session,
                stdin=stdin,
                stdout=stdout,
                input_func=input_func,
                interactive=stdin.isatty(),
            )
        return True

    def parse_metadata_filters(
        raw_tokens: list[str],
    ) -> tuple[int, int, dict[str, str]]:
        page = 0
        page_size = 20
        metadata: dict[str, str] = {}
        index = 0
        while index < len(raw_tokens):
            token = raw_tokens[index]
            if token == "--page" and index + 1 < len(raw_tokens):
                page = int(raw_tokens[index + 1])
                index += 2
                continue
            if token in {"--size", "--page-size"} and index + 1 < len(raw_tokens):
                page_size = int(raw_tokens[index + 1])
                index += 2
                continue
            if token == "--meta" and index + 1 < len(raw_tokens):
                metadata.update(_parse_metadata_pairs([raw_tokens[index + 1]]))
                index += 2
                continue
            raise ValueError(
                "Usage: /conversations list [--page N] [--size N] [--meta KEY=VALUE]"
            )
        return page, page_size, metadata

    tokens = shlex.split(argument)
    action = tokens[0].lower() if tokens else ""
    if not action:
        stdout.write(session.current_conversation_text() + "\n")
        stdout.flush()
        return False

    if action == "on":
        if not ensure_enabled(allow_resume=True):
            return False
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        _refresh_repl_screen(stdout, session, startup=False)
        if session.conversation_id is not None:
            stdout.write(f"Conversations enabled. Resumed {session.conversation_id}.\n")
        else:
            stdout.write("Conversations enabled. Conversation reset.\n")
        stdout.flush()
        return False

    if action == "off":
        session.disable_conversations()
        if (
            local_config is not None
            and session.backend_kind is BackendKind.LOCAL
            and session.server_url == local_config.server_url
        ):
            session.client = client_factory(local_config)
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        _refresh_repl_screen(stdout, session, startup=False)
        stdout.write("Conversations disabled. Conversation reset.\n")
        stdout.flush()
        return False

    if action == "new":
        if not session.conversations.enabled:
            stdout.write("Conversations mode is off.\n")
        else:
            session.reset()
            if session.conversation_registry is not None:
                session.conversation_registry.clear_last_active()
            _clear_attachments(
                repl_state,
                clear_images=True,
                clear_documents=True,
                clear_pending=True,
            )
            _refresh_repl_screen(stdout, session, startup=False)
            stdout.write("New Conversation will start on the next turn.\n")
        stdout.flush()
        return False

    if action == "store":
        value = tokens[1].lower() if len(tokens) > 1 else ""
        if value not in {"on", "off"}:
            stdout.write("Usage: /conversations store [on|off]\n")
        else:
            session.set_conversation_store(value == "on")
            stdout.write(f"Conversation store set to {value}. Conversation reset.\n")
        stdout.flush()
        return False

    if action in {"current", "status"}:
        stdout.write(session.current_conversation_text() + "\n")
        stdout.flush()
        return False

    if action == "id":
        stdout.write(f"{session.conversation_id or 'No active conversation id yet.'}\n")
        stdout.flush()
        return False

    if action == "list":
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            page, page_size, metadata = parse_metadata_filters(tokens[1:])
            text = session.list_remote_conversations(
                page=page,
                page_size=page_size,
                metadata=metadata,
            )
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        _print_paginated_text(
            text=text,
            stdout=stdout,
            stdin=stdin,
            prompt_label="conversations list",
        )
        return False

    if action == "bookmarks":
        _print_paginated_text(
            text=session.list_remote_conversations(bookmarks_only=True),
            stdout=stdout,
            stdin=stdin,
            prompt_label="conversations bookmarks",
        )
        return False

    if action == "show":
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            text = session.show_remote_conversation(
                tokens[1] if len(tokens) > 1 else None
            )
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        _print_paginated_text(
            text=text,
            stdout=stdout,
            stdin=stdin,
            prompt_label="conversations show",
        )
        return False

    if action == "use":
        reference = tokens[1] if len(tokens) > 1 else ""
        if not reference:
            stdout.write("Usage: /conversations use <conversation_id>\n")
            stdout.flush()
            return False
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            conversation_id = session.resolve_conversation_reference(reference)
            rendered = session.attach_remote_conversation(
                conversation_id,
                source="manual",
            )
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        _refresh_repl_screen(stdout, session, startup=False)
        stdout.write(rendered + "\n")
        stdout.flush()
        return False

    if action in {"history", "messages"}:
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            text = session.conversation_history_text(
                messages_only=action == "messages",
                conversation_id=tokens[1] if len(tokens) > 1 else None,
            )
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        _print_paginated_text(
            text=text,
            stdout=stdout,
            stdin=stdin,
            prompt_label=f"conversations {action}",
        )
        return False

    if action == "delete":
        try:
            rendered = session.delete_remote_conversation(
                tokens[1] if len(tokens) > 1 else None
            )
        except Exception as exc:
            stdout.write(f"[conversations] delete failed: {exc}\n")
            stdout.flush()
            return False
        _refresh_repl_screen(stdout, session, startup=False)
        stdout.write(rendered + "\n")
        stdout.flush()
        return False

    if action == "restart":
        from_entry_id = tokens[1] if len(tokens) > 1 else ""
        restart_reference: str | None = tokens[2] if len(tokens) > 2 else None
        if not from_entry_id:
            stdout.write("Usage: /conversations restart <entry_id> [conversation_id]\n")
            stdout.flush()
            return False
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            rendered = session.restart_remote_conversation(
                from_entry_id=from_entry_id,
                conversation_id=restart_reference,
            )
        except Exception as exc:
            stdout.write(f"[conversations] restart failed: {exc}\n")
            stdout.flush()
            return False
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        _refresh_repl_screen(stdout, session, startup=False)
        stdout.write(rendered + "\n")
        stdout.flush()
        return False

    if action == "set":
        field = tokens[1].lower() if len(tokens) > 1 else ""
        if field == "name" and len(tokens) > 2:
            session.set_pending_conversation_name(" ".join(tokens[2:]))
            stdout.write("Pending conversation name updated.\n")
        elif field == "description" and len(tokens) > 2:
            session.set_pending_conversation_description(" ".join(tokens[2:]))
            stdout.write("Pending conversation description updated.\n")
        elif field == "meta" and len(tokens) > 2:
            try:
                metadata = _parse_metadata_pairs(tokens[2:])
            except ValueError as exc:
                stdout.write(f"[conversations] {exc}\n")
            else:
                for key, value in metadata.items():
                    session.set_pending_conversation_metadata(key, value)
                stdout.write("Pending conversation metadata updated.\n")
        else:
            stdout.write(
                "Usage: /conversations set "
                "[name TEXT|description TEXT|meta KEY=VALUE ...]\n"
            )
        stdout.flush()
        return False

    if action == "unset":
        field = tokens[1].lower() if len(tokens) > 1 else ""
        if field == "name":
            session.clear_pending_conversation_name()
            stdout.write("Pending conversation name cleared.\n")
        elif field == "description":
            session.clear_pending_conversation_description()
            stdout.write("Pending conversation description cleared.\n")
        elif field == "meta":
            if len(tokens) > 2:
                session.clear_pending_conversation_metadata(tokens[2])
                stdout.write(f"Pending conversation metadata {tokens[2]} cleared.\n")
            else:
                session.clear_pending_conversation_metadata()
                stdout.write("Pending conversation metadata cleared.\n")
        elif field == "all":
            session.clear_pending_conversation_name()
            session.clear_pending_conversation_description()
            session.clear_pending_conversation_metadata()
            stdout.write("All pending conversation settings cleared.\n")
        else:
            stdout.write(
                "Usage: /conversations unset [name|description|meta <key>|all]\n"
            )
        stdout.flush()
        return False

    if action == "alias":
        if len(tokens) < 2:
            stdout.write("Usage: /conversations alias [<conversation_id>] <text>\n")
        else:
            try:
                if len(tokens) == 2:
                    if not session.conversation_id:
                        raise ValueError(
                            "No active conversation id yet. Use "
                            '"/conv alias <conversation_id> <text>" or start one first.'
                        )
                    reference = session.conversation_id
                    alias = tokens[1]
                else:
                    reference = tokens[1]
                    alias = " ".join(tokens[2:])
                stdout.write(
                    session.set_local_conversation_alias(reference, alias) + "\n"
                )
            except Exception as exc:
                stdout.write(f"[conversations] {exc}\n")
        stdout.flush()
        return False

    if action == "note":
        if len(tokens) < 3:
            stdout.write("Usage: /conversations note <conversation_id> <text>\n")
        else:
            try:
                stdout.write(
                    session.set_local_conversation_note(
                        tokens[1],
                        " ".join(tokens[2:]),
                    )
                    + "\n"
                )
            except Exception as exc:
                stdout.write(f"[conversations] {exc}\n")
        stdout.flush()
        return False

    if action == "tag":
        mode = tokens[1].lower() if len(tokens) > 1 else ""
        if mode not in {"add", "remove"} or len(tokens) < 4:
            stdout.write(
                "Usage: /conversations tag [add|remove] <conversation_id> <tag>\n"
            )
            stdout.flush()
            return False
        try:
            if mode == "add":
                rendered = session.add_local_conversation_tag(tokens[2], tokens[3])
            else:
                rendered = session.remove_local_conversation_tag(tokens[2], tokens[3])
            stdout.write(rendered + "\n")
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
        stdout.flush()
        return False

    if action == "forget":
        if len(tokens) < 2:
            stdout.write("Usage: /conversations forget <conversation_id>\n")
            stdout.flush()
            return False
        try:
            stdout.write(session.forget_local_conversation(tokens[1]) + "\n")
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
        stdout.flush()
        return False

    stdout.write(
        "Usage: /conversations "
        "[on|off|new|store on|store off|current|id|list|bookmarks|show|use|"
        "history|messages|delete|restart|set|unset|alias|note|tag|forget]\n"
    )
    stdout.flush()
    return False


def _maybe_resume_conversation(
    session: MistralSession,
    *,
    stdin: TextIO,
    stdout: TextIO,
    input_func: Callable[[str], str],
    interactive: bool,
) -> None:
    """Resume the last known stored conversation when configured to do so."""

    if not session.conversations.enabled or not session.conversations.store:
        return
    registry = session.conversation_registry
    if registry is None or not registry.last_active_conversation_id:
        return
    if session.conversation_id:
        return

    should_resume = False
    policy = session.conversations.resume_policy
    if policy == "last":
        should_resume = True
    elif policy == "prompt" and interactive:
        try:
            answer = input_func(
                "Resume the last Mistral Conversation "
                f"({registry.last_active_conversation_id})? [Y/n] "
            )
        except EOFError:
            answer = "n"
        should_resume = answer.strip().lower() not in {"n", "no"}

    if not should_resume:
        return

    try:
        session.attach_remote_conversation(
            registry.last_active_conversation_id,
            source="resume",
        )
    except Exception as exc:
        logger.warning(
            "Could not resume conversation id=%s error=%s",
            registry.last_active_conversation_id,
            exc,
        )
        registry.clear_last_active(registry.last_active_conversation_id)
        if interactive:
            stdout.write(
                "[conversations] Could not resume the last conversation; "
                "starting a new one instead.\n"
            )
            stdout.flush()


def _run_compact_command(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
) -> bool:
    tokens = shlex.split(argument)
    action = tokens[0].lower() if tokens else "now"
    if action in {"now", "run"}:
        try:
            stdout.write(session.compact_context().summary() + "\n")
        except Exception as exc:
            stdout.write(f"[compact] failed: {exc}\n")
        stdout.flush()
        return False
    if action in {"status", "show"}:
        stdout.write(session.context_status_text() + "\n")
        stdout.flush()
        return False
    if action == "auto":
        value = tokens[1].lower() if len(tokens) > 1 else ""
        if value not in {"on", "off"}:
            stdout.write("Usage: /compact auto [on|off]\n")
        else:
            session.configure_context(auto_compact=value == "on")
            stdout.write(f"Auto compact set to {value}.\n")
        stdout.flush()
        return False
    if action == "threshold":
        value = tokens[1] if len(tokens) > 1 else ""
        try:
            threshold = float(value.rstrip("%"))
        except ValueError:
            stdout.write("Usage: /compact threshold [0.1-0.99|10-99]\n")
        else:
            session.configure_context(threshold=threshold)
            stdout.write(
                f"Compact threshold set to {round(session.context.threshold * 100)}%.\n"
            )
        stdout.flush()
        return False
    if action == "reserve":
        value = tokens[1] if len(tokens) > 1 else ""
        try:
            reserve_tokens = int(value)
        except ValueError:
            stdout.write("Usage: /compact reserve [TOKENS]\n")
        else:
            session.configure_context(reserve_tokens=reserve_tokens)
            stdout.write(
                f"Context reserve set to {session.context.reserve_tokens} tokens.\n"
            )
        stdout.flush()
        return False
    if action in {"keep", "keep-turns"}:
        value = tokens[1] if len(tokens) > 1 else ""
        try:
            keep_turns = int(value)
        except ValueError:
            stdout.write("Usage: /compact keep [TURNS]\n")
        else:
            session.configure_context(keep_recent_turns=keep_turns)
            stdout.write(
                f"Compact keep turns set to {session.context.keep_recent_turns}.\n"
            )
        stdout.flush()
        return False

    stdout.write(
        "Usage: /compact [status|now|auto on|auto off|threshold N|reserve N|keep N]\n"
    )
    stdout.flush()
    return False


def _run_remote_command(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState,
    local_config: LocalMistralConfig | None,
    client_factory: Callable[[MistralConfig], Mistral],
) -> bool:
    normalized = argument.strip().lower()
    if not normalized:
        availability = (
            "available" if remote_api_key_available() else "missing MISTRAL_API_KEY"
        )
        stdout.write(f"Backend: {session.backend_kind.value}\n")
        stdout.write(f"Remote mode: {availability}\n")
        stdout.flush()
        return False

    if normalized == "on":
        try:
            remote_config = RemoteMistralConfig.from_env(
                timeout_ms=get_client_timeout_ms(session.client, DEFAULT_TIMEOUT_MS)
            )
        except RemoteAPIKeyError as exc:
            stdout.write(f"[remote] {exc}\n")
            stdout.flush()
            return False

        session.switch_backend(
            client=client_factory(remote_config),
            backend_kind=BackendKind.REMOTE,
            model_id=remote_config.model_id,
            server_url=None,
        )
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        _refresh_repl_screen(stdout, session, startup=False)
        stdout.write("Remote backend enabled. Conversation reset.\n")
        stdout.flush()
        return False

    if normalized == "off":
        if local_config is None:
            stdout.write("[remote] Local configuration is unavailable.\n")
            stdout.flush()
            return False
        session.switch_backend(
            client=client_factory(local_config),
            backend_kind=BackendKind.LOCAL,
            model_id=local_config.model_id,
            server_url=local_config.server_url,
        )
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        _refresh_repl_screen(stdout, session, startup=False)
        stdout.write("Local backend enabled. Conversation reset.\n")
        stdout.flush()
        return False

    stdout.write("Usage: /remote [on|off]\n")
    stdout.flush()
    return False


def _run_timeout_command(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
) -> bool:
    normalized = argument.strip().lower()
    if not normalized:
        stdout.write(f"Timeout: {session.timeout_ms} ms\n")
        stdout.flush()
        return False

    try:
        timeout_ms = _parse_timeout_ms(normalized)
    except ValueError as exc:
        stdout.write(f"[timeout] {exc}\n")
        stdout.flush()
        return False

    session.set_timeout_ms(timeout_ms)
    stdout.write(f"Timeout set to {timeout_ms} ms.\n")
    stdout.flush()
    return False


def _parse_timeout_ms(value: str) -> int:
    text = value.strip().lower()
    if not text:
        raise ValueError("Usage: /timeout [MILLISECONDS|SECONDSs|MINUTESm]")
    if text.endswith("ms"):
        amount = int(text[:-2].strip())
        multiplier = 1
    elif text.endswith("s"):
        amount = int(text[:-1].strip())
        multiplier = 1_000
    elif text.endswith("m"):
        amount = int(text[:-1].strip())
        multiplier = 60_000
    else:
        amount = int(text)
        multiplier = 1

    timeout_ms = amount * multiplier
    if timeout_ms < 1_000:
        raise ValueError("Timeout must be at least 1000 ms.")
    return timeout_ms


def _run_repl(
    session: MistralSession,
    *,
    local_config: LocalMistralConfig,
    client_factory: Callable[[MistralConfig], Mistral],
    input_func: Callable[[str], str],
    stdin: TextIO,
    stdout: TextIO,
    stream: bool,
    path_picker: PathPicker | None,
) -> int:
    logger.info("Starting interactive REPL")
    _refresh_repl_screen(stdout, session, startup=True)
    history = _InputHistory()
    repl_state = _ReplState()
    renderer: InteractiveTTYRenderer | None = None
    if (
        supports_full_terminal_ui(stdout)
        and stdin.isatty()
        and _is_default_input_func(input_func)
    ):
        renderer = InteractiveTTYRenderer(
            stream=stdout,
            status_provider=lambda: _repl_status_line(session, repl_state),
        )
        session.answer_writer = renderer.write_answer
        session.reasoning_writer = renderer.write_reasoning
        session.status_callback = renderer.show_status
    while True:
        try:
            line = _read_repl_line(
                prompt=_repl_prompt(repl_state),
                input_func=input_func,
                stdin=stdin,
                stdout=stdout,
                history=history,
                renderer=renderer,
            )
        except EOFError:
            stdout.write("\n")
            stdout.flush()
            logger.info("REPL exited on EOF")
            return 0
        except KeyboardInterrupt:
            stdout.write("\n")
            stdout.flush()
            logger.info("REPL input interrupted")
            continue

        stripped = line.strip()
        if not stripped:
            continue
        history.add(stripped)

        command = _parse_command(stripped)
        if command is not None:
            should_exit = _run_command(
                command[0],
                command[1],
                session,
                stdout,
                repl_state=repl_state,
                local_config=local_config,
                client_factory=client_factory,
                input_func=input_func,
                stdin=stdin,
                path_picker=path_picker,
            )
            if renderer is not None:
                renderer.finalize_output()
            if should_exit:
                if renderer is not None:
                    renderer.clear_overlay()
                    stdout.write("\r")
                    stdout.flush()
                return 0
            continue

        if repl_state.pending_attachment is not None:
            pending = repl_state.pending_attachment
            try:
                next_active_images = list(repl_state.active_images)
                next_active_documents = list(repl_state.active_documents)
                if pending.kind == "image":
                    next_active_images = [path.expanduser() for path in pending.paths]
                else:
                    next_active_documents = [
                        path.expanduser() for path in pending.paths
                    ]
                content = _build_active_attachment_message(
                    session,
                    prompt=stripped,
                    image_paths=next_active_images,
                    document_paths=next_active_documents,
                )
            except Exception as exc:
                stdout.write(
                    f"[{pending.kind}] could not prepare staged attachment: {exc}\n"
                )
                stdout.flush()
                repl_state.pending_attachment = None
                continue
            repl_state.active_images = next_active_images
            repl_state.active_documents = next_active_documents
            repl_state.pending_attachment = None
            result = session.send_content(content, stream=stream)
        elif repl_state.active_images or repl_state.active_documents:
            try:
                content = _build_active_attachment_message(
                    session,
                    prompt=stripped,
                    image_paths=repl_state.active_images,
                    document_paths=repl_state.active_documents,
                )
            except Exception as exc:
                stdout.write(
                    f"[attachments] could not prepare active attachments: {exc}\n"
                )
                stdout.flush()
                continue
            result = session.send_content(content, stream=stream)
        else:
            result = session.send(stripped, stream=stream)
        if renderer is not None:
            renderer.finalize_output()
        if result.cancelled:
            continue

    return 0


def _build_session(
    *,
    client_factory: Callable[[MistralConfig], Mistral],
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
        remote_config = RemoteMistralConfig.from_env(timeout_ms=config.timeout_ms)
        session.enable_conversations(
            client=client_factory(remote_config),
            model_id=remote_config.model_id,
            store=conversations.store,
        )
        session.apply_conversation_resume_policy(conversations.resume_policy)
    else:
        session.conversations = conversations
    return session


def main(
    argv: Sequence[str] | None = None,
    *,
    input_func: Callable[[str], str] = input,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    client_factory: Callable[[MistralConfig], Mistral] = build_client,
    path_picker: PathPicker | None = None,
) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        stdout.write(f"mistral4cli {__version__}\n")
        stdout.flush()
        return 0
    if not _ensure_supported_platform(stderr):
        return 1
    logging_config = _resolve_logging_config(args)
    configure_logging(logging_config)
    logger.info(
        "CLI start print_defaults=%s once=%s stream=%s tty=%s",
        args.print_defaults,
        args.once is not None,
        not args.no_stream,
        stdin.isatty(),
    )
    config, generation, system_prompt = _resolve_local_configs(args)
    conversations = _resolve_conversation_config(args)
    context = _resolve_context_config(args)
    reasoning_enabled = _resolve_reasoning_visibility(args)
    thinking_visible = _resolve_thinking_visibility(args)
    try:
        pending_conversation = PendingConversationSettings(
            name=(args.conversation_name or "").strip(),
            description=(args.conversation_description or "").strip(),
            metadata=_parse_metadata_pairs(args.conversation_meta),
        )
    except ValueError as exc:
        stderr.write(f"[conversations] {exc}\n")
        stderr.flush()
        return 1
    conversation_registry = _resolve_conversation_registry(args)
    tool_bridge = _build_tool_bridge(args, stderr)
    logging_summary = render_logging_summary(logging_config)

    if args.print_defaults:
        defaults_backend = BackendKind.LOCAL
        defaults_model = config.model_id
        defaults_server: str | None = config.server_url
        if conversations.enabled:
            defaults_backend = BackendKind.REMOTE
            defaults_model = REMOTE_MODEL_ID
            defaults_server = None
        stdout.write(
            render_defaults_summary(
                backend_kind=defaults_backend,
                model_id=defaults_model,
                server_url=defaults_server,
                timeout_ms=config.timeout_ms,
                generation=generation,
                stream_enabled=not args.no_stream,
                reasoning_enabled=reasoning_enabled,
                thinking_visible=thinking_visible,
                conversations=conversations,
                context=context,
                tool_summary=tool_bridge.runtime_summary(),
                logging_summary=logging_summary,
                stream=stdout,
            )
            + "\nSystem prompt:\n"
            + system_prompt
            + "\n"
        )
        stdout.flush()
        return 0

    stream = not args.no_stream
    if args.once is not None:
        session = _build_session(
            client_factory=client_factory,
            config=config,
            generation=generation,
            system_prompt=system_prompt,
            tool_bridge=tool_bridge,
            stdout=stdout,
            stream=stream,
            reasoning_enabled=reasoning_enabled,
            thinking_visible=thinking_visible,
            logging_summary=logging_summary,
            conversations=conversations,
            context=context,
            conversation_registry=conversation_registry,
            pending_conversation=pending_conversation,
        )
        _maybe_resume_conversation(
            session,
            stdin=stdin,
            stdout=stdout,
            input_func=input_func,
            interactive=False,
        )
        session.send(args.once, stream=stream)
        return 0

    if not stdin.isatty():
        piped_prompt = stdin.read().strip()
        if piped_prompt:
            session = _build_session(
                client_factory=client_factory,
                config=config,
                generation=generation,
                system_prompt=system_prompt,
                tool_bridge=tool_bridge,
                stdout=stdout,
                stream=stream,
                reasoning_enabled=reasoning_enabled,
                thinking_visible=thinking_visible,
                logging_summary=logging_summary,
                conversations=conversations,
                context=context,
                conversation_registry=conversation_registry,
                pending_conversation=pending_conversation,
            )
            _maybe_resume_conversation(
                session,
                stdin=stdin,
                stdout=stdout,
                input_func=input_func,
                interactive=False,
            )
            session.send(piped_prompt, stream=stream)
        return 0

    session = _build_session(
        client_factory=client_factory,
        config=config,
        generation=generation,
        system_prompt=system_prompt,
        tool_bridge=tool_bridge,
        stdout=stdout,
        stream=stream,
        reasoning_enabled=reasoning_enabled,
        thinking_visible=thinking_visible,
        logging_summary=logging_summary,
        conversations=conversations,
        context=context,
        conversation_registry=conversation_registry,
        pending_conversation=pending_conversation,
    )
    _maybe_resume_conversation(
        session,
        stdin=stdin,
        stdout=stdout,
        input_func=input_func,
        interactive=stdin.isatty(),
    )
    return _run_repl(
        session,
        local_config=config,
        client_factory=client_factory,
        input_func=input_func,
        stdin=stdin,
        stdout=stdout,
        stream=stream,
        path_picker=path_picker,
    )


if __name__ == "__main__":
    raise SystemExit(main())
