"""Command-line entrypoint for the Mistral Small 4 coding CLI."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TextIO

from mistralai.client import Mistral

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
from mistral4cli.local_mistral import (
    DEFAULT_API_KEY,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_MODE,
    DEFAULT_SERVER_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TOP_P,
    BackendKind,
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
from mistral4cli.mcp_bridge import (
    MCPConfig,
    MCPToolBridge,
    discover_mcp_config_path,
)
from mistral4cli.session import (
    DEFAULT_SYSTEM_PROMPT,
    MistralCodingSession,
    render_defaults_summary,
)
from mistral4cli.tooling import CompositeToolBridge, ToolBridge
from mistral4cli.ui import render_help_screen, render_welcome_banner


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
            "Interactive coding CLI for Mistral Small 4 local and remote backends."
        ),
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
        "--system-prompt",
        default=None,
        help="Override the default coding assistant system prompt.",
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
        "--print-defaults",
        action="store_true",
        help="Print the effective defaults and exit.",
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


def _resolve_remote_mcp_bridge(
    args: argparse.Namespace, stderr: TextIO
) -> MCPToolBridge | None:
    if args.no_mcp:
        return None

    config_path = discover_mcp_config_path(args.mcp_config)
    if config_path is None:
        return None

    try:
        config = MCPConfig.load(config_path)
    except FileNotFoundError:
        if args.mcp_config is not None or os.environ.get("MISTRAL_LOCAL_MCP_CONFIG"):
            stderr.write(f"[mcp] config not found: {config_path}\n")
            stderr.flush()
        return None
    except Exception as exc:
        stderr.write(f"[mcp] could not load {config_path}: {exc}\n")
        stderr.flush()
        return None

    if not config.configured:
        return None

    return MCPToolBridge(config)


def _build_tool_bridge(args: argparse.Namespace, stderr: TextIO) -> CompositeToolBridge:
    bridges: list[ToolBridge] = [LocalToolBridge()]
    remote_bridge = _resolve_remote_mcp_bridge(args, stderr)
    if remote_bridge is not None:
        bridges.append(remote_bridge)
    return CompositeToolBridge(bridges=bridges)


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


def _run_shell_shortcut(
    argument: str, session: MistralCodingSession, stdout: TextIO
) -> bool:
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


def _run_ls_shortcut(
    argument: str, session: MistralCodingSession, stdout: TextIO
) -> bool:
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


def _run_find_shortcut(
    argument: str, session: MistralCodingSession, stdout: TextIO
) -> bool:
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


def _run_edit_shortcut(
    argument: str, session: MistralCodingSession, stdout: TextIO
) -> bool:
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
    session: MistralCodingSession,
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
    session.send_content(message, stream=session.stream_enabled, disable_tools=True)
    return False


def _run_doc_shortcut(
    argument: str,
    session: MistralCodingSession,
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
    session.send_content(message, stream=session.stream_enabled, disable_tools=True)
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


def _print_banner(stdout: TextIO, session: MistralCodingSession) -> None:
    stdout.write(
        render_welcome_banner(session.describe_defaults(), stream=stdout) + "\n"
    )
    stdout.flush()


def _print_runtime_refresh(stdout: TextIO, session: MistralCodingSession) -> None:
    stdout.write(session.describe_defaults() + "\n")
    stdout.flush()


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
    prompt: str, stdin: TextIO, stdout: TextIO, history: _InputHistory
) -> str:
    """Read one REPL line with simple raw-mode editing and history navigation."""

    import termios
    import tty

    fileno = stdin.fileno()
    original_attrs = termios.tcgetattr(fileno)
    buffer = ""
    stdout.write(prompt)
    stdout.flush()
    try:
        tty.setraw(fileno)
        while True:
            char = stdin.read(1)
            if char == "":
                raise EOFError
            if char in {"\r", "\n"}:
                _write_tty_newline(stdout)
                history.reset_navigation()
                return buffer
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x04":
                if not buffer:
                    raise EOFError
                continue
            if char in {"\x7f", "\b"}:
                if buffer:
                    buffer = buffer[:-1]
                    _redraw_prompt_line(stdout, prompt, buffer)
                continue
            if char == "\x1b":
                next_char = stdin.read(1)
                if next_char != "[":
                    continue
                direction = stdin.read(1)
                if direction == "A":
                    buffer = history.previous(buffer)
                    _redraw_prompt_line(stdout, prompt, buffer)
                elif direction == "B":
                    buffer = history.next()
                    _redraw_prompt_line(stdout, prompt, buffer)
                continue
            if char.isprintable():
                buffer += char
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
) -> str:
    """Read one REPL line using raw TTY mode when available."""

    if stdin.isatty() and _is_default_input_func(input_func):
        return _read_tty_line(prompt, stdin, stdout, history)
    return input_func(prompt)


def _run_command(
    command: str,
    argument: str,
    session: MistralCodingSession,
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
    if command in {"help", "h", "?"}:
        stdout.write(
            render_help_screen(
                summary=session.describe_defaults(),
                tools=session.describe_tools().splitlines(),
                stream=stdout,
            )
            + "\n"
        )
        stdout.flush()
        return False
    if command == "tools":
        stdout.write(session.describe_tools() + "\n")
        stdout.flush()
        return False
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
            stdout.write("Visible reasoning enabled.\n")
        elif normalized in {"off", "false", "0"}:
            session.set_reasoning_visibility(False)
            stdout.write("Visible reasoning disabled.\n")
        elif normalized == "toggle":
            session.toggle_reasoning_visibility()
            stdout.write(session.reasoning_status_text() + "\n")
        else:
            stdout.write("Usage: /reasoning [on|off|toggle]\n")
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
    if command in {"exit", "quit", "q"}:
        return True
    if command in {"reset", "new"}:
        repl_state.pending_attachment = None
        session.reset()
        stdout.write("Conversation reset.\n")
        _print_runtime_refresh(stdout, session)
        return False
    if command == "defaults":
        stdout.write(session.describe_defaults() + "\n")
        stdout.flush()
        return False
    if command == "system":
        if argument:
            repl_state.pending_attachment = None
            session.set_system_prompt(argument)
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


def _run_remote_command(
    argument: str,
    session: MistralCodingSession,
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
        repl_state.pending_attachment = None
        stdout.write("Remote backend enabled. Conversation reset.\n")
        _print_runtime_refresh(stdout, session)
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
        repl_state.pending_attachment = None
        stdout.write("Local backend enabled. Conversation reset.\n")
        _print_runtime_refresh(stdout, session)
        return False

    stdout.write("Usage: /remote [on|off]\n")
    stdout.flush()
    return False


def _run_timeout_command(
    argument: str,
    session: MistralCodingSession,
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
    session: MistralCodingSession,
    *,
    local_config: LocalMistralConfig,
    client_factory: Callable[[MistralConfig], Mistral],
    input_func: Callable[[str], str],
    stdin: TextIO,
    stdout: TextIO,
    stream: bool,
    path_picker: PathPicker | None,
) -> int:
    _print_banner(stdout, session)
    history = _InputHistory()
    repl_state = _ReplState()
    while True:
        try:
            line = _read_repl_line(
                prompt="mistral4small> ",
                input_func=input_func,
                stdin=stdin,
                stdout=stdout,
                history=history,
            )
        except EOFError:
            stdout.write("\n")
            stdout.flush()
            return 0
        except KeyboardInterrupt:
            stdout.write("\n")
            stdout.flush()
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
            if should_exit:
                return 0
            continue

        if repl_state.pending_attachment is not None:
            pending = repl_state.pending_attachment
            try:
                if pending.kind == "image":
                    if session.backend_kind is BackendKind.REMOTE:
                        content = build_remote_image_message(
                            pending.paths, prompt=stripped
                        )
                    else:
                        content = build_image_message(pending.paths, prompt=stripped)
                else:
                    if session.backend_kind is BackendKind.REMOTE:
                        content = build_remote_document_message(
                            pending.paths, prompt=stripped
                        )
                    else:
                        content = build_document_message(pending.paths, prompt=stripped)
            except Exception as exc:
                stdout.write(
                    f"[{pending.kind}] could not prepare staged attachment: {exc}\n"
                )
                stdout.flush()
                repl_state.pending_attachment = None
                continue
            repl_state.pending_attachment = None
            result = session.send_content(content, stream=stream, disable_tools=True)
        else:
            result = session.send(stripped, stream=stream)
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
) -> MistralCodingSession:
    return MistralCodingSession(
        client=client_factory(config),
        backend_kind=BackendKind.LOCAL,
        model_id=config.model_id,
        server_url=config.server_url,
        generation=generation,
        system_prompt=system_prompt,
        tool_bridge=tool_bridge,
        stdout=stdout,
        stream_enabled=stream,
    )


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
    config, generation, system_prompt = _resolve_local_configs(args)
    tool_bridge = _build_tool_bridge(args, stderr)

    if args.print_defaults:
        stdout.write(
            render_defaults_summary(
                backend_kind=BackendKind.LOCAL,
                model_id=config.model_id,
                server_url=config.server_url,
                timeout_ms=config.timeout_ms,
                generation=generation,
                stream_enabled=not args.no_stream,
                reasoning_visible=True,
                tool_summary=tool_bridge.runtime_summary(),
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
