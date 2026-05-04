"""Linux-only command-line entrypoint for the dual-model Mistral CLI."""

from __future__ import annotations

import logging
import shutil  # noqa: F401
import sys
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from mistralcli import __version__
from mistralcli import cli_repl as _cli_repl_mod
from mistralcli.attachments import PathPicker
from mistralcli.cli_commands import (
    _maybe_resume_conversation,
    _parse_timeout_ms,
)
from mistralcli.cli_commands import (
    _run_command as _dispatch_command,
)
from mistralcli.cli_config import (
    _build_session,
    _build_tool_bridge,
    _parse_metadata_pairs,
    _resolve_context_config,
    _resolve_conversation_config,
    _resolve_conversation_registry,
    _resolve_local_configs,
    _resolve_logging_config,
    _resolve_reasoning_visibility,
    _resolve_remote_model_id,
    _resolve_thinking_visibility,
    build_parser,
)
from mistralcli.cli_repl import (
    _clear_screen_if_supported,
    _is_default_input_func,
    _linux_supported,
    _normalize_pasted_text,
    _print_paginated_text,
    _read_bracketed_paste,
    _read_escape_sequence,
    _read_repl_line,
    _read_tty_line,
    _refresh_repl_screen,
    _set_bracketed_paste,
    _write_tty_newline,
)
from mistralcli.cli_repl import (
    _run_repl as _run_repl_impl,
)
from mistralcli.cli_shortcuts import _normalize_inline_prompt
from mistralcli.cli_state import (
    _build_active_attachment_message,
    _InputHistory,
    _parse_command,
    _PendingAttachment,
    _repl_status_line,
    _ReplState,
)
from mistralcli.local_mistral import (
    REMOTE_MODEL_ID,
    BackendKind,
    LocalMistralConfig,
    MistralConfig,
    build_client,
)
from mistralcli.logging_config import configure_logging, render_logging_summary
from mistralcli.mistral_client import MistralClientProtocol
from mistralcli.session import (
    MistralSession,
    PendingConversationSettings,
    render_defaults_summary,
)

logger = logging.getLogger("mistralcli.cli")
LINUX_ONLY_MESSAGE = "This client is currently supported on Linux only."


def _ensure_supported_platform(stderr: TextIO) -> bool:
    """Fail fast when the CLI is launched outside Linux."""

    if _linux_supported():
        return True
    stderr.write(LINUX_ONLY_MESSAGE + "\n")
    stderr.flush()
    return False


def _run_command(
    command: str,
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState | None = None,
    local_config: LocalMistralConfig | None = None,
    client_factory: Callable[[MistralConfig], MistralClientProtocol] = build_client,
    remote_model_id: str = REMOTE_MODEL_ID,
    input_func: Callable[[str], str] = input,
    stdin: TextIO = sys.stdin,
    path_picker: PathPicker | None = None,
) -> bool:
    """Run one slash command with the historical cli.py defaults."""

    return _dispatch_command(
        command,
        argument,
        session,
        stdout,
        repl_state=repl_state,
        local_config=local_config,
        client_factory=client_factory,
        remote_model_id=remote_model_id,
        input_func=input_func,
        stdin=stdin,
        path_picker=path_picker,
        print_paginated_text=_print_paginated_text,
        refresh_repl_screen=lambda target_stdout, target_session: _refresh_repl_screen(
            target_stdout,
            target_session,
            startup=False,
        ),
    )


def _run_repl(*args: Any, **kwargs: Any) -> int:
    """Run the interactive REPL while honoring cli.py monkeypatches in tests."""

    original_is_default = _cli_repl_mod._is_default_input_func
    original_read_line = _cli_repl_mod._read_repl_line
    _cli_repl_mod._is_default_input_func = _is_default_input_func
    _cli_repl_mod._read_repl_line = _read_repl_line
    try:
        return _run_repl_impl(*args, **kwargs)
    finally:
        _cli_repl_mod._is_default_input_func = original_is_default
        _cli_repl_mod._read_repl_line = original_read_line


def main(
    argv: Sequence[str] | None = None,
    *,
    input_func: Callable[[str], str] = input,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    client_factory: Callable[[MistralConfig], MistralClientProtocol] = build_client,
    path_picker: PathPicker | None = None,
) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        stdout.write(f"mistralcli {__version__}\n")
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
    try:
        remote_model_id = _resolve_remote_model_id(args)
    except ValueError as exc:
        stderr.write(f"[remote] {exc}\n")
        stderr.flush()
        return 1
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
            defaults_model = remote_model_id
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
            remote_model_id=remote_model_id,
        )
        try:
            _maybe_resume_conversation(
                session,
                stdin=stdin,
                stdout=stdout,
                input_func=input_func,
                interactive=False,
            )
            session.send(args.once, stream=stream)
        finally:
            session.close()
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
                remote_model_id=remote_model_id,
            )
            try:
                _maybe_resume_conversation(
                    session,
                    stdin=stdin,
                    stdout=stdout,
                    input_func=input_func,
                    interactive=False,
                )
                session.send(piped_prompt, stream=stream)
            finally:
                session.close()
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
        remote_model_id=remote_model_id,
    )
    try:
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
            remote_model_id=remote_model_id,
            input_func=input_func,
            stdin=stdin,
            stdout=stdout,
            stream=stream,
            path_picker=path_picker,
        )
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LINUX_ONLY_MESSAGE",
    "_InputHistory",
    "_PendingAttachment",
    "_ReplState",
    "_build_active_attachment_message",
    "_clear_screen_if_supported",
    "_is_default_input_func",
    "_normalize_inline_prompt",
    "_normalize_pasted_text",
    "_parse_command",
    "_parse_timeout_ms",
    "_read_bracketed_paste",
    "_read_escape_sequence",
    "_read_repl_line",
    "_read_tty_line",
    "_refresh_repl_screen",
    "_repl_status_line",
    "_run_command",
    "_run_repl",
    "_set_bracketed_paste",
    "_write_tty_newline",
    "main",
]
