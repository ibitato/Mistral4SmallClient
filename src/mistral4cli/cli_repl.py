"""Interactive REPL loop and terminal helpers for the CLI."""

from __future__ import annotations

import logging
import re
import shutil
import sys
from collections.abc import Callable
from typing import TextIO

from mistral4cli.attachments import PathPicker
from mistral4cli.cli_commands import _run_command
from mistral4cli.cli_state import (
    _build_active_attachment_message,
    _InputHistory,
    _parse_command,
    _repl_prompt,
    _repl_status_line,
    _ReplState,
)
from mistral4cli.local_mistral import (
    LocalMistralConfig,
    MistralConfig,
    REMOTE_MODEL_ID,
)
from mistral4cli.mistral_client import MistralClientProtocol
from mistral4cli.session import MistralSession
from mistral4cli.ui import (
    CLEAR_SCREEN,
    InteractiveTTYRenderer,
    render_welcome_banner,
    supports_full_terminal_ui,
    terminal_recommendation,
)

logger = logging.getLogger("mistral4cli.cli")

BRACKETED_PASTE_ENABLE = "\x1b[?2004h"
BRACKETED_PASTE_DISABLE = "\x1b[?2004l"
BRACKETED_PASTE_START = "200~"
BRACKETED_PASTE_END = "\x1b[201~"


def _linux_supported() -> bool:
    """Return whether the current runtime platform is supported."""

    return sys.platform.startswith("linux")


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


def refresh_repl_screen(stdout: TextIO, session: MistralSession) -> None:
    """Refresh the interactive screen after a runtime-changing command."""

    _refresh_repl_screen(stdout, session, startup=False)


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


def _set_bracketed_paste(stdout: TextIO, *, enabled: bool) -> None:
    """Toggle bracketed paste mode for terminals that support it."""

    stdout.write(BRACKETED_PASTE_ENABLE if enabled else BRACKETED_PASTE_DISABLE)
    stdout.flush()


def _normalize_pasted_text(text: str) -> str:
    """Fold multiline pasted text into one prompt buffer line."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]*\n+[ \t]*", " ", normalized)
    normalized = normalized.replace("\t", " ")
    return normalized.strip()


def _read_bracketed_paste(stdin: TextIO) -> str:
    """Read one bracketed paste payload and return the normalized text."""

    chars: list[str] = []
    while True:
        char = stdin.read(1)
        if char == "":
            break
        chars.append(char)
        if "".join(chars[-len(BRACKETED_PASTE_END) :]) == BRACKETED_PASTE_END:
            del chars[-len(BRACKETED_PASTE_END) :]
            break
    return _normalize_pasted_text("".join(chars))


def _read_escape_sequence(stdin: TextIO) -> str:
    """Read one CSI escape sequence payload after the initial ESC byte."""

    next_char = stdin.read(1)
    if next_char != "[":
        return next_char
    payload = ""
    while True:
        char = stdin.read(1)
        if char == "":
            return payload
        payload += char
        if char.isalpha() or char == "~":
            return payload


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
        _set_bracketed_paste(stdout, enabled=True)
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
                sequence = _read_escape_sequence(stdin)
                if sequence == "A":
                    buffer = history.previous(buffer)
                    if renderer is not None:
                        renderer.render_input(prompt, buffer)
                    else:
                        _redraw_prompt_line(stdout, prompt, buffer)
                elif sequence == "B":
                    buffer = history.next()
                    if renderer is not None:
                        renderer.render_input(prompt, buffer)
                    else:
                        _redraw_prompt_line(stdout, prompt, buffer)
                elif sequence == BRACKETED_PASTE_START:
                    pasted = _read_bracketed_paste(stdin)
                    if pasted:
                        buffer += pasted
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
        _set_bracketed_paste(stdout, enabled=False)
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


def _run_repl(
    session: MistralSession,
    *,
    local_config: LocalMistralConfig,
    client_factory: Callable[[MistralConfig], MistralClientProtocol],
    remote_model_id: str = REMOTE_MODEL_ID,
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
                remote_model_id=remote_model_id,
                input_func=input_func,
                stdin=stdin,
                path_picker=path_picker,
                print_paginated_text=_print_paginated_text,
                refresh_repl_screen=refresh_repl_screen,
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
