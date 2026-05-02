"""Local slash-command shortcuts for files, tools, and attachments."""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Callable
from typing import TextIO

from mistralcli.attachments import (
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
from mistralcli.cli_state import _PendingAttachment, _ReplState, _set_active_attachment
from mistralcli.local_mistral import BackendKind
from mistralcli.session import MistralSession


def _split_shortcut_argument(argument: str) -> tuple[str, str | None]:
    head, separator, tail = argument.partition(" -- ")
    if separator:
        return head.strip(), tail.strip()
    return argument.strip(), None


def _parse_shortcut_options(
    parser: argparse.ArgumentParser,
    tokens: list[str],
    stdout: TextIO,
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
