"""Terminal rendering helpers for the Mistral Small 4 CLI."""

from __future__ import annotations

import os
import shutil
import textwrap
from collections.abc import Sequence
from typing import TextIO

from mistral4cli.local_mistral import (
    REMOTE_SERVER_LABEL,
    BackendKind,
    LocalGenerationConfig,
)

GREEN = "\x1b[38;5;82m"
ORANGE = "\x1b[38;5;208m"
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
CLEAR_SCREEN = "\x1b[2J\x1b[H"

ASCII_BANNER = (
    r" __  __ _     _             _        _ _  _           _ _ "
    "\n"
    r"|  \/  (_)___| |_ _ _ __ _| |___   | || || |__  __ _| | |"
    "\n"
    r"| |\/| | (_-<  _| '_/ _` | / / |  | __ || '_ \/ _` | | |"
    "\n"
    r"|_|  |_|_/__/\__|_| \__,_|_\_\_|  |_||_||_.__/\__,_|_|_|"
    "\n"
    r"                    [ Mistral4Small ]                    "
)


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def supports_full_terminal_ui(stream: TextIO) -> bool:
    """Return whether the current output stream supports the interactive TUI."""

    if os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def terminal_recommendation(*, stream: TextIO) -> str:
    """Return a short terminal recommendation when colors may degrade."""

    if not supports_full_terminal_ui(stream):
        return ""
    if os.environ.get("TERM") == "xterm-256color":
        return ""
    return _paint(
        (
            "Recommended terminal setting: TERM=xterm-256color for the intended "
            "retro colors."
        ),
        ORANGE,
        stream,
        bold=True,
    )


def _paint(text: str, color: str, stream: TextIO, *, bold: bool = False) -> str:
    if not _supports_color(stream):
        return text
    prefix = f"{BOLD if bold else ''}{color}"
    return f"{prefix}{text}{RESET}"


def _paint_reasoning(text: str, stream: TextIO) -> str:
    if not _supports_color(stream):
        return text
    return f"{DIM}{ITALIC}{ORANGE}{text}{RESET}"


def _paint_multiline(
    text: str, color: str, stream: TextIO, *, bold: bool = False
) -> str:
    return "\n".join(
        _paint(line, color, stream, bold=bold) for line in text.splitlines()
    )


def _terminal_width(stream: TextIO) -> int:
    try:
        return max(shutil.get_terminal_size().columns, 72)
    except OSError:
        return 100


def _render_table(
    title: str,
    rows: Sequence[tuple[str, str]],
    *,
    stream: TextIO,
) -> str:
    label_width = max(len(label) for label, _ in rows)
    total_width = min(_terminal_width(stream), 120)
    value_width = max(total_width - label_width - 7, 24)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"
    body = [border, f"| {title:<{label_width + value_width + 3}} |", border]
    for label, value in rows:
        wrapped = textwrap.wrap(value, width=value_width) or [""]
        body.append(f"| {label:<{label_width}} | {wrapped[0]:<{value_width}} |")
        for extra in wrapped[1:]:
            body.append(f"| {'':<{label_width}} | {extra:<{value_width}} |")
    body.append(border)
    return _paint_multiline("\n".join(body), GREEN, stream)


def render_runtime_summary(
    *,
    backend_kind: BackendKind,
    model_id: str,
    server_url: str | None,
    timeout_ms: int,
    generation: LocalGenerationConfig,
    stream_enabled: bool,
    reasoning_visible: bool,
    tool_summary: str,
    stream: TextIO,
) -> str:
    """Render a formatted runtime summary."""

    max_tokens = (
        "unset" if generation.max_tokens is None else str(generation.max_tokens)
    )
    prompt_mode = generation.prompt_mode or "unset"
    stream_mode = "on" if stream_enabled else "off"
    reasoning_mode = "on" if reasoning_visible else "off"
    rows = [
        ("Backend", backend_kind.value),
        ("Server", server_url or REMOTE_SERVER_LABEL),
        ("Model", model_id),
        ("Timeout", f"{timeout_ms} ms"),
        (
            "Sampling",
            (
                f"temperature={generation.temperature} "
                f"top_p={generation.top_p} "
                f"prompt_mode={prompt_mode} "
                f"max_tokens={max_tokens} "
                f"stream={stream_mode} "
                f"reasoning={reasoning_mode}"
            ),
        ),
        ("Tools", tool_summary),
    ]
    return _render_table("Mistral Small 4 CLI", rows, stream=stream)


def render_welcome_banner(summary: str, *, stream: TextIO) -> str:
    """Render the startup banner with retro colors when supported."""

    lines = [
        _paint_multiline(ASCII_BANNER, GREEN, stream, bold=True),
        _paint("Mistral4Small retro console", ORANGE, stream, bold=True),
        _paint(
            "Official SDK, local llama.cpp, optional Mistral cloud and MCP tools.",
            GREEN,
            stream,
        ),
        _paint(summary, GREEN, stream),
        _paint(
            "Type /help for actionable commands. Ctrl-C cancels the current answer.",
            DIM,
            stream,
        ),
    ]
    return "\n".join(lines)


def render_help_screen(
    *,
    summary: str,
    tools: Sequence[str] | None,
    stream: TextIO,
) -> str:
    """Render a concise but actionable help screen."""

    tool_lines = list(tools or [])
    runtime_section = [
        _paint("Runtime", ORANGE, stream, bold=True),
        summary,
    ]
    commands_section = [
        _paint("Commands", ORANGE, stream, bold=True),
        "/help        Show this screen.",
        "/defaults    Show the current runtime defaults.",
        "/tools       Inspect local OS tools and FireCrawl MCP tools.",
        "/run         Run a shell command via the local shell tool.",
        "/ls          List files and directories.",
        "/find        Search text in the project tree.",
        "/edit        Write text to a file.",
        (
            "/image       Pick one image in the terminal; with no --prompt, "
            "stage it for the next message."
        ),
        (
            "/doc         Pick one document in the terminal; with no --prompt, "
            "stage it for the next message."
        ),
        "/remote      Show, enable, or disable the Mistral cloud backend.",
        "/timeout     Show or set the active request timeout.",
        "/reasoning   Show, enable, disable, or toggle visible reasoning output.",
        "/reset       Clear the conversation but keep the system prompt.",
        "/system TXT  Replace the system prompt and reset the chat.",
        "/exit        Leave the REPL.",
    ]
    usage_section = [
        _paint("How to use", ORANGE, stream, bold=True),
        "Ask for code, debugging, refactors, explanations, or tests.",
        "For web-backed questions, mention that you want FireCrawl or sources.",
        (
            "The CLI always exposes local OS tools and auto-loads MCP tools "
            "from mcp.json when present."
        ),
        "Tool-assisted turns are managed automatically by the session.",
    ]
    shortcuts_section = [
        _paint("Shortcuts", ORANGE, stream, bold=True),
        "Up/Down arrows browse previous prompts.",
        "Ctrl-C cancels the current response without dropping the session.",
        "Ctrl-D exits the REPL.",
        (
            "In attachment pickers, Enter selects the highlighted file and [..] "
            "goes to the parent directory."
        ),
        "Visible reasoning is rendered in dim italic text when the backend emits it.",
    ]
    examples_section = [
        _paint("Examples", ORANGE, stream, bold=True),
        '  - "Explain this error and propose the smallest possible fix."',
        '  - "/run --cwd . -- git status"',
        '  - "/run --lines 40 --offset 40 -- git log --oneline"',
        '  - "/ls src"',
        '  - "/find --path src --limit 10 -- shell"',
        '  - "/edit notes.txt -- Replace this text."',
        '  - "/image" then type "Describe this image."',
        '  - "/doc" then type "Analyze this document."',
        '  - "/image --prompt Describe the selected image."',
        '  - "/doc --prompt Summarize the selected document."',
        '  - "/remote on"',
        '  - "/timeout 300000"',
        '  - "/reasoning off"',
        '  - "/reasoning toggle"',
        '  - "Use shell to run `git status` and summarize the result."',
        '  - "Read src/mistral4cli/cli.py and summarize the structure."',
        '  - "Search official documentation about X and summarize the API."',
        '  - "/system You are a strict Python 3.10 reviewer."',
        '  - "/reset"',
    ]
    tools_section = [_paint("Available tools", ORANGE, stream, bold=True)]
    if tool_lines:
        tools_section.extend(f"  - {line}" for line in tool_lines)
    else:
        tools_section.append("  - No tool catalogue loaded yet. Use /tools to refresh.")

    return "\n".join(
        [
            *runtime_section,
            "",
            *commands_section,
            "",
            *usage_section,
            "",
            *shortcuts_section,
            "",
            *examples_section,
            "",
            *tools_section,
        ]
    )


def render_tools_screen(tool_lines: Sequence[str], *, stream: TextIO) -> str:
    """Render a detailed tool status screen."""

    header = _paint("Available tools", ORANGE, stream, bold=True)
    if tool_lines:
        body = [f"  - {line}" for line in tool_lines]
    else:
        body = ["  - No MCP tools available."]
    return "\n".join([header, *body])


def render_reasoning_chunk(text: str, *, stream: TextIO) -> str:
    """Render one visible reasoning fragment for the terminal."""

    return _paint_reasoning(text, stream)
