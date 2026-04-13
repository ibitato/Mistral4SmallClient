"""Terminal rendering helpers for the Mistral Small 4 CLI."""

from __future__ import annotations

import os
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

ASCII_BANNER = (
    r" __  __ _       _        _            _     _____ _ _ "
    "\n"
    r"|  \/  (_) __ _| | ___  | | ___   ___| | __|  ___| (_)_ __  "
    "\n"
    r"| |\/| | |/ _` | |/ _ \ | |/ _ \ / __| |/ /| |_  | | | '_ \ "
    "\n"
    r"| |  | | | (_| | |  __/ | | (_) | (__|   < |  _| | | | | | |"
    "\n"
    r"|_|  |_|_|\__, |_|\___| |_|\___/ \___|_|\_\|_|   |_|_|_| |_|"
    "\n"
    r"          |___/                                              "
)


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


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


def render_runtime_summary(
    *,
    backend_kind: BackendKind,
    model_id: str,
    server_url: str | None,
    generation: LocalGenerationConfig,
    stream_enabled: bool,
    reasoning_visible: bool,
    tool_summary: str,
) -> str:
    """Render a compact runtime summary."""

    max_tokens = (
        "unset" if generation.max_tokens is None else str(generation.max_tokens)
    )
    prompt_mode = generation.prompt_mode or "unset"
    stream_mode = "on" if stream_enabled else "off"
    reasoning_mode = "on" if reasoning_visible else "off"
    lines = [
        "Mistral Small 4 CLI",
        f"Backend: {backend_kind.value}",
        f"Server: {server_url or REMOTE_SERVER_LABEL}",
        f"Model: {model_id}",
        (
            "Sampling: "
            f"temperature={generation.temperature} "
            f"top_p={generation.top_p} "
            f"prompt_mode={prompt_mode} "
            f"max_tokens={max_tokens} "
            f"stream={stream_mode} "
            f"reasoning={reasoning_mode}"
        ),
        f"Tools: {tool_summary}",
    ]
    return "\n".join(lines)


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
        "/image       Pick images and ask the model to analyze them.",
        "/doc         Pick documents and send page images for OCR analysis.",
        "/remote      Show, enable, or disable the Mistral cloud backend.",
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
        '  - "/image --prompt Describe the selected image."',
        '  - "/doc --prompt Summarize the file contents."',
        '  - "/remote on"',
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
