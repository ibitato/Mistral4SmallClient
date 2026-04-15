"""Terminal rendering helpers for the Mistral Small 4 CLI."""

from __future__ import annotations

import os
import re
import shutil
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PROMPT_CONTINUATION_PREFIX = "... "
ANSWER_TYPEWRITER_VISIBLE_CHARS = 48
ANSWER_TYPEWRITER_PAUSE_S = 0.008
REASONING_TYPEWRITER_VISIBLE_CHARS = 56
REASONING_TYPEWRITER_PAUSE_S = 0.006


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


def _paint_prompt(text: str, stream: TextIO) -> str:
    return _paint(text, GREEN, stream, bold=True)


def _paint_multiline(
    text: str, color: str, stream: TextIO, *, bold: bool = False
) -> str:
    return "\n".join(
        _paint(line, color, stream, bold=bold) for line in text.splitlines()
    )


def _terminal_width(stream: TextIO, *, minimum: int = 72) -> int:
    try:
        return max(shutil.get_terminal_size().columns, minimum)
    except OSError:
        return max(100, minimum)


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _truncate_plain_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return text[:1]
    if width <= 3:
        return text[:width]
    return f"{text[: width - 3]}..."


def _clear_rendered_block(stream: TextIO, line_count: int) -> None:
    if line_count <= 0:
        return
    stream.write("\r")
    for _ in range(line_count - 1):
        stream.write("\x1b[1A")
    for index in range(line_count):
        stream.write("\r\x1b[2K")
        if index < line_count - 1:
            stream.write("\x1b[1B")
    for _ in range(line_count - 1):
        stream.write("\x1b[1A")
    stream.write("\r")
    stream.flush()


def iter_typewriter_chunks(text: str, *, visible_chars: int) -> list[str]:
    """Split ANSI-decorated text into small visible chunks for TTY playback."""

    if not text:
        return []
    chunks: list[str] = []
    current: list[str] = []
    visible = 0
    index = 0
    while index < len(text):
        match = ANSI_ESCAPE_RE.match(text, index)
        if match is not None:
            current.append(match.group(0))
            index = match.end()
            continue
        char = text[index]
        current.append(char)
        index += 1
        if char == "\n":
            chunks.append("".join(current))
            current = []
            visible = 0
            continue
        visible += 1
        if visible >= visible_chars:
            chunks.append("".join(current))
            current = []
            visible = 0
    if current:
        chunks.append("".join(current))
    return chunks


def wrap_prompt_buffer(prompt: str, buffer: str, *, width: int) -> list[str]:
    """Wrap one logical REPL buffer into prompt-display lines."""

    available_width = max(width, len(prompt) + 8, len(PROMPT_CONTINUATION_PREFIX) + 8)
    content = buffer or ""
    wrapped = textwrap.wrap(
        content,
        width=available_width,
        initial_indent=prompt,
        subsequent_indent=PROMPT_CONTINUATION_PREFIX,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return [prompt]
    return wrapped


def paint_prompt_lines(
    lines: Sequence[str],
    *,
    prompt: str,
    stream: TextIO,
) -> list[str]:
    """Paint the REPL prompt prefixes without affecting wrap calculations."""

    painted: list[str] = []
    for line in lines:
        if line.startswith(prompt):
            painted.append(_paint_prompt(prompt, stream) + line[len(prompt) :])
            continue
        if line.startswith(PROMPT_CONTINUATION_PREFIX):
            painted.append(
                _paint_prompt(PROMPT_CONTINUATION_PREFIX, stream)
                + line[len(PROMPT_CONTINUATION_PREFIX) :]
            )
            continue
        painted.append(line)
    return painted


def _detect_hanging_indent(text: str) -> str:
    bullet_match = re.match(r"^(\s*(?:[-*+]\s+|\d+\.\s+))", text)
    if bullet_match is not None:
        return " " * len(bullet_match.group(1))
    return ""


def _looks_like_literal_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if text.startswith(("    ", "\t")):
        return True
    if stripped.startswith("```"):
        return True
    if text.lstrip() != text and stripped.startswith(("-", "*", "+")):
        return True
    if stripped.startswith(("{", "}", "[", "]")):
        return True
    if stripped.endswith(("{", "[", "}", "]", ",", ":")):
        return True
    return bool("|" in text and text.count("|") >= 2)


def _wrap_prose_line(text: str, *, width: int) -> list[str]:
    hanging_indent = _detect_hanging_indent(text)
    wrapper = textwrap.TextWrapper(
        width=width,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=False,
        break_on_hyphens=False,
        subsequent_indent=hanging_indent,
    )
    wrapped = wrapper.wrap(text)
    return wrapped or [text]


@dataclass(slots=True)
class SmartOutputWriter:
    """Incremental terminal writer that wraps normal prose safely."""

    stream: TextIO
    style: Callable[[str, TextIO], str] | None = None
    _pending: str = ""
    _in_fence: bool = False

    def feed(self, text: str) -> str:
        """Consume streamed text and return the wrapped terminal output."""

        if not text:
            return ""
        self._pending += text
        output: list[str] = []
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            output.extend(self._render_complete_line(line))
            output.append("\n")
        output.extend(self._emit_soft_wraps())
        return "".join(output)

    def finish(self) -> str:
        """Flush any pending text that was held for wrapping decisions."""

        if not self._pending:
            return ""
        line = self._pending
        self._pending = ""
        return "".join(self._render_complete_line(line))

    def _render_complete_line(self, line: str) -> list[str]:
        stripped = line.strip()
        literal = self._in_fence or _looks_like_literal_line(line)
        rendered_lines = (
            [line]
            if literal
            else _wrap_prose_line(
                line,
                width=_terminal_width(self.stream, minimum=20),
            )
        )
        if stripped.startswith("```"):
            self._in_fence = not self._in_fence
        return [self._style_line(item) for item in rendered_lines]

    def _emit_soft_wraps(self) -> list[str]:
        if (
            not self._pending
            or self._in_fence
            or _looks_like_literal_line(self._pending)
        ):
            return []
        output: list[str] = []
        width = _terminal_width(self.stream, minimum=20)
        while len(self._pending) > width:
            breakpoint = self._soft_break_index(self._pending, width)
            if breakpoint is None:
                break
            line = self._pending[:breakpoint].rstrip()
            remainder = self._pending[breakpoint:].lstrip()
            output.append(self._style_line(line))
            output.append("\n")
            self._pending = remainder
        return output

    def _soft_break_index(self, text: str, width: int) -> int | None:
        window = text[:width]
        whitespace_positions = [
            index for index, char in enumerate(window) if char.isspace()
        ]
        if not whitespace_positions:
            return None
        return whitespace_positions[-1] + 1

    def _style_line(self, text: str) -> str:
        if self.style is None or not text:
            return text
        return self.style(text, self.stream)


@dataclass(slots=True)
class InteractiveTTYRenderer:
    """Manage the wrapped REPL composer, status bar, and streamed output."""

    stream: TextIO
    status_provider: Callable[[], str]
    answer_writer: SmartOutputWriter = field(init=False)
    reasoning_writer: SmartOutputWriter = field(init=False)
    _overlay_lines: int = 0

    def __post_init__(self) -> None:
        self.answer_writer = SmartOutputWriter(stream=self.stream)
        self.reasoning_writer = SmartOutputWriter(
            stream=self.stream,
            style=lambda text, active_stream: _paint_reasoning(text, active_stream),
        )

    def render_input(self, prompt: str, buffer: str) -> None:
        """Draw only the wrapped input composer while the user is typing."""

        lines = wrap_prompt_buffer(
            prompt,
            buffer,
            width=_terminal_width(self.stream, minimum=20),
        )
        self._replace_overlay(
            paint_prompt_lines(lines, prompt=prompt, stream=self.stream)
        )

    def commit_input(self, prompt: str, buffer: str) -> None:
        """Replace the overlay with the committed input before a turn starts."""

        lines = wrap_prompt_buffer(
            prompt,
            buffer,
            width=_terminal_width(self.stream, minimum=20),
        )
        self.clear_overlay()
        painted = paint_prompt_lines(lines, prompt=prompt, stream=self.stream)
        self.stream.write("\n".join(painted) + "\n")
        self.stream.flush()

    def clear_overlay(self) -> None:
        """Remove the currently rendered composer and status-bar overlay."""

        _clear_rendered_block(self.stream, self._overlay_lines)
        self._overlay_lines = 0

    def show_status(self) -> None:
        """Render only the bottom status bar as the active overlay."""

        self._replace_overlay([self.render_status_bar()])

    def write_answer(self, text: str) -> None:
        """Write wrapped assistant answer text below the overlay."""

        self.clear_overlay()
        # Flush any pending reasoning fragment before the answer starts so the
        # visible chain-of-thought never gets stranded in the TTY writer buffer.
        reasoning_tail = self.reasoning_writer.finish()
        if reasoning_tail:
            self._write_typewriter(
                reasoning_tail,
                visible_chars=REASONING_TYPEWRITER_VISIBLE_CHARS,
                pause_s=REASONING_TYPEWRITER_PAUSE_S,
            )
        rendered = self.answer_writer.feed(text)
        if rendered:
            self._write_typewriter(
                rendered,
                visible_chars=ANSWER_TYPEWRITER_VISIBLE_CHARS,
                pause_s=ANSWER_TYPEWRITER_PAUSE_S,
            )

    def write_reasoning(self, text: str) -> None:
        """Write wrapped visible-reasoning text below the overlay."""

        self.clear_overlay()
        rendered = self.reasoning_writer.feed(text)
        if rendered:
            self._write_typewriter(
                rendered,
                visible_chars=REASONING_TYPEWRITER_VISIBLE_CHARS,
                pause_s=REASONING_TYPEWRITER_PAUSE_S,
            )

    def finalize_output(self) -> None:
        """Flush any pending output fragments before restoring the status bar."""

        self.clear_overlay()
        answer_tail = self.answer_writer.finish()
        reasoning_tail = self.reasoning_writer.finish()
        if reasoning_tail:
            self._write_typewriter(
                reasoning_tail,
                visible_chars=REASONING_TYPEWRITER_VISIBLE_CHARS,
                pause_s=REASONING_TYPEWRITER_PAUSE_S,
            )
        if answer_tail:
            self._write_typewriter(
                answer_tail,
                visible_chars=ANSWER_TYPEWRITER_VISIBLE_CHARS,
                pause_s=ANSWER_TYPEWRITER_PAUSE_S,
            )

    def render_status_bar(self) -> str:
        """Render the one-line bottom status bar for the current turn state."""

        width = _terminal_width(self.stream, minimum=20)
        # Keep one column free to avoid terminal auto-wrap when the status line
        # lands exactly on the last visible cell.
        content_width = max(1, width - 1)
        text = _truncate_plain_text(self.status_provider(), content_width)
        return _paint(text.ljust(content_width), ORANGE, self.stream, bold=True)

    def _replace_overlay(self, lines: Sequence[str]) -> None:
        self.clear_overlay()
        rendered = list(lines)
        if not rendered:
            return
        self.stream.write("\r")
        self.stream.write("\n".join(f"{line}\x1b[K" for line in rendered))
        self.stream.flush()
        self._overlay_lines = len(rendered)

    def _write_typewriter(
        self,
        text: str,
        *,
        visible_chars: int,
        pause_s: float,
    ) -> None:
        """Emit streamed output in fast ANSI-safe batches for a typewriter feel."""

        if not text:
            return
        chunks = iter_typewriter_chunks(text, visible_chars=visible_chars)
        if len(chunks) <= 1:
            self.stream.write(text)
            self.stream.flush()
            return
        for index, chunk in enumerate(chunks):
            self.stream.write(chunk)
            self.stream.flush()
            if index < len(chunks) - 1:
                time.sleep(pause_s)


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
    logging_summary: str,
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
        ("Logging", logging_summary),
    ]
    return _render_table("Mistral Small 4 CLI", rows, stream=stream)


def render_welcome_banner(summary: str, *, stream: TextIO) -> str:
    """Render the startup banner with retro colors when supported."""

    lines = [
        _paint_multiline(ASCII_BANNER, GREEN, stream, bold=True),
        _paint("Mistral4Small multimodal console", ORANGE, stream, bold=True),
        _paint(
            (
                "Official SDK, local llama.cpp, remote Mistral cloud, "
                "multimodal turns, and optional tools."
            ),
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
            "/image       Pick one image in the terminal; it stays active until "
            "you drop it."
        ),
        (
            "/doc         Pick one document in the terminal; it stays active "
            "until you drop it."
        ),
        "/drop        Clear all active and staged attachments.",
        "/dropimage   Clear active and staged image attachments.",
        "/dropdoc     Clear active and staged document attachments.",
        "/remote      Show, enable, or disable the Mistral cloud backend.",
        "/timeout     Show or set the active request timeout.",
        "/reasoning   Show, enable, disable, or toggle visible reasoning output.",
        "/reset       Clear the conversation but keep the system prompt.",
        "/system TXT  Replace the system prompt and reset the chat.",
        "/exit        Leave the REPL.",
    ]
    usage_section = [
        _paint("How to use", ORANGE, stream, bold=True),
        (
            "Ask questions, analyze documents or images, summarize, extract, "
            "compare, draft, translate, troubleshoot, or code."
        ),
        "For web-backed questions, mention that you want FireCrawl or sources.",
        (
            "The CLI always exposes local OS tools and auto-loads optional MCP "
            "tools from mcp.json when present."
        ),
        "Tool-assisted turns are available when the task needs them.",
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
        '  - "Summarize this document and extract action items."',
        '  - "Describe this image and list all visible text."',
        '  - "Compare the attached PDF with the image and highlight differences."',
        '  - "Translate the attached text into English and keep the structure."',
        '  - "Explain this error and propose the smallest possible fix."',
        '  - "/run --cwd . -- git status"',
        '  - "/run --lines 40 --offset 40 -- git log --oneline"',
        '  - "/ls /tmp"',
        '  - "/find --path docs --limit 10 -- mistral"',
        '  - "/edit notes.txt -- Replace this text."',
        '  - "/image" then type "Describe this image."',
        '  - "/doc" then type "Analyze this document."',
        '  - "/image --prompt Describe the selected image."',
        '  - "/doc --prompt Summarize the selected document."',
        '  - "/drop"',
        '  - "/remote on"',
        '  - "/timeout 300000"',
        '  - "/reasoning off"',
        '  - "/reasoning toggle"',
        '  - "Use shell to inspect /tmp and summarize what changed today."',
        '  - "Read README.md and summarize the main capabilities."',
        '  - "Search official documentation about X and summarize the API."',
        '  - "/system You are a concise multilingual assistant."',
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
