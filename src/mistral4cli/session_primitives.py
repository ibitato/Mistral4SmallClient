"""Shared types, constants, and pure helpers for interactive sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TextIO

from mistral4cli.local_mistral import (
    BackendKind,
    ContextConfig,
    ConversationConfig,
    LocalGenerationConfig,
)
from mistral4cli.mistral_client import MistralClientProtocol
from mistral4cli.ui import render_runtime_summary

DEFAULT_SYSTEM_PROMPT = "\n".join(
    [
        (
            "You are the assistant inside the Mistral4Cli Linux terminal "
            "client for using and testing Mistral Small 4 and Mistral Medium 3.5."
        ),
        (
            "Respond directly, stay focused on the user's goal, and prefer "
            "concrete results over speculation."
        ),
        "",
        "Environment rules:",
        (
            "- This client is supported on Linux only. Assume Linux paths, "
            "commands, and tooling semantics."
        ),
        (
            "- You always have access to these local tools: shell, read_file, "
            "write_file, list_dir, and search_text."
        ),
        "- You can also use MCP when external information is needed.",
        (
            "- Before asserting anything about the filesystem, system state, "
            "or tool-accessible facts, verify with tools whenever practical."
        ),
        (
            "- Tool results are authoritative. After a successful tool call, "
            "use that result instead of repeating the same call."
        ),
        "",
        "Tool selection rules:",
        (
            "- shell is the primary tool for OS inspection and command "
            "execution. Use it for rg/grep/find, git, processes, services, "
            "packages, env vars, permissions, logs, network state, /proc, "
            "/sys, and general Linux discovery."
        ),
        (
            "- search_text is only for searching text inside files under a "
            "specific workspace path. It returns one matching line per file "
            "and is not a replacement for shell grep/find or OS-wide search."
        ),
        "- list_dir is for directory orientation before deeper reads or searches.",
        "- read_file is for reading a specific known file after you know the path.",
        (
            "- write_file is only for creating or updating text on disk when "
            "the user asks for that outcome."
        ),
        "",
        "Examples:",
        '- "Find files mentioning timeout in src/" -> search_text with path=src.',
        '- "Check running nginx processes" -> shell.',
        '- "Search the OS for docker service files" -> shell.',
        '- "Show what is in /etc/systemd" -> list_dir or shell.',
        '- "Read pyproject.toml" -> read_file.',
        '- "Save this summary to notes.txt" -> write_file.',
        "",
        "Attachment rules:",
        (
            "- If the current user turn includes attached images or documents, "
            "analyze those attachments directly first."
        ),
        (
            "- Do not call shell, local tools, MCP, or external OCR/search "
            "tools for an attachment turn unless the user explicitly asks for "
            "that or the task clearly requires a tool action such as saving, "
            "exporting, or editing a file."
        ),
        (
            "- Do not claim the attachments are missing when the current "
            "message already contains them."
        ),
        (
            "- If the conversation includes attached images or documents, "
            "analyze them carefully before replying."
        ),
        "",
        "If context is missing, ask for the minimum needed before guessing.",
    ]
)

REASONING_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("[THINK]", "[/THINK]"),
    ("[think]", "[/think]"),
)

REMOTE_CONTEXT_WINDOWS = {
    "mistral-small-latest": 256_000,
    "mistral-small-2603": 256_000,
    "mistral-small-2603+1": 256_000,
    "mistral-small-4-0-26-03": 256_000,
    "mistral-medium-3.5": 256_000,
}


def render_defaults_summary(
    *,
    backend_kind: BackendKind,
    model_id: str,
    server_url: str | None,
    timeout_ms: int,
    generation: LocalGenerationConfig,
    stream_enabled: bool,
    reasoning_enabled: bool,
    thinking_visible: bool,
    conversations: ConversationConfig | None = None,
    context: ContextConfig | None = None,
    conversation_id: str | None = None,
    tool_summary: str,
    logging_summary: str,
    stream: TextIO,
) -> str:
    """Render the active runtime defaults as human-readable text."""

    return render_runtime_summary(
        backend_kind=backend_kind,
        model_id=model_id,
        server_url=server_url,
        timeout_ms=timeout_ms,
        generation=generation,
        stream_enabled=stream_enabled,
        reasoning_enabled=reasoning_enabled,
        thinking_visible=thinking_visible,
        conversations=conversations or ConversationConfig(),
        context=context or ContextConfig(),
        conversation_id=conversation_id,
        tool_summary=tool_summary,
        logging_summary=logging_summary,
        stream=stream,
    )


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Result of a single user turn."""

    content: str
    finish_reason: str
    reasoning: str = ""
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """Normalized token usage metadata for one turn or a session."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    max_context_tokens: int | None = None

    def merge(self, other: UsageSnapshot | None) -> UsageSnapshot:
        """Return the cumulative sum of this usage and another snapshot."""

        if other is None:
            return self
        return UsageSnapshot(
            prompt_tokens=_sum_optional_ints(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=_sum_optional_ints(
                self.completion_tokens,
                other.completion_tokens,
            ),
            total_tokens=_sum_optional_ints(self.total_tokens, other.total_tokens),
            max_context_tokens=self.max_context_tokens or other.max_context_tokens,
        )

    def is_empty(self) -> bool:
        """Return whether this usage snapshot carries any usable values."""

        return (
            self.prompt_tokens is None
            and self.completion_tokens is None
            and self.total_tokens is None
        )


@dataclass(frozen=True, slots=True)
class ContextStatus:
    """Estimated context state before sending a chat-completions turn."""

    estimated_tokens: int
    window_tokens: int
    threshold_tokens: int
    reserve_tokens: int
    auto_compact: bool


@dataclass(frozen=True, slots=True)
class CompactResult:
    """Result of a manual or automatic context compaction pass."""

    changed: bool
    reason: str
    before_tokens: int
    after_tokens: int
    window_tokens: int
    threshold_tokens: int

    def summary(self) -> str:
        """Return a concise user-facing compaction summary."""

        state = "compacted" if self.changed else "unchanged"
        return (
            f"Context {state}: {self.before_tokens}->{self.after_tokens}/"
            f"{self.window_tokens} tokens "
            f"(threshold={self.threshold_tokens}). {self.reason}"
        )


@dataclass(frozen=True, slots=True)
class SessionStatusSnapshot:
    """User-facing live status for the current interactive turn."""

    phase: str
    detail: str | None
    estimated_context: ContextStatus | None
    last_usage: UsageSnapshot | None
    cumulative_usage: UsageSnapshot | None


@dataclass(frozen=True, slots=True)
class PendingConversationSettings:
    """Pending remote conversation settings for the next start or restart."""

    name: str = ""
    description: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def active(self) -> bool:
        """Return whether any pending setting is configured."""

        return bool(self.name or self.description or self.metadata)


@dataclass(frozen=True, slots=True)
class _ModelTurn:
    """Intermediate result from one model call."""

    content: str
    finish_reason: str
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False
    error: bool = False


@dataclass(frozen=True, slots=True)
class _BackendState:
    """Snapshot of the non-Conversations backend that can be restored later."""

    client: MistralClientProtocol
    backend_kind: BackendKind
    model_id: str
    server_url: str | None


@dataclass(slots=True)
class _RenderedSegment:
    """Incremental answer or reasoning text rendered from one backend chunk."""

    kind: str
    text: str


@dataclass(slots=True)
class _DeferredAnswerBuffer:
    """Delay likely textual tool-call payloads until the turn completes."""

    enabled: bool
    _buffer: list[str] = field(default_factory=list)
    _streaming_passthrough: bool = False

    def feed(self, text: str) -> str:
        """Buffer likely JSON tool-call text until the turn is complete."""

        if not self.enabled or not text:
            return text
        if self._streaming_passthrough:
            return text

        self._buffer.append(text)
        combined = "".join(self._buffer)
        stripped = combined.lstrip()
        if not stripped:
            return ""
        if stripped.startswith(("```", "{", "[")):
            return ""

        self._streaming_passthrough = True
        return combined

    def finalize(self) -> str:
        """Return any buffered text that was never streamed to the terminal."""

        if not self.enabled or self._streaming_passthrough:
            return ""
        return "".join(self._buffer)


@dataclass(slots=True)
class _ReasoningParser:
    """Split a mixed text stream into visible answer and hidden reasoning."""

    in_reasoning: bool = False
    pending: str = ""
    answer_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)

    def feed(self, text: str) -> list[_RenderedSegment]:
        """Consume one text chunk and return renderable output segments."""

        segments: list[_RenderedSegment] = []
        if not text:
            return segments

        self.pending += text
        while self.pending:
            if self.in_reasoning:
                close_match = _find_first_tag(self.pending, _close_tags())
                if close_match is None:
                    emit, keep = _split_possible_tag_suffix(self.pending, _close_tags())
                    if emit:
                        self.reasoning_parts.append(emit)
                        segments.append(_RenderedSegment(kind="reasoning", text=emit))
                    self.pending = keep
                    break

                close_index, close_tag = close_match
                if close_index > 0:
                    emit = self.pending[:close_index]
                    self.reasoning_parts.append(emit)
                    segments.append(_RenderedSegment(kind="reasoning", text=emit))
                self.pending = self.pending[close_index + len(close_tag) :]
                self.in_reasoning = False
                continue

            open_match = _find_first_tag(self.pending, _open_tags())
            close_match = _find_first_tag(self.pending, _close_tags())
            if close_match is not None and (
                open_match is None or close_match[0] < open_match[0]
            ):
                close_index, close_tag = close_match
                if close_index > 0:
                    emit = self.pending[:close_index]
                    self.answer_parts.append(emit)
                    segments.append(_RenderedSegment(kind="answer", text=emit))
                self.pending = self.pending[close_index + len(close_tag) :]
                continue
            if open_match is None:
                emit, keep = _split_possible_tag_suffix(
                    self.pending,
                    [*_open_tags(), *_close_tags()],
                )
                if emit:
                    self.answer_parts.append(emit)
                    segments.append(_RenderedSegment(kind="answer", text=emit))
                self.pending = keep
                break

            open_index, open_tag = open_match
            if open_index > 0:
                emit = self.pending[:open_index]
                self.answer_parts.append(emit)
                segments.append(_RenderedSegment(kind="answer", text=emit))
            self.pending = self.pending[open_index + len(open_tag) :]
            self.in_reasoning = True

        return segments

    def finish(self) -> list[_RenderedSegment]:
        """Flush any trailing text after the stream ends."""

        if not self.pending:
            return []
        segment = _RenderedSegment(
            kind="reasoning" if self.in_reasoning else "answer",
            text=self.pending,
        )
        if self.in_reasoning:
            self.reasoning_parts.append(self.pending)
        else:
            self.answer_parts.append(self.pending)
        self.pending = ""
        return [segment]

    @property
    def answer(self) -> str:
        """Return the full answer text seen so far."""

        return "".join(self.answer_parts).strip()

    @property
    def reasoning(self) -> str:
        """Return the full reasoning text seen so far."""

        return "".join(self.reasoning_parts).strip()


@dataclass(slots=True)
class _ToolCallState:
    """Accumulator for streamed tool-call deltas."""

    index: int
    call_id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)

    def update(self, tool_call: Any) -> None:
        """Merge a chat-completions streaming delta into the accumulator."""

        call_id = _field(tool_call, "id")
        if call_id and call_id != "null":
            self.call_id = str(call_id)

        function = _field(tool_call, "function")
        if function is None:
            return

        name = _field(function, "name")
        if name:
            self.name = str(name)

        arguments = _field(function, "arguments")
        if arguments is None:
            return
        if isinstance(arguments, str):
            self.arguments_parts.append(arguments)
        else:
            self.arguments_parts.append(json.dumps(arguments, ensure_ascii=False))

    def to_tool_call(self) -> dict[str, Any]:
        """Render the accumulated delta as a stable tool-call payload."""

        arguments = "".join(self.arguments_parts).strip() or "{}"
        call_id = self.call_id or f"tool_call_{self.index}"
        name = self.name or f"tool_{self.index}"
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }


@dataclass(slots=True)
class _ConversationToolCallState:
    """Accumulator for Conversations API function-call events."""

    index: int
    call_id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)

    def update(self, event_data: Any) -> None:
        """Merge one Conversations event into the tool-call accumulator."""

        call_id = _field(event_data, "tool_call_id") or _field(event_data, "id")
        if call_id:
            self.call_id = str(call_id)
        name = _field(event_data, "name")
        if name:
            self.name = str(name)
        arguments = _field(event_data, "arguments")
        if arguments is None:
            return
        if isinstance(arguments, str):
            self.arguments_parts.append(arguments)
        else:
            self.arguments_parts.append(json.dumps(arguments, ensure_ascii=False))

    def to_tool_call(self) -> dict[str, Any]:
        """Render the accumulated event stream as a stable tool-call payload."""

        return {
            "id": self.call_id or f"tool_call_{self.index}",
            "type": "function",
            "function": {
                "name": self.name or f"tool_{self.index}",
                "arguments": "".join(self.arguments_parts).strip() or "{}",
            },
        }


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    normalized: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = _field(tool_call, "function")
        if function is None:
            continue
        arguments = _field(function, "arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append(
            {
                "id": _field(tool_call, "id", f"tool_call_{index}"),
                "type": _field(tool_call, "type", "function"),
                "function": {
                    "name": _field(function, "name", f"tool_{index}"),
                    "arguments": arguments or "{}",
                },
            }
        )
    return normalized


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += 8
        role = message.get("role")
        if isinstance(role, str):
            total += _estimate_text_tokens(role)
        total += _estimate_content_tokens(message.get("content"))
        if message.get("tool_calls"):
            total += _estimate_text_tokens(json.dumps(message["tool_calls"]))
        if message.get("name"):
            total += _estimate_text_tokens(str(message["name"]))
        if message.get("tool_call_id"):
            total += _estimate_text_tokens(str(message["tool_call_id"]))
    return total + 4


def _estimate_tools_tokens(tools: list[dict[str, Any]] | None) -> int:
    if not tools:
        return 0
    return _estimate_text_tokens(json.dumps(tools, ensure_ascii=False)) + 16


def _estimate_content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return _estimate_text_tokens(content)
    if not isinstance(content, list):
        return _estimate_text_tokens(str(content))
    total = 0
    for block in content:
        block_type = _field(block, "type", "block")
        total += _estimate_text_tokens(str(block_type)) + 4
        if block_type == "text":
            total += _estimate_text_tokens(str(_field(block, "text", "")))
        elif block_type in {"image_url", "document_url"}:
            total += _estimate_text_tokens(json.dumps(block, ensure_ascii=False))
        else:
            total += _estimate_text_tokens(json.dumps(block, ensure_ascii=False))
    return total


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _render_messages_for_compaction(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "message"))
        content = _render_content_for_compaction(message.get("content"))
        if message.get("tool_calls"):
            content = (
                f"{content}\nTool calls: "
                f"{json.dumps(message['tool_calls'], ensure_ascii=False)}"
            ).strip()
        if message.get("tool_call_id"):
            content = f"tool_call_id={message['tool_call_id']}\n{content}".strip()
        chunks.append(f"### {index}. {role}\n{content}")
    return "\n\n".join(chunks)


def _render_content_for_compaction(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    rendered: list[str] = []
    for block in content:
        block_type = str(_field(block, "type", "block"))
        if block_type == "text":
            rendered.append(str(_field(block, "text", "")))
        elif block_type in {"image_url", "document_url"}:
            rendered.append(f"[{block_type} attachment omitted from compact summary]")
        else:
            rendered.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(part for part in rendered if part)


def _sum_optional_ints(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _normalize_usage_snapshot(
    raw_usage: Any,
    *,
    max_context_tokens: int | None,
) -> UsageSnapshot | None:
    if raw_usage is None:
        return None
    prompt_tokens = _field(raw_usage, "prompt_tokens")
    completion_tokens = _field(raw_usage, "completion_tokens")
    total_tokens = _field(raw_usage, "total_tokens")
    if not any(
        value is not None for value in (prompt_tokens, completion_tokens, total_tokens)
    ):
        return None
    return UsageSnapshot(
        prompt_tokens=_coerce_optional_int(prompt_tokens),
        completion_tokens=_coerce_optional_int(completion_tokens),
        total_tokens=_coerce_optional_int(total_tokens),
        max_context_tokens=max_context_tokens,
    )


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata_matches(raw_metadata: Any, expected: dict[str, str]) -> bool:
    """Return whether remote metadata contains the expected key/value pairs."""

    if not expected:
        return True
    if not isinstance(raw_metadata, dict):
        return False
    for key, value in expected.items():
        if str(raw_metadata.get(key, "")).strip() != value:
            return False
    return True


def _split_possible_tag_suffix(text: str, tags: list[str]) -> tuple[str, str]:
    keep = 0
    for tag in tags:
        max_prefix = min(len(tag) - 1, len(text))
        for prefix_length in range(max_prefix, 0, -1):
            if text.endswith(tag[:prefix_length]):
                keep = max(keep, prefix_length)
                break
    if keep == 0:
        return text, ""
    return text[:-keep], text[-keep:]


def _find_first_tag(text: str, tags: list[str]) -> tuple[int, str] | None:
    matches = [(text.find(tag), tag) for tag in tags]
    present = [(index, tag) for index, tag in matches if index != -1]
    if not present:
        return None
    return min(present, key=lambda item: item[0])


def _open_tags() -> list[str]:
    return [open_tag for open_tag, _close_tag in REASONING_TAG_PAIRS]


def _close_tags() -> list[str]:
    return [close_tag for _open_tag, close_tag in REASONING_TAG_PAIRS]


@dataclass(frozen=True, slots=True)
class _ParsedReasoningText:
    segments: list[_RenderedSegment]
    answer: str
    reasoning: str


def _parse_reasoning_text(text: str) -> _ParsedReasoningText:
    parser = _ReasoningParser()
    segments = parser.feed(text)
    segments.extend(parser.finish())
    return _ParsedReasoningText(
        segments=segments,
        answer=parser.answer,
        reasoning=parser.reasoning,
    )


def _content_segments_from_value(content: Any) -> list[_RenderedSegment]:
    if isinstance(content, str):
        return _parse_reasoning_text(content).segments
    if not isinstance(content, list):
        return []

    segments: list[_RenderedSegment] = []
    for block in content:
        block_type = _field(block, "type")
        if block_type == "text":
            text = _field(block, "text")
            if isinstance(text, str) and text:
                segments.append(_RenderedSegment(kind="answer", text=text))
        elif block_type == "thinking":
            for item in _field(block, "thinking", []) or []:
                text = _field(item, "text")
                if isinstance(text, str) and text:
                    segments.append(_RenderedSegment(kind="reasoning", text=text))
    return segments


def _conversation_content_segments(content: Any) -> list[_RenderedSegment]:
    if isinstance(content, dict):
        return _content_segments_from_value([content])
    return _content_segments_from_value(content)


def _conversation_tool_call(output: Any, index: int) -> dict[str, Any]:
    arguments = _field(output, "arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": str(_field(output, "tool_call_id", f"tool_call_{index}")),
        "type": "function",
        "function": {
            "name": str(_field(output, "name", f"tool_{index}")),
            "arguments": arguments or "{}",
        },
    }


def _summarize_conversation_entry(entry: Any) -> str:
    content = _field(entry, "content")
    if content is None:
        content = _field(entry, "result")
    if content is None:
        content = _field(entry, "arguments")
    if isinstance(content, str):
        text = content
    elif content is None:
        text = ""
    else:
        text = json.dumps(_jsonable(content), ensure_ascii=False)
    text = " ".join(text.split())
    if len(text) > 180:
        return f"{text[:177]}..."
    return text


def _jsonable(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


def _join_segments(segments: list[_RenderedSegment], *, kind: str) -> str:
    return "".join(segment.text for segment in segments if segment.kind == kind).strip()


def _extract_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    cleaned = _strip_json_code_fence(text)
    if not cleaned:
        return []
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return _normalize_textual_tool_calls(payload)


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _normalize_textual_tool_calls(payload: Any) -> list[dict[str, Any]]:
    raw_calls: Any
    if isinstance(payload, dict) and "tool_calls" in payload:
        raw_calls = payload.get("tool_calls")
    else:
        raw_calls = payload

    if isinstance(raw_calls, dict):
        candidates = [raw_calls]
    elif isinstance(raw_calls, list):
        candidates = raw_calls
    else:
        return []

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        function = candidate.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = candidate.get("name")
            arguments = candidate.get("arguments", {})

        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(arguments, str):
            arguments_text = arguments
        else:
            arguments_text = json.dumps(arguments, ensure_ascii=False)
        normalized.append(
            {
                "id": str(candidate.get("id") or f"tool_call_{index}"),
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "arguments": arguments_text or "{}",
                },
            }
        )
    return normalized
