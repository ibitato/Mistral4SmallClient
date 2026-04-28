"""Interactive session management for the Mistral Small 4 CLI."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, TextIO, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mistralai.client import Mistral

from mistral4cli.conversation_registry import ConversationRecord, ConversationRegistry
from mistral4cli.local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    BackendKind,
    ContextConfig,
    ConversationConfig,
    LocalGenerationConfig,
    get_client_timeout_ms,
    set_client_timeout_ms,
)
from mistral4cli.mcp_bridge import MCPBridgeError, MCPToolResult
from mistral4cli.tooling import ToolBridge
from mistral4cli.ui import render_reasoning_chunk, render_runtime_summary

DEFAULT_SYSTEM_PROMPT = "\n".join(
    [
        (
            "You are the assistant inside the Mistral4Cli Linux terminal "
            "client for Mistral Small 4."
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

logger = logging.getLogger("mistral4cli.session")

REMOTE_CONTEXT_WINDOWS = {
    "mistral-small-latest": 256_000,
    "mistral-small-2603": 256_000,
    "mistral-small-2603+1": 256_000,
    "mistral-small-4-0-26-03": 256_000,
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
    client: Mistral
    backend_kind: BackendKind
    model_id: str
    server_url: str | None


@dataclass(slots=True)
class _RenderedSegment:
    kind: str
    text: str


@dataclass(slots=True)
class _DeferredAnswerBuffer:
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
    in_reasoning: bool = False
    pending: str = ""
    answer_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)

    def feed(self, text: str) -> list[_RenderedSegment]:
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
                    self.pending, [*_open_tags(), *_close_tags()]
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
        return "".join(self.answer_parts).strip()

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts).strip()


@dataclass(slots=True)
class _ToolCallState:
    """Accumulator for streamed tool-call deltas."""

    index: int
    call_id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)

    def update(self, tool_call: Any) -> None:
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
        return {
            "id": self.call_id or f"tool_call_{self.index}",
            "type": "function",
            "function": {
                "name": self.name or f"tool_{self.index}",
                "arguments": "".join(self.arguments_parts).strip() or "{}",
            },
        }


@dataclass(slots=True)
class MistralSession:
    """Stateful conversation helper for the Mistral Small 4 CLI."""

    client: Mistral
    backend_kind: BackendKind = BackendKind.LOCAL
    model_id: str = DEFAULT_MODEL_ID
    server_url: str | None = DEFAULT_SERVER_URL
    generation: LocalGenerationConfig = field(default_factory=LocalGenerationConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_bridge: ToolBridge | None = None
    stdout: TextIO | None = None
    stream_enabled: bool = True
    show_reasoning: bool = True
    show_thinking: bool = True
    conversations: ConversationConfig = field(default_factory=ConversationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    logging_summary: str = "debug=on level=DEBUG rotate=daily retention=2d"
    max_tool_rounds: int = 20
    conversation_registry: ConversationRegistry | None = None
    answer_writer: Callable[[str], None] | None = None
    reasoning_writer: Callable[[str], None] | None = None
    status_callback: Callable[[], None] | None = None
    messages: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _mcp_warning_shown: bool = field(init=False, repr=False, default=False)
    _status_phase: str = field(init=False, repr=False, default="idle")
    _status_detail: str | None = field(init=False, repr=False, default=None)
    _last_usage: UsageSnapshot | None = field(init=False, repr=False, default=None)
    _cumulative_usage: UsageSnapshot | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _turn_usage_accumulator: UsageSnapshot | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _cached_context_status: ContextStatus | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _context_status_dirty: bool = field(init=False, repr=False, default=True)
    _missing_reasoning_notice_shown: bool = field(
        init=False,
        repr=False,
        default=False,
    )
    conversation_id: str | None = field(init=False, repr=False, default=None)
    pending_conversation: PendingConversationSettings = field(
        init=False,
        repr=False,
        default_factory=PendingConversationSettings,
    )
    conversation_resume_source: str = field(init=False, repr=False, default="new")
    _previous_backend_state: _BackendState | None = field(
        init=False,
        repr=False,
        default=None,
    )

    def __post_init__(self) -> None:
        """Normalize the session state after dataclass initialization."""

        if self.stdout is None:
            import sys

            self.stdout = sys.stdout
        self.system_prompt = self.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.context = self.context.normalized()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if self.conversations.enabled and self.backend_kind is not BackendKind.REMOTE:
            self.conversations = ConversationConfig(enabled=False, store=True)
        logger.debug(
            "Session initialized backend=%s model=%s tools=%s conversations=%s",
            self.backend_kind.value,
            self.model_id,
            self.tool_bridge is not None,
            self.conversations.enabled,
        )

    def reset(self) -> None:
        """Reset the conversation to the system prompt."""

        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._mark_context_status_dirty()
        self.conversation_id = None
        self.conversation_resume_source = "new"
        self._missing_reasoning_notice_shown = False
        self._set_status("idle")
        logger.info(
            "Conversation reset backend=%s model=%s conversations=%s",
            self.backend_kind.value,
            self.model_id,
            self.conversations.enabled,
        )

    def set_system_prompt(self, system_prompt: str) -> None:
        """Replace the active system prompt and reset the conversation."""

        self.system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.reset()

    def describe_tool_status(self) -> str:
        """Return a compact tool status summary."""

        if self.tool_bridge is None:
            return "FireCrawl MCP: disabled"
        return self.tool_bridge.runtime_summary()

    def describe_tools(self) -> str:
        """Return a live tool catalog summary."""

        if self.tool_bridge is None:
            return "FireCrawl MCP: disabled"
        return self.tool_bridge.describe_tools()

    def describe_defaults(self) -> str:
        """Render the active runtime defaults as human-readable text."""

        assert self.stdout is not None
        return render_defaults_summary(
            backend_kind=self.backend_kind,
            model_id=self.model_id,
            server_url=self.server_url,
            timeout_ms=self.timeout_ms,
            generation=self._display_generation(),
            stream_enabled=self.stream_enabled,
            reasoning_enabled=self.show_reasoning,
            thinking_visible=self.show_thinking,
            conversations=self.conversations,
            context=self.context,
            conversation_id=self.conversation_id,
            tool_summary=self.describe_tool_status(),
            logging_summary=self.logging_summary,
            stream=self.stdout,
        )

    def switch_backend(
        self,
        *,
        client: Mistral,
        backend_kind: BackendKind,
        model_id: str,
        server_url: str | None,
    ) -> None:
        """Swap the active model backend and reset the conversation."""

        self.client = client
        self.backend_kind = backend_kind
        self.model_id = model_id
        self.server_url = server_url
        self.conversations = ConversationConfig(enabled=False, store=True)
        self.conversation_id = None
        self._previous_backend_state = None
        logger.info(
            "Backend switched backend=%s model=%s server=%s",
            backend_kind.value,
            model_id,
            server_url,
        )
        self.reset()

    def enable_conversations(
        self,
        *,
        client: Mistral,
        model_id: str,
        store: bool,
        server_url: str | None = None,
    ) -> None:
        """Enable Mistral Cloud Conversations mode and reset the active chat."""

        if not self.conversations.enabled:
            self._previous_backend_state = _BackendState(
                client=self.client,
                backend_kind=self.backend_kind,
                model_id=self.model_id,
                server_url=self.server_url,
            )
        self.client = client
        self.backend_kind = BackendKind.REMOTE
        self.model_id = model_id
        self.server_url = server_url
        self.conversations = ConversationConfig(
            enabled=True,
            store=store,
            resume_policy=self.conversations.resume_policy,
        )
        logger.info("Conversations enabled model=%s store=%s", model_id, store)
        self.reset()

    def disable_conversations(self) -> None:
        """Disable Conversations mode and restore the previous backend if known."""

        previous = self._previous_backend_state
        self.conversations = ConversationConfig(
            enabled=False,
            store=True,
            resume_policy=self.conversations.resume_policy,
        )
        self.conversation_id = None
        self._previous_backend_state = None
        self._missing_reasoning_notice_shown = False
        if previous is not None:
            self.client = previous.client
            self.backend_kind = previous.backend_kind
            self.model_id = previous.model_id
            self.server_url = previous.server_url
        logger.info("Conversations disabled backend=%s", self.backend_kind.value)
        self.reset()

    def set_conversation_store(self, store: bool) -> None:
        """Update Conversations storage policy and start a fresh conversation."""

        self.conversations = ConversationConfig(
            enabled=self.conversations.enabled,
            store=store,
            resume_policy=self.conversations.resume_policy,
        )
        self.reset()

    def conversations_status_text(self) -> str:
        """Return a user-facing Conversations status summary."""

        state = "on" if self.conversations.enabled else "off"
        store = "on" if self.conversations.store else "off"
        resume = self.conversations.resume_policy
        conversation_id = self.conversation_id or "not started"
        source = self.conversation_resume_source
        return (
            "Conversations: "
            f"{state} store={store} resume={resume} "
            f"source={source} conversation_id={conversation_id}"
        )

    def current_conversation_text(self) -> str:
        """Render the current active conversation and pending local state."""

        lines = [self.conversations_status_text()]
        if self.conversation_registry is not None and self.conversation_id:
            record = self.conversation_registry.get(self.conversation_id)
            if record is not None:
                lines.append(self._format_registry_record(record, include_note=True))
        pending = self.pending_conversation_text()
        if pending:
            lines.append(pending)
        return "\n".join(lines)

    def pending_conversation_text(self) -> str:
        """Return a compact summary of pending conversation settings."""

        if not self.pending_conversation.active():
            return ""
        parts: list[str] = []
        if self.pending_conversation.name:
            parts.append(f"name={self.pending_conversation.name}")
        if self.pending_conversation.description:
            parts.append(f"description={self.pending_conversation.description}")
        if self.pending_conversation.metadata:
            metadata = ", ".join(
                f"{key}={value}"
                for key, value in sorted(self.pending_conversation.metadata.items())
            )
            parts.append(f"metadata={metadata}")
        return "Pending conversation settings: " + " | ".join(parts)

    def set_pending_conversation_name(self, value: str) -> None:
        """Set the pending remote conversation name."""

        self.pending_conversation = replace(
            self.pending_conversation,
            name=value.strip(),
        )

    def set_pending_conversation_description(self, value: str) -> None:
        """Set the pending remote conversation description."""

        self.pending_conversation = replace(
            self.pending_conversation,
            description=value.strip(),
        )

    def set_pending_conversation_metadata(self, key: str, value: str) -> None:
        """Set one pending remote conversation metadata pair."""

        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("Metadata key cannot be empty.")
        metadata = dict(self.pending_conversation.metadata)
        metadata[normalized_key] = value.strip()
        self.pending_conversation = replace(
            self.pending_conversation,
            metadata=metadata,
        )

    def clear_pending_conversation_name(self) -> None:
        """Clear the pending remote conversation name."""

        self.pending_conversation = replace(self.pending_conversation, name="")

    def clear_pending_conversation_description(self) -> None:
        """Clear the pending remote conversation description."""

        self.pending_conversation = replace(
            self.pending_conversation,
            description="",
        )

    def clear_pending_conversation_metadata(self, key: str | None = None) -> None:
        """Clear one or all pending remote conversation metadata fields."""

        if key is None:
            metadata: dict[str, str] = {}
        else:
            metadata = dict(self.pending_conversation.metadata)
            metadata.pop(key.strip(), None)
        self.pending_conversation = replace(
            self.pending_conversation,
            metadata=metadata,
        )

    def apply_conversation_resume_policy(self, resume_policy: str) -> None:
        """Update the configured startup resume policy."""

        normalized = resume_policy.strip().lower()
        if normalized not in {"last", "new", "prompt"}:
            raise ValueError("Resume policy must be one of: last, new, prompt.")
        self.conversations = ConversationConfig(
            enabled=self.conversations.enabled,
            store=self.conversations.store,
            resume_policy=normalized,
        )

    def attach_remote_conversation(
        self,
        conversation_id: str,
        *,
        source: str = "manual",
    ) -> str:
        """Attach the current session to an existing remote conversation id."""

        normalized_id = conversation_id.strip()
        if not normalized_id:
            raise ValueError("Conversation id cannot be empty.")
        payload = self.client.beta.conversations.get(conversation_id=normalized_id)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._mark_context_status_dirty()
        self._sync_conversation_id(
            normalized_id,
            source=source,
            payload=payload,
        )
        return f"Attached to conversation {normalized_id}."

    def resolve_conversation_reference(self, reference: str | None) -> str:
        """Resolve a user reference to a concrete remote conversation id."""

        if reference is None or not reference.strip():
            if self.conversation_id:
                return self.conversation_id
            raise ValueError("No active conversation id yet.")
        normalized = reference.strip()
        if self.conversation_registry is None:
            return normalized
        record = self.conversation_registry.resolve_reference(normalized)
        if record is not None:
            return record.conversation_id
        return normalized

    def list_remote_conversations(
        self,
        *,
        page: int = 0,
        page_size: int = 20,
        metadata: dict[str, str] | None = None,
        bookmarks_only: bool = False,
    ) -> str:
        """List remote conversations with local overlay metadata."""

        if bookmarks_only:
            if self.conversation_registry is None:
                return "No local conversation registry is configured."
            records = self.conversation_registry.bookmarks()
            if not records:
                return "No bookmarked conversations."
            lines = ["Bookmarked conversations:"]
            for index, record in enumerate(records, start=1):
                lines.append(
                    f"{index}. "
                    f"{self._format_registry_record(record, include_note=True)}"
                )
            return "\n".join(lines)

        list_kwargs: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "metadata": {},
        }
        payload = self.client.beta.conversations.list(**list_kwargs)
        conversations = list(payload or [])
        if metadata:
            filtered: list[Any] = []
            for conversation in conversations:
                conversation_id = str(_field(conversation, "id", "") or "")
                if not conversation_id:
                    continue
                detailed = self.client.beta.conversations.get(
                    conversation_id=conversation_id
                )
                self._update_registry_from_remote_payload(conversation_id, detailed)
                effective_metadata = _field(detailed, "metadata", None)
                if (
                    effective_metadata is None
                    and self.conversation_registry is not None
                ):
                    local_record = self.conversation_registry.get(conversation_id)
                    if local_record is not None and local_record.remote_metadata:
                        effective_metadata = local_record.remote_metadata
                if _metadata_matches(effective_metadata, metadata):
                    filtered.append(detailed)
            conversations = filtered
        if not conversations:
            return "No remote conversations found."
        lines = [
            f"Remote conversations page={page} size={page_size}:",
        ]
        for index, conversation in enumerate(conversations, start=1):
            conversation_id = str(_field(conversation, "id", "") or "")
            self._update_registry_from_remote_payload(conversation_id, conversation)
            label = self._format_remote_conversation_summary(conversation)
            if self.conversation_id == conversation_id:
                label = f"{label} [active]"
            lines.append(f"{index}. {label}")
        return "\n".join(lines)

    def show_remote_conversation(self, reference: str | None = None) -> str:
        """Render detailed information for one remote conversation."""

        conversation_id = self.resolve_conversation_reference(reference)
        payload = self.client.beta.conversations.get(conversation_id=conversation_id)
        self._update_registry_from_remote_payload(conversation_id, payload)
        lines = [f"Conversation {conversation_id}:"]
        lines.extend(self._render_remote_conversation_details(payload))
        if self.conversation_registry is not None:
            record = self.conversation_registry.get(conversation_id)
            if record is not None:
                lines.append("Local registry:")
                lines.append(self._format_registry_record(record, include_note=True))
        return "\n".join(lines)

    def conversation_history_text(
        self,
        *,
        messages_only: bool = False,
        conversation_id: str | None = None,
    ) -> str:
        """Fetch and render the active remote Conversation history."""

        if not self.conversations.enabled:
            return "Conversations mode is off."
        target_id = self.resolve_conversation_reference(conversation_id)
        conversations = self.client.beta.conversations
        payload = (
            conversations.get_messages(conversation_id=target_id)
            if messages_only
            else conversations.get_history(conversation_id=target_id)
        )
        entries = _field(payload, "messages" if messages_only else "entries", []) or []
        if not entries:
            return "No conversation entries."
        lines = [f"Conversation {target_id}:"]
        for index, entry in enumerate(entries, start=1):
            entry_type = str(_field(entry, "type", "entry"))
            role = _field(entry, "role")
            entry_id = str(_field(entry, "id", "") or "")
            label = f"{index}. {entry_type}"
            if role:
                label = f"{label} {role}"
            if entry_id:
                label = f"{label} id={entry_id}"
            text = _summarize_conversation_entry(entry)
            lines.append(f"{label}: {text}" if text else label)
        if not messages_only:
            lines.append(
                'Hint: use "/conv restart <entry_id>" to branch from one history entry.'
            )
        return "\n".join(lines)

    def delete_remote_conversation(self, conversation_id: str | None = None) -> str:
        """Delete a remote Conversation and clear local state when active."""

        if not self.conversations.enabled:
            return "Conversations mode is off."
        target_id = self.resolve_conversation_reference(conversation_id)
        self.client.beta.conversations.delete(conversation_id=target_id)
        if self.conversation_registry is not None:
            self.conversation_registry.mark_deleted(target_id)
        if self.conversation_id == target_id:
            self.conversation_id = None
            self.conversation_resume_source = "new"
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self._mark_context_status_dirty()
        return f"Deleted conversation {target_id}."

    def restart_remote_conversation(
        self,
        *,
        from_entry_id: str,
        conversation_id: str | None = None,
    ) -> str:
        """Restart a remote conversation from one entry and switch to the new id."""

        target_id = self.resolve_conversation_reference(conversation_id)
        if not from_entry_id.strip():
            raise ValueError("Entry id cannot be empty.")
        parent_id = target_id
        response = self.client.beta.conversations.restart(
            conversation_id=target_id,
            from_entry_id=from_entry_id.strip(),
            inputs="",
            store=self.conversations.store,
            completion_args=cast(Any, self._conversation_completion_args()),
            metadata=cast(
                Any,
                self.pending_conversation.metadata
                if self.pending_conversation.metadata
                else None,
            ),
        )
        new_id = str(_field(response, "conversation_id", "") or "")
        if not new_id:
            raise RuntimeError("Restart returned no conversation id.")
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._mark_context_status_dirty()
        self._sync_conversation_id(
            new_id,
            source="restart",
            parent_conversation_id=parent_id,
        )
        outputs = _field(response, "outputs", []) or []
        summary = ""
        for output in outputs:
            if _field(output, "type") == "message.output":
                segments = _conversation_content_segments(_field(output, "content"))
                answer = _join_segments(segments, kind="answer")
                if answer:
                    summary = answer
                    break
        if summary:
            return (
                f"Restarted from {from_entry_id.strip()}. "
                f"Active conversation: {new_id}.\nAssistant: {summary}"
            )
        return f"Restarted from {from_entry_id.strip()}. Active conversation: {new_id}."

    def forget_local_conversation(self, reference: str) -> str:
        """Forget one local registry record without touching the remote object."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        resolved = self.conversation_registry.resolve_reference(reference.strip())
        conversation_id = (
            resolved.conversation_id if resolved is not None else reference.strip()
        )
        if not conversation_id:
            raise ValueError("Conversation id cannot be empty.")
        if not self.conversation_registry.forget(conversation_id):
            return f"No local record for conversation {conversation_id}."
        return f"Forgot local record for conversation {conversation_id}."

    def set_local_conversation_alias(self, reference: str, alias: str) -> str:
        """Set a local alias for one conversation."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        conversation_id = self.resolve_conversation_reference(reference)
        record = self.conversation_registry.set_alias(conversation_id, alias)
        return f"Alias for {record.conversation_id} set to {record.alias or '(empty)'}."

    def set_local_conversation_note(self, reference: str, note: str) -> str:
        """Set a local note for one conversation."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        conversation_id = self.resolve_conversation_reference(reference)
        self.conversation_registry.set_note(conversation_id, note)
        return f"Note updated for {conversation_id}."

    def add_local_conversation_tag(self, reference: str, tag: str) -> str:
        """Add one local tag to a conversation."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        conversation_id = self.resolve_conversation_reference(reference)
        record = self.conversation_registry.add_tag(conversation_id, tag)
        return f"Tags for {conversation_id}: {', '.join(record.tags) or '(none)'}."

    def remove_local_conversation_tag(self, reference: str, tag: str) -> str:
        """Remove one local tag from a conversation."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        conversation_id = self.resolve_conversation_reference(reference)
        record = self.conversation_registry.remove_tag(conversation_id, tag)
        return f"Tags for {conversation_id}: {', '.join(record.tags) or '(none)'}."

    def context_status(self) -> ContextStatus:
        """Return the current estimated context state for chat completions."""

        return self._compute_context_status()

    def _compute_context_status(self) -> ContextStatus:
        """Compute a fresh client-side estimate for the current chat history."""

        window_tokens = self._model_context_window() or self.context.local_window_tokens
        threshold_tokens = int(window_tokens * self.context.threshold)
        reserve_tokens = self._effective_context_reserve()
        estimated_tokens = self.estimate_context_tokens(self.messages) + reserve_tokens
        return ContextStatus(
            estimated_tokens=estimated_tokens,
            window_tokens=window_tokens,
            threshold_tokens=threshold_tokens,
            reserve_tokens=reserve_tokens,
            auto_compact=self.context.auto_compact,
        )

    def context_status_text(self) -> str:
        """Return a user-facing context policy summary."""

        status = self.context_status()
        auto = "on" if status.auto_compact else "off"
        threshold_percent = round(self.context.threshold * 100)
        mode = "backend-managed" if self.conversations.enabled else "client-managed"
        return (
            f"Context: {mode} estimate={status.estimated_tokens}/"
            f"{status.window_tokens} threshold={threshold_percent}% "
            f"reserve={status.reserve_tokens} "
            f"keep_turns={self.context.keep_recent_turns} "
            f"auto={auto}"
        )

    def configure_context(
        self,
        *,
        auto_compact: bool | None = None,
        threshold: float | None = None,
        reserve_tokens: int | None = None,
        keep_recent_turns: int | None = None,
    ) -> None:
        """Update mutable context policy knobs at runtime."""

        self.context = replace(
            self.context,
            auto_compact=(
                self.context.auto_compact if auto_compact is None else auto_compact
            ),
            threshold=self.context.threshold if threshold is None else threshold,
            reserve_tokens=(
                self.context.reserve_tokens
                if reserve_tokens is None
                else reserve_tokens
            ),
            keep_recent_turns=(
                self.context.keep_recent_turns
                if keep_recent_turns is None
                else keep_recent_turns
            ),
        ).normalized()
        self._mark_context_status_dirty()

    def estimate_context_tokens(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Estimate request tokens when backend tokenizers are unavailable."""

        active_messages = self.messages if messages is None else messages
        return _estimate_messages_tokens(active_messages) + _estimate_tools_tokens(
            tools
        )

    def compact_context(self) -> CompactResult:
        """Summarize old chat-completions history and keep recent turns."""

        before_tokens = self.estimate_context_tokens(self.messages)
        window_tokens = self._model_context_window() or self.context.local_window_tokens
        threshold_tokens = int(window_tokens * self.context.threshold)
        if self.conversations.enabled:
            return CompactResult(
                changed=False,
                reason="Conversations mode stores context on the Mistral backend.",
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                window_tokens=window_tokens,
                threshold_tokens=threshold_tokens,
            )

        older_messages, recent_messages = self._split_compactable_history()
        if not older_messages:
            return CompactResult(
                changed=False,
                reason="Nothing old enough to compact.",
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                window_tokens=window_tokens,
                threshold_tokens=threshold_tokens,
            )

        summary = self._summarize_messages_for_compaction(older_messages)
        if not summary.strip():
            return CompactResult(
                changed=False,
                reason="The model did not return a compaction summary.",
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                window_tokens=window_tokens,
                threshold_tokens=threshold_tokens,
            )

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "assistant",
                "content": (
                    "[Compacted previous context]\n"
                    f"{summary.strip()}\n"
                    "[End compacted context]"
                ),
            },
            *recent_messages,
        ]
        self._mark_context_status_dirty()
        after_tokens = self.estimate_context_tokens(self.messages)
        logger.info(
            "Context compacted before_tokens=%s after_tokens=%s window=%s",
            before_tokens,
            after_tokens,
            window_tokens,
        )
        return CompactResult(
            changed=True,
            reason="Old turns were summarized and recent turns were preserved.",
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            window_tokens=window_tokens,
            threshold_tokens=threshold_tokens,
        )

    @property
    def timeout_ms(self) -> int:
        """Return the active request timeout in milliseconds."""

        return get_client_timeout_ms(self.client)

    def set_timeout_ms(self, timeout_ms: int) -> None:
        """Update the active request timeout in milliseconds."""

        set_client_timeout_ms(self.client, timeout_ms)

    def visible_reasoning_supported(self) -> bool:
        """Return whether the active backend can render visible reasoning."""

        return True

    def reasoning_status_text(self) -> str:
        """Return a user-facing reasoning-request status string."""

        state = "on" if self.show_reasoning else "off"
        if self.backend_kind is BackendKind.REMOTE:
            if self.conversations.enabled:
                return (
                    "Reasoning request: "
                    f"{state} (remote Conversations, requested best-effort)"
                )
            return f"Reasoning request: {state} (remote SDK)"
        return f"Reasoning request: {state} (local backend)"

    def thinking_status_text(self) -> str:
        """Return a user-facing thinking-render status string."""

        state = "on" if self.show_thinking else "off"
        return f"Thinking display: {state}"

    def set_reasoning_visibility(self, visible: bool) -> None:
        """Enable or disable reasoning requests to the backend."""

        self.show_reasoning = visible
        self._missing_reasoning_notice_shown = False

    def toggle_reasoning_visibility(self) -> bool:
        """Toggle backend reasoning requests and return the new state."""

        self.show_reasoning = not self.show_reasoning
        self._missing_reasoning_notice_shown = False
        return self.show_reasoning

    def set_thinking_visibility(self, visible: bool) -> None:
        """Enable or disable local rendering of thinking blocks."""

        self.show_thinking = visible
        self._missing_reasoning_notice_shown = False

    def toggle_thinking_visibility(self) -> bool:
        """Toggle local thinking rendering and return the new state."""

        self.show_thinking = not self.show_thinking
        self._missing_reasoning_notice_shown = False
        return self.show_thinking

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute a tool through the active bridge."""

        return self._call_tool_bridge(public_name, arguments)

    def status_snapshot(self) -> SessionStatusSnapshot:
        """Return the current live status for interactive UI rendering."""

        max_context_tokens = self._model_context_window()
        estimated_context = self._status_context_snapshot()
        last_usage = self._with_context_window(self._last_usage, max_context_tokens)
        combined_cumulative = self._cumulative_usage
        if self._turn_usage_accumulator is not None:
            combined_cumulative = self._turn_usage_accumulator.merge(
                combined_cumulative
            )
        cumulative_usage = self._with_context_window(
            combined_cumulative, max_context_tokens
        )
        return SessionStatusSnapshot(
            phase=self._status_phase,
            detail=self._status_detail,
            estimated_context=estimated_context,
            last_usage=last_usage,
            cumulative_usage=cumulative_usage,
        )

    def send_content(
        self,
        content: str | list[dict[str, Any]],
        *,
        stream: bool = True,
        disable_tools: bool = False,
    ) -> TurnResult:
        """Send a text or multimodal user turn and update the conversation."""

        normalized = self._normalize_user_content(content)
        if normalized is None:
            return TurnResult(content="", finish_reason="empty", cancelled=False)

        logger.debug(
            "Sending turn stream=%s disable_tools=%s attachments=%s content=%s",
            stream,
            disable_tools,
            self._has_attachment_blocks(normalized),
            self._content_summary(normalized),
        )
        if self.conversations.enabled:
            return self._send_conversation_content(
                normalized,
                stream=stream,
                disable_tools=disable_tools,
            )

        if not self._prepare_context_for_turn(normalized, disable_tools=disable_tools):
            return TurnResult(content="", finish_reason="context_overflow")

        message_start = len(self.messages)
        self.messages.append({"role": "user", "content": normalized})
        self._mark_context_status_dirty()
        self._last_usage = None
        self._turn_usage_accumulator = None
        self._set_status("thinking")
        try:
            seen_tool_calls: set[str] = set()
            tool_rounds_executed = 0
            tools = [] if disable_tools else self._resolve_tools()
            if not tools:
                turn = self._send_single_turn(stream=stream, tools=None)
                if turn.cancelled or turn.error:
                    self._rollback_to(message_start)
                    self._turn_usage_accumulator = None
                    if turn.cancelled:
                        self._set_status("interrupted")
                    else:
                        self._set_status("error")
                if not turn.cancelled and not turn.error:
                    self._commit_assistant_message(turn)
                    self._commit_turn_usage()
                    self._set_status("done")
                return TurnResult(
                    content=turn.content,
                    finish_reason=turn.finish_reason,
                    reasoning=turn.reasoning,
                    cancelled=turn.cancelled,
                )

            for _ in range(self.max_tool_rounds):
                turn = self._send_single_turn(stream=stream, tools=tools)
                if turn.cancelled or turn.error:
                    self._rollback_to(message_start)
                    self._turn_usage_accumulator = None
                    if turn.cancelled:
                        self._set_status("interrupted")
                    else:
                        self._set_status("error")
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=turn.cancelled,
                    )

                if turn.finish_reason != "tool_calls" or not turn.tool_calls:
                    self._commit_assistant_message(turn)
                    self._commit_turn_usage()
                    self._set_status("done")
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=False,
                    )

                self._commit_assistant_message(turn)
                for call in turn.tool_calls:
                    function = call["function"]
                    name = str(function["name"])
                    try:
                        arguments = self._parse_tool_arguments(
                            function.get("arguments")
                        )
                    except json.JSONDecodeError as exc:
                        result = MCPToolResult(
                            text=(
                                f"[tool-error] invalid JSON arguments for {name}: {exc}"
                            ),
                            is_error=True,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        result = MCPToolResult(
                            text=f"[tool-error] {exc}", is_error=True
                        )
                    else:
                        signature = self._tool_call_signature(name, arguments)
                        if signature in seen_tool_calls:
                            logger.warning(
                                (
                                    "Blocked repeated identical tool call "
                                    "name=%s arguments=%s"
                                ),
                                name,
                                self._summarize_tool_arguments(arguments),
                            )
                            self.messages.append(
                                self._tool_message(
                                    call=call,
                                    result=MCPToolResult(
                                        text=(
                                            "[tool-error] repeated identical tool call "
                                            "blocked; use the prior tool result"
                                        ),
                                        is_error=True,
                                        structured_content={
                                            "status": "error",
                                            "tool": name,
                                            "code": "repeated_identical_tool_call",
                                            "arguments": arguments,
                                        },
                                    ),
                                )
                            )
                            self._mark_context_status_dirty()
                            self._print(
                                "[error] repeated identical tool call blocked\n"
                            )
                            self._turn_usage_accumulator = None
                            self._set_status("error")
                            return TurnResult(
                                content="",
                                finish_reason="error",
                                cancelled=False,
                            )
                        seen_tool_calls.add(signature)
                        logger.debug(
                            "Executing tool name=%s arguments=%s",
                            name,
                            self._summarize_tool_arguments(arguments),
                        )
                        self._set_status("tool", detail=name)
                        result = self._call_tool_bridge(name, arguments)
                        self._set_status("thinking")
                        logger.debug(
                            "Tool result name=%s error=%s structured=%s",
                            name,
                            result.is_error,
                            result.structured_content is not None,
                        )
                    self.messages.append(self._tool_message(call=call, result=result))
                    self._mark_context_status_dirty()
                tool_rounds_executed += 1

            if tool_rounds_executed > 0:
                logger.warning(
                    "Tool loop limit reached; forcing final answer max_rounds=%s",
                    self.max_tool_rounds,
                )
                final_turn = self._send_single_turn(stream=stream, tools=None)
                if final_turn.cancelled or final_turn.error:
                    self._rollback_to(message_start)
                    self._turn_usage_accumulator = None
                    if final_turn.cancelled:
                        self._set_status("interrupted")
                    else:
                        self._set_status("error")
                    return TurnResult(
                        content=final_turn.content,
                        finish_reason=final_turn.finish_reason,
                        reasoning=final_turn.reasoning,
                        cancelled=final_turn.cancelled,
                    )
                self._commit_assistant_message(final_turn)
                self._commit_turn_usage()
                self._set_status("done")
                return TurnResult(
                    content=final_turn.content,
                    finish_reason=final_turn.finish_reason,
                    reasoning=final_turn.reasoning,
                    cancelled=False,
                )

            self._rollback_to(message_start)
            self._turn_usage_accumulator = None
            self._print("[error] tool loop limit reached\n")
            self._set_status("error")
            logger.error("Tool loop limit reached max_rounds=%s", self.max_tool_rounds)
            return TurnResult(content="", finish_reason="error", cancelled=False)
        except KeyboardInterrupt:
            self._rollback_to(message_start)
            self._turn_usage_accumulator = None
            self._print("\n[interrupted]\n")
            self._set_status("interrupted")
            logger.info("Turn interrupted by user")
            return TurnResult(content="", finish_reason="cancelled", cancelled=True)

    def send(self, user_text: str, *, stream: bool = True) -> TurnResult:
        """Send one text user turn and update the conversation history."""

        return self.send_content(user_text, stream=stream)

    def _send_conversation_content(
        self,
        content: str | list[dict[str, Any]],
        *,
        stream: bool,
        disable_tools: bool,
    ) -> TurnResult:
        message_start = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        self._mark_context_status_dirty()
        self._last_usage = None
        self._turn_usage_accumulator = None
        self._set_status("thinking")
        pending_tool_calls: list[dict[str, Any]] = []
        pending_tool_inputs: list[dict[str, Any]] = []
        try:
            seen_tool_calls: set[str] = set()
            inputs = self._conversation_user_inputs(content)
            tools = [] if disable_tools else self._resolve_tools()
            for _ in range(self.max_tool_rounds + 1):
                turn = self._send_conversation_turn(
                    inputs=inputs,
                    stream=stream,
                    tools=tools,
                )
                if turn.cancelled or turn.error:
                    if turn.cancelled and turn.tool_calls:
                        self._complete_pending_conversation_tool_calls(turn.tool_calls)
                    self._rollback_to(message_start)
                    self._turn_usage_accumulator = None
                    self._set_status("interrupted" if turn.cancelled else "error")
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=turn.cancelled,
                    )
                if turn.finish_reason != "tool_calls" or not turn.tool_calls:
                    self._commit_assistant_message(turn)
                    self._commit_turn_usage()
                    self._set_status("done")
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=False,
                    )

                pending_tool_calls = list(turn.tool_calls)
                self._commit_assistant_message(turn)
                tool_inputs = pending_tool_inputs = []
                for call in turn.tool_calls:
                    function = call["function"]
                    name = str(function["name"])
                    try:
                        arguments = self._parse_tool_arguments(
                            function.get("arguments")
                        )
                    except Exception as exc:
                        result = MCPToolResult(
                            text=f"[tool-error] invalid arguments for {name}: {exc}",
                            is_error=True,
                        )
                    else:
                        signature = self._tool_call_signature(name, arguments)
                        if signature in seen_tool_calls:
                            result = MCPToolResult(
                                text=(
                                    "[tool-error] repeated identical tool call "
                                    "blocked; use the prior tool result"
                                ),
                                is_error=True,
                                structured_content={
                                    "status": "error",
                                    "tool": name,
                                    "code": "repeated_identical_tool_call",
                                    "arguments": arguments,
                                },
                            )
                            self.messages.append(
                                self._tool_message(call=call, result=result)
                            )
                            self._mark_context_status_dirty()
                            self._print(
                                "[error] repeated identical tool call blocked\n"
                            )
                            self._turn_usage_accumulator = None
                            self._set_status("error")
                            return TurnResult(
                                content="",
                                finish_reason="error",
                                cancelled=False,
                            )
                        seen_tool_calls.add(signature)
                        self._set_status("tool", detail=name)
                        result = self._call_tool_bridge(name, arguments)
                        self._set_status("thinking")
                    self.messages.append(self._tool_message(call=call, result=result))
                    self._mark_context_status_dirty()
                    tool_inputs.append(
                        {
                            "type": "function.result",
                            "tool_call_id": call["id"],
                            "result": self._render_tool_result(result),
                        }
                    )
                inputs = tool_inputs
                pending_tool_calls = []
                pending_tool_inputs = []

            self._rollback_to(message_start)
            self._turn_usage_accumulator = None
            self._print("[error] tool loop limit reached\n")
            self._set_status("error")
            return TurnResult(content="", finish_reason="error", cancelled=False)
        except KeyboardInterrupt:
            self._complete_pending_conversation_tool_calls(
                pending_tool_calls,
                completed_inputs=pending_tool_inputs,
            )
            self._rollback_to(message_start)
            self._turn_usage_accumulator = None
            self._print("\n[interrupted]\n")
            self._set_status("interrupted")
            return TurnResult(content="", finish_reason="cancelled", cancelled=True)

    def _send_conversation_turn(
        self,
        *,
        inputs: str | list[dict[str, Any]] | None,
        stream: bool,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        if stream:
            return self._send_conversation_streaming(inputs=inputs, tools=tools)
        return self._send_conversation_non_streaming(inputs=inputs, tools=tools)

    def _conversation_user_inputs(
        self, content: str | list[dict[str, Any]]
    ) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        return [{"type": "message.input", "role": "user", "content": content}]

    def _conversation_completion_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "response_format": {"type": "text"},
            "reasoning_effort": "high" if self.show_reasoning else "none",
        }
        if self.generation.max_tokens is not None:
            args["max_tokens"] = self.generation.max_tokens
        return args

    def _conversation_start_kwargs(
        self,
        *,
        inputs: str | list[dict[str, Any]],
        stream: bool,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "inputs": inputs,
            "model": self.model_id,
            "instructions": self.system_prompt,
            "store": self.conversations.store,
            "completion_args": self._conversation_completion_args(),
        }
        if self.pending_conversation.name:
            kwargs["name"] = self.pending_conversation.name
        if self.pending_conversation.description:
            kwargs["description"] = self.pending_conversation.description
        if self.pending_conversation.metadata:
            kwargs["metadata"] = self.pending_conversation.metadata
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _conversation_append_kwargs(
        self,
        *,
        inputs: str | list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if self.conversation_id is None:
            raise RuntimeError("Conversation append requires a conversation id.")
        return {
            "conversation_id": self.conversation_id,
            "inputs": inputs,
            "store": self.conversations.store,
            "completion_args": self._conversation_completion_args(),
        }

    def _send_conversation_non_streaming(
        self,
        *,
        inputs: str | list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        try:
            conversations = self.client.beta.conversations
            if self.conversation_id is None:
                if inputs is None:
                    raise RuntimeError("Conversation start requires inputs.")
                response = conversations.start(
                    **self._conversation_start_kwargs(
                        inputs=inputs,
                        stream=False,
                        tools=tools,
                    )
                )
            else:
                response = conversations.append(
                    **self._conversation_append_kwargs(inputs=inputs)
                )
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(content="", finish_reason="cancelled", cancelled=True)
        except Exception as exc:
            self._print(f"[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        return self._handle_conversation_response(response)

    def _send_conversation_streaming(
        self,
        *,
        inputs: str | list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        parser = _ReasoningParser()
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_states: dict[str, _ConversationToolCallState] = {}
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        usage_snapshot: Any = None
        error_message: str | None = None

        try:
            conversations = self.client.beta.conversations
            if self.conversation_id is None:
                if inputs is None:
                    raise RuntimeError("Conversation start requires inputs.")
                stream = conversations.start_stream(
                    **self._conversation_start_kwargs(
                        inputs=inputs,
                        stream=True,
                        tools=tools,
                    )
                )
            else:
                stream = conversations.append_stream(
                    **self._conversation_append_kwargs(inputs=inputs)
                )
            with stream as active_stream:
                for event in active_stream:
                    event_name = str(_field(event, "event", ""))
                    data = _field(event, "data")
                    if event_name == "conversation.response.started":
                        conversation_id = _field(data, "conversation_id")
                        if conversation_id and self.conversations.store:
                            self._sync_conversation_id(
                                str(conversation_id),
                                source="stream",
                            )
                        continue
                    if event_name == "conversation.response.done":
                        usage_snapshot = _field(data, "usage", usage_snapshot)
                        continue
                    if event_name == "conversation.response.error":
                        error_message = str(
                            _field(data, "message", "conversation error")
                        )
                        continue
                    if event_name == "function.call.delta":
                        call_id = str(_field(data, "tool_call_id", "") or "")
                        if not call_id:
                            call_id = str(_field(data, "id", len(tool_states)) or "")
                        state = tool_states.setdefault(
                            call_id,
                            _ConversationToolCallState(index=len(tool_states)),
                        )
                        state.update(data)
                        continue
                    if event_name != "message.output.delta":
                        continue
                    for segment in _conversation_content_segments(
                        _field(data, "content")
                    ):
                        if segment.kind == "reasoning":
                            self._set_status("answering")
                            self._print_reasoning(segment.text)
                            reasoning_parts.append(segment.text)
                            reasoning_printed = True
                        else:
                            for parsed in parser.feed(segment.text):
                                if parsed.kind == "reasoning":
                                    self._set_status("answering")
                                    self._print_reasoning(parsed.text)
                                    reasoning_parts.append(parsed.text)
                                    reasoning_printed = True
                                else:
                                    self._set_status("answering")
                                    answer_parts.append(parsed.text)
                                    answer_started = self._print_answer_separator(
                                        reasoning_printed=reasoning_printed,
                                        answer_started=answer_started,
                                    )
                                    self._print(parsed.text)
                        printed_anything = True
        except KeyboardInterrupt:
            tool_calls = [
                state.to_tool_call() for _, state in sorted(tool_states.items())
            ]
            self._print("\n[interrupted]\n")
            return _ModelTurn(
                content="".join(answer_parts).strip() or parser.answer,
                reasoning="".join(reasoning_parts).strip() or parser.reasoning,
                finish_reason="cancelled",
                tool_calls=tool_calls,
                cancelled=True,
            )
        except Exception as exc:
            self._print(f"\n[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        if error_message:
            self._print(f"\n[error] {error_message}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(usage_snapshot)
        for segment in parser.finish():
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_parts.append(segment.text)
                reasoning_printed = True
            else:
                self._set_status("answering")
                answer_parts.append(segment.text)
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
                printed_anything = True

        content = "".join(answer_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
        self._finalize_remote_reasoning(
            reasoning=reasoning,
            finish_reason="tool_calls" if tool_calls else "stop",
            has_answer_text=bool(content),
        )
        if printed_anything and content and not content.endswith("\n"):
            self._print("\n")
        return _ModelTurn(
            content=content,
            finish_reason="tool_calls" if tool_calls else "stop",
            reasoning=reasoning,
            tool_calls=tool_calls,
        )

    def _handle_conversation_response(self, response: Any) -> _ModelTurn:
        conversation_id = _field(response, "conversation_id")
        if conversation_id and self.conversations.store:
            self._sync_conversation_id(
                str(conversation_id),
                source="response",
                payload=response,
                fallback_metadata=(
                    self.pending_conversation.metadata
                    if self.pending_conversation.metadata
                    else None
                ),
            )
        self._record_usage(_field(response, "usage"))
        outputs = _field(response, "outputs", []) or []
        segments: list[_RenderedSegment] = []
        tool_calls: list[dict[str, Any]] = []
        for output in outputs:
            output_type = _field(output, "type")
            if output_type == "message.output":
                segments.extend(
                    _conversation_content_segments(_field(output, "content"))
                )
            elif output_type == "function.call":
                tool_calls.append(_conversation_tool_call(output, len(tool_calls)))

        content = _join_segments(segments, kind="answer")
        reasoning = _join_segments(segments, kind="reasoning")
        if tool_calls:
            return _ModelTurn(
                content=content,
                finish_reason="tool_calls",
                reasoning=reasoning,
                tool_calls=tool_calls,
            )

        reasoning_printed = False
        answer_started = False
        for segment in segments:
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_printed = True
            else:
                self._set_status("answering")
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
        self._finalize_remote_reasoning(
            reasoning=reasoning,
            finish_reason="stop",
            has_answer_text=bool(content),
        )
        if segments and content and not content.endswith("\n"):
            self._print("\n")
        return _ModelTurn(content=content, finish_reason="stop", reasoning=reasoning)

    def _request_kwargs(
        self,
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": self.messages,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "stream": stream,
            "response_format": {"type": "text"},
        }
        if self.generation.max_tokens is not None:
            kwargs["max_tokens"] = self.generation.max_tokens
        if self.backend_kind is BackendKind.REMOTE:
            kwargs["reasoning_effort"] = "high" if self.show_reasoning else "none"
        else:
            prompt_mode = self._effective_prompt_mode()
            if prompt_mode is not None:
                kwargs["prompt_mode"] = prompt_mode
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _should_use_raw_chat(self) -> bool:
        """Return whether the local raw chat endpoint should be used."""

        return (
            self.backend_kind is BackendKind.LOCAL
            and self.show_reasoning
            and self.show_thinking
            and isinstance(self.client, Mistral)
        )

    def _chat_endpoint(self) -> str:
        if not self.server_url:
            raise RuntimeError("The raw chat endpoint is only available in local mode.")
        return f"{self.server_url.rstrip('/')}/v1/chat/completions"

    def _open_raw_request(self, payload: dict[str, Any]) -> Any:
        """Open a raw HTTP request against the local OpenAI-compatible chat API."""

        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self._chat_endpoint(),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout_s = max(1.0, self.timeout_ms / 1000)
        try:
            return urlopen(request, timeout=timeout_s)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"raw chat request failed with HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"raw chat request failed: {exc.reason}") from exc

    def _print(self, text: str) -> None:
        if self.answer_writer is not None:
            self.answer_writer(text)
            return
        assert self.stdout is not None
        self.stdout.write(text)
        self.stdout.flush()

    def _print_reasoning(self, text: str) -> None:
        if not text or not self.show_thinking:
            return
        if self.reasoning_writer is not None:
            self.reasoning_writer(text)
            return
        assert self.stdout is not None
        self.stdout.write(render_reasoning_chunk(text, stream=self.stdout))
        self.stdout.flush()

    def _finalize_remote_reasoning(
        self,
        *,
        reasoning: str,
        finish_reason: str,
        has_answer_text: bool,
    ) -> None:
        if self.backend_kind is not BackendKind.REMOTE:
            return
        if reasoning.strip():
            self._missing_reasoning_notice_shown = False
            return
        if (
            not self.show_reasoning
            or not self.show_thinking
            or finish_reason
            in {
                "tool_calls",
                "cancelled",
                "error",
            }
        ):
            return
        if not has_answer_text or self._missing_reasoning_notice_shown:
            return
        if self.conversations.enabled:
            message = (
                "[reasoning] Reasoning was requested, but Mistral "
                "Conversations returned no thinking blocks for this turn.\n"
            )
        else:
            message = (
                "[reasoning] Reasoning was requested, but the remote "
                "backend returned no thinking blocks for this turn.\n"
            )
        if has_answer_text:
            message = "\n" + message
        self._print(message)
        logger.warning(
            "Reasoning requested but not returned backend=%s conversations=%s",
            self.backend_kind.value,
            self.conversations.enabled,
        )
        self._missing_reasoning_notice_shown = True

    def _print_answer_separator(
        self, *, reasoning_printed: bool, answer_started: bool
    ) -> bool:
        """Separate visible reasoning from the final answer once per turn."""

        if reasoning_printed and not answer_started:
            self._print("\n\n")
            return True
        return answer_started

    def _rollback_to(self, message_count: int) -> None:
        self.messages = self.messages[:message_count]
        self._mark_context_status_dirty()

    def _commit_assistant_message(self, turn: _ModelTurn) -> None:
        if turn.tool_calls:
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": turn.tool_calls,
            }
            if turn.content:
                assistant_message["content"] = turn.content
            self.messages.append(assistant_message)
            self._mark_context_status_dirty()
            return

        if turn.content:
            self.messages.append({"role": "assistant", "content": turn.content})
            self._mark_context_status_dirty()
            logger.info(
                (
                    "Assistant message committed finish_reason=%s "
                    "content_len=%s reasoning_len=%s"
                ),
                turn.finish_reason,
                len(turn.content),
                len(turn.reasoning),
            )

    def _sync_conversation_id(
        self,
        conversation_id: str,
        *,
        source: str,
        payload: Any | None = None,
        parent_conversation_id: str | None = None,
        fallback_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update the active conversation id and synchronize the local registry."""

        normalized_id = conversation_id.strip()
        if not normalized_id:
            return
        previous_id = self.conversation_id
        self.conversation_id = normalized_id
        self.conversation_resume_source = source
        if self.conversation_registry is None:
            return
        if previous_id and previous_id != normalized_id:
            self.conversation_registry.migrate_conversation_id(
                previous_id, normalized_id
            )
        if payload is not None:
            self._update_registry_from_remote_payload(
                normalized_id,
                payload,
                fallback_metadata=fallback_metadata,
            )
        else:
            self.conversation_registry.update_remote_snapshot(
                normalized_id,
                remote_name=self.pending_conversation.name or None,
                remote_description=self.pending_conversation.description or None,
                remote_metadata=(
                    self.pending_conversation.metadata
                    if self.pending_conversation.metadata
                    else None
                ),
                remote_kind="model",
                remote_model=self.model_id,
                parent_conversation_id=parent_conversation_id,
            )
        self.conversation_registry.remember_active(normalized_id)

    def _update_registry_from_remote_payload(
        self,
        conversation_id: str,
        payload: Any,
        *,
        parent_conversation_id: str | None = None,
        fallback_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist remote conversation metadata into the local registry."""

        if self.conversation_registry is None or not conversation_id.strip():
            return
        metadata = _field(payload, "metadata", None)
        if metadata is None and fallback_metadata:
            metadata = fallback_metadata
        self.conversation_registry.update_remote_snapshot(
            conversation_id,
            remote_name=_field(payload, "name", None),
            remote_description=_field(payload, "description", None),
            remote_metadata=metadata,
            remote_kind=("agent" if _field(payload, "agent_id") else "model"),
            remote_model=_field(payload, "model", None),
            remote_agent_id=_field(payload, "agent_id", None),
            created_at=_field(payload, "created_at", None),
            updated_at=_field(payload, "updated_at", None),
            parent_conversation_id=parent_conversation_id,
        )

    def _format_remote_conversation_summary(self, payload: Any) -> str:
        """Render a one-line summary for a remote conversation entity."""

        conversation_id = str(_field(payload, "id", "") or "")
        kind = "agent" if _field(payload, "agent_id") else "model"
        target = str(
            _field(payload, "agent_id", "") or _field(payload, "model", "") or "unknown"
        )
        name = str(_field(payload, "name", "") or "")
        description = str(_field(payload, "description", "") or "")
        created_at = str(_field(payload, "created_at", "") or "")
        updated_at = str(_field(payload, "updated_at", "") or "")
        details = [
            conversation_id,
            kind,
            target,
        ]
        if name:
            details.append(f'name="{name}"')
        if description:
            details.append(f'description="{description}"')
        if created_at:
            details.append(f"created={created_at}")
        if updated_at:
            details.append(f"updated={updated_at}")
        if self.conversation_registry is not None:
            record = self.conversation_registry.get(conversation_id)
            if record is not None:
                overlay = self._format_registry_record(record, include_note=False)
                if overlay:
                    details.append(f"local[{overlay}]")
        return " | ".join(part for part in details if part)

    def _render_remote_conversation_details(self, payload: Any) -> list[str]:
        """Render a multi-line detail view for one remote conversation."""

        lines = [
            f"kind: {'agent' if _field(payload, 'agent_id') else 'model'}",
            f"model: {_field(payload, 'model', '') or '(none)'}",
            f"agent_id: {_field(payload, 'agent_id', '') or '(none)'}",
            f"name: {_field(payload, 'name', '') or '(none)'}",
            f"description: {_field(payload, 'description', '') or '(none)'}",
            f"created_at: {_field(payload, 'created_at', '') or '(unknown)'}",
            f"updated_at: {_field(payload, 'updated_at', '') or '(unknown)'}",
        ]
        metadata = _field(payload, "metadata", None)
        if metadata:
            lines.append(
                "metadata: " + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            )
        else:
            lines.append("metadata: (none)")
        return lines

    def _format_registry_record(
        self,
        record: ConversationRecord,
        *,
        include_note: bool,
    ) -> str:
        """Render the local overlay metadata for one registry entry."""

        parts = [record.conversation_id]
        if record.alias:
            parts.append(f"alias={record.alias}")
        if record.tags:
            parts.append(f"tags={','.join(record.tags)}")
        if include_note and record.note:
            parts.append(f"note={record.note}")
        if record.remote_name:
            parts.append(f"remote_name={record.remote_name}")
        if record.remote_metadata:
            metadata = ",".join(
                f"{key}={value}"
                for key, value in sorted(record.remote_metadata.items())
            )
            parts.append(f"metadata={metadata}")
        if record.parent_conversation_id:
            parts.append(f"parent={record.parent_conversation_id}")
        if record.deleted:
            parts.append("deleted=yes")
        return " | ".join(parts)

    def _parse_tool_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            text = raw_arguments.strip()
            if not text:
                return {}
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Tool arguments must decode to an object")
            return cast(dict[str, Any], parsed)
        raise TypeError(f"Unsupported tool arguments payload: {type(raw_arguments)!r}")

    def _tool_message(
        self, *, call: dict[str, Any], result: MCPToolResult
    ) -> dict[str, Any]:
        function = call["function"]
        content = self._render_tool_result(result)
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": function["name"],
            "content": content,
        }

    def _render_tool_result(self, result: MCPToolResult) -> str:
        parts: list[str] = []
        if result.structured_content is not None:
            parts.append(
                json.dumps(
                    result.structured_content, ensure_ascii=False, sort_keys=True
                )
            )
        if result.text:
            parts.append(result.text)
        return "\n\n".join(parts)

    def _cancelled_tool_result(self, call: dict[str, Any]) -> MCPToolResult:
        """Build a synthetic tool result used to unblock interrupted Conversations."""

        function = call["function"]
        return MCPToolResult(
            text="[tool-error] cancelled by user",
            is_error=True,
            structured_content={
                "status": "error",
                "tool": str(function["name"]),
                "code": "cancelled_by_user",
                "cancelled": True,
            },
        )

    def _complete_pending_conversation_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        completed_inputs: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send cancellation tool results to unblock an interrupted Conversation."""

        if not tool_calls or self.conversation_id is None:
            return
        inputs = list(completed_inputs or [])
        completed_ids = {
            str(item.get("tool_call_id", ""))
            for item in inputs
            if isinstance(item, dict)
        }
        for call in tool_calls:
            call_id = str(call["id"])
            if call_id in completed_ids:
                continue
            inputs.append(
                {
                    "type": "function.result",
                    "tool_call_id": call_id,
                    "result": self._render_tool_result(
                        self._cancelled_tool_result(call)
                    ),
                }
            )
        if not inputs:
            return
        try:
            self.client.beta.conversations.append(
                **self._conversation_append_kwargs(inputs=inputs)
            )
            logger.warning(
                "Completed pending Conversations tool calls after interruption "
                "conversation_id=%s count=%s",
                self.conversation_id,
                len(inputs),
            )
        except Exception as exc:
            logger.warning(
                "Failed to complete pending Conversations tool calls after "
                "interruption conversation_id=%s error=%s",
                self.conversation_id,
                exc,
            )

    def _call_tool_bridge(
        self, public_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        assert self.tool_bridge is not None
        try:
            return self.tool_bridge.call_tool(public_name, arguments)
        except MCPBridgeError as exc:
            return MCPToolResult(text=f"[tool-error] {exc}", is_error=True)

    def _tool_call_signature(self, name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _content_summary(self, content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return f"text(len={len(content)})"
        return f"blocks(len={len(content)})"

    def _summarize_tool_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        summarized: dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                if key in {"content", "prompt", "system_prompt", "api_key"}:
                    summarized[key] = f"<str len={len(value)}>"
                elif len(value) > 120:
                    summarized[key] = f"{value[:117]}..."
                else:
                    summarized[key] = value
                continue
            if isinstance(value, list):
                summarized[key] = f"<list len={len(value)}>"
                continue
            if isinstance(value, dict):
                summarized[key] = f"<dict keys={sorted(value)}>"
                continue
            summarized[key] = value
        return summarized

    def _resolve_tools(self) -> list[dict[str, Any]]:
        if self.tool_bridge is None:
            return []
        try:
            return self.tool_bridge.to_mistral_tools()
        except MCPBridgeError as exc:
            if not self._mcp_warning_shown:
                self._print(f"[mcp] {exc}\n")
                self._mcp_warning_shown = True
            return []

    def _has_attachment_blocks(self, content: str | list[dict[str, Any]]) -> bool:
        if isinstance(content, str):
            return False
        return any(
            block.get("type") in {"image_url", "document_url"} for block in content
        )

    def _send_single_turn(
        self,
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        if self._should_use_raw_chat():
            if stream:
                return self._send_streaming_raw(tools=tools)
            return self._send_non_streaming_raw(tools=tools)
        if stream:
            return self._send_streaming(tools=tools)
        return self._send_non_streaming(tools=tools)

    def _prepare_context_for_turn(
        self,
        content: str | list[dict[str, Any]],
        *,
        disable_tools: bool,
    ) -> bool:
        pending_messages = [*self.messages, {"role": "user", "content": content}]
        tools = [] if disable_tools else self._resolve_tools()
        window_tokens = self._model_context_window() or self.context.local_window_tokens
        threshold_tokens = int(window_tokens * self.context.threshold)
        reserve_tokens = self._effective_context_reserve()
        estimated_tokens = (
            self.estimate_context_tokens(pending_messages, tools=tools) + reserve_tokens
        )
        logger.debug(
            "Context estimate tokens=%s window=%s threshold=%s reserve=%s auto=%s",
            estimated_tokens,
            window_tokens,
            threshold_tokens,
            reserve_tokens,
            self.context.auto_compact,
        )
        if estimated_tokens < threshold_tokens:
            return True

        if self.context.auto_compact:
            self._print(
                f"[compact] estimated context {estimated_tokens}/"
                f"{window_tokens} reached threshold {threshold_tokens}; "
                "compacting old turns...\n"
            )
            try:
                result = self.compact_context()
            except Exception as exc:
                logger.exception("Automatic context compaction failed")
                self._print(f"[compact] failed: {exc}\n")
            else:
                self._print(f"[compact] {result.summary()}\n")
            pending_messages = [*self.messages, {"role": "user", "content": content}]
            estimated_tokens = (
                self.estimate_context_tokens(pending_messages, tools=tools)
                + reserve_tokens
            )
            if estimated_tokens < window_tokens:
                return True

        if estimated_tokens >= window_tokens:
            self._print(
                f"[context] estimated prompt {estimated_tokens}/"
                f"{window_tokens} exceeds the configured context window. "
                "Run /compact or raise the configured window before retrying.\n"
            )
            self._set_status("error")
            logger.warning(
                "Context overflow blocked estimated=%s window=%s",
                estimated_tokens,
                window_tokens,
            )
            return False

        if not self.context.auto_compact:
            self._print(
                f"[context] estimated context {estimated_tokens}/"
                f"{window_tokens} is above the "
                f"{round(self.context.threshold * 100)}% threshold. "
                "Run /compact or enable /compact auto on.\n"
            )
        return True

    def _effective_context_reserve(self) -> int:
        if self.generation.max_tokens is None:
            return self.context.reserve_tokens
        return max(self.context.reserve_tokens, self.generation.max_tokens)

    def _split_compactable_history(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        body = self.messages[1:]
        if not body:
            return [], []
        user_seen = 0
        split_index = len(body)
        for index in range(len(body) - 1, -1, -1):
            if body[index].get("role") == "user":
                user_seen += 1
                if user_seen >= self.context.keep_recent_turns:
                    split_index = index
                    break
        if user_seen < self.context.keep_recent_turns:
            return [], body
        return body[:split_index], body[split_index:]

    def _summarize_messages_for_compaction(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        rendered_history = _render_messages_for_compaction(messages)
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "You compact chat history for Mistral4Cli. Summarize durable "
                    "facts, decisions, file paths, commands and outputs, tool "
                    "results, user preferences, and open tasks. Keep the summary "
                    "concise, factual, and in English. Do not invent details."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the old conversation below so a future assistant "
                    "can continue without the full transcript.\n\n"
                    f"{rendered_history}"
                ),
            },
        ]
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": summary_messages,
            "temperature": min(self.generation.temperature, 0.2),
            "top_p": self.generation.top_p,
            "stream": False,
            "response_format": {"type": "text"},
            "max_tokens": self.context.summary_max_tokens,
        }
        if self.backend_kind is BackendKind.REMOTE:
            kwargs["reasoning_effort"] = "none"
        else:
            prompt_mode = self._effective_prompt_mode()
            if prompt_mode is not None:
                kwargs["prompt_mode"] = prompt_mode
        response = self.client.chat.complete(**kwargs)
        choice = response.choices[0]
        message = choice.message
        if message is None:
            return ""
        content_value = message.content
        if isinstance(content_value, str):
            return content_value.strip()
        return _join_segments(
            _content_segments_from_value(content_value), kind="answer"
        )

    def _send_non_streaming_raw(
        self, *, tools: list[dict[str, Any]] | None
    ) -> _ModelTurn:
        payload = self._request_kwargs(stream=False, tools=tools)
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        try:
            with self._open_raw_request(payload) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(content="", finish_reason="cancelled", cancelled=True)
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(raw.get("usage"))

        choice = raw["choices"][0]
        message = choice.get("message", {})
        raw_content = message.get("content") or ""
        raw_reasoning = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        parsed = _parse_reasoning_text(raw_content)
        content = parsed.answer
        reasoning = str(raw_reasoning).strip() or parsed.reasoning
        if not tool_calls and tools:
            tool_calls = _extract_tool_calls_from_text(raw_content)
            if tool_calls:
                finish_reason = "tool_calls"
                content = ""

        if tool_calls:
            return _ModelTurn(
                content=content,
                finish_reason=finish_reason,
                reasoning=reasoning,
                tool_calls=tool_calls,
            )

        if reasoning:
            self._set_status("answering")
            self._print_reasoning(reasoning)
            printed_anything = True
            reasoning_printed = True
        for segment in parsed.segments:
            if segment.kind == "answer":
                self._set_status("answering")
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
                printed_anything = True
            elif segment.kind == "reasoning":
                reasoning_printed = True
        if printed_anything and not content.endswith("\n"):
            self._print("\n")
        elif finish_reason == "length":
            self._print("[truncated response without text]\n")

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason,
            reasoning=reasoning,
        )

    def _send_streaming_raw(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        payload = self._request_kwargs(stream=True, tools=tools)
        finish_reason = ""
        tool_states: dict[int, _ToolCallState] = {}
        parser = _ReasoningParser()
        deferred_answer = _DeferredAnswerBuffer(enabled=bool(tools))
        reasoning_parts: list[str] = []
        answer_parts: list[str] = []
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        usage_snapshot: Any = None

        try:
            with self._open_raw_request(payload) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_line = line[6:]
                    if data_line == "[DONE]":
                        break
                    event = json.loads(data_line)
                    choices = event.get("choices") or []
                    if not choices:
                        usage = event.get("usage")
                        if usage is not None:
                            usage_snapshot = usage
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    usage = event.get("usage")
                    if usage is not None:
                        usage_snapshot = usage
                    reasoning_delta = delta.get("reasoning_content")
                    if isinstance(reasoning_delta, str) and reasoning_delta:
                        self._set_status("answering")
                        reasoning_parts.append(reasoning_delta)
                        self._print_reasoning(reasoning_delta)
                        printed_anything = True
                        reasoning_printed = True
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        for segment in parser.feed(content):
                            if segment.kind == "reasoning":
                                self._set_status("answering")
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                            else:
                                self._set_status("answering")
                                answer_parts.append(segment.text)
                                display_text = deferred_answer.feed(segment.text)
                                if display_text:
                                    answer_started = self._print_answer_separator(
                                        reasoning_printed=reasoning_printed,
                                        answer_started=answer_started,
                                    )
                                    self._print(display_text)
                            printed_anything = True
                    for tool_call in delta.get("tool_calls") or []:
                        index = int(_field(tool_call, "index", 0) or 0)
                        state = tool_states.setdefault(
                            index, _ToolCallState(index=index)
                        )
                        state.update(tool_call)
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(
                content=parser.answer,
                reasoning="".join(reasoning_parts).strip() or parser.reasoning,
                finish_reason="cancelled",
                cancelled=True,
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"\n[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(usage_snapshot)

        for segment in parser.finish():
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_printed = True
            else:
                self._set_status("answering")
                answer_parts.append(segment.text)
                display_text = deferred_answer.feed(segment.text)
                if display_text:
                    answer_started = self._print_answer_separator(
                        reasoning_printed=reasoning_printed,
                        answer_started=answer_started,
                    )
                    self._print(display_text)
            printed_anything = True

        content = "".join(answer_parts).strip() or parser.answer
        reasoning = "".join(reasoning_parts).strip() or parser.reasoning
        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
        if not tool_calls and tools:
            tool_calls = _extract_tool_calls_from_text(content)
            if tool_calls:
                finish_reason = "tool_calls"
                content = ""

        deferred_tail = deferred_answer.finalize()
        if deferred_tail and not tool_calls:
            self._set_status("answering")
            answer_started = self._print_answer_separator(
                reasoning_printed=reasoning_printed,
                answer_started=answer_started,
            )
            self._print(deferred_tail)
            printed_anything = True

        if printed_anything and not content.endswith("\n"):
            self._print("\n")
        if finish_reason == "length" and not content:
            self._print("[truncated response without text]\n")

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason or "stop",
            reasoning=reasoning,
            tool_calls=tool_calls,
        )

    def _send_non_streaming(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        try:
            response = self.client.chat.complete(
                **self._request_kwargs(stream=False, tools=tools)
            )
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(content="", finish_reason="cancelled", cancelled=True)
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(getattr(response, "usage", None))
        choice = response.choices[0]
        message = choice.message
        if message is None:
            self._print("[error] empty response message\n")
            return _ModelTurn(content="", finish_reason="error", error=True)
        content_value = message.content
        finish_reason = choice.finish_reason or "stop"
        tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None))
        segments = _content_segments_from_value(content_value)
        content = _join_segments(segments, kind="answer")
        reasoning = _join_segments(segments, kind="reasoning")
        if not tool_calls and tools:
            if isinstance(content_value, str):
                tool_calls = _extract_tool_calls_from_text(content_value)
            if tool_calls:
                finish_reason = "tool_calls"
                content = ""
        reasoning_printed = False
        answer_started = False

        if tool_calls:
            return _ModelTurn(
                content=content,
                finish_reason=finish_reason,
                reasoning=reasoning,
                tool_calls=tool_calls,
            )

        for segment in segments:
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_printed = True
            else:
                self._set_status("answering")
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
        if segments and not content.endswith("\n"):
            self._print("\n")
        elif finish_reason == "length":
            self._print("[truncated response without text]\n")
        self._finalize_remote_reasoning(
            reasoning=reasoning,
            finish_reason=finish_reason,
            has_answer_text=bool(content),
        )

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason,
            reasoning=reasoning,
        )

    def _send_streaming(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        finish_reason = ""
        tool_states: dict[int, _ToolCallState] = {}
        parser = _ReasoningParser()
        deferred_answer = _DeferredAnswerBuffer(enabled=bool(tools))
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_snapshot: Any = None

        try:
            stream = self.client.chat.stream(
                **self._request_kwargs(stream=True, tools=tools)
            )
            with stream as active_stream:
                for event in active_stream:
                    data = getattr(event, "data", None)
                    if not data or not getattr(data, "choices", None):
                        usage = getattr(data, "usage", None)
                        if usage is not None:
                            usage_snapshot = usage
                        continue
                    choice = data.choices[0]
                    usage = getattr(data, "usage", None)
                    if usage is not None:
                        usage_snapshot = usage
                    finish_reason = choice.finish_reason or finish_reason
                    delta = choice.delta
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        for segment in parser.feed(content):
                            if segment.kind == "reasoning":
                                self._set_status("answering")
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                                reasoning_parts.append(segment.text)
                            else:
                                self._set_status("answering")
                                answer_parts.append(segment.text)
                                display_text = deferred_answer.feed(segment.text)
                                if display_text:
                                    answer_started = self._print_answer_separator(
                                        reasoning_printed=reasoning_printed,
                                        answer_started=answer_started,
                                    )
                                    self._print(display_text)
                            printed_anything = True
                    elif isinstance(content, list):
                        for segment in _content_segments_from_value(content):
                            if segment.kind == "reasoning":
                                self._set_status("answering")
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                                reasoning_parts.append(segment.text)
                            else:
                                self._set_status("answering")
                                answer_parts.append(segment.text)
                                display_text = deferred_answer.feed(segment.text)
                                if display_text:
                                    answer_started = self._print_answer_separator(
                                        reasoning_printed=reasoning_printed,
                                        answer_started=answer_started,
                                    )
                                    self._print(display_text)
                            printed_anything = True
                    for tool_call in getattr(delta, "tool_calls", None) or []:
                        index = int(getattr(tool_call, "index", 0) or 0)
                        state = tool_states.setdefault(
                            index, _ToolCallState(index=index)
                        )
                        state.update(tool_call)
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(
                content=("".join(answer_parts).strip() or parser.answer),
                reasoning=("".join(reasoning_parts).strip() or parser.reasoning),
                finish_reason="cancelled",
                cancelled=True,
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"\n[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(usage_snapshot)

        for segment in parser.finish():
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_printed = True
                reasoning_parts.append(segment.text)
            else:
                self._set_status("answering")
                answer_parts.append(segment.text)
                display_text = deferred_answer.feed(segment.text)
                if display_text:
                    answer_started = self._print_answer_separator(
                        reasoning_printed=reasoning_printed,
                        answer_started=answer_started,
                    )
                    self._print(display_text)
            printed_anything = True

        content = "".join(answer_parts).strip()
        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
        if not tool_calls and tools:
            tool_calls = _extract_tool_calls_from_text(content)
            if tool_calls:
                finish_reason = "tool_calls"
                content = ""

        deferred_tail = deferred_answer.finalize()
        if deferred_tail and not tool_calls:
            self._set_status("answering")
            answer_started = self._print_answer_separator(
                reasoning_printed=reasoning_printed,
                answer_started=answer_started,
            )
            self._print(deferred_tail)
            printed_anything = True

        if printed_anything and not content.endswith("\n"):
            self._print("\n")
        if finish_reason == "length" and not content:
            self._print("[truncated response without text]\n")
        self._finalize_remote_reasoning(
            reasoning="".join(reasoning_parts).strip(),
            finish_reason=finish_reason or "stop",
            has_answer_text=bool(content),
        )

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason or "stop",
            reasoning="".join(reasoning_parts).strip(),
            tool_calls=tool_calls,
        )

    def _normalize_user_content(
        self, content: str | list[dict[str, Any]]
    ) -> str | list[dict[str, Any]] | None:
        if isinstance(content, str):
            clean_text = content.strip()
            if not clean_text:
                return None
            return clean_text
        if not content:
            return None
        return content

    def _effective_prompt_mode(self) -> str | None:
        if self.backend_kind is BackendKind.REMOTE:
            return None
        if not self.show_reasoning:
            return None
        return self.generation.prompt_mode

    def _display_generation(self) -> LocalGenerationConfig:
        return replace(self.generation, prompt_mode=self._effective_prompt_mode())

    def _set_status(self, phase: str, detail: str | None = None) -> None:
        if self._status_phase == phase and self._status_detail == detail:
            return
        self._status_phase = phase
        self._status_detail = detail
        if self.status_callback is not None:
            self.status_callback()

    def _record_usage(self, raw_usage: Any) -> None:
        usage = _normalize_usage_snapshot(
            raw_usage,
            max_context_tokens=self._model_context_window(),
        )
        if usage is None or usage.is_empty():
            return
        self._last_usage = usage
        self._turn_usage_accumulator = usage.merge(self._turn_usage_accumulator)
        if self.status_callback is not None:
            self.status_callback()

    def _mark_context_status_dirty(self) -> None:
        self._context_status_dirty = True

    def _status_context_snapshot(self) -> ContextStatus | None:
        if self.conversations.enabled:
            return None
        if self._context_status_dirty or self._cached_context_status is None:
            self._cached_context_status = self._compute_context_status()
            self._context_status_dirty = False
        return self._cached_context_status

    def _model_context_window(self) -> int | None:
        if self.backend_kind is not BackendKind.REMOTE:
            return self.context.local_window_tokens
        normalized_model = self.model_id.strip().lower()
        return REMOTE_CONTEXT_WINDOWS.get(
            normalized_model,
            self.context.remote_window_tokens,
        )

    def _with_context_window(
        self,
        usage: UsageSnapshot | None,
        max_context_tokens: int | None,
    ) -> UsageSnapshot | None:
        if usage is None:
            return None
        return replace(usage, max_context_tokens=max_context_tokens)

    def _commit_turn_usage(self) -> None:
        if self._turn_usage_accumulator is None:
            return
        self._cumulative_usage = self._turn_usage_accumulator.merge(
            self._cumulative_usage
        )
        self._turn_usage_accumulator = None


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


MistralCodingSession = MistralSession
