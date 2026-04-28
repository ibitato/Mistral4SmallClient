# ruff: noqa: F401
from __future__ import annotations

import io
import os
import pty
import re
import select
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mistral4cli import __version__
from mistral4cli.attachments import (
    build_image_message,
    build_remote_document_message,
    build_remote_image_message,
)
from mistral4cli.cli import (
    LINUX_ONLY_MESSAGE,
    _build_active_attachment_message,
    _clear_screen_if_supported,
    _InputHistory,
    _parse_command,
    _PendingAttachment,
    _refresh_repl_screen,
    _repl_status_line,
    _ReplState,
    _run_command,
    _run_repl,
    _write_tty_newline,
    main,
)
from mistral4cli.conversation_registry import ConversationRegistry
from mistral4cli.local_mistral import (
    DEFAULT_TIMEOUT_MS,
    BackendKind,
    ContextConfig,
    ConversationConfig,
    LocalGenerationConfig,
    LocalMistralConfig,
    RemoteMistralConfig,
    build_client,
)
from mistral4cli.local_tools import LocalToolBridge
from mistral4cli.logging_config import DEFAULT_LOG_RETENTION_DAYS
from mistral4cli.mcp_bridge import MCPToolResult
from mistral4cli.session import (
    DEFAULT_SYSTEM_PROMPT,
    MistralCodingSession,
    MistralSession,
    UsageSnapshot,
)
from mistral4cli.ui import (
    CLEAR_SCREEN,
    CYAN,
    GREEN,
    ORANGE,
    RED,
    RESET,
    InteractiveTTYRenderer,
    SmartOutputWriter,
    iter_typewriter_chunks,
    paint_prompt_lines,
    render_help_screen,
    render_welcome_banner,
    terminal_recommendation,
    wrap_prompt_buffer,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "internet"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class FakeStdin(io.StringIO):
    def __init__(self, value: str = "", *, tty: bool = False) -> None:
        super().__init__(value)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class FakeTTYOutput(io.StringIO):
    def isatty(self) -> bool:
        return True


@dataclass(slots=True)
class FakeToolFunction:
    name: str
    arguments: str


@dataclass(slots=True)
class FakeToolCall:
    id: str = "tool_call_1"
    type: str = "function"
    index: int = 0
    function: FakeToolFunction = field(
        default_factory=lambda: FakeToolFunction(name="tool", arguments="{}")
    )


@dataclass(slots=True)
class FakeMessage:
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass(slots=True)
class FakeDelta:
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass(slots=True)
class FakeChoice:
    message: FakeMessage = field(default_factory=FakeMessage)
    delta: FakeDelta = field(default_factory=FakeDelta)
    finish_reason: str | None = None


@dataclass(slots=True)
class FakeResponse:
    choices: list[FakeChoice]
    usage: object | None = None


@dataclass(slots=True)
class FakeEvent:
    data: FakeResponse


@dataclass(slots=True)
class FakeUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class FakeStream:
    def __init__(
        self, events: list[FakeEvent], interrupt_after: int | None = None
    ) -> None:
        self.events = events
        self.interrupt_after = interrupt_after
        self.closed = False

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.closed = True

    def __iter__(self) -> Iterator[FakeEvent]:
        for index, event in enumerate(self.events):
            if self.interrupt_after is not None and index >= self.interrupt_after:
                raise KeyboardInterrupt
            yield event


class FakeChat:
    def __init__(
        self,
        *,
        complete_text: str = "ok",
        complete_responses: list[FakeResponse] | None = None,
        stream_chunks: list[str] | None = None,
        complete_interrupt_once: bool = False,
        interrupt_after: int | None = None,
    ) -> None:
        self.complete_text = complete_text
        self.complete_responses = complete_responses or []
        self.stream_chunks = stream_chunks or ["ok"]
        self.complete_interrupt_once = complete_interrupt_once
        self.interrupt_after = interrupt_after
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.last_stream: FakeStream | None = None

    def complete(self, **kwargs: Any) -> FakeResponse:
        self.complete_calls.append(kwargs)
        if self.complete_interrupt_once:
            self.complete_interrupt_once = False
            raise KeyboardInterrupt
        if self.complete_responses:
            return self.complete_responses.pop(0)
        return FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(content=self.complete_text),
                    finish_reason="stop",
                )
            ]
        )

    def stream(self, **kwargs: Any) -> FakeStream:
        self.stream_calls.append(kwargs)
        events = [
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(content=chunk),
                            finish_reason="stop"
                            if index == len(self.stream_chunks) - 1
                            else None,
                        )
                    ]
                )
            )
            for index, chunk in enumerate(self.stream_chunks)
        ]
        self.last_stream = FakeStream(events, interrupt_after=self.interrupt_after)
        return self.last_stream


class FakeClient:
    def __init__(
        self,
        *,
        complete_text: str = "ok",
        complete_responses: list[FakeResponse] | None = None,
        stream_chunks: list[str] | None = None,
        complete_interrupt_once: bool = False,
        interrupt_after: int | None = None,
    ) -> None:
        self.chat: Any = FakeChat(
            complete_text=complete_text,
            complete_responses=complete_responses,
            stream_chunks=stream_chunks,
            complete_interrupt_once=complete_interrupt_once,
            interrupt_after=interrupt_after,
        )
        self.beta: Any = None


@dataclass(slots=True)
class FakeConversationOutput:
    type: str
    content: Any = None
    tool_call_id: str = "tool_call_1"
    name: str = "tool"
    arguments: str = "{}"


@dataclass(slots=True)
class FakeConversationResponse:
    conversation_id: str
    outputs: list[FakeConversationOutput]
    usage: FakeUsage | None = None


@dataclass(slots=True)
class FakeConversationEntity:
    id: str
    model: str = "mistral-small-latest"
    agent_id: str = ""
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] | None = None
    created_at: str | datetime = "2026-04-28T00:00:00Z"
    updated_at: str | datetime = "2026-04-28T00:00:00Z"


@dataclass(slots=True)
class FakeConversationEvent:
    event: str
    data: Any


@dataclass(slots=True)
class FakeConversationStarted:
    conversation_id: str


@dataclass(slots=True)
class FakeConversationDone:
    usage: FakeUsage | None = None


@dataclass(slots=True)
class FakeConversationMessageDelta:
    content: Any


@dataclass(slots=True)
class FakeConversationFunctionDelta:
    tool_call_id: str
    name: str
    arguments: str


class FakeConversationStream:
    def __init__(
        self,
        events: list[FakeConversationEvent],
        interrupt_after: int | None = None,
    ) -> None:
        self.events = events
        self.interrupt_after = interrupt_after
        self.closed = False

    def __enter__(self) -> FakeConversationStream:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.closed = True

    def __iter__(self) -> Iterator[FakeConversationEvent]:
        for index, event in enumerate(self.events):
            if self.interrupt_after is not None and index >= self.interrupt_after:
                raise KeyboardInterrupt
            yield event


class FakeConversations:
    def __init__(
        self,
        *,
        responses: list[FakeConversationResponse] | None = None,
        stream_events: list[FakeConversationEvent] | None = None,
        entities: list[FakeConversationEntity] | None = None,
        stream_interrupt_after: int | None = None,
    ) -> None:
        self.responses = responses or [
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "ok"}],
                    )
                ],
            )
        ]
        self.stream_events = stream_events or []
        self.stream_interrupt_after = stream_interrupt_after
        self.entities = entities or [
            FakeConversationEntity(
                id="conv_1",
                name="Primary conversation",
                description="Tracked in tests",
                metadata={"suite": "cli"},
            )
        ]
        self.start_calls: list[dict[str, Any]] = []
        self.append_calls: list[dict[str, Any]] = []
        self.start_stream_calls: list[dict[str, Any]] = []
        self.append_stream_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.restart_calls: list[dict[str, Any]] = []

    def start(self, **kwargs: Any) -> FakeConversationResponse:
        self.start_calls.append(kwargs)
        return self.responses.pop(0)

    def append(self, **kwargs: Any) -> FakeConversationResponse:
        self.append_calls.append(kwargs)
        return self.responses.pop(0)

    def start_stream(self, **kwargs: Any) -> FakeConversationStream:
        self.start_stream_calls.append(kwargs)
        return FakeConversationStream(
            self.stream_events,
            interrupt_after=self.stream_interrupt_after,
        )

    def append_stream(self, **kwargs: Any) -> FakeConversationStream:
        self.append_stream_calls.append(kwargs)
        return FakeConversationStream(
            self.stream_events,
            interrupt_after=self.stream_interrupt_after,
        )

    def list(self, **kwargs: Any) -> list[FakeConversationEntity]:
        self.list_calls.append(kwargs)
        return list(self.entities)

    def get(self, **kwargs: Any) -> FakeConversationEntity:
        self.get_calls.append(kwargs)
        conversation_id = str(kwargs["conversation_id"])
        for entity in self.entities:
            if entity.id == conversation_id:
                return entity
        raise RuntimeError(f"unknown conversation {conversation_id}")

    def get_history(self, **kwargs: Any) -> object:
        return type(
            "History",
            (),
            {
                "entries": [
                    type(
                        "Entry",
                        (),
                        {
                            "id": "entry_1",
                            "type": "message.input",
                            "role": "user",
                            "content": "hello",
                        },
                    )()
                ]
            },
        )()

    def get_messages(self, **kwargs: Any) -> object:
        return type(
            "Messages",
            (),
            {
                "messages": [
                    type(
                        "Message",
                        (),
                        {
                            "id": "message_1",
                            "type": "message.output",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "ok"}],
                        },
                    )()
                ]
            },
        )()

    def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)

    def restart(self, **kwargs: Any) -> FakeConversationResponse:
        self.restart_calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return FakeConversationResponse(
            conversation_id="conv_restart",
            outputs=[
                FakeConversationOutput(
                    type="message.output",
                    content=[{"type": "text", "text": "branched"}],
                )
            ],
        )


class FakeBeta:
    def __init__(self, conversations: FakeConversations) -> None:
        self.conversations = conversations


class FakeConversationClient(FakeClient):
    def __init__(self, conversations: FakeConversations | None = None) -> None:
        super().__init__()
        self.beta: Any = FakeBeta(conversations or FakeConversations())


class FakeRawHTTPResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self) -> FakeRawHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class FakeRawStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = [(line + "\n").encode("utf-8") for line in lines]

    def __enter__(self) -> FakeRawStreamResponse:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.lines)


class FakeToolBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def runtime_summary(self) -> str:
        return "FireCrawl MCP: auto-tools on (tests/mcp.json)"

    def describe_tools(self) -> str:
        return "\n".join(
            [
                self.runtime_summary(),
                "Tools:",
                "  - web_search: Search the web.",
            ]
        )

    def to_mistral_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        self.calls.append((public_name, arguments))
        return MCPToolResult(
            text='{"results":[{"title":"Example","url":"https://example.com"}]}',
            is_error=False,
        )


class FakeLongToolBridge(FakeToolBridge):
    def describe_tools(self) -> str:
        return "\n".join(f"tool line {index}" for index in range(20))


class InterruptingToolBridge(FakeToolBridge):
    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        self.calls.append((public_name, arguments))
        raise KeyboardInterrupt


__all__ = [
    "ANSI_ESCAPE_RE",
    "CLEAR_SCREEN",
    "CYAN",
    "DEFAULT_LOG_RETENTION_DAYS",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TIMEOUT_MS",
    "FIXTURE_DIR",
    "GREEN",
    "LINUX_ONLY_MESSAGE",
    "ORANGE",
    "RED",
    "RESET",
    "Any",
    "BackendKind",
    "ContextConfig",
    "ConversationConfig",
    "ConversationRegistry",
    "FakeBeta",
    "FakeChat",
    "FakeChoice",
    "FakeClient",
    "FakeConversationClient",
    "FakeConversationDone",
    "FakeConversationEntity",
    "FakeConversationEvent",
    "FakeConversationFunctionDelta",
    "FakeConversationMessageDelta",
    "FakeConversationOutput",
    "FakeConversationResponse",
    "FakeConversationStarted",
    "FakeConversations",
    "FakeDelta",
    "FakeEvent",
    "FakeLongToolBridge",
    "FakeMessage",
    "FakeRawHTTPResponse",
    "FakeRawStreamResponse",
    "FakeResponse",
    "FakeStdin",
    "FakeStream",
    "FakeTTYOutput",
    "FakeToolBridge",
    "FakeToolCall",
    "FakeToolFunction",
    "FakeUsage",
    "InteractiveTTYRenderer",
    "InterruptingToolBridge",
    "LocalGenerationConfig",
    "LocalMistralConfig",
    "LocalToolBridge",
    "MCPToolResult",
    "MistralCodingSession",
    "MistralSession",
    "Path",
    "RemoteMistralConfig",
    "SmartOutputWriter",
    "UsageSnapshot",
    "_InputHistory",
    "_PendingAttachment",
    "_ReplState",
    "__version__",
    "_build_active_attachment_message",
    "_clear_screen_if_supported",
    "_parse_command",
    "_refresh_repl_screen",
    "_repl_status_line",
    "_run_command",
    "_run_repl",
    "_write_tty_newline",
    "build_client",
    "build_image_message",
    "build_remote_document_message",
    "build_remote_image_message",
    "datetime",
    "field",
    "io",
    "iter_typewriter_chunks",
    "main",
    "os",
    "paint_prompt_lines",
    "pty",
    "re",
    "render_help_screen",
    "render_welcome_banner",
    "select",
    "signal",
    "subprocess",
    "sys",
    "terminal_recommendation",
    "time",
    "timezone",
    "wrap_prompt_buffer",
]
