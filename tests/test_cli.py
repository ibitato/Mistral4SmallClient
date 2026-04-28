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

    def __iter__(self):
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
        self.chat = FakeChat(
            complete_text=complete_text,
            complete_responses=complete_responses,
            stream_chunks=stream_chunks,
            complete_interrupt_once=complete_interrupt_once,
            interrupt_after=interrupt_after,
        )


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
    created_at: str = "2026-04-28T00:00:00Z"
    updated_at: str = "2026-04-28T00:00:00Z"


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

    def __iter__(self):
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
        self.beta = FakeBeta(conversations or FakeConversations())


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

    def __iter__(self):
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


def test_main_returns_zero_without_interactive_input() -> None:
    output = io.StringIO()
    exit_code = main(
        [],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: FakeClient(),
    )

    assert exit_code == 0
    assert output.getvalue() == ""


def test_version_flag_prints_package_version() -> None:
    output = io.StringIO()

    exit_code = main(
        ["--version"],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: FakeClient(),
    )

    assert exit_code == 0
    assert output.getvalue() == f"mistral4cli {__version__}\n"


def test_short_version_flag_prints_package_version() -> None:
    output = io.StringIO()

    exit_code = main(
        ["-v"],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: FakeClient(),
    )

    assert exit_code == 0
    assert output.getvalue() == f"mistral4cli {__version__}\n"


def test_main_rejects_non_linux_before_print_defaults(monkeypatch: Any) -> None:
    output = io.StringIO()
    error = io.StringIO()
    client_factory_called = False

    def client_factory(_config: Any) -> FakeClient:
        nonlocal client_factory_called
        client_factory_called = True
        return FakeClient()

    monkeypatch.setattr("mistral4cli.cli.sys.platform", "darwin")

    exit_code = main(
        ["--print-defaults", "--no-mcp"],
        stdin=FakeStdin(""),
        stdout=output,
        stderr=error,
        client_factory=client_factory,
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    assert error.getvalue() == LINUX_ONLY_MESSAGE + "\n"
    assert client_factory_called is False


def test_main_rejects_non_linux_before_once(monkeypatch: Any) -> None:
    output = io.StringIO()
    error = io.StringIO()
    client_factory_called = False

    def client_factory(_config: Any) -> FakeClient:
        nonlocal client_factory_called
        client_factory_called = True
        return FakeClient()

    monkeypatch.setattr("mistral4cli.cli.sys.platform", "win32")

    exit_code = main(
        ["--once", "Return only ok.", "--no-mcp"],
        stdin=FakeStdin(""),
        stdout=output,
        stderr=error,
        client_factory=client_factory,
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    assert error.getvalue() == LINUX_ONLY_MESSAGE + "\n"
    assert client_factory_called is False


def test_main_rejects_non_linux_before_interactive_start(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    error = io.StringIO()

    monkeypatch.setattr("mistral4cli.cli.sys.platform", "darwin")

    exit_code = main(
        ["--no-mcp"],
        stdin=FakeStdin("", tty=True),
        stdout=output,
        stderr=error,
        client_factory=lambda _config: FakeClient(),
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    assert error.getvalue() == LINUX_ONLY_MESSAGE + "\n"


def test_print_defaults_shows_mistral_small_4_defaults() -> None:
    output = io.StringIO()
    client_factory_called = False

    def client_factory(_config: Any) -> FakeClient:
        nonlocal client_factory_called
        client_factory_called = True
        return FakeClient()

    exit_code = main(
        ["--print-defaults", "--no-mcp"],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=client_factory,
    )

    assert exit_code == 0
    assert client_factory_called is False
    rendered = output.getvalue()
    assert "Mistral Small 4 CLI" in rendered
    assert "| Backend" in rendered
    assert "local" in rendered
    assert "Local OS tools: ready" in rendered
    assert "| Timeout" in rendered
    assert f"{DEFAULT_TIMEOUT_MS} ms" in rendered
    assert "temperature=0.7" in rendered
    assert "top_p=0.95" in rendered
    assert "prompt_mode=reasoning" in rendered
    assert "reasoning=on" in rendered
    assert "stream=on" in rendered
    assert "| Context" in rendered
    assert "auto_compact=on" in rendered
    assert "threshold=90%" in rendered
    assert "local_window=262144" in rendered
    assert "| Logging" in rendered
    assert "debug=on" in rendered
    assert f"retention={DEFAULT_LOG_RETENTION_DAYS}d" in rendered


def test_print_defaults_applies_context_cli_options() -> None:
    output = io.StringIO()

    exit_code = main(
        [
            "--print-defaults",
            "--no-mcp",
            "--no-auto-compact",
            "--compact-threshold",
            "85",
            "--context-reserve-tokens",
            "4096",
            "--context-keep-turns",
            "4",
        ],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: FakeClient(),
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "auto_compact=off" in rendered
    assert "threshold=85%" in rendered
    assert "reserve=4096" in rendered
    assert "keep_turns=4" in rendered


def test_print_defaults_applies_reasoning_cli_option() -> None:
    output = io.StringIO()

    exit_code = main(
        ["--print-defaults", "--no-mcp", "--no-reasoning"],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: FakeClient(),
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "reasoning=off" in rendered
    assert "thinking=on" in rendered


def test_once_uses_effective_defaults_and_prints_answer() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")

    exit_code = main(
        ["--once", "Return only the word ok.", "--no-stream", "--no-mcp"],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: fake_client,
    )

    assert exit_code == 0
    assert output.getvalue() == "ok\n"
    assert len(fake_client.chat.complete_calls) == 1
    call = fake_client.chat.complete_calls[0]
    assert call["temperature"] == 0.7
    assert call["top_p"] == 0.95
    assert call["prompt_mode"] == "reasoning"
    assert "max_tokens" not in call


def test_once_can_start_in_conversations_mode(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    fake_client = FakeConversationClient()
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    exit_code = main(
        [
            "--conversations",
            "--once",
            "Return only ok.",
            "--no-stream",
            "--no-mcp",
            "--conversation-index",
            str(tmp_path / "conversations.json"),
        ],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: fake_client,
    )

    assert exit_code == 0
    assert "ok\n" in output.getvalue()
    assert "Mistral Conversations returned no thinking blocks" in output.getvalue()
    assert len(fake_client.beta.conversations.start_calls) == 1
    call = fake_client.beta.conversations.start_calls[0]
    assert call["inputs"] == "Return only ok."
    assert call["model"] == "mistral-small-latest"
    assert call["store"] is True
    assert call["completion_args"]["reasoning_effort"] == "high"
    assert fake_client.chat.complete_calls == []


def test_once_can_start_in_conversations_mode_with_reasoning_disabled(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    fake_client = FakeConversationClient()
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    exit_code = main(
        [
            "--conversations",
            "--no-reasoning",
            "--once",
            "Return only ok.",
            "--no-stream",
            "--no-mcp",
            "--conversation-index",
            str(tmp_path / "conversations.json"),
        ],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: fake_client,
    )

    assert exit_code == 0
    call = fake_client.beta.conversations.start_calls[0]
    assert call["completion_args"]["reasoning_effort"] == "none"


def test_print_defaults_includes_conversation_resume_policy() -> None:
    output = io.StringIO()

    exit_code = main(
        ["--print-defaults", "--no-mcp", "--conversations"],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: FakeConversationClient(),
    )

    assert exit_code == 0
    assert "resume=last" in output.getvalue()


def test_once_conversations_applies_pending_creation_metadata(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    fake_client = FakeConversationClient()
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    exit_code = main(
        [
            "--conversations",
            "--conversation-name",
            "Release review",
            "--conversation-description",
            "Track rollout notes",
            "--conversation-meta",
            "ticket=OPS-42",
            "--conversation-meta",
            "owner=dlopez",
            "--once",
            "Return only ok.",
            "--no-stream",
            "--no-mcp",
            "--conversation-index",
            str(tmp_path / "conversations.json"),
        ],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: fake_client,
    )

    assert exit_code == 0
    call = fake_client.beta.conversations.start_calls[0]
    assert call["name"] == "Release review"
    assert call["description"] == "Track rollout notes"
    assert call["metadata"] == {"ticket": "OPS-42", "owner": "dlopez"}


def test_once_conversations_auto_resumes_last_registry_entry(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    registry = ConversationRegistry.load(tmp_path / "conversations.json")
    registry.update_remote_snapshot(
        "conv_1",
        remote_name="Saved remote conversation",
        remote_model="mistral-small-latest",
    )
    registry.remember_active("conv_1")
    fake_client = FakeConversationClient()
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    exit_code = main(
        [
            "--conversations",
            "--conversation-index",
            str(registry.path),
            "--once",
            "Return only ok.",
            "--no-stream",
            "--no-mcp",
        ],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: fake_client,
    )

    assert exit_code == 0
    assert fake_client.beta.conversations.start_calls == []
    assert len(fake_client.beta.conversations.append_calls) == 1
    assert fake_client.beta.conversations.append_calls[0]["conversation_id"] == "conv_1"


def test_main_creates_debug_log_file_by_default(tmp_path: Path) -> None:
    output = io.StringIO()

    exit_code = main(
        ["--no-mcp", "--log-dir", str(tmp_path)],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: FakeClient(),
    )

    assert exit_code == 0
    log_file = tmp_path / "mistral4cli.log"
    assert log_file.exists()
    rendered = log_file.read_text(encoding="utf-8")
    assert "INFO mistral4cli Logging configured" in rendered
    assert "INFO mistral4cli.cli CLI start" in rendered
    assert "DEBUG mistral4cli.cli Built tool bridge count=1" in rendered


def test_help_and_banner_are_actionable_and_retro() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )

    banner = render_welcome_banner(session.describe_defaults(), stream=output)
    help_text = render_help_screen(
        summary=session.describe_defaults(),
        tools=session.describe_tools().splitlines(),
        stream=output,
    )

    assert "Mistral4Small multimodal console" in banner
    assert "Type /help for actionable commands" in banner
    assert "Mistral cloud" in banner
    assert "+-" in banner
    assert "| Backend" in banner
    assert "| Model" in banner
    assert "/status" in help_text
    assert "/tools" in help_text
    assert "FireCrawl MCP" in help_text
    assert "/run" in help_text
    assert "/edit" in help_text
    assert "/find" in help_text
    assert "/ls" in help_text
    assert "/image" in help_text
    assert "/doc" in help_text
    assert "/drop" in help_text
    assert "/dropdoc" in help_text
    assert "/dropimage" in help_text
    assert "/remote" in help_text
    assert "/conv" in help_text
    assert "/compact" in help_text
    assert "/timeout" in help_text
    assert "/reasoning" in help_text
    assert "/thinking" in help_text
    assert "Search official documentation" in help_text
    assert "Describe this image and list all visible text." in help_text
    assert "Ctrl-C cancels the current response" in help_text


def test_default_system_prompt_hardens_tool_selection_rules() -> None:
    assert "This client is supported on Linux only." in DEFAULT_SYSTEM_PROMPT
    assert "shell is the primary tool for OS inspection" in DEFAULT_SYSTEM_PROMPT
    assert (
        "search_text is only for searching text inside files" in DEFAULT_SYSTEM_PROMPT
    )
    assert '"Check running nginx processes" -> shell.' in DEFAULT_SYSTEM_PROMPT
    assert '"Find files mentioning timeout in src/" -> search_text with path=src.' in (
        DEFAULT_SYSTEM_PROMPT
    )
    assert "Tool results are authoritative." in DEFAULT_SYSTEM_PROMPT


def test_session_backward_compatibility_alias_points_to_canonical_class() -> None:
    assert MistralCodingSession is MistralSession


def test_help_command_prints_without_pager_in_non_tty() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )

    should_exit = _run_command(
        "help",
        "",
        session,
        output,
        stdin=FakeStdin("q\n"),
    )

    rendered = output.getvalue()
    assert should_exit is False
    assert "[help] Press Enter for more, q to quit:" not in rendered
    assert "/exit" in rendered


def test_help_command_paginates_in_tty(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    monkeypatch.setenv("TERM", "xterm-256color")
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )
    monkeypatch.setattr(
        "mistral4cli.cli.shutil.get_terminal_size",
        lambda: os.terminal_size((100, 12)),
    )

    should_exit = _run_command(
        "help",
        "",
        session,
        output,
        stdin=FakeStdin("\n" * 10, tty=True),
    )

    rendered = output.getvalue()
    assert should_exit is False
    assert "[help] Press Enter for more, q to quit:" in rendered
    assert "/exit" in rendered


def test_help_command_can_quit_pager_early(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    monkeypatch.setenv("TERM", "xterm-256color")
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )
    monkeypatch.setattr(
        "mistral4cli.cli.shutil.get_terminal_size",
        lambda: os.terminal_size((100, 12)),
    )

    should_exit = _run_command(
        "help",
        "",
        session,
        output,
        stdin=FakeStdin("q\n", tty=True),
    )

    rendered = output.getvalue()
    assert should_exit is False
    assert "[help] Press Enter for more, q to quit:" in rendered
    assert "/exit" not in rendered


def test_tools_command_prints_without_pager_in_non_tty() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeLongToolBridge(),
        stdout=output,
    )

    should_exit = _run_command(
        "tools",
        "",
        session,
        output,
        stdin=FakeStdin("q\n"),
    )

    rendered = output.getvalue()
    assert should_exit is False
    assert "[tools] Press Enter for more, q to quit:" not in rendered
    assert "tool line 19" in rendered


def test_tools_command_paginates_in_tty(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    monkeypatch.setenv("TERM", "xterm-256color")
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeLongToolBridge(),
        stdout=output,
    )
    monkeypatch.setattr(
        "mistral4cli.cli.shutil.get_terminal_size",
        lambda: os.terminal_size((100, 8)),
    )

    should_exit = _run_command(
        "tools",
        "",
        session,
        output,
        stdin=FakeStdin("\n" * 10, tty=True),
    )

    rendered = output.getvalue()
    assert should_exit is False
    assert "[tools] Press Enter for more, q to quit:" in rendered
    assert "tool line 19" in rendered


def test_tools_command_can_quit_pager_early(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    monkeypatch.setenv("TERM", "xterm-256color")
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeLongToolBridge(),
        stdout=output,
    )
    monkeypatch.setattr(
        "mistral4cli.cli.shutil.get_terminal_size",
        lambda: os.terminal_size((100, 8)),
    )

    should_exit = _run_command(
        "tools",
        "",
        session,
        output,
        stdin=FakeStdin("q\n", tty=True),
    )

    rendered = output.getvalue()
    assert should_exit is False
    assert "[tools] Press Enter for more, q to quit:" in rendered
    assert "tool line 0" in rendered
    assert "tool line 19" not in rendered


def test_terminal_recommendation_prefers_xterm_256color(monkeypatch: Any) -> None:
    output = FakeTTYOutput()

    monkeypatch.setenv("TERM", "xterm-256color")

    assert terminal_recommendation(stream=output) == ""


def test_terminal_recommendation_warns_for_xterm(monkeypatch: Any) -> None:
    output = FakeTTYOutput()

    monkeypatch.setenv("TERM", "xterm")

    recommendation = terminal_recommendation(stream=output)

    assert "TERM=xterm-256color" in recommendation


def test_clear_screen_is_interactive_only(monkeypatch: Any) -> None:
    non_tty = io.StringIO()
    tty_output = FakeTTYOutput()

    monkeypatch.setenv("TERM", "xterm-256color")

    _clear_screen_if_supported(non_tty)
    _clear_screen_if_supported(tty_output)

    assert non_tty.getvalue() == ""
    assert tty_output.getvalue() == CLEAR_SCREEN


def test_refresh_repl_screen_clears_and_warns_before_banner(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    monkeypatch.setenv("TERM", "xterm")

    _refresh_repl_screen(output, session, startup=True)

    rendered = output.getvalue()
    assert rendered.startswith(CLEAR_SCREEN)
    assert "TERM=xterm-256color" in rendered
    assert "Mistral4Small multimodal console" in rendered


def test_tool_command_and_session_tool_loop() -> None:
    output = io.StringIO()
    tool_call = FakeToolCall(
        function=FakeToolFunction(name="web_search", arguments='{"query":"mcp"}')
    )
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(tool_calls=[tool_call]),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="Found a source."),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )
    bridge = FakeToolBridge()
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=bridge,
        stdout=output,
    )

    result = session.send("Busca MCP.", stream=False)

    assert result.cancelled is False
    assert result.finish_reason == "stop"
    assert result.content == "Found a source."
    assert bridge.calls == [("web_search", {"query": "mcp"})]
    assert session.messages[-2]["role"] == "tool"
    assert '{"results":' in session.messages[-2]["content"]
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "Found a source.",
    }


def test_textual_json_tool_call_fallback_executes_tool_loop() -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            content=(
                                "```json\n"
                                '{\n  "name": "web_search",\n'
                                '  "arguments": {"query": "mcp"}\n}\n'
                                "```"
                            )
                        ),
                        finish_reason="stop",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="Found a source."),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )
    bridge = FakeToolBridge()
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=bridge,
        stdout=output,
    )

    result = session.send("Busca MCP.", stream=False)

    assert result.cancelled is False
    assert result.finish_reason == "stop"
    assert result.content == "Found a source."
    assert bridge.calls == [("web_search", {"query": "mcp"})]
    assert "```json" not in output.getvalue()
    assert session.messages[-2]["role"] == "tool"
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "Found a source.",
    }


def test_textual_json_tool_call_fallback_executes_shell_tool(tmp_path: Path) -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            content=(
                                "```json\n"
                                "{\n"
                                '  "name": "shell",\n'
                                '  "arguments": {\n'
                                '    "command": "printf ok",\n'
                                '    "cwd": ".",\n'
                                '    "max_lines": 20\n'
                                "  }\n"
                                "}\n"
                                "```"
                            )
                        ),
                        finish_reason="stop",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="shell tool executed"),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    result = session.send("Run a shell command.", stream=False)

    assert result.cancelled is False
    assert result.finish_reason == "stop"
    assert result.content == "shell tool executed"
    assert session.messages[-2]["role"] == "tool"
    assert '"tool": "shell"' in session.messages[-2]["content"]
    assert "exit_code=0" in session.messages[-2]["content"]
    assert "ok" in session.messages[-2]["content"]
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "shell tool executed",
    }


def test_textual_json_tool_call_fallback_executes_read_file_tool(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    notes = tmp_path / "notes.txt"
    notes.write_text("hello from file\n", encoding="utf-8")
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            content=(
                                "{\n"
                                '  "name": "read_file",\n'
                                '  "arguments": {\n'
                                '    "path": "notes.txt"\n'
                                "  }\n"
                                "}"
                            )
                        ),
                        finish_reason="stop",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="read tool executed"),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    result = session.send("Read a file.", stream=False)

    assert result.cancelled is False
    assert result.finish_reason == "stop"
    assert result.content == "read tool executed"
    assert session.messages[-2]["role"] == "tool"
    assert '"tool": "read_file"' in session.messages[-2]["content"]
    assert "hello from file" in session.messages[-2]["content"]
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "read tool executed",
    }


def test_textual_json_tool_call_fallback_executes_search_text_tool(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    notes = tmp_path / "notes.txt"
    notes.write_text("needle\nhaystack\n", encoding="utf-8")
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            content=(
                                '{\n  "tool_calls": [\n'
                                "    {\n"
                                '      "name": "search_text",\n'
                                '      "arguments": {\n'
                                '        "query": "needle",\n'
                                '        "path": "."\n'
                                "      }\n"
                                "    }\n"
                                "  ]\n"
                                "}"
                            )
                        ),
                        finish_reason="stop",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="search tool executed"),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    result = session.send("Search the repo.", stream=False)

    assert result.cancelled is False
    assert result.finish_reason == "stop"
    assert result.content == "search tool executed"
    assert session.messages[-2]["role"] == "tool"
    assert "notes.txt" in session.messages[-2]["content"]
    assert "needle" in session.messages[-2]["content"]
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "search tool executed",
    }


def test_repeated_identical_tool_call_is_blocked(tmp_path: Path) -> None:
    output = io.StringIO()
    tool_call = FakeToolCall(
        function=FakeToolFunction(
            name="write_file",
            arguments='{"path":"notes.txt","content":"hello"}',
        )
    )
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(tool_calls=[tool_call]),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(tool_calls=[tool_call]),
                        finish_reason="tool_calls",
                    )
                ]
            ),
        ]
    )
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    result = session.send("Write a file.", stream=False)

    assert result.finish_reason == "error"
    assert result.cancelled is False
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert "repeated identical tool call blocked" in output.getvalue()
    assert '"code": "repeated_identical_tool_call"' in session.messages[-1]["content"]


def test_tool_loop_limit_forces_final_answer_after_last_tool(tmp_path: Path) -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            tool_calls=[
                                FakeToolCall(
                                    function=FakeToolFunction(
                                        name="list_dir",
                                        arguments='{"path":"/tmp","max_entries":50}',
                                    )
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            tool_calls=[
                                FakeToolCall(
                                    function=FakeToolFunction(
                                        name="list_dir",
                                        arguments='{"path":"/tmp","max_entries":100}',
                                    )
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            tool_calls=[
                                FakeToolCall(
                                    function=FakeToolFunction(
                                        name="shell",
                                        arguments=(
                                            '{"command":"ls -lt /tmp/*.txt '
                                            '2>/dev/null | head -20"}'
                                        ),
                                    )
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            tool_calls=[
                                FakeToolCall(
                                    function=FakeToolFunction(
                                        name="read_file",
                                        arguments='{"path":"/tmp/escritura.txt"}',
                                    )
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="Aqui tienes el texto final."),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    result = session.send("Lee el txt de hoy en /tmp.", stream=False)

    assert result.finish_reason == "stop"
    assert result.content == "Aqui tienes el texto final."
    assert "[error] tool loop limit reached" not in output.getvalue()
    assert len(fake_client.chat.complete_calls) == 5
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "Aqui tienes el texto final.",
    }


def test_stream_cancel_does_not_commit_partial_assistant_turn() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["hello", " world"], interrupt_after=1)
    session = MistralSession(
        client=fake_client, generation=LocalGenerationConfig(), stdout=output
    )

    result = session.send("Haz una respuesta larga.", stream=True)

    assert result.cancelled is True
    assert result.content == "hello"
    assert fake_client.chat.last_stream is not None
    assert fake_client.chat.last_stream.closed is True
    assert session.messages == [{"role": "system", "content": session.system_prompt}]
    assert "[interrupted]" in output.getvalue()


def test_streaming_textual_json_tool_call_fallback_executes_tool_loop() -> None:
    output = io.StringIO()
    bridge = FakeToolBridge()

    first_stream = FakeStream(
        [
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(content="```json\n"),
                            finish_reason=None,
                        )
                    ]
                )
            ),
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(
                                content=(
                                    '{\n  "name": "web_search",\n'
                                    '  "arguments": {"query": "mcp"}\n}\n```'
                                )
                            ),
                            finish_reason="stop",
                        )
                    ]
                )
            ),
        ]
    )
    second_stream = FakeStream(
        [
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(content="Found a source."),
                            finish_reason="stop",
                        )
                    ]
                )
            )
        ]
    )

    class SequencedChat:
        def __init__(self) -> None:
            self.stream_calls: list[dict[str, Any]] = []
            self.complete_calls: list[dict[str, Any]] = []
            self._streams = [first_stream, second_stream]

        def stream(self, **kwargs: Any) -> FakeStream:
            self.stream_calls.append(kwargs)
            return self._streams.pop(0)

    class SequencedClient:
        def __init__(self) -> None:
            self.chat = SequencedChat()

    session = MistralSession(
        client=SequencedClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=bridge,
        stdout=output,
    )

    result = session.send("Busca MCP.", stream=True)

    assert result.cancelled is False
    assert result.finish_reason == "stop"
    assert result.content == "Found a source."
    assert bridge.calls == [("web_search", {"query": "mcp"})]
    assert "```json" not in output.getvalue()
    assert "Found a source." in output.getvalue()


def test_non_stream_cancel_does_not_break_followup() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_interrupt_once=True, complete_text="ok")
    session = MistralSession(
        client=fake_client, generation=LocalGenerationConfig(), stdout=output
    )

    first = session.send("Haz una respuesta larga.", stream=False)
    second = session.send("Return only ok.", stream=False)

    assert first.cancelled is True
    assert first.finish_reason == "cancelled"
    assert first.content == ""
    assert second.cancelled is False
    assert second.finish_reason == "stop"
    assert second.content == "ok"
    assert session.messages[0] == {"role": "system", "content": session.system_prompt}
    assert session.messages[1]["role"] == "user"
    assert len([m for m in session.messages if m["role"] == "user"]) == 1
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}
    assert "[interrupted]" in output.getvalue()


def test_raw_stream_cancel_rolls_back_user_turn_and_allows_followup(
    monkeypatch: Any,
) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    responses = iter(
        [
            FakeRawStreamResponse(
                [
                    (
                        'data: {"choices":[{"finish_reason":null,"delta":'
                        '{"content":"hola"}}]}'
                    ),
                ]
            ),
            FakeRawHTTPResponse(
                '{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}'
            ),
        ]
    )

    monkeypatch.setattr(MistralSession, "_should_use_raw_chat", lambda self: True)

    def open_raw_request(self: MistralSession, payload: dict[str, Any]) -> object:
        response = next(responses)
        if isinstance(response, FakeRawStreamResponse):

            class InterruptingResponse(FakeRawStreamResponse):
                def __iter__(self_inner):
                    iterator = super().__iter__()
                    yield next(iterator)
                    raise KeyboardInterrupt

            return InterruptingResponse(
                [line.decode("utf-8").rstrip("\n") for line in response.lines]
            )
        return response

    monkeypatch.setattr(MistralSession, "_open_raw_request", open_raw_request)

    first = session.send("Primera pregunta.", stream=True)
    second = session.send("Return only ok.", stream=False)

    assert first.cancelled is True
    assert second.cancelled is False
    assert second.content == "ok"
    assert session.messages == [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "Return only ok."},
        {"role": "assistant", "content": "ok"},
    ]


def test_model_error_rolls_back_failed_turn() -> None:
    output = io.StringIO()

    class ErrorChat(FakeChat):
        def complete(self, **kwargs: Any) -> FakeResponse:
            self.complete_calls.append(kwargs)
            raise RuntimeError("boom")

    class ErrorClient:
        def __init__(self) -> None:
            self.chat = ErrorChat()

    session = MistralSession(
        client=ErrorClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("first prompt", stream=False)

    assert result.finish_reason == "error"
    assert result.cancelled is False
    assert session.messages == [
        {"role": "system", "content": session.system_prompt},
    ]
    assert "[error] boom" in output.getvalue()


def test_parse_command_supports_system_reset_and_tools() -> None:
    assert _parse_command("/system change the tone") == (
        "system",
        "change the tone",
    )
    assert _parse_command(":reset") == ("reset", "")
    assert _parse_command("/tools") == ("tools", "")
    assert _parse_command("/image --prompt describe") == (
        "image",
        "--prompt describe",
    )
    assert _parse_command("/doc --prompt resume") == ("doc", "--prompt resume")
    assert _parse_command("/drop") == ("drop", "")
    assert _parse_command("/dropdoc") == ("dropdoc", "")
    assert _parse_command("/dropimage") == ("dropimage", "")
    assert _parse_command("/remote on") == ("remote", "on")
    assert _parse_command("/conv on") == ("conv", "on")
    assert _parse_command("/conversations new") == ("conversations", "new")
    assert _parse_command("/compact threshold 85") == ("compact", "threshold 85")
    assert _parse_command("/status") == ("status", "")
    assert _parse_command("/timeout 5m") == ("timeout", "5m")
    assert _parse_command("/reasoning off") == ("reasoning", "off")
    assert _parse_command("/thinking off") == ("thinking", "off")
    assert _parse_command("/run --cwd . -- git status") == (
        "run",
        "--cwd . -- git status",
    )
    assert _parse_command("/find --path src -- shell") == (
        "find",
        "--path src -- shell",
    )


def test_compact_status_and_runtime_configuration_command() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        context=ContextConfig(
            threshold=0.9,
            reserve_tokens=8192,
            local_window_tokens=262_144,
        ),
        stdout=output,
    )

    _run_command("compact", "status", session, output)
    _run_command("compact", "threshold 85", session, output)
    _run_command("compact", "auto off", session, output)
    _run_command("compact", "reserve 4096", session, output)
    _run_command("compact", "keep 4", session, output)

    rendered = output.getvalue()
    assert "Context: client-managed" in rendered
    assert "threshold=90%" in rendered
    assert "Compact threshold set to 85%." in rendered
    assert "Auto compact set to off." in rendered
    assert "Context reserve set to 4096 tokens." in rendered
    assert "Compact keep turns set to 4." in rendered
    assert session.context.threshold == 0.85
    assert session.context.auto_compact is False
    assert session.context.reserve_tokens == 4096
    assert session.context.keep_recent_turns == 4


def test_status_command_shows_dynamic_session_snapshot() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(max_tokens=128),
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.conversation_id = "conv_123"
    session._last_usage = UsageSnapshot(total_tokens=321, max_context_tokens=256_000)
    session._cumulative_usage = UsageSnapshot(total_tokens=654)
    repl_state = _ReplState()
    repl_state.active_images = [Path("/tmp/example.png")]
    repl_state.active_documents = [Path("/tmp/example.pdf")]

    assert _run_command("status", "", session, output, repl_state=repl_state) is False

    rendered = output.getvalue()
    assert "Session status:" in rendered
    assert (
        "Runtime: backend=remote server=Mistral Cloud model=mistral-small-latest"
        in rendered
    )
    assert "Response: stream=on reasoning=on thinking=on timeout=300000ms" in rendered
    assert "Conversations: mode=on store=on resume=last id=conv_123" in rendered
    assert "Context: est:backend last:321/256000 usage:654" in rendered
    assert "Attachments: images=1 documents=1" in rendered


def test_manual_compact_summarizes_old_history_and_preserves_recent_turns() -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="Old decisions and paths."),
                        finish_reason="stop",
                    )
                ]
            )
        ]
    )
    session = MistralSession(
        client=fake_client,
        context=ContextConfig(keep_recent_turns=1),
        stdout=output,
    )
    session.messages = [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ]

    result = session.compact_context()

    assert result.changed is True
    assert fake_client.chat.complete_calls[0]["max_tokens"] == 2048
    assert "[Compacted previous context]" in session.messages[1]["content"]
    assert "Old decisions and paths." in session.messages[1]["content"]
    assert session.messages[-2:] == [
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ]


def test_auto_compact_runs_before_over_threshold_turn() -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="Short summary."),
                        finish_reason="stop",
                    )
                ]
            ),
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="final answer"),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )
    session = MistralSession(
        client=fake_client,
        context=ContextConfig(
            threshold=0.1,
            reserve_tokens=0,
            local_window_tokens=10_000,
            keep_recent_turns=1,
        ),
        stdout=output,
    )
    session.messages = [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "old " * 3000},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ]

    result = session.send("continue", stream=False)

    assert result.content == "final answer"
    assert len(fake_client.chat.complete_calls) == 2
    final_messages = fake_client.chat.complete_calls[1]["messages"]
    assert "[Compacted previous context]" in final_messages[1]["content"]
    assert final_messages[-2] == {"role": "user", "content": "continue"}
    assert "[compact] estimated context" in output.getvalue()


def test_context_overflow_blocks_turn_when_auto_compact_is_off() -> None:
    output = io.StringIO()
    fake_client = FakeClient()
    session = MistralSession(
        client=fake_client,
        context=ContextConfig(
            auto_compact=False,
            threshold=0.5,
            reserve_tokens=0,
            local_window_tokens=1200,
        ),
        stdout=output,
    )

    result = session.send("x" * 10_000, stream=False)

    assert result.finish_reason == "context_overflow"
    assert fake_client.chat.complete_calls == []
    assert "exceeds the configured context window" in output.getvalue()
    assert session.messages == [{"role": "system", "content": session.system_prompt}]


def test_non_conversations_store_off_still_sends_tools() -> None:
    output = io.StringIO()
    fake_client = FakeClient()
    session = MistralSession(
        client=fake_client,
        conversations=ConversationConfig(enabled=False, store=False),
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )

    result = session.send("search", stream=False)

    assert result.content == "ok"
    assert fake_client.chat.complete_calls
    assert fake_client.chat.complete_calls[0]["tools"][0]["function"]["name"] == (
        "web_search"
    )


def test_input_history_supports_up_down_navigation() -> None:
    history = _InputHistory()
    history.add("first")
    history.add("second")

    assert history.previous("") == "second"
    assert history.previous("") == "first"
    assert history.previous("") == "first"
    assert history.next() == "second"
    assert history.next() == ""
    assert history.next() == ""


def test_input_history_restores_draft_after_navigation() -> None:
    history = _InputHistory()
    history.add("first")
    history.add("second")

    assert history.previous("draft command") == "second"
    assert history.previous("ignored") == "first"
    assert history.next() == "second"
    assert history.next() == "draft command"


def test_input_history_skips_empty_and_consecutive_duplicates() -> None:
    history = _InputHistory()

    history.add("")
    history.add("first")
    history.add("first")
    history.add("second")

    assert history.entries == ["first", "second"]


def test_wrap_prompt_buffer_uses_continuation_prefix_for_long_input() -> None:
    lines = wrap_prompt_buffer(
        "M4S> ",
        (
            "This is a deliberately long prompt that should wrap into multiple "
            "display lines for the interactive composer."
        ),
        width=48,
    )

    assert len(lines) >= 2
    assert lines[0].startswith("M4S> ")
    assert lines[1].startswith("... ")


def test_paint_prompt_lines_styles_prompt_prefixes(monkeypatch: Any) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = FakeTTYOutput()

    painted = paint_prompt_lines(
        ["M4S> hola", "... mundo"],
        prompt="M4S> ",
        stream=output,
    )

    assert GREEN in painted[0]
    assert "M4S> hola" in ANSI_ESCAPE_RE.sub("", painted[0])
    assert GREEN in painted[1]
    assert "... mundo" in ANSI_ESCAPE_RE.sub("", painted[1])


def test_smart_output_writer_wraps_prose_without_breaking_words(
    monkeypatch: Any,
) -> None:
    output = io.StringIO()
    writer = SmartOutputWriter(stream=output)
    monkeypatch.setattr(
        "mistral4cli.ui.shutil.get_terminal_size",
        lambda: os.terminal_size((28, 24)),
    )

    rendered = writer.feed("Alpha beta gamma delta epsilon zeta eta theta iota kappa")
    rendered += writer.finish()

    lines = rendered.splitlines()
    assert len(lines) >= 2
    assert "epsi-\n" not in rendered
    assert all(len(line) <= 28 for line in lines)


def test_smart_output_writer_preserves_fenced_code_blocks(monkeypatch: Any) -> None:
    output = io.StringIO()
    writer = SmartOutputWriter(stream=output)
    monkeypatch.setattr(
        "mistral4cli.ui.shutil.get_terminal_size",
        lambda: os.terminal_size((20, 24)),
    )

    rendered = writer.feed(
        "```python\nvery_long_identifier = another_identifier\n```\n"
    )

    assert rendered == "```python\nvery_long_identifier = another_identifier\n```\n"


def test_smart_output_writer_colors_fenced_code_blocks_in_tty(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = FakeTTYOutput()
    writer = SmartOutputWriter(
        stream=output,
        literal_style=lambda text, active_stream: f"{CYAN}{text}{RESET}",
    )

    rendered = writer.feed("```python\nprint('ok')\n```\n")

    assert f"{CYAN}```python{RESET}" in rendered
    assert f"{CYAN}print('ok'){RESET}" in rendered
    assert f"{CYAN}```{RESET}" in rendered


def test_interactive_tty_renderer_keeps_prose_plain_and_colors_code(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = FakeTTYOutput()
    renderer = InteractiveTTYRenderer(
        stream=output,
        status_provider=lambda: "answering | local | model | est:- | last:- | usage:-",
    )

    renderer.write_answer("Before\n```python\nprint('ok')\n```\nAfter")
    renderer.finalize_output()

    rendered = output.getvalue()
    plain = ANSI_ESCAPE_RE.sub("", rendered)
    assert "Before" in plain
    assert "After" in plain
    assert "```python" in plain
    assert f"{CYAN}```python" in rendered
    assert f"{CYAN}print('ok')" in rendered


def test_smart_output_writer_can_render_markdown_rule_in_tty(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = FakeTTYOutput()
    writer = SmartOutputWriter(
        stream=output,
        markdown_rule_style=lambda text, active_stream: f"{ORANGE}{text}{RESET}",
        render_markdown_rules=True,
    )
    monkeypatch.setattr(
        "mistral4cli.ui.shutil.get_terminal_size",
        lambda: os.terminal_size((20, 24)),
    )

    rendered = writer.feed("Before\n---\nAfter\n")
    plain = ANSI_ESCAPE_RE.sub("", rendered)

    assert "Before" in plain
    assert "After" in plain
    assert "\n-------------------\n" in plain
    assert f"{ORANGE}-------------------{RESET}" in rendered


def test_markdown_rule_is_not_rendered_inside_fence(monkeypatch: Any) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = FakeTTYOutput()
    writer = SmartOutputWriter(
        stream=output,
        literal_style=lambda text, active_stream: f"{CYAN}{text}{RESET}",
        markdown_rule_style=lambda text, active_stream: f"{ORANGE}{text}{RESET}",
        render_markdown_rules=True,
    )

    rendered = writer.feed("```md\n---\n```\n")
    plain = ANSI_ESCAPE_RE.sub("", rendered)

    assert "---" in plain
    assert "-------------------" not in plain


def test_iter_typewriter_chunks_preserves_ansi_sequences_and_newlines() -> None:
    text = f"{ORANGE}hola{RESET}\nmundo"

    chunks = iter_typewriter_chunks(text, visible_chars=2)

    assert "".join(chunks) == text
    assert any(chunk.endswith("\n") for chunk in chunks)
    assert any(ORANGE in chunk for chunk in chunks)


def test_renderer_typewriter_batches_multichunk_answer(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    renderer = InteractiveTTYRenderer(
        stream=output,
        status_provider=(
            lambda: "answering | local | model | est:- | last:- | usage:-"
        ),
    )
    pauses: list[float] = []
    monkeypatch.setattr("mistral4cli.ui.ANSWER_TYPEWRITER_VISIBLE_CHARS", 4)
    monkeypatch.setattr("mistral4cli.ui.ANSWER_TYPEWRITER_PAUSE_S", 0.001)
    monkeypatch.setattr("mistral4cli.ui.time.sleep", pauses.append)

    renderer.write_answer("abcdefghijkl")
    renderer.finalize_output()

    assert output.getvalue() == "abcdefghijkl"
    assert pauses == [0.001, 0.001]


def test_renderer_typewriter_skips_pause_after_newline(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    renderer = InteractiveTTYRenderer(
        stream=output,
        status_provider=(
            lambda: "answering | local | model | est:- | last:- | usage:-"
        ),
    )
    pauses: list[float] = []
    monkeypatch.setattr("mistral4cli.ui.ANSWER_TYPEWRITER_VISIBLE_CHARS", 4)
    monkeypatch.setattr("mistral4cli.ui.ANSWER_TYPEWRITER_PAUSE_S", 0.001)
    monkeypatch.setattr("mistral4cli.ui.time.sleep", pauses.append)

    renderer.write_answer("abcd\nefgh")
    renderer.finalize_output()

    assert output.getvalue() == "abcd\nefgh"
    assert pauses == []


def test_repl_status_line_shows_phase_attachments_and_usage() -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=FakeUsage(prompt_tokens=12, completion_tokens=6, total_tokens=18),
            )
        ]
    )
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    repl_state = _ReplState(
        pending_attachment=_PendingAttachment(
            kind="document",
            summary="w3c-dummy.pdf",
            paths=[FIXTURE_DIR / "w3c-dummy.pdf"],
        ),
        active_images=[FIXTURE_DIR / "wikimedia-demo.png"],
    )

    session.send("Return only ok.", stream=False)

    context_status = session.context_status()
    rendered = _repl_status_line(session, repl_state)
    assert "done" in rendered
    assert "stage:document" in rendered
    assert "img:1" in rendered
    assert (
        f"est:{context_status.estimated_tokens}/{context_status.window_tokens}"
        in rendered
    )
    assert "last:18/256000" in rendered
    assert "usage:18" in rendered


def test_repl_status_line_marks_conversations_context_as_backend_managed() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        conversations=ConversationConfig(enabled=True, store=True),
        stdout=output,
    )

    rendered = _repl_status_line(session, _ReplState())

    assert "conv:on" in rendered
    assert "est:backend" in rendered
    assert "last:-" in rendered
    assert "usage:-" in rendered


def test_status_bar_leaves_one_column_to_avoid_terminal_autowrap(
    monkeypatch: Any,
) -> None:
    output = FakeTTYOutput()
    renderer = InteractiveTTYRenderer(
        stream=output,
        status_provider=(
            lambda: (
                "idle | local | model | reasoning:on | thinking:on | "
                "est:- | last:- | usage:-"
            )
        ),
    )
    monkeypatch.setattr(
        "mistral4cli.ui.shutil.get_terminal_size",
        lambda: os.terminal_size((20, 24)),
    )

    rendered = renderer.render_status_bar()
    plain = ANSI_ESCAPE_RE.sub("", rendered)

    assert len(plain) == 19


def test_render_input_omits_status_bar_while_user_is_typing(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = FakeTTYOutput()
    renderer = InteractiveTTYRenderer(
        stream=output,
        status_provider=(
            lambda: ("thinking... | local | model | est:- | last:- | usage:-")
        ),
    )
    monkeypatch.setattr(
        "mistral4cli.ui.shutil.get_terminal_size",
        lambda: os.terminal_size((40, 24)),
    )

    renderer.render_input("M4S> ", "hola")

    rendered = output.getvalue()
    plain = ANSI_ESCAPE_RE.sub("", rendered)
    assert "M4S> hola" in plain
    assert "thinking..." not in plain
    assert GREEN in rendered


def test_renderer_flushes_pending_reasoning_before_answer(
    monkeypatch: Any,
) -> None:
    output = FakeTTYOutput()
    fake_client = FakeClient(complete_text="<think>plan first</think>ok")
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    renderer = InteractiveTTYRenderer(
        stream=output,
        status_provider=(
            lambda: "answering | local | model | est:- | last:- | usage:-"
        ),
    )
    session.answer_writer = renderer.write_answer
    session.reasoning_writer = renderer.write_reasoning
    monkeypatch.setattr("mistral4cli.ui._terminal_width", lambda *_args, **_kwargs: 200)

    result = session.send("Return ok.", stream=False)
    renderer.finalize_output()

    plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
    assert result.reasoning == "plan first"
    assert result.content == "ok"
    assert "plan first\n\nok\n" in plain


def test_write_tty_newline_emits_crlf() -> None:
    output = io.StringIO()

    _write_tty_newline(output)

    assert output.getvalue() == "\r\n"


def test_repl_quit_with_renderer_restores_column_zero(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    output = FakeTTYOutput()
    monkeypatch.setenv("TERM", "xterm-256color")
    session = MistralSession(
        client=FakeClient(complete_text="ok"),
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )
    monkeypatch.setattr("mistral4cli.cli._is_default_input_func", lambda _func: True)
    monkeypatch.setattr(
        "mistral4cli.cli._read_repl_line",
        lambda **_kwargs: "/quit",
    )

    exit_code = _run_repl(
        session,
        local_config=LocalMistralConfig(),
        client_factory=lambda _config: FakeClient(),
        input_func=lambda _prompt: "/quit",
        stdin=FakeStdin("", tty=True),
        stdout=output,
        stream=False,
        path_picker=None,
    )

    assert exit_code == 0
    assert output.getvalue().endswith("\r")


def test_tty_repl_quit_exits_without_lingering_process_group(tmp_path: Path) -> None:
    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"] = "xterm-256color"
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = (
        f"{src_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mistral4cli",
            "--no-mcp",
            "--log-dir",
            str(tmp_path),
        ],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)
    prompt_seen = False
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd not in ready:
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(master_fd, 65536)
            if not chunk:
                break
            if b"M4S>" in chunk:
                prompt_seen = True
                break
        assert prompt_seen, "interactive prompt did not appear before timeout"

        os.write(master_fd, b"/quit\n")
        assert process.wait(timeout=10) == 0

        time.sleep(0.2)
        group_scan = subprocess.run(
            ["ps", "-o", "pid=", "-g", str(process.pid)],
            capture_output=True,
            text=True,
        )
        assert group_scan.returncode != 0 or group_scan.stdout.strip() == ""
    finally:
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        finally:
            os.close(master_fd)


def test_remote_request_uses_reasoning_effort_and_omits_prompt_mode() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return only ok.", stream=False)

    assert result.content == "ok"
    assert "prompt_mode" not in fake_client.chat.complete_calls[0]
    assert fake_client.chat.complete_calls[0]["reasoning_effort"] == "high"


def test_remote_non_stream_usage_is_tracked_in_status_snapshot() -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=FakeUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        ]
    )
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return only ok.", stream=False)

    snapshot = session.status_snapshot()
    assert result.content == "ok"
    assert snapshot.last_usage is not None
    assert snapshot.last_usage.total_tokens == 15
    assert snapshot.last_usage.max_context_tokens == 256_000
    assert snapshot.cumulative_usage is not None
    assert snapshot.cumulative_usage.total_tokens == 15


def test_remote_stream_usage_is_tracked_in_status_snapshot() -> None:
    output = io.StringIO()

    class UsageStreamChat:
        def __init__(self) -> None:
            self.complete_calls: list[dict[str, Any]] = []
            self.stream_calls: list[dict[str, Any]] = []
            self.last_stream = FakeStream(
                [
                    FakeEvent(
                        data=FakeResponse(
                            choices=[
                                FakeChoice(
                                    delta=FakeDelta(content="ok"),
                                    finish_reason="stop",
                                )
                            ],
                            usage=FakeUsage(
                                prompt_tokens=20,
                                completion_tokens=10,
                                total_tokens=30,
                            ),
                        )
                    )
                ]
            )

        def stream(self, **kwargs: Any) -> FakeStream:
            self.stream_calls.append(kwargs)
            return self.last_stream

    class UsageStreamClient:
        def __init__(self) -> None:
            self.chat = UsageStreamChat()

    session = MistralSession(
        client=UsageStreamClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return only ok.", stream=True)

    snapshot = session.status_snapshot()
    assert result.content == "ok"
    assert snapshot.last_usage is not None
    assert snapshot.last_usage.total_tokens == 30
    assert snapshot.cumulative_usage is not None
    assert snapshot.cumulative_usage.total_tokens == 30


def test_remote_request_disables_reasoning_effort_when_hidden() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=False,
    )

    result = session.send("Return only ok.", stream=False)

    assert result.content == "ok"
    assert "prompt_mode" not in fake_client.chat.complete_calls[0]
    assert fake_client.chat.complete_calls[0]["reasoning_effort"] == "none"


def test_remote_request_keeps_reasoning_when_thinking_is_hidden() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=True,
        show_thinking=False,
    )

    result = session.send("Return only ok.", stream=False)

    assert result.content == "ok"
    assert fake_client.chat.complete_calls[0]["reasoning_effort"] == "high"


def test_remote_stream_request_uses_reasoning_effort_and_omits_prompt_mode() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["ok"])
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return only ok.", stream=True)

    assert result.content == "ok"
    assert len(fake_client.chat.stream_calls) == 1
    call = fake_client.chat.stream_calls[0]
    assert "prompt_mode" not in call
    assert call["reasoning_effort"] == "high"


def test_remote_stream_request_disables_reasoning_effort_when_hidden() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["ok"])
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=False,
    )

    result = session.send("Return only ok.", stream=True)

    assert result.content == "ok"
    assert len(fake_client.chat.stream_calls) == 1
    call = fake_client.chat.stream_calls[0]
    assert "prompt_mode" not in call
    assert call["reasoning_effort"] == "none"


def test_conversations_session_starts_then_appends() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "first"}],
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "second"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    first = session.send("hello", stream=False)
    second = session.send("again", stream=False)

    assert first.content == "first"
    assert second.content == "second"
    assert session.conversation_id == "conv_1"
    assert conversations.start_calls[0]["inputs"] == "hello"
    assert conversations.start_calls[0]["completion_args"]["reasoning_effort"] == "high"
    assert conversations.append_calls[0]["conversation_id"] == "conv_1"
    assert conversations.append_calls[0]["inputs"] == "again"
    assert (
        conversations.append_calls[0]["completion_args"]["reasoning_effort"] == "high"
    )


def test_conversations_session_start_and_append_disable_reasoning_effort() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "first"}],
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "second"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        show_reasoning=False,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.set_reasoning_visibility(False)

    first = session.send("hello", stream=False)
    second = session.send("again", stream=False)

    assert first.content == "first"
    assert second.content == "second"
    assert conversations.start_calls[0]["completion_args"]["reasoning_effort"] == "none"
    assert (
        conversations.append_calls[0]["completion_args"]["reasoning_effort"] == "none"
    )


def test_conversations_keep_reasoning_effort_when_thinking_is_hidden() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_hidden_thinking",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "ok"}],
                    )
                ],
            )
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        show_reasoning=True,
        show_thinking=False,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("hello", stream=False)

    assert result.content == "ok"
    assert conversations.start_calls[0]["completion_args"]["reasoning_effort"] == "high"
    assert "Mistral Conversations returned no thinking blocks" not in output.getvalue()


def test_conversations_streaming_records_usage_and_text() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        stream_events=[
            FakeConversationEvent(
                event="conversation.response.started",
                data=FakeConversationStarted(conversation_id="conv_stream"),
            ),
            FakeConversationEvent(
                event="message.output.delta",
                data=FakeConversationMessageDelta(
                    content={"type": "text", "text": "ok"}
                ),
            ),
            FakeConversationEvent(
                event="conversation.response.done",
                data=FakeConversationDone(
                    usage=FakeUsage(
                        prompt_tokens=10,
                        completion_tokens=2,
                        total_tokens=12,
                    )
                ),
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("hello", stream=True)

    assert result.content == "ok"
    assert "ok\n" in output.getvalue()
    assert "Mistral Conversations returned no thinking blocks" in output.getvalue()
    assert session.conversation_id == "conv_stream"
    assert (
        conversations.start_stream_calls[0]["completion_args"]["reasoning_effort"]
        == "high"
    )
    assert session.status_snapshot().last_usage is not None
    assert session.status_snapshot().last_usage.total_tokens == 12


def test_conversations_append_stream_disables_reasoning_effort_when_hidden() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        stream_events=[
            FakeConversationEvent(
                event="message.output.delta",
                data=FakeConversationMessageDelta(
                    content={"type": "text", "text": "ok"}
                ),
            ),
            FakeConversationEvent(
                event="conversation.response.done",
                data=FakeConversationDone(
                    usage=FakeUsage(
                        prompt_tokens=8,
                        completion_tokens=2,
                        total_tokens=10,
                    )
                ),
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        show_reasoning=False,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.set_reasoning_visibility(False)
    session.conversation_id = "conv_stream"

    result = session.send("hello", stream=True)

    assert result.content == "ok"
    assert conversations.append_stream_calls[0]["conversation_id"] == "conv_stream"
    assert (
        conversations.append_stream_calls[0]["completion_args"]["reasoning_effort"]
        == "none"
    )


def test_conversations_tool_call_executes_bridge_and_appends_result() -> None:
    output = io.StringIO()
    tool_bridge = FakeToolBridge()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="function.call",
                        tool_call_id="tool_call_1",
                        name="web_search",
                        arguments='{"query":"mistral"}',
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "done"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        tool_bridge=tool_bridge,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("search", stream=False)

    assert result.content == "done"
    assert tool_bridge.calls == [("web_search", {"query": "mistral"})]
    assert conversations.append_calls[0]["inputs"][0]["type"] == "function.result"
    assert conversations.append_calls[0]["inputs"][0]["tool_call_id"] == "tool_call_1"


def test_conversations_tool_interrupt_appends_cancel_result_and_recovers() -> None:
    output = io.StringIO()
    tool_bridge = InterruptingToolBridge()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="function.call",
                        tool_call_id="tool_call_1",
                        name="web_search",
                        arguments='{"query":"mistral"}',
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "cancelled cleanup"}],
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "recovered"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        tool_bridge=tool_bridge,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    first = session.send("search", stream=False)
    second = session.send("try again", stream=False)

    assert first.cancelled is True
    assert first.finish_reason == "cancelled"
    assert second.cancelled is False
    assert second.content == "recovered"
    assert conversations.append_calls[0]["inputs"][0]["type"] == "function.result"
    assert conversations.append_calls[0]["inputs"][0]["tool_call_id"] == "tool_call_1"
    assert "cancelled_by_user" in conversations.append_calls[0]["inputs"][0]["result"]
    assert conversations.append_calls[1]["inputs"] == "try again"


def test_conversations_stream_interrupt_with_tool_call_appends_cancel_result() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        stream_events=[
            FakeConversationEvent(
                "conversation.response.started",
                FakeConversationStarted("conv_tools"),
            ),
            FakeConversationEvent(
                "function.call.delta",
                FakeConversationFunctionDelta(
                    tool_call_id="tool_call_1",
                    name="web_search",
                    arguments='{"query":"mistral"}',
                ),
            ),
            FakeConversationEvent(
                "conversation.response.done",
                FakeConversationDone(),
            ),
        ],
        responses=[
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "cleanup"}],
                    )
                ],
            )
        ],
        stream_interrupt_after=2,
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("search", stream=True)

    assert result.cancelled is True
    assert result.finish_reason == "cancelled"
    assert conversations.append_calls[0]["inputs"][0]["type"] == "function.result"
    assert conversations.append_calls[0]["inputs"][0]["tool_call_id"] == "tool_call_1"
    assert "cancelled_by_user" in conversations.append_calls[0]["inputs"][0]["result"]


def test_remote_command_requires_api_key(monkeypatch: Any) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    should_exit = _run_command(
        "remote",
        "on",
        session,
        output,
        local_config=LocalMistralConfig(),
        client_factory=lambda _config: FakeClient(),
    )

    assert should_exit is False
    assert session.backend_kind is BackendKind.LOCAL
    assert "[remote] Set MISTRAL_API_KEY" in output.getvalue()


def test_conversations_command_requires_api_key(monkeypatch: Any) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    should_exit = _run_command(
        "conv",
        "on",
        session,
        output,
        repl_state=_ReplState(),
        local_config=LocalMistralConfig(),
        client_factory=lambda _config: FakeConversationClient(),
    )

    assert should_exit is False
    assert session.conversations.enabled is False
    assert "[conversations] Set MISTRAL_API_KEY" in output.getvalue()


def test_conversations_command_enables_and_resets(monkeypatch: Any) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )
    session.messages.append({"role": "user", "content": "stale"})
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    should_exit = _run_command(
        "conv",
        "on",
        session,
        output,
        repl_state=_ReplState(),
        local_config=LocalMistralConfig(),
        client_factory=lambda _config: FakeConversationClient(),
    )

    assert should_exit is False
    assert session.conversations.enabled is True
    assert session.backend_kind is BackendKind.REMOTE
    assert session.messages == [{"role": "system", "content": session.system_prompt}]
    assert "Conversations enabled. Conversation reset." in output.getvalue()


def test_conversations_command_store_new_history_and_delete() -> None:
    output = io.StringIO()
    conversations = FakeConversations()
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.conversation_id = "conv_1"

    assert _run_command("conv", "store off", session, output) is False
    assert session.conversations.store is False
    assert session.conversation_id is None
    session.conversation_id = "conv_1"
    assert (
        _run_command("conv", "history", session, output, stdin=FakeStdin("")) is False
    )
    assert "message.input user id=entry_1: hello" in output.getvalue()
    assert '"/conv restart <entry_id>"' in output.getvalue()
    assert (
        _run_command("conv", "messages", session, output, stdin=FakeStdin("")) is False
    )
    assert "message.output assistant" in output.getvalue()
    assert _run_command("conv", "delete", session, output) is False
    assert conversations.delete_calls == [{"conversation_id": "conv_1"}]
    assert session.conversation_id is None


def test_conversations_command_list_show_and_use(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        entities=[
            FakeConversationEntity(
                id="conv_1",
                name="Primary conversation",
                description="Tracked in tests",
                metadata={"suite": "cli"},
            ),
            FakeConversationEntity(
                id="conv_2",
                name="Other conversation",
                description="Should be filtered out",
                metadata={"suite": "other"},
            ),
        ]
    )
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
        conversation_registry=ConversationRegistry.load(
            tmp_path / "conversations.json"
        ),
    )
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    assert (
        _run_command(
            "conv",
            "list --page 0 --size 5 --meta suite=cli",
            session,
            output,
            repl_state=_ReplState(),
            local_config=LocalMistralConfig(),
            client_factory=lambda _config: FakeConversationClient(conversations),
            stdin=FakeStdin(""),
        )
        is False
    )
    assert conversations.list_calls == [{"page": 0, "page_size": 5, "metadata": {}}]
    assert conversations.get_calls[:2] == [
        {"conversation_id": "conv_1"},
        {"conversation_id": "conv_2"},
    ]
    assert "Primary conversation" in output.getvalue()
    assert "Other conversation" not in output.getvalue()
    output.seek(0)
    output.truncate(0)

    assert (
        _run_command(
            "conv",
            "show conv_1",
            session,
            output,
            repl_state=_ReplState(),
            local_config=LocalMistralConfig(),
            client_factory=lambda _config: FakeConversationClient(conversations),
            stdin=FakeStdin(""),
        )
        is False
    )
    assert "Tracked in tests" in output.getvalue()
    output.seek(0)
    output.truncate(0)

    assert (
        _run_command(
            "conv",
            "use conv_1",
            session,
            output,
            repl_state=_ReplState(),
            local_config=LocalMistralConfig(),
            client_factory=lambda _config: FakeConversationClient(conversations),
            stdin=FakeStdin(""),
        )
        is False
    )
    assert session.conversation_id == "conv_1"


def test_conversations_command_list_without_metadata_omits_filter(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    conversations = FakeConversations()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
        conversation_registry=ConversationRegistry.load(
            tmp_path / "conversations.json"
        ),
    )
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    assert (
        _run_command(
            "conv",
            "list",
            session,
            output,
            repl_state=_ReplState(),
            local_config=LocalMistralConfig(),
            client_factory=lambda _config: FakeConversationClient(conversations),
            stdin=FakeStdin(""),
        )
        is False
    )
    assert conversations.list_calls == [{"page": 0, "page_size": 20, "metadata": {}}]
    assert "Remote conversations page=0 size=20:" in output.getvalue()


def test_conversations_command_handles_datetime_timestamps(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    timestamp = datetime(2026, 4, 28, 9, 30, tzinfo=timezone.utc)
    conversations = FakeConversations(
        entities=[
            FakeConversationEntity(
                id="conv_1",
                name="Primary conversation",
                created_at=timestamp,
                updated_at=timestamp,
            )
        ]
    )
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
        conversation_registry=ConversationRegistry.load(
            tmp_path / "conversations.json"
        ),
    )
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    assert (
        _run_command(
            "conv",
            "list",
            session,
            output,
            repl_state=_ReplState(),
            local_config=LocalMistralConfig(),
            client_factory=lambda _config: FakeConversationClient(conversations),
            stdin=FakeStdin(""),
        )
        is False
    )
    assert "Primary conversation" in output.getvalue()

    record = session.conversation_registry.get("conv_1")
    assert record is not None
    assert record.created_at == timestamp.isoformat()
    assert record.updated_at == timestamp.isoformat()


def test_conversations_command_pending_settings_and_bookmarks(tmp_path: Path) -> None:
    output = io.StringIO()
    registry = ConversationRegistry.load(tmp_path / "conversations.json")
    session = MistralSession(
        client=FakeConversationClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        conversation_registry=registry,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.conversation_id = "conv_1"
    registry.remember_active("conv_1")

    assert _run_command("conv", "set name Release review", session, output) is False
    assert _run_command("conv", "set meta owner=dlopez", session, output) is False
    assert session.pending_conversation.name == "Release review"
    assert session.pending_conversation.metadata == {"owner": "dlopez"}
    assert _run_command("conv", "alias release-review", session, output) is False
    assert _run_command("conv", "alias conv_1 release-review", session, output) is False
    assert _run_command("conv", "tag add conv_1 ops", session, output) is False
    output.seek(0)
    output.truncate(0)

    assert (
        _run_command(
            "conv",
            "bookmarks",
            session,
            output,
            stdin=FakeStdin(""),
        )
        is False
    )
    rendered = output.getvalue()
    assert "release-review" in rendered
    assert "ops" in rendered


def test_conversations_command_use_resolves_local_alias(tmp_path: Path) -> None:
    output = io.StringIO()
    conversations = FakeConversations()
    registry = ConversationRegistry.load(tmp_path / "conversations.json")
    registry.set_alias("conv_1", "release-review")
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        conversation_registry=registry,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    assert (
        _run_command(
            "conv",
            "use release-review",
            session,
            output,
            repl_state=_ReplState(),
        )
        is False
    )
    assert conversations.get_calls[-1] == {"conversation_id": "conv_1"}
    assert session.conversation_id == "conv_1"
    assert "Attached to conversation conv_1." in output.getvalue()


def test_conversations_command_alias_without_active_id_errors(tmp_path: Path) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeConversationClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        conversation_registry=ConversationRegistry.load(
            tmp_path / "conversations.json"
        ),
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    assert _run_command("conv", "alias release-review", session, output) is False
    assert "No active conversation id yet." in output.getvalue()


def test_conversations_command_status_id_store_and_new(tmp_path: Path) -> None:
    output = io.StringIO()
    registry = ConversationRegistry.load(tmp_path / "conversations.json")
    session = MistralSession(
        client=FakeConversationClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        conversation_registry=registry,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.conversation_id = "conv_1"
    registry.remember_active("conv_1")

    assert _run_command("conv", "", session, output) is False
    assert "Conversations: on store=on resume=last" in output.getvalue()
    output.seek(0)
    output.truncate(0)

    assert _run_command("conv", "current", session, output) is False
    assert "conversation_id=conv_1" in output.getvalue()
    output.seek(0)
    output.truncate(0)

    assert _run_command("conv", "status", session, output) is False
    assert "conversation_id=conv_1" in output.getvalue()
    output.seek(0)
    output.truncate(0)

    assert _run_command("conv", "id", session, output) is False
    assert output.getvalue().strip() == "conv_1"
    output.seek(0)
    output.truncate(0)

    assert _run_command("conv", "store maybe", session, output) is False
    assert "Usage: /conversations store [on|off]" in output.getvalue()
    output.seek(0)
    output.truncate(0)

    assert _run_command("conv", "new", session, output) is False
    assert session.conversation_id is None
    assert registry.last_active_conversation_id == ""
    assert "New Conversation will start on the next turn." in output.getvalue()
    output.seek(0)
    output.truncate(0)

    assert _run_command("conv", "off", session, output) is False
    assert session.conversations.enabled is False
    assert "Conversations disabled. Conversation reset." in output.getvalue()


def test_conversations_command_note_tag_remove_unset_and_forget(tmp_path: Path) -> None:
    output = io.StringIO()
    registry = ConversationRegistry.load(tmp_path / "conversations.json")
    session = MistralSession(
        client=FakeConversationClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        conversation_registry=registry,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.conversation_id = "conv_1"
    registry.remember_active("conv_1")
    registry.set_alias("conv_1", "primary")
    registry.add_tag("conv_1", "ops")

    assert (
        _run_command("conv", "note primary keep this thread", session, output) is False
    )
    assert registry.get("conv_1").note == "keep this thread"
    assert _run_command("conv", "tag remove primary ops", session, output) is False
    assert registry.get("conv_1").tags == []
    assert _run_command("conv", "set name Release review", session, output) is False
    assert (
        _run_command("conv", "set description Track rollout notes", session, output)
        is False
    )
    assert (
        _run_command("conv", "set meta owner=dlopez env=prod", session, output) is False
    )
    assert session.pending_conversation.name == "Release review"
    assert session.pending_conversation.description == "Track rollout notes"
    assert session.pending_conversation.metadata == {
        "owner": "dlopez",
        "env": "prod",
    }
    assert _run_command("conv", "unset name", session, output) is False
    assert session.pending_conversation.name == ""
    assert _run_command("conv", "unset meta owner", session, output) is False
    assert session.pending_conversation.metadata == {"env": "prod"}
    assert _run_command("conv", "unset description", session, output) is False
    assert session.pending_conversation.description == ""
    assert _run_command("conv", "unset all", session, output) is False
    assert session.pending_conversation.active() is False
    assert _run_command("conv", "forget primary", session, output) is False
    assert registry.get("conv_1") is None


def test_conversations_command_restart_switches_to_new_conversation(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_branch",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "branched"}],
                    )
                ],
            )
        ],
        entities=[
            FakeConversationEntity(id="conv_1"),
            FakeConversationEntity(id="conv_branch", description="Branch"),
        ],
    )
    registry = ConversationRegistry.load(tmp_path / "conversations.json")
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        conversation_registry=registry,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.conversation_id = "conv_1"
    registry.remember_active("conv_1")

    assert _run_command("conv", "restart entry_1", session, output) is False
    assert conversations.restart_calls[0]["conversation_id"] == "conv_1"
    assert conversations.restart_calls[0]["from_entry_id"] == "entry_1"
    assert conversations.restart_calls[0]["inputs"] == ""
    assert session.conversation_id == "conv_branch"
    assert registry.get("conv_branch") is not None
    assert registry.get("conv_branch").parent_conversation_id == "conv_1"
    assert "Active conversation: conv_branch." in output.getvalue()


def test_remote_command_switches_backend_and_resets_conversation(
    monkeypatch: Any,
) -> None:
    output = io.StringIO()
    captured: list[object] = []
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    session.messages.append({"role": "user", "content": "stale"})
    monkeypatch.setenv("MISTRAL_API_KEY", "example-mistral-key")

    def client_factory(config: object) -> FakeClient:
        captured.append(config)
        return FakeClient()

    should_exit = _run_command(
        "remote",
        "on",
        session,
        output,
        local_config=LocalMistralConfig(),
        client_factory=client_factory,
    )

    assert should_exit is False
    assert len(captured) == 1
    assert isinstance(captured[0], RemoteMistralConfig)
    assert session.backend_kind is BackendKind.REMOTE
    assert session.model_id == "mistral-small-latest"
    assert session.server_url is None
    assert session.messages == [
        {"role": "system", "content": session.system_prompt},
    ]
    assert "Remote backend enabled. Conversation reset." in output.getvalue()
    assert "| Backend" in output.getvalue()
    assert "remote" in output.getvalue()
    assert "Mistral Cloud" in output.getvalue()


def test_remote_command_clears_screen_in_tty(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    monkeypatch.setenv("MISTRAL_API_KEY", "example-mistral-key")
    monkeypatch.setenv("TERM", "xterm-256color")

    should_exit = _run_command(
        "remote",
        "on",
        session,
        output,
        local_config=LocalMistralConfig(),
        client_factory=lambda _config: FakeClient(),
    )

    assert should_exit is False
    assert output.getvalue().startswith(CLEAR_SCREEN)


def test_remote_off_switches_back_to_local() -> None:
    output = io.StringIO()
    captured: list[object] = []
    local_config = LocalMistralConfig()
    session = MistralSession(
        client=FakeClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    def client_factory(config: object) -> FakeClient:
        captured.append(config)
        return FakeClient()

    should_exit = _run_command(
        "remote",
        "off",
        session,
        output,
        local_config=local_config,
        client_factory=client_factory,
    )

    assert should_exit is False
    assert len(captured) == 1
    assert captured[0] == local_config
    assert session.backend_kind is BackendKind.LOCAL
    assert session.model_id == local_config.model_id
    assert session.server_url == local_config.server_url
    assert "Local backend enabled. Conversation reset." in output.getvalue()
    assert "| Backend" in output.getvalue()
    assert local_config.server_url in output.getvalue()


def test_reset_command_reprints_runtime_summary() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    session.messages.append({"role": "user", "content": "stale"})

    should_exit = _run_command("reset", "", session, output)

    assert should_exit is False
    assert session.messages == [{"role": "system", "content": session.system_prompt}]
    rendered = output.getvalue()
    assert "Conversation reset." in rendered
    assert "| Backend" in rendered
    assert "| Timeout" in rendered


def test_reset_command_clears_active_attachments() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    repl_state = _ReplState(
        pending_attachment=_PendingAttachment(
            kind="document",
            summary="w3c-dummy.pdf",
            paths=[FIXTURE_DIR / "w3c-dummy.pdf"],
        ),
        active_images=[FIXTURE_DIR / "wikimedia-demo.png"],
        active_documents=[FIXTURE_DIR / "w3c-dummy.pdf"],
    )

    should_exit = _run_command("reset", "", session, output, repl_state=repl_state)

    assert should_exit is False
    assert repl_state.pending_attachment is None
    assert repl_state.active_images == []
    assert repl_state.active_documents == []


def test_reset_command_clears_screen_in_tty(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    monkeypatch.setenv("TERM", "xterm-256color")

    should_exit = _run_command("reset", "", session, output)

    assert should_exit is False
    assert output.getvalue().startswith(CLEAR_SCREEN)


def test_system_command_clears_screen_when_it_resets(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    monkeypatch.setenv("TERM", "xterm-256color")

    should_exit = _run_command("system", "You are terse.", session, output)

    assert should_exit is False
    assert output.getvalue().startswith(CLEAR_SCREEN)
    assert session.system_prompt == "You are terse."


def test_drop_commands_clear_attachment_state() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    repl_state = _ReplState(
        pending_attachment=_PendingAttachment(
            kind="image",
            summary="wikimedia-demo.png",
            paths=[FIXTURE_DIR / "wikimedia-demo.png"],
        ),
        active_images=[FIXTURE_DIR / "wikimedia-demo.png"],
        active_documents=[FIXTURE_DIR / "w3c-dummy.pdf"],
    )

    assert (
        _run_command("dropimage", "", session, output, repl_state=repl_state) is False
    )
    assert repl_state.pending_attachment is None
    assert repl_state.active_images == []
    assert repl_state.active_documents == [FIXTURE_DIR / "w3c-dummy.pdf"]

    assert _run_command("dropdoc", "", session, output, repl_state=repl_state) is False
    assert repl_state.active_documents == []

    repl_state.active_images = [FIXTURE_DIR / "wikimedia-demo.png"]
    repl_state.active_documents = [FIXTURE_DIR / "w3c-dummy.pdf"]
    assert _run_command("drop", "", session, output, repl_state=repl_state) is False
    assert repl_state.active_images == []
    assert repl_state.active_documents == []
    assert "Active attachments cleared." in output.getvalue()


def test_timeout_command_reports_current_timeout() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("timeout", "", session, output) is False
    assert f"Timeout: {DEFAULT_TIMEOUT_MS} ms" in output.getvalue()


def test_timeout_command_updates_timeout_in_minutes() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("timeout", "5m", session, output) is False
    assert session.timeout_ms == 300_000
    assert "Timeout set to 300000 ms." in output.getvalue()


def test_timeout_command_rejects_too_small_value() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("timeout", "500", session, output) is False
    assert "[timeout] Timeout must be at least 1000 ms." in output.getvalue()


def test_shortcuts_call_local_tools(tmp_path: Path) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    assert _run_command("edit", "notes.txt -- hello world", session, output) is False
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello world"

    assert (
        _run_command(
            "run",
            "--cwd . --lines 4 -- printf 'one\\ntwo\\nthree\\n'",
            session,
            output,
        )
        is False
    )
    assert "exit_code=0" in output.getvalue()
    assert "[more output available" in output.getvalue()

    assert _run_command("ls", ".", session, output) is False
    assert "notes.txt" in output.getvalue()

    assert _run_command("find", "--path . --limit 5 -- hello", session, output) is False
    assert "notes.txt" in output.getvalue()


def test_image_shortcut_uses_picker_and_multimodal_payload(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    image = FIXTURE_DIR / "wikimedia-demo.png"
    fake_client = FakeClient(stream_chunks=["ok"])
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    should_exit = _run_command(
        "image",
        "--prompt Describe the image.",
        session,
        output,
        input_func=lambda _prompt: "",
        path_picker=lambda **_kwargs: [image],
    )

    assert should_exit is False
    assert "selected" in output.getvalue()
    assert "ok" in output.getvalue()
    assert len(fake_client.chat.stream_calls) == 1
    call = fake_client.chat.stream_calls[0]
    content = call["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Describe the image." in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_doc_shortcut_without_prompt_stages_attachment_for_next_turn(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"
    fake_client = FakeClient(stream_chunks=["ok"])
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )
    repl_state = _ReplState()

    should_exit = _run_command(
        "doc",
        "",
        session,
        output,
        repl_state=repl_state,
        input_func=lambda _prompt: "",
        path_picker=lambda **_kwargs: [pdf_file],
    )

    assert should_exit is False
    assert repl_state.pending_attachment is not None
    assert repl_state.pending_attachment.kind == "document"
    assert fake_client.chat.stream_calls == []
    assert "attachment staged" in output.getvalue()


def test_active_attachment_message_can_combine_image_and_document(
    tmp_path: Path,
) -> None:
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=io.StringIO(),
    )

    content = _build_active_attachment_message(
        session,
        prompt="Compare both attachments.",
        image_paths=[FIXTURE_DIR / "wikimedia-demo.png"],
        document_paths=[FIXTURE_DIR / "w3c-dummy.pdf"],
    )

    assert content[0]["type"] == "text"
    assert "Active images:" in content[0]["text"]
    assert "Active documents:" in content[0]["text"]
    assert any(block.get("type") == "image_url" for block in content[1:])
    assert any(
        block.get("type") == "text"
        and "Document 1: w3c-dummy.pdf" in str(block.get("text"))
        for block in content[1:]
    )


def test_local_active_document_is_reinjected_on_followup_turn(tmp_path: Path) -> None:
    output = io.StringIO()
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )
    prompts = iter(["/doc", "Describe the document.", "What text does it contain?"])

    def input_func(_prompt: str) -> str:
        try:
            return next(prompts)
        except StopIteration as exc:
            raise EOFError from exc

    exit_code = _run_repl(
        session,
        local_config=LocalMistralConfig(),
        client_factory=lambda _config: FakeClient(),
        input_func=input_func,
        stdin=FakeStdin(""),
        stdout=output,
        stream=False,
        path_picker=lambda **_kwargs: [pdf_file],
    )

    assert exit_code == 0
    assert len(fake_client.chat.complete_calls) == 2
    first_call = fake_client.chat.complete_calls[0]
    second_call = fake_client.chat.complete_calls[1]
    first_user_messages = [
        message["content"]
        for message in first_call["messages"]
        if message["role"] == "user"
    ]
    second_user_messages = [
        message["content"]
        for message in second_call["messages"]
        if message["role"] == "user"
    ]
    first_content = first_user_messages[0]
    second_content = second_user_messages[-1]
    assert isinstance(first_content, list)
    assert isinstance(second_content, list)
    assert "tools" in first_call
    assert "tools" in second_call
    assert any(block.get("type") == "image_url" for block in first_content[1:])
    assert any(block.get("type") == "image_url" for block in second_content[1:])


def test_remote_active_document_is_reinjected_on_followup_turn(tmp_path: Path) -> None:
    output = io.StringIO()
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )
    prompts = iter(["/doc", "Describe the document.", "What text does it contain?"])

    def input_func(_prompt: str) -> str:
        try:
            return next(prompts)
        except StopIteration as exc:
            raise EOFError from exc

    exit_code = _run_repl(
        session,
        local_config=LocalMistralConfig(),
        client_factory=lambda _config: FakeClient(),
        input_func=input_func,
        stdin=FakeStdin(""),
        stdout=output,
        stream=False,
        path_picker=lambda **_kwargs: [pdf_file],
    )

    assert exit_code == 0
    assert len(fake_client.chat.complete_calls) == 2
    first_user_messages = [
        message["content"]
        for message in fake_client.chat.complete_calls[0]["messages"]
        if message["role"] == "user"
    ]
    second_user_messages = [
        message["content"]
        for message in fake_client.chat.complete_calls[1]["messages"]
        if message["role"] == "user"
    ]
    first_content = first_user_messages[0]
    second_content = second_user_messages[-1]
    assert isinstance(first_content, list)
    assert isinstance(second_content, list)
    assert "tools" in fake_client.chat.complete_calls[0]
    assert "tools" in fake_client.chat.complete_calls[1]
    assert any(block.get("type") == "document_url" for block in first_content[1:])
    assert any(block.get("type") == "document_url" for block in second_content[1:])


def test_attachment_turns_keep_tools_available_by_default(tmp_path: Path) -> None:
    output = io.StringIO()
    image = FIXTURE_DIR / "wikimedia-demo.png"
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    result = session.send_content(
        build_image_message([image], prompt="Describe the image."),
        stream=False,
    )

    assert result.content == "ok"
    assert "tools" in fake_client.chat.complete_calls[0]


def test_remote_image_turns_use_cloud_shape_and_keep_tools_available(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    image = FIXTURE_DIR / "wikimedia-demo.png"
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    result = session.send_content(
        build_remote_image_message([image], prompt="Describe the image."),
        stream=False,
    )

    assert result.content == "ok"
    call = fake_client.chat.complete_calls[0]
    content = call["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"
    assert isinstance(content[1]["image_url"], str)
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert "tools" in call


def test_remote_document_turns_use_document_url_and_keep_tools_available(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    result = session.send_content(
        build_remote_document_message([pdf_file], prompt="Analyze the document."),
        stream=False,
    )

    assert result.content == "ok"
    call = fake_client.chat.complete_calls[0]
    content = call["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "document_url"
    assert content[1]["document_url"].startswith("data:application/pdf;base64,")
    assert content[1]["document_name"] == "w3c-dummy.pdf"
    assert "tools" in call


def test_visible_reasoning_is_rendered_but_not_committed() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="<think>check file</think>ok")
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return ok.", stream=False)

    assert result.finish_reason == "stop"
    assert result.reasoning == "check file"
    assert result.content == "ok"
    assert "check file" in output.getvalue()
    assert "check file\n\nok\n" in output.getvalue()
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}


def test_uppercase_think_reasoning_is_rendered_but_not_committed() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="[THINK]plan[/THINK]ok")
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return ok.", stream=False)

    assert result.reasoning == "plan"
    assert result.content == "ok"
    assert "plan" in output.getvalue()
    assert "plan\n\nok\n" in output.getvalue()
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}


def test_raw_reasoning_content_is_rendered_and_committed_cleanly(
    monkeypatch: Any,
) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    monkeypatch.setattr(MistralSession, "_should_use_raw_chat", lambda self: True)
    monkeypatch.setattr(
        MistralSession,
        "_open_raw_request",
        lambda self, payload: FakeRawHTTPResponse(
            '{"choices":[{"finish_reason":"stop","message":{"role":"assistant",'
            '"content":"ok","reasoning_content":"plan first"}}]}'
        ),
    )

    result = session.send("Return ok.", stream=False)

    assert result.reasoning == "plan first"
    assert result.content == "ok"
    assert "plan first" in output.getvalue()
    assert "plan first\n\nok\n" in output.getvalue()
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}


def test_raw_stream_reasoning_content_is_rendered_and_committed_cleanly(
    monkeypatch: Any,
) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    monkeypatch.setattr(MistralSession, "_should_use_raw_chat", lambda self: True)
    monkeypatch.setattr(
        MistralSession,
        "_open_raw_request",
        lambda self, payload: FakeRawStreamResponse(
            [
                (
                    'data: {"choices":[{"finish_reason":null,"delta":'
                    '{"role":"assistant","content":null}}]}'
                ),
                (
                    'data: {"choices":[{"finish_reason":null,"delta":'
                    '{"reasoning_content":"plan "}}]}'
                ),
                (
                    'data: {"choices":[{"finish_reason":null,"delta":'
                    '{"reasoning_content":"first"}}]}'
                ),
                'data: {"choices":[{"finish_reason":"stop","delta":{"content":"ok"}}]}',
                "data: [DONE]",
            ]
        ),
    )

    result = session.send("Return ok.", stream=True)

    assert result.reasoning == "plan first"
    assert result.content == "ok"
    assert "plan first" in output.getvalue()
    assert "plan first\n\nok\n" in output.getvalue()
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}


def test_thinking_display_can_be_hidden_without_disabling_reasoning() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["<think>hidden</think>", "ok"])
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=True,
        show_thinking=False,
    )

    result = session.send("Return ok.", stream=True)

    assert result.reasoning == "hidden"
    assert result.content == "ok"
    assert "hidden" not in output.getvalue()
    assert output.getvalue().endswith("ok\n")


def test_doc_shortcut_uses_picker_and_document_payload(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    docx_file = FIXTURE_DIR / "pywordform-sample_form.docx"
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"
    fake_client = FakeClient(stream_chunks=["ok"])
    session = MistralSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    should_exit = _run_command(
        "doc",
        "--prompt Summarize the file.",
        session,
        output,
        input_func=lambda _prompt: "",
        path_picker=lambda **_kwargs: [docx_file, pdf_file],
    )

    assert should_exit is False
    assert "selected" in output.getvalue()
    assert "ok" in output.getvalue()
    assert len(fake_client.chat.stream_calls) == 1
    call = fake_client.chat.stream_calls[0]
    content = call["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Summarize the file." in content[0]["text"]
    assert any(
        block["type"] == "text" and "pywordform-sample_form.docx" in block["text"]
        for block in content
    )
    assert any(
        block["type"] == "text" and "w3c-dummy.pdf" in block["text"]
        for block in content
    )
    assert sum(1 for block in content if block["type"] == "image_url") >= 2


def test_reasoning_command_updates_session_state() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("reasoning", "", session, output) is False
    assert "Reasoning request: on (local backend)" in output.getvalue()

    output.truncate(0)
    output.seek(0)
    assert _run_command("reasoning", "off", session, output) is False
    assert session.show_reasoning is False
    assert "Reasoning request disabled." in output.getvalue()

    output.truncate(0)
    output.seek(0)
    assert _run_command("reasoning", "toggle", session, output) is False
    assert session.show_reasoning is True
    assert "Reasoning request: on (local backend)" in output.getvalue()


def test_thinking_command_updates_render_state_without_touching_reasoning() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("thinking", "", session, output) is False
    assert "Thinking display: on" in output.getvalue()

    output.truncate(0)
    output.seek(0)
    assert _run_command("thinking", "off", session, output) is False
    assert session.show_thinking is False
    assert session.show_reasoning is True
    assert "Thinking display disabled." in output.getvalue()

    output.truncate(0)
    output.seek(0)
    assert _run_command("thinking", "toggle", session, output) is False
    assert session.show_thinking is True
    assert session.show_reasoning is True
    assert "Thinking display: on" in output.getvalue()


def test_remote_reasoning_command_reports_remote_backend() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("reasoning", "", session, output) is False
    assert "Reasoning request: on (remote SDK)" in output.getvalue()


def test_remote_conversations_reasoning_command_reports_best_effort() -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeConversationClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    assert _run_command("reasoning", "", session, output) is False
    assert (
        "Reasoning request: on (remote Conversations, requested best-effort)"
        in output.getvalue()
    )


def test_remote_structured_reasoning_is_rendered_and_committed_cleanly() -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(
                            content=[
                                {
                                    "type": "thinking",
                                    "thinking": [
                                        {"type": "text", "text": "plan first"}
                                    ],
                                },
                                {"type": "text", "text": "ok"},
                            ]
                        ),
                        finish_reason="stop",
                    )
                ]
            )
        ]
    )
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return ok.", stream=False)

    assert result.reasoning == "plan first"
    assert result.content == "ok"
    assert output.getvalue() == "plan first\n\nok\n"
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}


def test_remote_stream_reasoning_is_rendered_and_committed_cleanly() -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[],
        stream_chunks=[],
    )
    fake_client.chat.stream_chunks = []
    fake_client.chat.last_stream = FakeStream(
        [
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(content=""),
                            finish_reason=None,
                        )
                    ]
                )
            ),
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(
                                content=[
                                    {
                                        "type": "thinking",
                                        "thinking": [{"type": "text", "text": "plan "}],
                                    }
                                ]
                            ),
                            finish_reason=None,
                        )
                    ]
                )
            ),
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(
                                content=[
                                    {
                                        "type": "thinking",
                                        "thinking": [{"type": "text", "text": "first"}],
                                    }
                                ]
                            ),
                            finish_reason=None,
                        )
                    ]
                )
            ),
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(content=[{"type": "text", "text": "ok"}]),
                            finish_reason="stop",
                        )
                    ]
                )
            ),
        ]
    )
    fake_client.chat.stream = lambda **kwargs: fake_client.chat.last_stream  # type: ignore[method-assign]
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return ok.", stream=True)

    assert result.reasoning == "plan first"
    assert result.content == "ok"
    assert output.getvalue() == "plan first\n\nok\n"
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}


def test_conversations_structured_reasoning_is_rendered_and_committed_cleanly() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_reasoning",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": "plan first"}],
                            },
                            {"type": "text", "text": "ok"},
                        ],
                    )
                ],
            )
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("Return ok.", stream=False)

    assert result.reasoning == "plan first"
    assert result.content == "ok"
    assert output.getvalue() == "plan first\n\nok\n"


def test_conversations_missing_reasoning_prints_best_effort_notice_once() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_missing_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "first"}],
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_missing_2",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "second"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    first = session.send("one", stream=False)
    second = session.send("two", stream=False)

    assert first.reasoning == ""
    assert second.reasoning == ""
    assert (
        output.getvalue().count("Mistral Conversations returned no thinking blocks")
        == 1
    )


def test_conversations_reasoning_disabled_requests_none_and_skips_notice() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_no_reasoning",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "ok"}],
                    )
                ],
            )
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=False,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.set_reasoning_visibility(False)

    result = session.send("Return ok.", stream=False)

    assert result.reasoning == ""
    assert result.content == "ok"
    assert conversations.start_calls[0]["completion_args"]["reasoning_effort"] == "none"
    assert "Mistral Conversations returned no thinking blocks" not in output.getvalue()


def test_image_shortcut_reports_invalid_selection_without_crashing(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    invalid_file = tmp_path / "notes.md"
    invalid_file.write_text("not an image", encoding="utf-8")
    session = MistralSession(
        client=FakeClient(stream_chunks=["ok"]),
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    should_exit = _run_command(
        "image",
        "",
        session,
        output,
        input_func=lambda _prompt: "",
        path_picker=lambda **_kwargs: [invalid_file],
    )

    assert should_exit is False
    assert "could not prepare attachment" in output.getvalue()


def test_doc_shortcut_reports_invalid_selection_without_crashing(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    invalid_file = tmp_path / "archive.zip"
    invalid_file.write_text("not a document", encoding="utf-8")
    session = MistralSession(
        client=FakeClient(stream_chunks=["ok"]),
        generation=LocalGenerationConfig(),
        tool_bridge=LocalToolBridge(root=tmp_path),
        stdout=output,
    )

    should_exit = _run_command(
        "doc",
        "",
        session,
        output,
        input_func=lambda _prompt: "",
        path_picker=lambda **_kwargs: [invalid_file],
    )

    assert should_exit is False
    assert "could not prepare attachment" in output.getvalue()
