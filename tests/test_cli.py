from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mistral4cli.attachments import (
    build_image_message,
    build_remote_document_message,
    build_remote_image_message,
)
from mistral4cli.cli import (
    _clear_screen_if_supported,
    _InputHistory,
    _parse_command,
    _refresh_repl_screen,
    _ReplState,
    _run_command,
    _write_tty_newline,
    main,
)
from mistral4cli.local_mistral import (
    DEFAULT_TIMEOUT_MS,
    BackendKind,
    LocalGenerationConfig,
    LocalMistralConfig,
    RemoteMistralConfig,
)
from mistral4cli.local_tools import LocalToolBridge
from mistral4cli.mcp_bridge import MCPToolResult
from mistral4cli.session import MistralCodingSession
from mistral4cli.ui import (
    CLEAR_SCREEN,
    render_help_screen,
    render_welcome_banner,
    terminal_recommendation,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "internet"


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


@dataclass(slots=True)
class FakeEvent:
    data: FakeResponse


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


def test_help_and_banner_are_actionable_and_retro() -> None:
    output = io.StringIO()
    session = MistralCodingSession(
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

    assert "Mistral4Small retro console" in banner
    assert "Type /help for actionable commands" in banner
    assert "Mistral cloud" in banner
    assert "+-" in banner
    assert "| Backend" in banner
    assert "| Model" in banner
    assert "/tools" in help_text
    assert "FireCrawl MCP" in help_text
    assert "/run" in help_text
    assert "/edit" in help_text
    assert "/find" in help_text
    assert "/ls" in help_text
    assert "/image" in help_text
    assert "/doc" in help_text
    assert "/remote" in help_text
    assert "/timeout" in help_text
    assert "/reasoning" in help_text
    assert "Search official documentation" in help_text
    assert "Ctrl-C cancels the current response" in help_text


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
    session = MistralCodingSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    monkeypatch.setenv("TERM", "xterm")

    _refresh_repl_screen(output, session, startup=True)

    rendered = output.getvalue()
    assert rendered.startswith(CLEAR_SCREEN)
    assert "TERM=xterm-256color" in rendered
    assert "Mistral4Small retro console" in rendered


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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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


def test_stream_cancel_does_not_commit_partial_assistant_turn() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["hello", " world"], interrupt_after=1)
    session = MistralCodingSession(
        client=fake_client, generation=LocalGenerationConfig(), stdout=output
    )

    result = session.send("Haz una respuesta larga.", stream=True)

    assert result.cancelled is True
    assert result.content == "hello"
    assert fake_client.chat.last_stream is not None
    assert fake_client.chat.last_stream.closed is True
    assert session.messages == [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "Haz una respuesta larga."},
    ]
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

    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}
    assert "[interrupted]" in output.getvalue()


def test_model_error_rolls_back_failed_turn() -> None:
    output = io.StringIO()

    class ErrorChat(FakeChat):
        def complete(self, **kwargs: Any) -> FakeResponse:
            self.complete_calls.append(kwargs)
            raise RuntimeError("boom")

    class ErrorClient:
        def __init__(self) -> None:
            self.chat = ErrorChat()

    session = MistralCodingSession(
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
    assert _parse_command("/remote on") == ("remote", "on")
    assert _parse_command("/timeout 5m") == ("timeout", "5m")
    assert _parse_command("/reasoning off") == ("reasoning", "off")
    assert _parse_command("/run --cwd . -- git status") == (
        "run",
        "--cwd . -- git status",
    )
    assert _parse_command("/find --path src -- shell") == (
        "find",
        "--path src -- shell",
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


def test_write_tty_newline_emits_crlf() -> None:
    output = io.StringIO()

    _write_tty_newline(output)

    assert output.getvalue() == "\r\n"


def test_remote_request_uses_reasoning_effort_and_omits_prompt_mode() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")
    session = MistralCodingSession(
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


def test_remote_request_disables_reasoning_effort_when_hidden() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")
    session = MistralCodingSession(
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


def test_remote_command_requires_api_key(monkeypatch: Any) -> None:
    output = io.StringIO()
    session = MistralCodingSession(
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


def test_remote_command_switches_backend_and_resets_conversation(
    monkeypatch: Any,
) -> None:
    output = io.StringIO()
    captured: list[object] = []
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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


def test_reset_command_clears_screen_in_tty(monkeypatch: Any) -> None:
    output = FakeTTYOutput()
    session = MistralCodingSession(
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
    session = MistralCodingSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    monkeypatch.setenv("TERM", "xterm-256color")

    should_exit = _run_command("system", "You are terse.", session, output)

    assert should_exit is False
    assert output.getvalue().startswith(CLEAR_SCREEN)
    assert session.system_prompt == "You are terse."


def test_timeout_command_reports_current_timeout() -> None:
    output = io.StringIO()
    session = MistralCodingSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("timeout", "", session, output) is False
    assert f"Timeout: {DEFAULT_TIMEOUT_MS} ms" in output.getvalue()


def test_timeout_command_updates_timeout_in_minutes() -> None:
    output = io.StringIO()
    session = MistralCodingSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("timeout", "5m", session, output) is False
    assert session.timeout_ms == 300_000
    assert "Timeout set to 300000 ms." in output.getvalue()


def test_timeout_command_rejects_too_small_value() -> None:
    output = io.StringIO()
    session = MistralCodingSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("timeout", "500", session, output) is False
    assert "[timeout] Timeout must be at least 1000 ms." in output.getvalue()


def test_shortcuts_call_local_tools(tmp_path: Path) -> None:
    output = io.StringIO()
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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


def test_attachment_turns_do_not_offer_tools(tmp_path: Path) -> None:
    output = io.StringIO()
    image = FIXTURE_DIR / "wikimedia-demo.png"
    fake_client = FakeClient(complete_text="ok")
    session = MistralCodingSession(
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
    assert "tools" not in fake_client.chat.complete_calls[0]


def test_remote_image_turns_use_cloud_shape_and_disable_tools(tmp_path: Path) -> None:
    output = io.StringIO()
    image = FIXTURE_DIR / "wikimedia-demo.png"
    fake_client = FakeClient(complete_text="ok")
    session = MistralCodingSession(
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
        disable_tools=True,
    )

    assert result.content == "ok"
    call = fake_client.chat.complete_calls[0]
    content = call["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"
    assert isinstance(content[1]["image_url"], str)
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert "tools" not in call


def test_remote_document_turns_use_document_url_and_disable_tools(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"
    fake_client = FakeClient(complete_text="ok")
    session = MistralCodingSession(
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
        disable_tools=True,
    )

    assert result.content == "ok"
    call = fake_client.chat.complete_calls[0]
    content = call["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "document_url"
    assert content[1]["document_url"].startswith("data:application/pdf;base64,")
    assert content[1]["document_name"] == "w3c-dummy.pdf"
    assert "tools" not in call


def test_visible_reasoning_is_rendered_but_not_committed() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="<think>check file</think>ok")
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    monkeypatch.setattr(MistralCodingSession, "_should_use_raw_chat", lambda self: True)
    monkeypatch.setattr(
        MistralCodingSession,
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
    session = MistralCodingSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    monkeypatch.setattr(MistralCodingSession, "_should_use_raw_chat", lambda self: True)
    monkeypatch.setattr(
        MistralCodingSession,
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


def test_reasoning_can_be_hidden() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["<think>hidden</think>", "ok"])
    session = MistralCodingSession(
        client=fake_client,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=False,
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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("reasoning", "", session, output) is False
    assert "Visible reasoning: on (local raw endpoint)" in output.getvalue()

    output.truncate(0)
    output.seek(0)
    assert _run_command("reasoning", "off", session, output) is False
    assert session.show_reasoning is False
    assert "Visible reasoning disabled." in output.getvalue()

    output.truncate(0)
    output.seek(0)
    assert _run_command("reasoning", "toggle", session, output) is False
    assert session.show_reasoning is True
    assert "Visible reasoning: on (local raw endpoint)" in output.getvalue()


def test_remote_reasoning_command_reports_remote_backend() -> None:
    output = io.StringIO()
    session = MistralCodingSession(
        client=FakeClient(),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("reasoning", "", session, output) is False
    assert "Visible reasoning: on (remote SDK)" in output.getvalue()


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
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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


def test_image_shortcut_reports_invalid_selection_without_crashing(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    invalid_file = tmp_path / "notes.md"
    invalid_file.write_text("not an image", encoding="utf-8")
    session = MistralCodingSession(
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
    session = MistralCodingSession(
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
