from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mistral4cli.cli import _parse_command, _run_command, main
from mistral4cli.local_mistral import LocalGenerationConfig
from mistral4cli.local_tools import LocalToolBridge
from mistral4cli.mcp_bridge import MCPToolResult
from mistral4cli.session import MistralCodingSession
from mistral4cli.ui import render_help_screen, render_welcome_banner

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "internet"


class FakeStdin(io.StringIO):
    def __init__(self, value: str = "", *, tty: bool = False) -> None:
        super().__init__(value)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


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
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass(slots=True)
class FakeDelta:
    content: str | None = None
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
    assert "Mistral Small 4 local CLI" in rendered
    assert "Local OS tools: ready" in rendered
    assert "temperature=0.7" in rendered
    assert "top_p=0.95" in rendered
    assert "prompt_mode=reasoning" in rendered
    assert "stream=on" in rendered


def test_once_uses_effective_defaults_and_prints_answer() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")

    exit_code = main(
        ["--once", "Devuelve solo la palabra ok.", "--no-stream", "--no-mcp"],
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
    assert "/tools" in help_text
    assert "FireCrawl MCP" in help_text
    assert "/run" in help_text
    assert "/edit" in help_text
    assert "/find" in help_text
    assert "/ls" in help_text
    assert "/image" in help_text
    assert "/doc" in help_text
    assert "Search official documentation" in help_text
    assert "Ctrl-C cancels the current response" in help_text


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


def test_stream_cancel_does_not_commit_partial_assistant_turn() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["hola", " mundo"], interrupt_after=1)
    session = MistralCodingSession(
        client=fake_client, generation=LocalGenerationConfig(), stdout=output
    )

    result = session.send("Haz una respuesta larga.", stream=True)

    assert result.cancelled is True
    assert result.content == "hola"
    assert fake_client.chat.last_stream is not None
    assert fake_client.chat.last_stream.closed is True
    assert session.messages == [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "Haz una respuesta larga."},
    ]
    assert "[interrupted]" in output.getvalue()


def test_non_stream_cancel_does_not_break_followup() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_interrupt_once=True, complete_text="ok")
    session = MistralCodingSession(
        client=fake_client, generation=LocalGenerationConfig(), stdout=output
    )

    first = session.send("Haz una respuesta larga.", stream=False)
    second = session.send("Devuelve solo ok.", stream=False)

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
    assert _parse_command("/run --cwd . -- git status") == (
        "run",
        "--cwd . -- git status",
    )
    assert _parse_command("/find --path src -- shell") == (
        "find",
        "--path src -- shell",
    )


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
