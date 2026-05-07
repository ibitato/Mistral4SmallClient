# ruff: noqa: F403, F405
import copy
import json
from collections.abc import Iterator
from typing import cast

from tests.cli_support import *


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
        "mistralcli.cli.shutil.get_terminal_size",
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
        "mistralcli.cli.shutil.get_terminal_size",
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
        "mistralcli.cli.shutil.get_terminal_size",
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
        "mistralcli.cli.shutil.get_terminal_size",
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
    assert "Mistral dual-model multimodal console" in rendered


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


def test_repeated_identical_tool_call_reuses_prior_result(tmp_path: Path) -> None:
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
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="write tool executed"),
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

    result = session.send("Write a file.", stream=False)

    assert result.finish_reason == "stop"
    assert result.cancelled is False
    assert result.content == "write tool executed"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert "repeated identical tool call blocked" not in output.getvalue()
    assert '"code": "reused_identical_tool_result"' in session.messages[-2]["content"]
    assert '"reused": true' in session.messages[-2]["content"]


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
        client=cast(Any, SequencedClient()),
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
                def __iter__(self) -> Iterator[bytes]:
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


def test_local_tool_turn_without_final_answer_inserts_placeholder_and_allows_followup(
    monkeypatch: Any,
) -> None:
    output = io.StringIO()
    tool_call = FakeToolCall(
        function=FakeToolFunction(
            name="web_search",
            arguments='{"query":"mcp"}',
        )
    )
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )
    responses = iter(
        [
            FakeRawHTTPResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": tool_call.id,
                                            "type": tool_call.type,
                                            "function": {
                                                "name": tool_call.function.name,
                                                "arguments": (
                                                    tool_call.function.arguments
                                                ),
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                )
            ),
            FakeRawHTTPResponse(
                '{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":""}}]}'
            ),
            FakeRawHTTPResponse(
                '{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}'
            ),
        ]
    )

    monkeypatch.setattr(MistralSession, "_should_use_raw_chat", lambda self: True)
    monkeypatch.setattr(
        MistralSession,
        "_open_raw_request",
        lambda self, payload: next(responses),
    )

    first = session.send("Busca MCP.", stream=False)
    second = session.send("Return only ok.", stream=False)

    assert first.cancelled is False
    assert first.finish_reason == "stop"
    assert second.content == "ok"
    assert session.messages == [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "Busca MCP."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": "web_search",
            "content": '{"results":[{"title":"Example","url":"https://example.com"}]}',
        },
        {
            "role": "assistant",
            "content": (
                "[Previous tool run completed without a final assistant answer. "
                "Continue from the preserved tool results above.]"
            ),
        },
        {"role": "user", "content": "Return only ok."},
        {"role": "assistant", "content": "ok"},
    ]


def test_local_raw_history_repair_closes_trailing_tool_turn_before_next_prompt(
    monkeypatch: Any,
) -> None:
    output = io.StringIO()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )
    session.messages = [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "Busca algo."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tool_call_1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"command":"echo test"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tool_call_1",
            "name": "shell",
            "content": '{"stdout":"test"}',
        },
    ]
    captured_payloads: list[dict[str, Any]] = []

    monkeypatch.setattr(MistralSession, "_should_use_raw_chat", lambda self: True)

    def open_raw_request(self: MistralSession, payload: dict[str, Any]) -> object:
        captured_payloads.append(copy.deepcopy(payload))
        return FakeRawHTTPResponse(
            '{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}'
        )

    monkeypatch.setattr(MistralSession, "_open_raw_request", open_raw_request)

    result = session.send("Return only ok.", stream=False)

    assert result.content == "ok"
    assert captured_payloads[0]["messages"] == [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "Busca algo."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tool_call_1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"command":"echo test"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tool_call_1",
            "name": "shell",
            "content": '{"stdout":"test"}',
        },
        {
            "role": "assistant",
            "content": (
                "[Previous tool run completed without a final assistant answer. "
                "Continue from the preserved tool results above.]"
            ),
        },
        {"role": "user", "content": "Return only ok."},
    ]
    assert "[repair] restored local raw-chat history" in output.getvalue()


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
        client=cast(Any, ErrorClient()),
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
