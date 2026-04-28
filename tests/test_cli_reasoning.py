# ruff: noqa: F403, F405
from tests.cli_support import *


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


def test_local_reasoning_keeps_raw_transport_when_thinking_is_hidden() -> None:
    session = MistralSession(
        client=build_client(
            LocalMistralConfig(
                api_key="local",
                model_id="unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL",
                server_url="http://127.0.0.1:8080",
                timeout_ms=DEFAULT_TIMEOUT_MS,
            )
        ),
        generation=LocalGenerationConfig(),
        stdout=io.StringIO(),
        show_reasoning=True,
        show_thinking=False,
    )

    assert session._should_use_raw_chat() is True


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
    fake_client.chat.stream = lambda **kwargs: fake_client.chat.last_stream
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
