# ruff: noqa: F403, F405
from tests.cli_support import *


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
