# ruff: noqa: F403, F405
from tests.cli_support import *


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


def test_status_command_renders_in_red_on_tty(monkeypatch: Any) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = FakeTTYOutput()
    session = MistralSession(
        client=FakeClient(),
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    assert _run_command("status", "", session, output, repl_state=_ReplState()) is False

    rendered = output.getvalue()
    assert RED in rendered
    assert "Session status:" in ANSI_ESCAPE_RE.sub("", rendered)


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
