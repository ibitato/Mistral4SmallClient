# ruff: noqa: F403, F405
from tests.cli_support import *


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
