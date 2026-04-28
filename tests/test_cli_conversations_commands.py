# ruff: noqa: F403, F405
from tests.cli_support import *


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
