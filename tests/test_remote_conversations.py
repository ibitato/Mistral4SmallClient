from __future__ import annotations

import io
import os
from contextlib import suppress
from time import monotonic, sleep
from uuid import uuid4

import pytest

from mistral4cli.conversation_registry import ConversationRegistry
from mistral4cli.local_mistral import (
    BackendKind,
    LocalGenerationConfig,
    RemoteMistralConfig,
    build_client,
)
from mistral4cli.session import MistralSession, PendingConversationSettings

pytestmark = pytest.mark.remote


def _wait_for_remote_listing(session: MistralSession, metadata: dict[str, str]) -> str:
    """Poll the remote list endpoint briefly for eventually consistent metadata."""

    deadline = monotonic() + 6.0
    last_text = "No remote conversations found."
    while monotonic() < deadline:
        last_text = session.list_remote_conversations(metadata=metadata)
        if "No remote conversations found." not in last_text:
            return last_text
        sleep(1.0)
    return last_text


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="Set MISTRAL_API_KEY to enable remote Mistral Conversations tests.",
)
def test_remote_conversations_start_and_append_smoke() -> None:
    config = RemoteMistralConfig.from_env()
    client = build_client(config)

    conversation_id = ""
    try:
        started = client.beta.conversations.start(
            model=config.model_id,
            instructions="Return short literal answers.",
            inputs="Return only the word ok.",
            store=True,
            completion_args={
                "max_tokens": 16,
                "temperature": 0.0,
                "reasoning_effort": "high",
            },
        )

        assert started.conversation_id
        assert started.outputs
        conversation_id = started.conversation_id

        followup = client.beta.conversations.append(
            conversation_id=conversation_id,
            inputs="Return only the word done.",
            store=True,
            completion_args={
                "max_tokens": 16,
                "temperature": 0.0,
                "reasoning_effort": "high",
            },
        )

        assert followup.conversation_id == conversation_id
        assert followup.outputs
    finally:
        if conversation_id:
            client.beta.conversations.delete(conversation_id=conversation_id)


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="Set MISTRAL_API_KEY to enable remote Mistral Conversations tests.",
)
def test_remote_conversations_without_reasoning_smoke() -> None:
    config = RemoteMistralConfig.from_env()
    client = build_client(config)

    conversation_id = ""
    try:
        started = client.beta.conversations.start(
            model=config.model_id,
            instructions="Return short literal answers.",
            inputs="Return only the word ok.",
            store=True,
            completion_args={
                "max_tokens": 16,
                "temperature": 0.0,
                "reasoning_effort": "none",
            },
        )

        assert started.conversation_id
        assert started.outputs
        conversation_id = started.conversation_id

        followup = client.beta.conversations.append(
            conversation_id=conversation_id,
            inputs="Return only the word done.",
            store=True,
            completion_args={
                "max_tokens": 16,
                "temperature": 0.0,
                "reasoning_effort": "none",
            },
        )

        assert followup.conversation_id == conversation_id
        assert followup.outputs
    finally:
        if conversation_id:
            client.beta.conversations.delete(conversation_id=conversation_id)


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="Set MISTRAL_API_KEY to enable remote Mistral Conversations tests.",
)
def test_remote_conversations_session_management_smoke(tmp_path) -> None:
    config = RemoteMistralConfig.from_env()
    client = build_client(config)
    registry = ConversationRegistry.load(tmp_path / "conversations.json")
    session = MistralSession(
        client=client,
        backend_kind=BackendKind.REMOTE,
        model_id=config.model_id,
        server_url=None,
        generation=LocalGenerationConfig(
            temperature=0.0,
            top_p=1.0,
            prompt_mode=None,
            max_tokens=16,
        ),
        stdout=io.StringIO(),
        show_reasoning=False,
        conversation_registry=registry,
    )
    session.enable_conversations(
        client=client,
        model_id=config.model_id,
        store=True,
    )
    unique = uuid4().hex[:10]
    session.pending_conversation = PendingConversationSettings(
        name=f"Smoke {unique}",
        description="Remote Conversations management smoke test.",
        metadata={"suite": "remote", "run": unique},
    )
    created_ids: list[str] = []

    try:
        result = session.send("Return only the word ok.", stream=False)
        assert result.finish_reason == "stop"
        assert result.cancelled is False
        assert session.conversation_id
        conversation_id = session.conversation_id
        created_ids.append(conversation_id)

        list_text = session.list_remote_conversations()
        assert "Remote conversations page=0 size=20:" in list_text
        assert conversation_id in list_text

        filtered_text = _wait_for_remote_listing(session, {"run": unique})
        assert conversation_id in filtered_text

        show_text = session.show_remote_conversation(conversation_id)
        assert f"Conversation {conversation_id}:" in show_text
        assert unique in show_text

        history_text = session.conversation_history_text(
            conversation_id=conversation_id
        )
        assert f"Conversation {conversation_id}:" in history_text
        assert 'Hint: use "/conv restart <entry_id>"' in history_text

        messages_text = session.conversation_history_text(
            messages_only=True,
            conversation_id=conversation_id,
        )
        assert f"Conversation {conversation_id}:" in messages_text

        history = client.beta.conversations.get_history(conversation_id=conversation_id)
        first_entry = next(
            (entry for entry in history.entries if getattr(entry, "id", None)),
            None,
        )
        assert first_entry is not None

        restart_text = session.restart_remote_conversation(
            from_entry_id=str(first_entry.id),
            conversation_id=conversation_id,
        )
        assert "Restarted from" in restart_text
        assert session.conversation_id
        branch_id = session.conversation_id
        assert branch_id != conversation_id
        created_ids.append(branch_id)

        branch_show_text = session.show_remote_conversation(branch_id)
        assert f"Conversation {branch_id}:" in branch_show_text

        delete_text = session.delete_remote_conversation(branch_id)
        assert delete_text == f"Deleted conversation {branch_id}."
        created_ids.remove(branch_id)
    finally:
        for conversation_id in reversed(created_ids):
            with suppress(Exception):
                client.beta.conversations.delete(conversation_id=conversation_id)
