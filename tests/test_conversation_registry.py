from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mistral4cli.conversation_registry import (
    ConversationRegistry,
    default_registry_path,
)


def test_default_registry_path_uses_state_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "/tmp/m4s-state")
    path = default_registry_path()

    assert path == Path("/tmp/m4s-state/mistral4cli/conversations.json")


def test_registry_round_trip_and_alias_resolution(tmp_path: Path) -> None:
    path = tmp_path / "conversations.json"
    registry = ConversationRegistry.load(path)

    registry.update_remote_snapshot(
        "conv_1",
        remote_name="Primary",
        remote_model="mistral-small-latest",
    )
    registry.set_alias("conv_1", "primary")
    registry.set_note("conv_1", "keep this thread")
    registry.add_tag("conv_1", "ops")
    registry.remember_active("conv_1")

    reloaded = ConversationRegistry.load(path)

    assert reloaded.last_active_conversation_id == "conv_1"
    resolved = reloaded.resolve_reference("primary")
    record = reloaded.get("conv_1")
    assert resolved is not None
    assert record is not None
    assert resolved.conversation_id == "conv_1"
    assert record.note == "keep this thread"
    assert record.tags == ["ops"]


def test_registry_migrates_ids_and_keeps_parent_links(tmp_path: Path) -> None:
    path = tmp_path / "conversations.json"
    registry = ConversationRegistry.load(path)

    registry.update_remote_snapshot("conv_old", remote_name="Old")
    registry.update_remote_snapshot(
        "conv_child",
        remote_name="Child",
        parent_conversation_id="conv_old",
    )
    registry.remember_active("conv_old")

    registry.migrate_conversation_id("conv_old", "conv_new")

    assert registry.get("conv_old") is None
    child = registry.get("conv_child")
    assert registry.get("conv_new") is not None
    assert child is not None
    assert child.parent_conversation_id == "conv_new"
    assert registry.last_active_conversation_id == "conv_new"


def test_registry_normalizes_datetime_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "conversations.json"
    registry = ConversationRegistry.load(path)
    created_at = datetime(2026, 4, 28, 9, 30, tzinfo=timezone.utc)
    updated_at = datetime(2026, 4, 28, 10, 15, tzinfo=timezone.utc)

    registry.update_remote_snapshot(
        "conv_1",
        remote_name="Primary",
        created_at=created_at,
        updated_at=updated_at,
    )

    record = registry.get("conv_1")
    assert record is not None
    assert record.created_at == created_at.isoformat()
    assert record.updated_at == updated_at.isoformat()
