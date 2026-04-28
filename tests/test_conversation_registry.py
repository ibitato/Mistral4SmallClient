from __future__ import annotations

from pathlib import Path

from mistral4cli.conversation_registry import (
    ConversationRegistry,
    default_registry_path,
)


def test_default_registry_path_uses_state_directory(monkeypatch) -> None:
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
    assert reloaded.resolve_reference("primary") is not None
    assert reloaded.resolve_reference("primary").conversation_id == "conv_1"
    assert reloaded.get("conv_1").note == "keep this thread"
    assert reloaded.get("conv_1").tags == ["ops"]


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
    assert registry.get("conv_new") is not None
    assert registry.get("conv_child").parent_conversation_id == "conv_new"
    assert registry.last_active_conversation_id == "conv_new"
