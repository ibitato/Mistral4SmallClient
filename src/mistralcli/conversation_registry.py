"""Persistent local registry for remote Mistral Conversations."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _timestamp_now() -> str:
    """Return the current UTC time in ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    """Normalize SDK and JSON scalar values into stored text."""

    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value).strip()


def default_registry_path() -> Path:
    """Return the default persistent registry path."""

    xdg_state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / "mistralcli" / "conversations.json"
    return Path.home() / ".local" / "state" / "mistralcli" / "conversations.json"


@dataclass(slots=True)
class ConversationRecord:
    """Local metadata overlay for one remote conversation."""

    conversation_id: str
    alias: str = ""
    tags: list[str] = field(default_factory=list)
    note: str = ""
    remote_name: str = ""
    remote_description: str = ""
    remote_metadata: dict[str, Any] = field(default_factory=dict)
    remote_kind: str = "model"
    remote_model: str = ""
    remote_agent_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str = ""
    parent_conversation_id: str = ""
    deleted: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConversationRecord:
        """Build a record from serialized JSON state."""

        return cls(
            conversation_id=str(payload.get("conversation_id", "")).strip(),
            alias=str(payload.get("alias", "")).strip(),
            tags=[
                str(tag).strip() for tag in payload.get("tags", []) if str(tag).strip()
            ],
            note=str(payload.get("note", "")).strip(),
            remote_name=str(payload.get("remote_name", "")).strip(),
            remote_description=str(payload.get("remote_description", "")).strip(),
            remote_metadata=dict(payload.get("remote_metadata", {}) or {}),
            remote_kind=str(payload.get("remote_kind", "model") or "model").strip(),
            remote_model=str(payload.get("remote_model", "")).strip(),
            remote_agent_id=str(payload.get("remote_agent_id", "")).strip(),
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            last_used_at=str(payload.get("last_used_at", "")).strip(),
            parent_conversation_id=str(
                payload.get("parent_conversation_id", "")
            ).strip(),
            deleted=bool(payload.get("deleted", False)),
        )


class ConversationRegistry:
    """Persistent local registry for organizing remote conversations."""

    def __init__(
        self,
        *,
        path: Path,
        records: dict[str, ConversationRecord] | None = None,
        last_active_conversation_id: str = "",
    ) -> None:
        self.path = path
        self.records = records or {}
        self.last_active_conversation_id = last_active_conversation_id

    @classmethod
    def load(cls, path: str | Path | None = None) -> ConversationRegistry:
        """Load a registry from disk, or create an empty one if missing."""

        resolved_path = (
            Path(path).expanduser() if path is not None else default_registry_path()
        )
        if not resolved_path.exists():
            return cls(path=resolved_path)
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        raw_records = payload.get("records", []) if isinstance(payload, dict) else []
        records: dict[str, ConversationRecord] = {}
        for raw_record in raw_records:
            if not isinstance(raw_record, dict):
                continue
            record = ConversationRecord.from_dict(raw_record)
            if record.conversation_id:
                records[record.conversation_id] = record
        return cls(
            path=resolved_path,
            records=records,
            last_active_conversation_id=str(
                payload.get("last_active_conversation_id", "")
            ).strip()
            if isinstance(payload, dict)
            else "",
        )

    def save(self) -> None:
        """Persist the current registry state to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "last_active_conversation_id": self.last_active_conversation_id,
            "records": [
                asdict(record)
                for record in sorted(
                    self.records.values(),
                    key=lambda item: (
                        item.last_used_at or "",
                        item.updated_at or "",
                        item.created_at or "",
                        item.conversation_id,
                    ),
                    reverse=True,
                )
            ],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def get(self, conversation_id: str) -> ConversationRecord | None:
        """Return the local record for one remote conversation."""

        return self.records.get(conversation_id.strip())

    def ensure_record(self, conversation_id: str) -> ConversationRecord:
        """Return an existing record or create a new empty one."""

        normalized_id = conversation_id.strip()
        if not normalized_id:
            raise ValueError("Conversation id cannot be empty.")
        record = self.records.get(normalized_id)
        if record is None:
            record = ConversationRecord(conversation_id=normalized_id)
            self.records[normalized_id] = record
        return record

    def remember_active(self, conversation_id: str) -> ConversationRecord:
        """Mark one remote conversation as the last active entry."""

        record = self.ensure_record(conversation_id)
        now = _timestamp_now()
        if not record.created_at:
            record.created_at = now
        record.last_used_at = now
        self.last_active_conversation_id = record.conversation_id
        self.save()
        return record

    def clear_last_active(self, conversation_id: str | None = None) -> None:
        """Clear the last active pointer, optionally only if it matches."""

        if (
            conversation_id is None
            or self.last_active_conversation_id == conversation_id
        ):
            self.last_active_conversation_id = ""
            self.save()

    def update_remote_snapshot(
        self,
        conversation_id: str,
        *,
        remote_name: Any | None = None,
        remote_description: Any | None = None,
        remote_metadata: dict[str, Any] | None = None,
        remote_kind: Any | None = None,
        remote_model: Any | None = None,
        remote_agent_id: Any | None = None,
        created_at: Any | None = None,
        updated_at: Any | None = None,
        parent_conversation_id: Any | None = None,
        mark_deleted: bool | None = None,
    ) -> ConversationRecord:
        """Update the remote-facing snapshot for one record."""

        record = self.ensure_record(conversation_id)
        if remote_name is not None:
            record.remote_name = _normalize_text(remote_name)
        if remote_description is not None:
            record.remote_description = _normalize_text(remote_description)
        if remote_metadata is not None:
            record.remote_metadata = dict(remote_metadata)
        if remote_kind is not None:
            record.remote_kind = _normalize_text(remote_kind) or "model"
        if remote_model is not None:
            record.remote_model = _normalize_text(remote_model)
        if remote_agent_id is not None:
            record.remote_agent_id = _normalize_text(remote_agent_id)
        if created_at is not None:
            record.created_at = _normalize_text(created_at)
        if updated_at is not None:
            record.updated_at = _normalize_text(updated_at)
        if parent_conversation_id is not None:
            record.parent_conversation_id = _normalize_text(parent_conversation_id)
        if mark_deleted is not None:
            record.deleted = mark_deleted
        self.save()
        return record

    def migrate_conversation_id(self, old_id: str, new_id: str) -> ConversationRecord:
        """Move local metadata when the backend rotates a conversation id."""

        normalized_old = old_id.strip()
        normalized_new = new_id.strip()
        if not normalized_old or not normalized_new:
            raise ValueError("Both conversation ids are required for migration.")
        if normalized_old == normalized_new:
            return self.ensure_record(normalized_new)

        old_record = self.records.pop(normalized_old, None)
        if old_record is None:
            record = self.ensure_record(normalized_new)
        else:
            old_record.conversation_id = normalized_new
            record = old_record
            self.records[normalized_new] = record
        for candidate in self.records.values():
            if candidate.parent_conversation_id == normalized_old:
                candidate.parent_conversation_id = normalized_new
        if self.last_active_conversation_id == normalized_old:
            self.last_active_conversation_id = normalized_new
        self.save()
        return record

    def mark_deleted(self, conversation_id: str) -> None:
        """Mark a conversation as deleted locally."""

        record = self.ensure_record(conversation_id)
        record.deleted = True
        if self.last_active_conversation_id == record.conversation_id:
            self.last_active_conversation_id = ""
        self.save()

    def forget(self, conversation_id: str) -> bool:
        """Delete one record from local state only."""

        normalized_id = conversation_id.strip()
        removed = self.records.pop(normalized_id, None) is not None
        if self.last_active_conversation_id == normalized_id:
            self.last_active_conversation_id = ""
        if removed:
            self.save()
        return removed

    def set_alias(self, conversation_id: str, alias: str) -> ConversationRecord:
        """Set the local alias for a conversation."""

        record = self.ensure_record(conversation_id)
        record.alias = alias.strip()
        self.save()
        return record

    def set_note(self, conversation_id: str, note: str) -> ConversationRecord:
        """Set the local free-form note for a conversation."""

        record = self.ensure_record(conversation_id)
        record.note = note.strip()
        self.save()
        return record

    def add_tag(self, conversation_id: str, tag: str) -> ConversationRecord:
        """Add a local tag to a conversation."""

        normalized_tag = tag.strip()
        if not normalized_tag:
            raise ValueError("Tag cannot be empty.")
        record = self.ensure_record(conversation_id)
        if normalized_tag not in record.tags:
            record.tags.append(normalized_tag)
            record.tags.sort()
            self.save()
        return record

    def remove_tag(self, conversation_id: str, tag: str) -> ConversationRecord:
        """Remove a local tag from a conversation."""

        record = self.ensure_record(conversation_id)
        record.tags = [item for item in record.tags if item != tag.strip()]
        self.save()
        return record

    def resolve_reference(self, reference: str) -> ConversationRecord | None:
        """Resolve a user reference by exact id or unique alias."""

        normalized = reference.strip()
        if not normalized:
            return None
        direct = self.records.get(normalized)
        if direct is not None:
            return direct
        matches = [
            record for record in self.records.values() if record.alias == normalized
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def bookmarks(self) -> list[ConversationRecord]:
        """Return locally curated conversations."""

        return sorted(
            [
                record
                for record in self.records.values()
                if record.alias or record.tags or record.note
            ],
            key=lambda item: (
                item.last_used_at or "",
                item.updated_at or "",
                item.created_at or "",
                item.conversation_id,
            ),
            reverse=True,
        )
