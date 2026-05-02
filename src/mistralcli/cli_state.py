"""Shared REPL state and status rendering helpers for the CLI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mistralcli.attachments import (
    build_document_message,
    build_image_message,
    build_remote_document_message,
    build_remote_image_message,
)
from mistralcli.local_mistral import BackendKind
from mistralcli.session import (
    ContextStatus,
    MistralSession,
    SessionStatusSnapshot,
    UsageSnapshot,
)


@dataclass(slots=True)
class _InputHistory:
    """In-memory REPL input history with up/down navigation state."""

    entries: list[str] = field(default_factory=list)
    browse_index: int | None = None
    draft: str = ""

    def add(self, line: str) -> None:
        """Append a non-empty line unless it duplicates the last entry."""

        clean = line.strip()
        if not clean:
            self.reset_navigation()
            return
        if not self.entries or self.entries[-1] != clean:
            self.entries.append(clean)
        self.reset_navigation()

    def previous(self, current_buffer: str) -> str:
        """Move one step back in history."""

        if not self.entries:
            return current_buffer
        if self.browse_index is None:
            self.draft = current_buffer
            self.browse_index = len(self.entries) - 1
        elif self.browse_index > 0:
            self.browse_index -= 1
        return self.entries[self.browse_index]

    def next(self) -> str:
        """Move one step forward in history."""

        if self.browse_index is None:
            return self.draft
        if self.browse_index < len(self.entries) - 1:
            self.browse_index += 1
            return self.entries[self.browse_index]
        draft = self.draft
        self.reset_navigation()
        return draft

    def reset_navigation(self) -> None:
        """Reset history navigation state after a committed line."""

        self.browse_index = None
        self.draft = ""


@dataclass(slots=True)
class _PendingAttachment:
    """Attachment selection staged for the next user prompt."""

    kind: str
    summary: str
    paths: list[Path]


@dataclass(slots=True)
class _ReplState:
    """Mutable REPL state that survives between commands."""

    pending_attachment: _PendingAttachment | None = None
    active_images: list[Path] = field(default_factory=list)
    active_documents: list[Path] = field(default_factory=list)


def _set_active_attachment(
    repl_state: _ReplState,
    *,
    kind: str,
    paths: Sequence[Path],
) -> None:
    active_paths = [path.expanduser() for path in paths]
    if kind == "image":
        repl_state.active_images = active_paths
        return
    repl_state.active_documents = active_paths


def _clear_attachments(
    repl_state: _ReplState,
    *,
    clear_images: bool,
    clear_documents: bool,
    clear_pending: bool = True,
) -> None:
    if clear_pending and repl_state.pending_attachment is not None:
        pending_kind = repl_state.pending_attachment.kind
        if (pending_kind == "image" and clear_images) or (
            pending_kind == "document" and clear_documents
        ):
            repl_state.pending_attachment = None
    if clear_images:
        repl_state.active_images = []
    if clear_documents:
        repl_state.active_documents = []


def _repl_prompt(repl_state: _ReplState) -> str:
    tokens: list[str] = []
    if repl_state.pending_attachment is not None:
        tokens.append(f"stage:{repl_state.pending_attachment.kind}")
    if repl_state.active_images:
        tokens.append(f"img:{len(repl_state.active_images)}")
    if repl_state.active_documents:
        tokens.append(f"doc:{len(repl_state.active_documents)}")
    if not tokens:
        return "MC> "
    return f"MC[{','.join(tokens)}]> "


def _format_estimated_context_for_status(
    status: ContextStatus | None,
    *,
    conversations_enabled: bool,
) -> str:
    if conversations_enabled:
        return "est:backend"
    if status is None:
        return "est:-"
    return f"est:{status.estimated_tokens}/{status.window_tokens}"


def _format_usage_for_status(usage: UsageSnapshot | None) -> str:
    if usage is None or usage.total_tokens is None:
        return "last:-"
    if usage.max_context_tokens is None:
        return f"last:{usage.total_tokens}/?"
    return f"last:{usage.total_tokens}/{usage.max_context_tokens}"


def _format_session_total_for_status(usage: UsageSnapshot | None) -> str:
    if usage is None or usage.total_tokens is None:
        return "usage:-"
    return f"usage:{usage.total_tokens}"


def _status_phase_label(snapshot: SessionStatusSnapshot) -> str:
    if snapshot.phase == "tool" and snapshot.detail:
        return f"tool:{snapshot.detail}"
    if snapshot.phase == "thinking":
        return "thinking..."
    return snapshot.phase


def _repl_status_line(session: MistralSession, repl_state: _ReplState) -> str:
    snapshot = session.status_snapshot()
    parts = [
        _status_phase_label(snapshot),
        session.backend_kind.value,
        session.model_id,
        f"reasoning:{'on' if session.show_reasoning else 'off'}",
        f"thinking:{'on' if session.show_thinking else 'off'}",
        f"conv:{'on' if session.conversations.enabled else 'off'}",
    ]
    if session.conversations.enabled and session.conversation_id:
        parts.append(f"cid:{session.conversation_id[:8]}")
    if repl_state.pending_attachment is not None:
        parts.append(f"stage:{repl_state.pending_attachment.kind}")
    if repl_state.active_images:
        parts.append(f"img:{len(repl_state.active_images)}")
    if repl_state.active_documents:
        parts.append(f"doc:{len(repl_state.active_documents)}")
    parts.append(
        _format_estimated_context_for_status(
            snapshot.estimated_context,
            conversations_enabled=session.conversations.enabled,
        )
    )
    parts.append(_format_usage_for_status(snapshot.last_usage))
    parts.append(_format_session_total_for_status(snapshot.cumulative_usage))
    return " | ".join(parts)


def _render_session_status(
    session: MistralSession,
    repl_state: _ReplState,
) -> str:
    snapshot = session.status_snapshot()
    phase = _status_phase_label(snapshot)
    server = "Mistral Cloud"
    if session.backend_kind is BackendKind.LOCAL:
        server = session.server_url or "local server"

    lines = ["Session status:"]
    lines.append(f"Phase: {phase}")
    lines.append(
        "Runtime: "
        f"backend={session.backend_kind.value} "
        f"server={server} "
        f"model={session.model_id}"
    )
    lines.append(
        "Response: "
        f"stream={'on' if session.stream_enabled else 'off'} "
        f"reasoning={'on' if session.show_reasoning else 'off'} "
        f"thinking={'on' if session.show_thinking else 'off'} "
        f"timeout={session.timeout_ms}ms"
    )
    lines.append(
        "Conversations: "
        f"mode={'on' if session.conversations.enabled else 'off'} "
        f"store={'on' if session.conversations.store else 'off'} "
        f"resume={session.conversations.resume_policy} "
        f"id={session.conversation_id or 'not started'}"
    )
    estimated = _format_estimated_context_for_status(
        snapshot.estimated_context,
        conversations_enabled=session.conversations.enabled,
    )
    last_usage = _format_usage_for_status(snapshot.last_usage)
    total_usage = _format_session_total_for_status(snapshot.cumulative_usage)
    lines.append(f"Context: {estimated} {last_usage} {total_usage}")
    attachment_parts: list[str] = []
    if repl_state.pending_attachment is not None:
        attachment_parts.append(f"stage={repl_state.pending_attachment.kind}")
    if repl_state.active_images:
        attachment_parts.append(f"images={len(repl_state.active_images)}")
    if repl_state.active_documents:
        attachment_parts.append(f"documents={len(repl_state.active_documents)}")
    if attachment_parts:
        lines.append("Attachments: " + " ".join(attachment_parts))
    pending = session.pending_conversation_text()
    if pending:
        lines.append(pending)
    return "\n".join(lines)


def _build_active_attachment_message(
    session: MistralSession,
    *,
    prompt: str,
    image_paths: Sequence[Path],
    document_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    if not image_paths and not document_paths:
        raise ValueError("At least one active attachment is required")

    text_lines = [prompt.strip()]
    if image_paths:
        text_lines.extend(["", "Active images:"])
        text_lines.extend(f"- {path.name}" for path in image_paths)
    if document_paths:
        text_lines.extend(["", "Active documents:"])
        text_lines.extend(f"- {path.name}" for path in document_paths)
        if session.backend_kind is BackendKind.REMOTE:
            text_lines.append("")
            text_lines.append("The active documents are attached natively.")
        else:
            text_lines.append("")
            text_lines.append("The active documents are attached as OCR images.")

    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(text_lines)}]
    if image_paths:
        image_message = (
            build_remote_image_message(image_paths, prompt=prompt)
            if session.backend_kind is BackendKind.REMOTE
            else build_image_message(image_paths, prompt=prompt)
        )
        content.extend(image_message[1:])
    if document_paths:
        document_message = (
            build_remote_document_message(document_paths, prompt=prompt)
            if session.backend_kind is BackendKind.REMOTE
            else build_document_message(document_paths, prompt=prompt)
        )
        content.extend(document_message[1:])
    return content


def _parse_command(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped[0] not in {"/", ":"}:
        return None
    command_body = stripped[1:].strip()
    if not command_body:
        return None
    command, _, argument = command_body.partition(" ")
    return command.lower(), argument.strip()
