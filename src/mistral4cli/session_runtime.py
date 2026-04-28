# mypy: disable-error-code="assignment,attr-defined,has-type"
# mypy: disable-error-code="no-any-return,no-untyped-def"
"""Runtime, status, and remote-conversation management for sessions."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, cast

from mistralai.client import Mistral

from mistral4cli.local_mistral import (
    BackendKind,
    ConversationConfig,
    get_client_timeout_ms,
    set_client_timeout_ms,
)
from mistral4cli.mcp_bridge import MCPToolResult
from mistral4cli.session_primitives import (
    DEFAULT_SYSTEM_PROMPT,
    REMOTE_CONTEXT_WINDOWS,
    SessionStatusSnapshot,
    _BackendState,
    _conversation_content_segments,
    _field,
    _join_segments,
    _metadata_matches,
    _normalize_usage_snapshot,
    _summarize_conversation_entry,
    render_defaults_summary,
)

logger = logging.getLogger("mistral4cli.session")


class SessionRuntimeMixin:
    """Own backend switching, session defaults, and conversation registry state."""

    def __post_init__(self) -> None:
        """Normalize the session state after dataclass initialization."""

        if self.stdout is None:
            import sys

            self.stdout = sys.stdout
        self.system_prompt = self.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.context = self.context.normalized()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if self.conversations.enabled and self.backend_kind is not BackendKind.REMOTE:
            self.conversations = ConversationConfig(enabled=False, store=True)
        logger.debug(
            "Session initialized backend=%s model=%s tools=%s conversations=%s",
            self.backend_kind.value,
            self.model_id,
            self.tool_bridge is not None,
            self.conversations.enabled,
        )

    def reset(self) -> None:
        """Reset the conversation to the system prompt."""

        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._mark_context_status_dirty()
        self.conversation_id = None
        self.conversation_resume_source = "new"
        self._missing_reasoning_notice_shown = False
        self._set_status("idle")
        logger.info(
            "Conversation reset backend=%s model=%s conversations=%s",
            self.backend_kind.value,
            self.model_id,
            self.conversations.enabled,
        )

    def set_system_prompt(self, system_prompt: str) -> None:
        """Replace the active system prompt and reset the conversation."""

        self.system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.reset()

    def describe_tool_status(self) -> str:
        """Return a compact tool status summary."""

        if self.tool_bridge is None:
            return "FireCrawl MCP: disabled"
        return self.tool_bridge.runtime_summary()

    def describe_tools(self) -> str:
        """Return a live tool catalog summary."""

        if self.tool_bridge is None:
            return "FireCrawl MCP: disabled"
        return self.tool_bridge.describe_tools()

    def describe_defaults(self) -> str:
        """Render the active runtime defaults as human-readable text."""

        assert self.stdout is not None
        return render_defaults_summary(
            backend_kind=self.backend_kind,
            model_id=self.model_id,
            server_url=self.server_url,
            timeout_ms=self.timeout_ms,
            generation=self._display_generation(),
            stream_enabled=self.stream_enabled,
            reasoning_enabled=self.show_reasoning,
            thinking_visible=self.show_thinking,
            conversations=self.conversations,
            context=self.context,
            conversation_id=self.conversation_id,
            tool_summary=self.describe_tool_status(),
            logging_summary=self.logging_summary,
            stream=self.stdout,
        )

    def switch_backend(
        self,
        *,
        client: Mistral,
        backend_kind: BackendKind,
        model_id: str,
        server_url: str | None,
    ) -> None:
        """Swap the active model backend and reset the conversation."""

        self.client = client
        self.backend_kind = backend_kind
        self.model_id = model_id
        self.server_url = server_url
        self.conversations = ConversationConfig(enabled=False, store=True)
        self.conversation_id = None
        self._previous_backend_state = None
        logger.info(
            "Backend switched backend=%s model=%s server=%s",
            backend_kind.value,
            model_id,
            server_url,
        )
        self.reset()

    def enable_conversations(
        self,
        *,
        client: Mistral,
        model_id: str,
        store: bool,
        server_url: str | None = None,
    ) -> None:
        """Enable Mistral Cloud Conversations mode and reset the active chat."""

        if not self.conversations.enabled:
            self._previous_backend_state = _BackendState(
                client=self.client,
                backend_kind=self.backend_kind,
                model_id=self.model_id,
                server_url=self.server_url,
            )
        self.client = client
        self.backend_kind = BackendKind.REMOTE
        self.model_id = model_id
        self.server_url = server_url
        self.conversations = ConversationConfig(
            enabled=True,
            store=store,
            resume_policy=self.conversations.resume_policy,
        )
        logger.info("Conversations enabled model=%s store=%s", model_id, store)
        self.reset()

    def disable_conversations(self) -> None:
        """Disable Conversations mode and restore the previous backend if known."""

        previous = self._previous_backend_state
        self.conversations = ConversationConfig(
            enabled=False,
            store=True,
            resume_policy=self.conversations.resume_policy,
        )
        self.conversation_id = None
        self._previous_backend_state = None
        self._missing_reasoning_notice_shown = False
        if previous is not None:
            self.client = previous.client
            self.backend_kind = previous.backend_kind
            self.model_id = previous.model_id
            self.server_url = previous.server_url
        logger.info("Conversations disabled backend=%s", self.backend_kind.value)
        self.reset()

    def set_conversation_store(self, store: bool) -> None:
        """Update Conversations storage policy and start a fresh conversation."""

        self.conversations = ConversationConfig(
            enabled=self.conversations.enabled,
            store=store,
            resume_policy=self.conversations.resume_policy,
        )
        self.reset()

    def conversations_status_text(self) -> str:
        """Return a user-facing Conversations status summary."""

        state = "on" if self.conversations.enabled else "off"
        store = "on" if self.conversations.store else "off"
        resume = self.conversations.resume_policy
        conversation_id = self.conversation_id or "not started"
        source = self.conversation_resume_source
        return (
            "Conversations: "
            f"{state} store={store} resume={resume} "
            f"source={source} conversation_id={conversation_id}"
        )

    def current_conversation_text(self) -> str:
        """Render the current active conversation and pending local state."""

        lines = [self.conversations_status_text()]
        if self.conversation_registry is not None and self.conversation_id:
            record = self.conversation_registry.get(self.conversation_id)
            if record is not None:
                lines.append(self._format_registry_record(record, include_note=True))
        pending = self.pending_conversation_text()
        if pending:
            lines.append(pending)
        return "\n".join(lines)

    def pending_conversation_text(self) -> str:
        """Return a compact summary of pending conversation settings."""

        if not self.pending_conversation.active():
            return ""
        parts: list[str] = []
        if self.pending_conversation.name:
            parts.append(f"name={self.pending_conversation.name}")
        if self.pending_conversation.description:
            parts.append(f"description={self.pending_conversation.description}")
        if self.pending_conversation.metadata:
            metadata = ", ".join(
                f"{key}={value}"
                for key, value in sorted(self.pending_conversation.metadata.items())
            )
            parts.append(f"metadata={metadata}")
        return "Pending conversation settings: " + " | ".join(parts)

    def set_pending_conversation_name(self, value: str) -> None:
        """Set the pending remote conversation name."""

        self.pending_conversation = replace(
            self.pending_conversation,
            name=value.strip(),
        )

    def set_pending_conversation_description(self, value: str) -> None:
        """Set the pending remote conversation description."""

        self.pending_conversation = replace(
            self.pending_conversation,
            description=value.strip(),
        )

    def set_pending_conversation_metadata(self, key: str, value: str) -> None:
        """Set one pending remote conversation metadata pair."""

        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("Metadata key cannot be empty.")
        metadata = dict(self.pending_conversation.metadata)
        metadata[normalized_key] = value.strip()
        self.pending_conversation = replace(
            self.pending_conversation,
            metadata=metadata,
        )

    def clear_pending_conversation_name(self) -> None:
        """Clear the pending remote conversation name."""

        self.pending_conversation = replace(self.pending_conversation, name="")

    def clear_pending_conversation_description(self) -> None:
        """Clear the pending remote conversation description."""

        self.pending_conversation = replace(
            self.pending_conversation,
            description="",
        )

    def clear_pending_conversation_metadata(self, key: str | None = None) -> None:
        """Clear one or all pending remote conversation metadata fields."""

        if key is None:
            metadata: dict[str, str] = {}
        else:
            metadata = dict(self.pending_conversation.metadata)
            metadata.pop(key.strip(), None)
        self.pending_conversation = replace(
            self.pending_conversation,
            metadata=metadata,
        )

    def apply_conversation_resume_policy(self, resume_policy: str) -> None:
        """Update the configured startup resume policy."""

        normalized = resume_policy.strip().lower()
        if normalized not in {"last", "new", "prompt"}:
            raise ValueError("Resume policy must be one of: last, new, prompt.")
        self.conversations = ConversationConfig(
            enabled=self.conversations.enabled,
            store=self.conversations.store,
            resume_policy=normalized,
        )

    def attach_remote_conversation(
        self,
        conversation_id: str,
        *,
        source: str = "manual",
    ) -> str:
        """Attach the current session to an existing remote conversation id."""

        normalized_id = conversation_id.strip()
        if not normalized_id:
            raise ValueError("Conversation id cannot be empty.")
        payload = self.client.beta.conversations.get(conversation_id=normalized_id)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._mark_context_status_dirty()
        self._sync_conversation_id(
            normalized_id,
            source=source,
            payload=payload,
        )
        return f"Attached to conversation {normalized_id}."

    def resolve_conversation_reference(self, reference: str | None) -> str:
        """Resolve a user reference to a concrete remote conversation id."""

        if reference is None or not reference.strip():
            if self.conversation_id:
                return self.conversation_id
            raise ValueError("No active conversation id yet.")
        normalized = reference.strip()
        if self.conversation_registry is None:
            return normalized
        record = self.conversation_registry.resolve_reference(normalized)
        if record is not None:
            return record.conversation_id
        return normalized

    def list_remote_conversations(
        self,
        *,
        page: int = 0,
        page_size: int = 20,
        metadata: dict[str, str] | None = None,
        bookmarks_only: bool = False,
    ) -> str:
        """List remote conversations with local overlay metadata."""

        if bookmarks_only:
            if self.conversation_registry is None:
                return "No local conversation registry is configured."
            records = self.conversation_registry.bookmarks()
            if not records:
                return "No bookmarked conversations."
            lines = ["Bookmarked conversations:"]
            for index, record in enumerate(records, start=1):
                lines.append(
                    f"{index}. "
                    f"{self._format_registry_record(record, include_note=True)}"
                )
            return "\n".join(lines)

        payload = self.client.beta.conversations.list(
            page=page,
            page_size=page_size,
            metadata={},
        )
        conversations = list(payload or [])
        if metadata:
            filtered: list[Any] = []
            for conversation in conversations:
                conversation_id = str(_field(conversation, "id", "") or "")
                if not conversation_id:
                    continue
                detailed = self.client.beta.conversations.get(
                    conversation_id=conversation_id
                )
                self._update_registry_from_remote_payload(conversation_id, detailed)
                effective_metadata = _field(detailed, "metadata", None)
                if (
                    effective_metadata is None
                    and self.conversation_registry is not None
                ):
                    local_record = self.conversation_registry.get(conversation_id)
                    if local_record is not None and local_record.remote_metadata:
                        effective_metadata = local_record.remote_metadata
                if _metadata_matches(effective_metadata, metadata):
                    filtered.append(detailed)
            conversations = filtered
        if not conversations:
            return "No remote conversations found."
        lines = [
            f"Remote conversations page={page} size={page_size}:",
        ]
        for index, conversation in enumerate(conversations, start=1):
            conversation_id = str(_field(conversation, "id", "") or "")
            self._update_registry_from_remote_payload(conversation_id, conversation)
            label = self._format_remote_conversation_summary(conversation)
            if self.conversation_id == conversation_id:
                label = f"{label} [active]"
            lines.append(f"{index}. {label}")
        return "\n".join(lines)

    def show_remote_conversation(self, reference: str | None = None) -> str:
        """Render detailed information for one remote conversation."""

        conversation_id = self.resolve_conversation_reference(reference)
        payload = self.client.beta.conversations.get(conversation_id=conversation_id)
        self._update_registry_from_remote_payload(conversation_id, payload)
        lines = [f"Conversation {conversation_id}:"]
        lines.extend(self._render_remote_conversation_details(payload))
        if self.conversation_registry is not None:
            record = self.conversation_registry.get(conversation_id)
            if record is not None:
                lines.append("Local registry:")
                lines.append(self._format_registry_record(record, include_note=True))
        return "\n".join(lines)

    def conversation_history_text(
        self,
        *,
        messages_only: bool = False,
        conversation_id: str | None = None,
    ) -> str:
        """Fetch and render the active remote Conversation history."""

        if not self.conversations.enabled:
            return "Conversations mode is off."
        target_id = self.resolve_conversation_reference(conversation_id)
        conversations = self.client.beta.conversations
        payload = (
            conversations.get_messages(conversation_id=target_id)
            if messages_only
            else conversations.get_history(conversation_id=target_id)
        )
        entries = _field(payload, "messages" if messages_only else "entries", []) or []
        if not entries:
            return "No conversation entries."
        lines = [f"Conversation {target_id}:"]
        for index, entry in enumerate(entries, start=1):
            entry_type = str(_field(entry, "type", "entry"))
            role = _field(entry, "role")
            entry_id = str(_field(entry, "id", "") or "")
            label = f"{index}. {entry_type}"
            if role:
                label = f"{label} {role}"
            if entry_id:
                label = f"{label} id={entry_id}"
            text = _summarize_conversation_entry(entry)
            lines.append(f"{label}: {text}" if text else label)
        if not messages_only:
            lines.append(
                'Hint: use "/conv restart <entry_id>" to branch from one history entry.'
            )
        return "\n".join(lines)

    def delete_remote_conversation(self, conversation_id: str | None = None) -> str:
        """Delete a remote Conversation and clear local state when active."""

        if not self.conversations.enabled:
            return "Conversations mode is off."
        target_id = self.resolve_conversation_reference(conversation_id)
        self.client.beta.conversations.delete(conversation_id=target_id)
        if self.conversation_registry is not None:
            self.conversation_registry.mark_deleted(target_id)
        if self.conversation_id == target_id:
            self.conversation_id = None
            self.conversation_resume_source = "new"
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self._mark_context_status_dirty()
        return f"Deleted conversation {target_id}."

    def restart_remote_conversation(
        self,
        *,
        from_entry_id: str,
        conversation_id: str | None = None,
    ) -> str:
        """Restart a remote conversation from one entry and switch to the new id."""

        target_id = self.resolve_conversation_reference(conversation_id)
        if not from_entry_id.strip():
            raise ValueError("Entry id cannot be empty.")
        parent_id = target_id
        response = self.client.beta.conversations.restart(
            conversation_id=target_id,
            from_entry_id=from_entry_id.strip(),
            inputs="",
            store=self.conversations.store,
            completion_args=cast(Any, self._conversation_completion_args()),
            metadata=cast(
                Any,
                self.pending_conversation.metadata
                if self.pending_conversation.metadata
                else None,
            ),
        )
        new_id = str(_field(response, "conversation_id", "") or "")
        if not new_id:
            raise RuntimeError("Restart returned no conversation id.")
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._mark_context_status_dirty()
        self._sync_conversation_id(
            new_id,
            source="restart",
            parent_conversation_id=parent_id,
        )
        outputs = _field(response, "outputs", []) or []
        summary = ""
        for output in outputs:
            if _field(output, "type") == "message.output":
                segments = _conversation_content_segments(_field(output, "content"))
                answer = _join_segments(segments, kind="answer")
                if answer:
                    summary = answer
                    break
        if summary:
            return (
                f"Restarted from {from_entry_id.strip()}. "
                f"Active conversation: {new_id}.\nAssistant: {summary}"
            )
        return f"Restarted from {from_entry_id.strip()}. Active conversation: {new_id}."

    def forget_local_conversation(self, reference: str) -> str:
        """Forget one local registry record without touching the remote object."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        resolved = self.conversation_registry.resolve_reference(reference.strip())
        conversation_id = (
            resolved.conversation_id if resolved is not None else reference.strip()
        )
        if not conversation_id:
            raise ValueError("Conversation id cannot be empty.")
        if not self.conversation_registry.forget(conversation_id):
            return f"No local record for conversation {conversation_id}."
        return f"Forgot local record for conversation {conversation_id}."

    def set_local_conversation_alias(self, reference: str, alias: str) -> str:
        """Set a local alias for one conversation."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        conversation_id = self.resolve_conversation_reference(reference)
        record = self.conversation_registry.set_alias(conversation_id, alias)
        return f"Alias for {record.conversation_id} set to {record.alias or '(empty)'}."

    def set_local_conversation_note(self, reference: str, note: str) -> str:
        """Set a local note for one conversation."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        conversation_id = self.resolve_conversation_reference(reference)
        self.conversation_registry.set_note(conversation_id, note)
        return f"Note updated for {conversation_id}."

    def add_local_conversation_tag(self, reference: str, tag: str) -> str:
        """Add one local tag to a conversation."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        conversation_id = self.resolve_conversation_reference(reference)
        record = self.conversation_registry.add_tag(conversation_id, tag)
        return f"Tags for {conversation_id}: {', '.join(record.tags) or '(none)'}."

    def remove_local_conversation_tag(self, reference: str, tag: str) -> str:
        """Remove one local tag from a conversation."""

        if self.conversation_registry is None:
            return "No local conversation registry is configured."
        conversation_id = self.resolve_conversation_reference(reference)
        record = self.conversation_registry.remove_tag(conversation_id, tag)
        return f"Tags for {conversation_id}: {', '.join(record.tags) or '(none)'}."

    @property
    def timeout_ms(self) -> int:
        """Return the active request timeout in milliseconds."""

        return get_client_timeout_ms(self.client)

    def set_timeout_ms(self, timeout_ms: int) -> None:
        """Update the active request timeout in milliseconds."""

        set_client_timeout_ms(self.client, timeout_ms)

    def visible_reasoning_supported(self) -> bool:
        """Return whether the active backend can render visible reasoning."""

        return True

    def reasoning_status_text(self) -> str:
        """Return a user-facing reasoning-request status string."""

        state = "on" if self.show_reasoning else "off"
        if self.backend_kind is BackendKind.REMOTE:
            if self.conversations.enabled:
                return (
                    "Reasoning request: "
                    f"{state} (remote Conversations, requested best-effort)"
                )
            return f"Reasoning request: {state} (remote SDK)"
        return f"Reasoning request: {state} (local backend)"

    def thinking_status_text(self) -> str:
        """Return a user-facing thinking-render status string."""

        state = "on" if self.show_thinking else "off"
        return f"Thinking display: {state}"

    def set_reasoning_visibility(self, visible: bool) -> None:
        """Enable or disable reasoning requests to the backend."""

        self.show_reasoning = visible
        self._missing_reasoning_notice_shown = False

    def toggle_reasoning_visibility(self) -> bool:
        """Toggle backend reasoning requests and return the new state."""

        self.show_reasoning = not self.show_reasoning
        self._missing_reasoning_notice_shown = False
        return self.show_reasoning

    def set_thinking_visibility(self, visible: bool) -> None:
        """Enable or disable local rendering of thinking blocks."""

        self.show_thinking = visible
        self._missing_reasoning_notice_shown = False

    def toggle_thinking_visibility(self) -> bool:
        """Toggle local thinking rendering and return the new state."""

        self.show_thinking = not self.show_thinking
        self._missing_reasoning_notice_shown = False
        return self.show_thinking

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute a tool through the active bridge."""

        return self._call_tool_bridge(public_name, arguments)

    def status_snapshot(self) -> SessionStatusSnapshot:
        """Return the current live status for interactive UI rendering."""

        max_context_tokens = self._model_context_window()
        estimated_context = self._status_context_snapshot()
        last_usage = self._with_context_window(self._last_usage, max_context_tokens)
        combined_cumulative = self._cumulative_usage
        if self._turn_usage_accumulator is not None:
            combined_cumulative = self._turn_usage_accumulator.merge(
                combined_cumulative
            )
        cumulative_usage = self._with_context_window(
            combined_cumulative, max_context_tokens
        )
        return SessionStatusSnapshot(
            phase=self._status_phase,
            detail=self._status_detail,
            estimated_context=estimated_context,
            last_usage=last_usage,
            cumulative_usage=cumulative_usage,
        )

    def _normalize_user_content(
        self, content: str | list[dict[str, Any]]
    ) -> str | list[dict[str, Any]] | None:
        if isinstance(content, str):
            clean_text = content.strip()
            if not clean_text:
                return None
            return clean_text
        if not content:
            return None
        return content

    def _effective_prompt_mode(self) -> str | None:
        if self.backend_kind is BackendKind.REMOTE:
            return None
        if not self.show_reasoning:
            return None
        return self.generation.prompt_mode

    def _display_generation(self):
        return replace(self.generation, prompt_mode=self._effective_prompt_mode())

    def _set_status(self, phase: str, detail: str | None = None) -> None:
        if self._status_phase == phase and self._status_detail == detail:
            return
        self._status_phase = phase
        self._status_detail = detail
        if self.status_callback is not None:
            self.status_callback()

    def _record_usage(self, raw_usage: Any) -> None:
        usage = _normalize_usage_snapshot(
            raw_usage,
            max_context_tokens=self._model_context_window(),
        )
        if usage is None or usage.is_empty():
            return
        self._last_usage = usage
        self._turn_usage_accumulator = usage.merge(self._turn_usage_accumulator)
        if self.status_callback is not None:
            self.status_callback()

    def _mark_context_status_dirty(self) -> None:
        self._context_status_dirty = True

    def _status_context_snapshot(self):
        if self.conversations.enabled:
            return None
        if self._context_status_dirty or self._cached_context_status is None:
            self._cached_context_status = self._compute_context_status()
            self._context_status_dirty = False
        return self._cached_context_status

    def _model_context_window(self) -> int | None:
        if self.backend_kind is not BackendKind.REMOTE:
            return self.context.local_window_tokens
        normalized_model = self.model_id.strip().lower()
        return REMOTE_CONTEXT_WINDOWS.get(
            normalized_model,
            self.context.remote_window_tokens,
        )

    def _with_context_window(self, usage, max_context_tokens: int | None):
        if usage is None:
            return None
        return replace(usage, max_context_tokens=max_context_tokens)

    def _commit_turn_usage(self) -> None:
        if self._turn_usage_accumulator is None:
            return
        self._cumulative_usage = self._turn_usage_accumulator.merge(
            self._cumulative_usage
        )
        self._turn_usage_accumulator = None
