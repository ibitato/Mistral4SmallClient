# mypy: disable-error-code="attr-defined,has-type,no-any-return,no-untyped-def"
# pyright: reportAttributeAccessIssue=false
"""Context estimation, compaction, and transport selection for sessions."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from mistral4cli.local_mistral import BackendKind
from mistral4cli.session_primitives import (
    CompactResult,
    ContextStatus,
    _content_segments_from_value,
    _estimate_messages_tokens,
    _estimate_tools_tokens,
    _join_segments,
    _render_messages_for_compaction,
)

logger = logging.getLogger("mistral4cli.session")


class SessionContextMixin:
    """Estimate and compact client-managed context before model turns."""

    def context_status(self) -> ContextStatus:
        """Return the current estimated context state for chat completions."""

        return self._compute_context_status()

    def _compute_context_status(self) -> ContextStatus:
        """Compute a fresh client-side estimate for the current chat history."""

        window_tokens = self._model_context_window() or self.context.local_window_tokens
        threshold_tokens = int(window_tokens * self.context.threshold)
        reserve_tokens = self._effective_context_reserve()
        estimated_tokens = self.estimate_context_tokens(self.messages) + reserve_tokens
        return ContextStatus(
            estimated_tokens=estimated_tokens,
            window_tokens=window_tokens,
            threshold_tokens=threshold_tokens,
            reserve_tokens=reserve_tokens,
            auto_compact=self.context.auto_compact,
        )

    def context_status_text(self) -> str:
        """Return a user-facing context policy summary."""

        status = self.context_status()
        auto = "on" if status.auto_compact else "off"
        threshold_percent = round(self.context.threshold * 100)
        mode = "backend-managed" if self.conversations.enabled else "client-managed"
        return (
            f"Context: {mode} estimate={status.estimated_tokens}/"
            f"{status.window_tokens} threshold={threshold_percent}% "
            f"reserve={status.reserve_tokens} "
            f"keep_turns={self.context.keep_recent_turns} "
            f"auto={auto}"
        )

    def configure_context(
        self,
        *,
        auto_compact: bool | None = None,
        threshold: float | None = None,
        reserve_tokens: int | None = None,
        keep_recent_turns: int | None = None,
    ) -> None:
        """Update mutable context policy knobs at runtime."""

        self.context = replace(
            self.context,
            auto_compact=(
                self.context.auto_compact if auto_compact is None else auto_compact
            ),
            threshold=self.context.threshold if threshold is None else threshold,
            reserve_tokens=(
                self.context.reserve_tokens
                if reserve_tokens is None
                else reserve_tokens
            ),
            keep_recent_turns=(
                self.context.keep_recent_turns
                if keep_recent_turns is None
                else keep_recent_turns
            ),
        ).normalized()
        self._mark_context_status_dirty()

    def estimate_context_tokens(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Estimate request tokens when backend tokenizers are unavailable."""

        active_messages = self.messages if messages is None else messages
        return _estimate_messages_tokens(active_messages) + _estimate_tools_tokens(
            tools
        )

    def compact_context(self) -> CompactResult:
        """Summarize old chat-completions history and keep recent turns."""

        before_tokens = self.estimate_context_tokens(self.messages)
        window_tokens = self._model_context_window() or self.context.local_window_tokens
        threshold_tokens = int(window_tokens * self.context.threshold)
        if self.conversations.enabled:
            return CompactResult(
                changed=False,
                reason="Conversations mode stores context on the Mistral backend.",
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                window_tokens=window_tokens,
                threshold_tokens=threshold_tokens,
            )

        older_messages, recent_messages = self._split_compactable_history()
        if not older_messages:
            return CompactResult(
                changed=False,
                reason="Nothing old enough to compact.",
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                window_tokens=window_tokens,
                threshold_tokens=threshold_tokens,
            )

        summary = self._summarize_messages_for_compaction(older_messages)
        if not summary.strip():
            return CompactResult(
                changed=False,
                reason="The model did not return a compaction summary.",
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                window_tokens=window_tokens,
                threshold_tokens=threshold_tokens,
            )

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "assistant",
                "content": (
                    "[Compacted previous context]\n"
                    f"{summary.strip()}\n"
                    "[End compacted context]"
                ),
            },
            *recent_messages,
        ]
        self._mark_context_status_dirty()
        after_tokens = self.estimate_context_tokens(self.messages)
        logger.info(
            "Context compacted before_tokens=%s after_tokens=%s window=%s",
            before_tokens,
            after_tokens,
            window_tokens,
        )
        return CompactResult(
            changed=True,
            reason="Old turns were summarized and recent turns were preserved.",
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            window_tokens=window_tokens,
            threshold_tokens=threshold_tokens,
        )

    def _send_single_turn(
        self,
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None,
    ):
        if self._should_use_raw_chat():
            if stream:
                return self._send_streaming_raw(tools=tools)
            return self._send_non_streaming_raw(tools=tools)
        if stream:
            return self._send_streaming(tools=tools)
        return self._send_non_streaming(tools=tools)

    def _prepare_context_for_turn(
        self,
        content: str | list[dict[str, Any]],
        *,
        disable_tools: bool,
    ) -> bool:
        pending_messages = [*self.messages, {"role": "user", "content": content}]
        tools = [] if disable_tools else self._resolve_tools()
        window_tokens = self._model_context_window() or self.context.local_window_tokens
        threshold_tokens = int(window_tokens * self.context.threshold)
        reserve_tokens = self._effective_context_reserve()
        estimated_tokens = (
            self.estimate_context_tokens(pending_messages, tools=tools) + reserve_tokens
        )
        logger.debug(
            "Context estimate tokens=%s window=%s threshold=%s reserve=%s auto=%s",
            estimated_tokens,
            window_tokens,
            threshold_tokens,
            reserve_tokens,
            self.context.auto_compact,
        )
        if estimated_tokens < threshold_tokens:
            return True

        if self.context.auto_compact:
            self._print(
                f"[compact] estimated context {estimated_tokens}/"
                f"{window_tokens} reached threshold {threshold_tokens}; "
                "compacting old turns...\n"
            )
            try:
                result = self.compact_context()
            except Exception as exc:
                logger.exception("Automatic context compaction failed")
                self._print(f"[compact] failed: {exc}\n")
            else:
                self._print(f"[compact] {result.summary()}\n")
            pending_messages = [*self.messages, {"role": "user", "content": content}]
            estimated_tokens = (
                self.estimate_context_tokens(pending_messages, tools=tools)
                + reserve_tokens
            )
            if estimated_tokens < window_tokens:
                return True

        if estimated_tokens >= window_tokens:
            self._print(
                f"[context] estimated prompt {estimated_tokens}/"
                f"{window_tokens} exceeds the configured context window. "
                "Run /compact or raise the configured window before retrying.\n"
            )
            self._set_status("error")
            logger.warning(
                "Context overflow blocked estimated=%s window=%s",
                estimated_tokens,
                window_tokens,
            )
            return False

        if not self.context.auto_compact:
            self._print(
                f"[context] estimated context {estimated_tokens}/"
                f"{window_tokens} is above the "
                f"{round(self.context.threshold * 100)}% threshold. "
                "Run /compact or enable /compact auto on.\n"
            )
        return True

    def _effective_context_reserve(self) -> int:
        if self.generation.max_tokens is None:
            return self.context.reserve_tokens
        return max(self.context.reserve_tokens, self.generation.max_tokens)

    def _split_compactable_history(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        body = self.messages[1:]
        if not body:
            return [], []
        user_seen = 0
        split_index = len(body)
        for index in range(len(body) - 1, -1, -1):
            if body[index].get("role") == "user":
                user_seen += 1
                if user_seen >= self.context.keep_recent_turns:
                    split_index = index
                    break
        if user_seen < self.context.keep_recent_turns:
            return [], body
        return body[:split_index], body[split_index:]

    def _summarize_messages_for_compaction(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        rendered_history = _render_messages_for_compaction(messages)
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "You compact chat history for Mistral4Cli. Summarize durable "
                    "facts, decisions, file paths, commands and outputs, tool "
                    "results, user preferences, and open tasks. Keep the summary "
                    "concise, factual, and in English. Do not invent details."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the old conversation below so a future assistant "
                    "can continue without the full transcript.\n\n"
                    f"{rendered_history}"
                ),
            },
        ]
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": summary_messages,
            "temperature": min(self.generation.temperature, 0.2),
            "top_p": self.generation.top_p,
            "stream": False,
            "response_format": {"type": "text"},
            "max_tokens": self.context.summary_max_tokens,
        }
        if self.backend_kind is BackendKind.REMOTE:
            kwargs["reasoning_effort"] = "none"
        else:
            prompt_mode = self._effective_prompt_mode()
            if prompt_mode is not None:
                kwargs["prompt_mode"] = prompt_mode
        response = self.client.chat.complete(**kwargs)
        choice = response.choices[0]
        message = choice.message
        if message is None:
            return ""
        content_value = message.content
        if isinstance(content_value, str):
            return content_value.strip()
        return _join_segments(
            _content_segments_from_value(content_value), kind="answer"
        )
