# mypy: disable-error-code="attr-defined,has-type,no-any-return"
"""Turn orchestration, tool execution, and output helpers for sessions."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from mistral4cli.conversation_registry import ConversationRecord
from mistral4cli.local_mistral import BackendKind
from mistral4cli.mcp_bridge import MCPBridgeError, MCPToolResult
from mistral4cli.session_primitives import TurnResult, _field, _ModelTurn
from mistral4cli.ui import render_reasoning_chunk

logger = logging.getLogger("mistral4cli.session")


class SessionToolsMixin:
    """Coordinate user turns, tool loops, and visible terminal output."""

    def send_content(
        self,
        content: str | list[dict[str, Any]],
        *,
        stream: bool = True,
        disable_tools: bool = False,
    ) -> TurnResult:
        """Send a text or multimodal user turn and update the conversation."""

        normalized = self._normalize_user_content(content)
        if normalized is None:
            return TurnResult(content="", finish_reason="empty", cancelled=False)

        logger.debug(
            "Sending turn stream=%s disable_tools=%s attachments=%s content=%s",
            stream,
            disable_tools,
            self._has_attachment_blocks(normalized),
            self._content_summary(normalized),
        )
        if self.conversations.enabled:
            return self._send_conversation_content(
                normalized,
                stream=stream,
                disable_tools=disable_tools,
            )

        if not self._prepare_context_for_turn(normalized, disable_tools=disable_tools):
            return TurnResult(content="", finish_reason="context_overflow")

        message_start = len(self.messages)
        self.messages.append({"role": "user", "content": normalized})
        self._mark_context_status_dirty()
        self._last_usage = None
        self._turn_usage_accumulator = None
        self._set_status("thinking")
        try:
            seen_tool_calls: set[str] = set()
            tool_rounds_executed = 0
            tools = [] if disable_tools else self._resolve_tools()
            if not tools:
                turn = self._send_single_turn(stream=stream, tools=None)
                if turn.cancelled or turn.error:
                    self._rollback_to(message_start)
                    self._turn_usage_accumulator = None
                    if turn.cancelled:
                        self._set_status("interrupted")
                    else:
                        self._set_status("error")
                if not turn.cancelled and not turn.error:
                    self._commit_assistant_message(turn)
                    self._commit_turn_usage()
                    self._set_status("done")
                return TurnResult(
                    content=turn.content,
                    finish_reason=turn.finish_reason,
                    reasoning=turn.reasoning,
                    cancelled=turn.cancelled,
                )

            for _ in range(self.max_tool_rounds):
                turn = self._send_single_turn(stream=stream, tools=tools)
                if turn.cancelled or turn.error:
                    self._rollback_to(message_start)
                    self._turn_usage_accumulator = None
                    if turn.cancelled:
                        self._set_status("interrupted")
                    else:
                        self._set_status("error")
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=turn.cancelled,
                    )

                if turn.finish_reason != "tool_calls" or not turn.tool_calls:
                    self._commit_assistant_message(turn)
                    self._commit_turn_usage()
                    self._set_status("done")
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=False,
                    )

                self._commit_assistant_message(turn)
                for call in turn.tool_calls:
                    function = call["function"]
                    name = str(function["name"])
                    try:
                        arguments = self._parse_tool_arguments(
                            function.get("arguments")
                        )
                    except json.JSONDecodeError as exc:
                        result = MCPToolResult(
                            text=(
                                f"[tool-error] invalid JSON arguments for {name}: {exc}"
                            ),
                            is_error=True,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        result = MCPToolResult(
                            text=f"[tool-error] {exc}",
                            is_error=True,
                        )
                    else:
                        signature = self._tool_call_signature(name, arguments)
                        if signature in seen_tool_calls:
                            logger.warning(
                                (
                                    "Blocked repeated identical tool call "
                                    "name=%s arguments=%s"
                                ),
                                name,
                                self._summarize_tool_arguments(arguments),
                            )
                            self.messages.append(
                                self._tool_message(
                                    call=call,
                                    result=MCPToolResult(
                                        text=(
                                            "[tool-error] repeated identical tool call "
                                            "blocked; use the prior tool result"
                                        ),
                                        is_error=True,
                                        structured_content={
                                            "status": "error",
                                            "tool": name,
                                            "code": "repeated_identical_tool_call",
                                            "arguments": arguments,
                                        },
                                    ),
                                )
                            )
                            self._mark_context_status_dirty()
                            self._print(
                                "[error] repeated identical tool call blocked\n"
                            )
                            self._turn_usage_accumulator = None
                            self._set_status("error")
                            return TurnResult(
                                content="",
                                finish_reason="error",
                                cancelled=False,
                            )
                        seen_tool_calls.add(signature)
                        logger.debug(
                            "Executing tool name=%s arguments=%s",
                            name,
                            self._summarize_tool_arguments(arguments),
                        )
                        self._set_status("tool", detail=name)
                        result = self._call_tool_bridge(name, arguments)
                        self._set_status("thinking")
                        logger.debug(
                            "Tool result name=%s error=%s structured=%s",
                            name,
                            result.is_error,
                            result.structured_content is not None,
                        )
                    self.messages.append(self._tool_message(call=call, result=result))
                    self._mark_context_status_dirty()
                tool_rounds_executed += 1

            if tool_rounds_executed > 0:
                logger.warning(
                    "Tool loop limit reached; forcing final answer max_rounds=%s",
                    self.max_tool_rounds,
                )
                final_turn = self._send_single_turn(stream=stream, tools=None)
                if final_turn.cancelled or final_turn.error:
                    self._rollback_to(message_start)
                    self._turn_usage_accumulator = None
                    if final_turn.cancelled:
                        self._set_status("interrupted")
                    else:
                        self._set_status("error")
                    return TurnResult(
                        content=final_turn.content,
                        finish_reason=final_turn.finish_reason,
                        reasoning=final_turn.reasoning,
                        cancelled=final_turn.cancelled,
                    )
                self._commit_assistant_message(final_turn)
                self._commit_turn_usage()
                self._set_status("done")
                return TurnResult(
                    content=final_turn.content,
                    finish_reason=final_turn.finish_reason,
                    reasoning=final_turn.reasoning,
                    cancelled=False,
                )

            self._rollback_to(message_start)
            self._turn_usage_accumulator = None
            self._print("[error] tool loop limit reached\n")
            self._set_status("error")
            logger.error("Tool loop limit reached max_rounds=%s", self.max_tool_rounds)
            return TurnResult(content="", finish_reason="error", cancelled=False)
        except KeyboardInterrupt:
            self._rollback_to(message_start)
            self._turn_usage_accumulator = None
            self._print("\n[interrupted]\n")
            self._set_status("interrupted")
            logger.info("Turn interrupted by user")
            return TurnResult(content="", finish_reason="cancelled", cancelled=True)

    def send(self, user_text: str, *, stream: bool = True) -> TurnResult:
        """Send one text user turn and update the conversation history."""

        return self.send_content(user_text, stream=stream)

    def _send_conversation_content(
        self,
        content: str | list[dict[str, Any]],
        *,
        stream: bool,
        disable_tools: bool,
    ) -> TurnResult:
        message_start = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        self._mark_context_status_dirty()
        self._last_usage = None
        self._turn_usage_accumulator = None
        self._set_status("thinking")
        pending_tool_calls: list[dict[str, Any]] = []
        pending_tool_inputs: list[dict[str, Any]] = []
        try:
            seen_tool_calls: set[str] = set()
            inputs = self._conversation_user_inputs(content)
            tools = [] if disable_tools else self._resolve_tools()
            for _ in range(self.max_tool_rounds + 1):
                turn = self._send_conversation_turn(
                    inputs=inputs,
                    stream=stream,
                    tools=tools,
                )
                if turn.cancelled or turn.error:
                    if turn.cancelled and turn.tool_calls:
                        self._complete_pending_conversation_tool_calls(turn.tool_calls)
                    self._rollback_to(message_start)
                    self._turn_usage_accumulator = None
                    self._set_status("interrupted" if turn.cancelled else "error")
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=turn.cancelled,
                    )
                if turn.finish_reason != "tool_calls" or not turn.tool_calls:
                    self._commit_assistant_message(turn)
                    self._commit_turn_usage()
                    self._set_status("done")
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=False,
                    )

                pending_tool_calls = list(turn.tool_calls)
                self._commit_assistant_message(turn)
                tool_inputs = pending_tool_inputs = []
                for call in turn.tool_calls:
                    function = call["function"]
                    name = str(function["name"])
                    try:
                        arguments = self._parse_tool_arguments(
                            function.get("arguments")
                        )
                    except Exception as exc:
                        result = MCPToolResult(
                            text=f"[tool-error] invalid arguments for {name}: {exc}",
                            is_error=True,
                        )
                    else:
                        signature = self._tool_call_signature(name, arguments)
                        if signature in seen_tool_calls:
                            result = MCPToolResult(
                                text=(
                                    "[tool-error] repeated identical tool call "
                                    "blocked; use the prior tool result"
                                ),
                                is_error=True,
                                structured_content={
                                    "status": "error",
                                    "tool": name,
                                    "code": "repeated_identical_tool_call",
                                    "arguments": arguments,
                                },
                            )
                            self.messages.append(
                                self._tool_message(call=call, result=result)
                            )
                            self._mark_context_status_dirty()
                            self._print(
                                "[error] repeated identical tool call blocked\n"
                            )
                            self._turn_usage_accumulator = None
                            self._set_status("error")
                            return TurnResult(
                                content="",
                                finish_reason="error",
                                cancelled=False,
                            )
                        seen_tool_calls.add(signature)
                        self._set_status("tool", detail=name)
                        result = self._call_tool_bridge(name, arguments)
                        self._set_status("thinking")
                    self.messages.append(self._tool_message(call=call, result=result))
                    self._mark_context_status_dirty()
                    tool_inputs.append(
                        {
                            "type": "function.result",
                            "tool_call_id": call["id"],
                            "result": self._render_tool_result(result),
                        }
                    )
                inputs = tool_inputs
                pending_tool_calls = []
                pending_tool_inputs = []

            self._rollback_to(message_start)
            self._turn_usage_accumulator = None
            self._print("[error] tool loop limit reached\n")
            self._set_status("error")
            return TurnResult(content="", finish_reason="error", cancelled=False)
        except KeyboardInterrupt:
            self._complete_pending_conversation_tool_calls(
                pending_tool_calls,
                completed_inputs=pending_tool_inputs,
            )
            self._rollback_to(message_start)
            self._turn_usage_accumulator = None
            self._print("\n[interrupted]\n")
            self._set_status("interrupted")
            return TurnResult(content="", finish_reason="cancelled", cancelled=True)

    def _print(self, text: str) -> None:
        if self.answer_writer is not None:
            self.answer_writer(text)
            return
        assert self.stdout is not None
        self.stdout.write(text)
        self.stdout.flush()

    def _print_reasoning(self, text: str) -> None:
        if not text or not self.show_thinking:
            return
        if self.reasoning_writer is not None:
            self.reasoning_writer(text)
            return
        assert self.stdout is not None
        self.stdout.write(render_reasoning_chunk(text, stream=self.stdout))
        self.stdout.flush()

    def _finalize_remote_reasoning(
        self,
        *,
        reasoning: str,
        finish_reason: str,
        has_answer_text: bool,
    ) -> None:
        if self.backend_kind is not BackendKind.REMOTE:
            return
        if reasoning.strip():
            self._missing_reasoning_notice_shown = False
            return
        if (
            not self.show_reasoning
            or not self.show_thinking
            or finish_reason in {"tool_calls", "cancelled", "error"}
        ):
            return
        if not has_answer_text or self._missing_reasoning_notice_shown:
            return
        if self.conversations.enabled:
            message = (
                "[reasoning] Reasoning was requested, but Mistral "
                "Conversations returned no thinking blocks for this turn.\n"
            )
        else:
            message = (
                "[reasoning] Reasoning was requested, but the remote "
                "backend returned no thinking blocks for this turn.\n"
            )
        if has_answer_text:
            message = "\n" + message
        self._print(message)
        logger.warning(
            "Reasoning requested but not returned backend=%s conversations=%s",
            self.backend_kind.value,
            self.conversations.enabled,
        )
        self._missing_reasoning_notice_shown = True

    def _print_answer_separator(
        self,
        *,
        reasoning_printed: bool,
        answer_started: bool,
    ) -> bool:
        """Separate visible reasoning from the final answer once per turn."""

        if reasoning_printed and not answer_started:
            self._print("\n\n")
            return True
        return answer_started

    def _rollback_to(self, message_count: int) -> None:
        self.messages = self.messages[:message_count]
        self._mark_context_status_dirty()

    def _commit_assistant_message(self, turn: _ModelTurn) -> None:
        if turn.tool_calls:
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": turn.tool_calls,
            }
            if turn.content:
                assistant_message["content"] = turn.content
            self.messages.append(assistant_message)
            self._mark_context_status_dirty()
            return

        if turn.content:
            self.messages.append({"role": "assistant", "content": turn.content})
            self._mark_context_status_dirty()
            logger.info(
                (
                    "Assistant message committed finish_reason=%s "
                    "content_len=%s reasoning_len=%s"
                ),
                turn.finish_reason,
                len(turn.content),
                len(turn.reasoning),
            )

    def _sync_conversation_id(
        self,
        conversation_id: str,
        *,
        source: str,
        payload: Any | None = None,
        parent_conversation_id: str | None = None,
        fallback_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update the active conversation id and synchronize the local registry."""

        normalized_id = conversation_id.strip()
        if not normalized_id:
            return
        previous_id = self.conversation_id
        self.conversation_id = normalized_id
        self.conversation_resume_source = source
        if self.conversation_registry is None:
            return
        if previous_id and previous_id != normalized_id:
            self.conversation_registry.migrate_conversation_id(
                previous_id,
                normalized_id,
            )
        if payload is not None:
            self._update_registry_from_remote_payload(
                normalized_id,
                payload,
                fallback_metadata=fallback_metadata,
            )
        else:
            self.conversation_registry.update_remote_snapshot(
                normalized_id,
                remote_name=self.pending_conversation.name or None,
                remote_description=self.pending_conversation.description or None,
                remote_metadata=(
                    self.pending_conversation.metadata
                    if self.pending_conversation.metadata
                    else None
                ),
                remote_kind="model",
                remote_model=self.model_id,
                parent_conversation_id=parent_conversation_id,
            )
        self.conversation_registry.remember_active(normalized_id)

    def _update_registry_from_remote_payload(
        self,
        conversation_id: str,
        payload: Any,
        *,
        parent_conversation_id: str | None = None,
        fallback_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist remote conversation metadata into the local registry."""

        if self.conversation_registry is None or not conversation_id.strip():
            return
        metadata = _field(payload, "metadata", None)
        if metadata is None and fallback_metadata:
            metadata = fallback_metadata
        self.conversation_registry.update_remote_snapshot(
            conversation_id,
            remote_name=_field(payload, "name", None),
            remote_description=_field(payload, "description", None),
            remote_metadata=metadata,
            remote_kind=("agent" if _field(payload, "agent_id") else "model"),
            remote_model=_field(payload, "model", None),
            remote_agent_id=_field(payload, "agent_id", None),
            created_at=_field(payload, "created_at", None),
            updated_at=_field(payload, "updated_at", None),
            parent_conversation_id=parent_conversation_id,
        )

    def _format_remote_conversation_summary(self, payload: Any) -> str:
        """Render a one-line summary for a remote conversation entity."""

        conversation_id = str(_field(payload, "id", "") or "")
        kind = "agent" if _field(payload, "agent_id") else "model"
        target = str(
            _field(payload, "agent_id", "") or _field(payload, "model", "") or "unknown"
        )
        name = str(_field(payload, "name", "") or "")
        description = str(_field(payload, "description", "") or "")
        created_at = str(_field(payload, "created_at", "") or "")
        updated_at = str(_field(payload, "updated_at", "") or "")
        details = [conversation_id, kind, target]
        if name:
            details.append(f'name="{name}"')
        if description:
            details.append(f'description="{description}"')
        if created_at:
            details.append(f"created={created_at}")
        if updated_at:
            details.append(f"updated={updated_at}")
        if self.conversation_registry is not None:
            record = self.conversation_registry.get(conversation_id)
            if record is not None:
                overlay = self._format_registry_record(record, include_note=False)
                if overlay:
                    details.append(f"local[{overlay}]")
        return " | ".join(part for part in details if part)

    def _render_remote_conversation_details(self, payload: Any) -> list[str]:
        """Render a multi-line detail view for one remote conversation."""

        lines = [
            f"kind: {'agent' if _field(payload, 'agent_id') else 'model'}",
            f"model: {_field(payload, 'model', '') or '(none)'}",
            f"agent_id: {_field(payload, 'agent_id', '') or '(none)'}",
            f"name: {_field(payload, 'name', '') or '(none)'}",
            f"description: {_field(payload, 'description', '') or '(none)'}",
            f"created_at: {_field(payload, 'created_at', '') or '(unknown)'}",
            f"updated_at: {_field(payload, 'updated_at', '') or '(unknown)'}",
        ]
        metadata = _field(payload, "metadata", None)
        if metadata:
            lines.append(
                "metadata: " + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            )
        else:
            lines.append("metadata: (none)")
        return lines

    def _format_registry_record(
        self,
        record: ConversationRecord,
        *,
        include_note: bool,
    ) -> str:
        """Render the local overlay metadata for one registry entry."""

        parts = [record.conversation_id]
        if record.alias:
            parts.append(f"alias={record.alias}")
        if record.tags:
            parts.append(f"tags={','.join(record.tags)}")
        if include_note and record.note:
            parts.append(f"note={record.note}")
        if record.remote_name:
            parts.append(f"remote_name={record.remote_name}")
        if record.remote_metadata:
            metadata = ",".join(
                f"{key}={value}"
                for key, value in sorted(record.remote_metadata.items())
            )
            parts.append(f"metadata={metadata}")
        if record.parent_conversation_id:
            parts.append(f"parent={record.parent_conversation_id}")
        if record.deleted:
            parts.append("deleted=yes")
        return " | ".join(parts)

    def _parse_tool_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            text = raw_arguments.strip()
            if not text:
                return {}
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Tool arguments must decode to an object")
            return cast(dict[str, Any], parsed)
        raise TypeError(f"Unsupported tool arguments payload: {type(raw_arguments)!r}")

    def _tool_message(
        self,
        *,
        call: dict[str, Any],
        result: MCPToolResult,
    ) -> dict[str, Any]:
        function = call["function"]
        content = self._render_tool_result(result)
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": function["name"],
            "content": content,
        }

    def _render_tool_result(self, result: MCPToolResult) -> str:
        parts: list[str] = []
        if result.structured_content is not None:
            parts.append(
                json.dumps(
                    result.structured_content,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if result.text:
            parts.append(result.text)
        return "\n\n".join(parts)

    def _cancelled_tool_result(self, call: dict[str, Any]) -> MCPToolResult:
        """Build a synthetic tool result used to unblock interrupted Conversations."""

        function = call["function"]
        return MCPToolResult(
            text="[tool-error] cancelled by user",
            is_error=True,
            structured_content={
                "status": "error",
                "tool": str(function["name"]),
                "code": "cancelled_by_user",
                "cancelled": True,
            },
        )

    def _complete_pending_conversation_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        completed_inputs: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send cancellation tool results to unblock an interrupted Conversation."""

        if not tool_calls or self.conversation_id is None:
            return
        inputs = list(completed_inputs or [])
        completed_ids = {
            str(item.get("tool_call_id", ""))
            for item in inputs
            if isinstance(item, dict)
        }
        for call in tool_calls:
            call_id = str(call["id"])
            if call_id in completed_ids:
                continue
            inputs.append(
                {
                    "type": "function.result",
                    "tool_call_id": call_id,
                    "result": self._render_tool_result(
                        self._cancelled_tool_result(call)
                    ),
                }
            )
        if not inputs:
            return
        try:
            self.client.beta.conversations.append(
                **self._conversation_append_kwargs(inputs=inputs)
            )
            logger.warning(
                "Completed pending Conversations tool calls after interruption "
                "conversation_id=%s count=%s",
                self.conversation_id,
                len(inputs),
            )
        except Exception as exc:
            logger.warning(
                "Failed to complete pending Conversations tool calls after "
                "interruption conversation_id=%s error=%s",
                self.conversation_id,
                exc,
            )

    def _call_tool_bridge(
        self,
        public_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        assert self.tool_bridge is not None
        try:
            return self.tool_bridge.call_tool(public_name, arguments)
        except MCPBridgeError as exc:
            return MCPToolResult(text=f"[tool-error] {exc}", is_error=True)

    def _tool_call_signature(self, name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _content_summary(self, content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return f"text(len={len(content)})"
        return f"blocks(len={len(content)})"

    def _summarize_tool_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        summarized: dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                if key in {"content", "prompt", "system_prompt", "api_key"}:
                    summarized[key] = f"<str len={len(value)}>"
                elif len(value) > 120:
                    summarized[key] = f"{value[:117]}..."
                else:
                    summarized[key] = value
                continue
            if isinstance(value, list):
                summarized[key] = f"<list len={len(value)}>"
                continue
            if isinstance(value, dict):
                summarized[key] = f"<dict keys={sorted(value)}>"
                continue
            summarized[key] = value
        return summarized

    def _resolve_tools(self) -> list[dict[str, Any]]:
        if self.tool_bridge is None:
            return []
        try:
            return self.tool_bridge.to_mistral_tools()
        except MCPBridgeError as exc:
            if not self._mcp_warning_shown:
                self._print(f"[mcp] {exc}\n")
                self._mcp_warning_shown = True
            return []

    def _has_attachment_blocks(self, content: str | list[dict[str, Any]]) -> bool:
        if isinstance(content, str):
            return False
        return any(
            block.get("type") in {"image_url", "document_url"} for block in content
        )
