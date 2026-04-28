# mypy: disable-error-code="attr-defined"
"""Conversations API transport helpers for interactive sessions."""

from __future__ import annotations

import logging
from typing import Any

from mistral4cli.session_primitives import (
    _conversation_content_segments,
    _conversation_tool_call,
    _ConversationToolCallState,
    _field,
    _join_segments,
    _ModelTurn,
    _ReasoningParser,
    _RenderedSegment,
)

logger = logging.getLogger("mistral4cli.session")


class SessionConversationsMixin:
    """Send turns through Mistral Conversations and normalize the results."""

    def _send_conversation_turn(
        self,
        *,
        inputs: str | list[dict[str, Any]] | None,
        stream: bool,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        if stream:
            return self._send_conversation_streaming(inputs=inputs, tools=tools)
        return self._send_conversation_non_streaming(inputs=inputs, tools=tools)

    def _conversation_user_inputs(
        self,
        content: str | list[dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        return [{"type": "message.input", "role": "user", "content": content}]

    def _conversation_completion_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "response_format": {"type": "text"},
            "reasoning_effort": "high" if self.show_reasoning else "none",
        }
        if self.generation.max_tokens is not None:
            args["max_tokens"] = self.generation.max_tokens
        return args

    def _conversation_start_kwargs(
        self,
        *,
        inputs: str | list[dict[str, Any]],
        stream: bool,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "inputs": inputs,
            "model": self.model_id,
            "instructions": self.system_prompt,
            "store": self.conversations.store,
            "completion_args": self._conversation_completion_args(),
        }
        if self.pending_conversation.name:
            kwargs["name"] = self.pending_conversation.name
        if self.pending_conversation.description:
            kwargs["description"] = self.pending_conversation.description
        if self.pending_conversation.metadata:
            kwargs["metadata"] = self.pending_conversation.metadata
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _conversation_append_kwargs(
        self,
        *,
        inputs: str | list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if self.conversation_id is None:
            raise RuntimeError("Conversation append requires a conversation id.")
        return {
            "conversation_id": self.conversation_id,
            "inputs": inputs,
            "store": self.conversations.store,
            "completion_args": self._conversation_completion_args(),
        }

    def _send_conversation_non_streaming(
        self,
        *,
        inputs: str | list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        try:
            conversations = self.client.beta.conversations
            if self.conversation_id is None:
                if inputs is None:
                    raise RuntimeError("Conversation start requires inputs.")
                response = conversations.start(
                    **self._conversation_start_kwargs(
                        inputs=inputs,
                        stream=False,
                        tools=tools,
                    )
                )
            else:
                response = conversations.append(
                    **self._conversation_append_kwargs(inputs=inputs)
                )
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(content="", finish_reason="cancelled", cancelled=True)
        except Exception as exc:
            self._print(f"[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        return self._handle_conversation_response(response)

    def _send_conversation_streaming(
        self,
        *,
        inputs: str | list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        parser = _ReasoningParser()
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_states: dict[str, _ConversationToolCallState] = {}
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        usage_snapshot: Any = None
        error_message: str | None = None

        try:
            conversations = self.client.beta.conversations
            if self.conversation_id is None:
                if inputs is None:
                    raise RuntimeError("Conversation start requires inputs.")
                stream = conversations.start_stream(
                    **self._conversation_start_kwargs(
                        inputs=inputs,
                        stream=True,
                        tools=tools,
                    )
                )
            else:
                stream = conversations.append_stream(
                    **self._conversation_append_kwargs(inputs=inputs)
                )
            with stream as active_stream:
                for event in active_stream:
                    event_name = str(_field(event, "event", ""))
                    data = _field(event, "data")
                    if event_name == "conversation.response.started":
                        conversation_id = _field(data, "conversation_id")
                        if conversation_id and self.conversations.store:
                            self._sync_conversation_id(
                                str(conversation_id),
                                source="stream",
                            )
                        continue
                    if event_name == "conversation.response.done":
                        usage_snapshot = _field(data, "usage", usage_snapshot)
                        continue
                    if event_name == "conversation.response.error":
                        error_message = str(
                            _field(data, "message", "conversation error")
                        )
                        continue
                    if event_name == "function.call.delta":
                        call_id = str(_field(data, "tool_call_id", "") or "")
                        if not call_id:
                            call_id = str(_field(data, "id", len(tool_states)) or "")
                        state = tool_states.setdefault(
                            call_id,
                            _ConversationToolCallState(index=len(tool_states)),
                        )
                        state.update(data)
                        continue
                    if event_name != "message.output.delta":
                        continue
                    for segment in _conversation_content_segments(
                        _field(data, "content")
                    ):
                        if segment.kind == "reasoning":
                            self._set_status("answering")
                            self._print_reasoning(segment.text)
                            reasoning_parts.append(segment.text)
                            reasoning_printed = True
                        else:
                            for parsed in parser.feed(segment.text):
                                if parsed.kind == "reasoning":
                                    self._set_status("answering")
                                    self._print_reasoning(parsed.text)
                                    reasoning_parts.append(parsed.text)
                                    reasoning_printed = True
                                else:
                                    self._set_status("answering")
                                    answer_parts.append(parsed.text)
                                    answer_started = self._print_answer_separator(
                                        reasoning_printed=reasoning_printed,
                                        answer_started=answer_started,
                                    )
                                    self._print(parsed.text)
                        printed_anything = True
        except KeyboardInterrupt:
            tool_calls = [
                state.to_tool_call() for _, state in sorted(tool_states.items())
            ]
            self._print("\n[interrupted]\n")
            return _ModelTurn(
                content="".join(answer_parts).strip() or parser.answer,
                reasoning="".join(reasoning_parts).strip() or parser.reasoning,
                finish_reason="cancelled",
                tool_calls=tool_calls,
                cancelled=True,
            )
        except Exception as exc:
            self._print(f"\n[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        if error_message:
            self._print(f"\n[error] {error_message}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(usage_snapshot)
        for segment in parser.finish():
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_parts.append(segment.text)
                reasoning_printed = True
            else:
                self._set_status("answering")
                answer_parts.append(segment.text)
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
                printed_anything = True

        content = "".join(answer_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
        self._finalize_remote_reasoning(
            reasoning=reasoning,
            finish_reason="tool_calls" if tool_calls else "stop",
            has_answer_text=bool(content),
        )
        if printed_anything and content and not content.endswith("\n"):
            self._print("\n")
        return _ModelTurn(
            content=content,
            finish_reason="tool_calls" if tool_calls else "stop",
            reasoning=reasoning,
            tool_calls=tool_calls,
        )

    def _handle_conversation_response(self, response: Any) -> _ModelTurn:
        conversation_id = _field(response, "conversation_id")
        if conversation_id and self.conversations.store:
            self._sync_conversation_id(
                str(conversation_id),
                source="response",
                payload=response,
                fallback_metadata=(
                    self.pending_conversation.metadata
                    if self.pending_conversation.metadata
                    else None
                ),
            )
        self._record_usage(_field(response, "usage"))
        outputs = _field(response, "outputs", []) or []
        segments: list[_RenderedSegment] = []
        tool_calls: list[dict[str, Any]] = []
        for output in outputs:
            output_type = _field(output, "type")
            if output_type == "message.output":
                segments.extend(
                    _conversation_content_segments(_field(output, "content"))
                )
            elif output_type == "function.call":
                tool_calls.append(_conversation_tool_call(output, len(tool_calls)))

        content = _join_segments(segments, kind="answer")
        reasoning = _join_segments(segments, kind="reasoning")
        if tool_calls:
            return _ModelTurn(
                content=content,
                finish_reason="tool_calls",
                reasoning=reasoning,
                tool_calls=tool_calls,
            )

        reasoning_printed = False
        answer_started = False
        for segment in segments:
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_printed = True
            else:
                self._set_status("answering")
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
        self._finalize_remote_reasoning(
            reasoning=reasoning,
            finish_reason="stop",
            has_answer_text=bool(content),
        )
        if segments and content and not content.endswith("\n"):
            self._print("\n")
        return _ModelTurn(content=content, finish_reason="stop", reasoning=reasoning)
