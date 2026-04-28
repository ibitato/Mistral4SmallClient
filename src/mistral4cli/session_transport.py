# mypy: disable-error-code="attr-defined"
"""Local raw chat and SDK transport implementations for sessions."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mistralai.client import Mistral

from mistral4cli.local_mistral import BackendKind
from mistral4cli.session_primitives import (
    _content_segments_from_value,
    _DeferredAnswerBuffer,
    _extract_tool_calls_from_text,
    _field,
    _join_segments,
    _ModelTurn,
    _normalize_tool_calls,
    _parse_reasoning_text,
    _ReasoningParser,
    _ToolCallState,
)


class SessionTransportMixin:
    """Send turns through local raw chat or the SDK chat-completions surface."""

    def _request_kwargs(
        self,
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": self.messages,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "stream": stream,
            "response_format": {"type": "text"},
        }
        if self.generation.max_tokens is not None:
            kwargs["max_tokens"] = self.generation.max_tokens
        if self.backend_kind is BackendKind.REMOTE:
            kwargs["reasoning_effort"] = "high" if self.show_reasoning else "none"
        else:
            prompt_mode = self._effective_prompt_mode()
            if prompt_mode is not None:
                kwargs["prompt_mode"] = prompt_mode
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _should_use_raw_chat(self) -> bool:
        """Return whether the local raw chat endpoint should be used."""

        return (
            self.backend_kind is BackendKind.LOCAL
            and self.show_reasoning
            and isinstance(self.client, Mistral)
        )

    def _chat_endpoint(self) -> str:
        if not self.server_url:
            raise RuntimeError("The raw chat endpoint is only available in local mode.")
        return f"{self.server_url.rstrip('/')}/v1/chat/completions"

    def _open_raw_request(self, payload: dict[str, Any]) -> Any:
        """Open a raw HTTP request against the local OpenAI-compatible chat API."""

        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self._chat_endpoint(),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout_s = max(1.0, self.timeout_ms / 1000)
        try:
            return urlopen(request, timeout=timeout_s)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"raw chat request failed with HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"raw chat request failed: {exc.reason}") from exc

    def _send_non_streaming_raw(
        self,
        *,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        payload = self._request_kwargs(stream=False, tools=tools)
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        try:
            with self._open_raw_request(payload) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(content="", finish_reason="cancelled", cancelled=True)
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(raw.get("usage"))

        choice = raw["choices"][0]
        message = choice.get("message", {})
        raw_content = message.get("content") or ""
        raw_reasoning = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        parsed = _parse_reasoning_text(raw_content)
        content = parsed.answer
        reasoning = str(raw_reasoning).strip() or parsed.reasoning
        if not tool_calls and tools:
            tool_calls = _extract_tool_calls_from_text(raw_content)
            if tool_calls:
                finish_reason = "tool_calls"
                content = ""

        if tool_calls:
            return _ModelTurn(
                content=content,
                finish_reason=finish_reason,
                reasoning=reasoning,
                tool_calls=tool_calls,
            )

        if reasoning:
            self._set_status("answering")
            self._print_reasoning(reasoning)
            printed_anything = True
            reasoning_printed = True
        for segment in parsed.segments:
            if segment.kind == "answer":
                self._set_status("answering")
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
                printed_anything = True
            elif segment.kind == "reasoning":
                reasoning_printed = True
        if printed_anything and not content.endswith("\n"):
            self._print("\n")
        elif finish_reason == "length":
            self._print("[truncated response without text]\n")

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason,
            reasoning=reasoning,
        )

    def _send_streaming_raw(
        self,
        *,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        payload = self._request_kwargs(stream=True, tools=tools)
        finish_reason = ""
        tool_states: dict[int, _ToolCallState] = {}
        parser = _ReasoningParser()
        deferred_answer = _DeferredAnswerBuffer(enabled=bool(tools))
        reasoning_parts: list[str] = []
        answer_parts: list[str] = []
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        usage_snapshot: Any = None

        try:
            with self._open_raw_request(payload) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_line = line[6:]
                    if data_line == "[DONE]":
                        break
                    event = json.loads(data_line)
                    choices = event.get("choices") or []
                    if not choices:
                        usage = event.get("usage")
                        if usage is not None:
                            usage_snapshot = usage
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    usage = event.get("usage")
                    if usage is not None:
                        usage_snapshot = usage
                    reasoning_delta = delta.get("reasoning_content")
                    if isinstance(reasoning_delta, str) and reasoning_delta:
                        self._set_status("answering")
                        reasoning_parts.append(reasoning_delta)
                        self._print_reasoning(reasoning_delta)
                        printed_anything = True
                        reasoning_printed = True
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        for segment in parser.feed(content):
                            if segment.kind == "reasoning":
                                self._set_status("answering")
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                            else:
                                self._set_status("answering")
                                answer_parts.append(segment.text)
                                display_text = deferred_answer.feed(segment.text)
                                if display_text:
                                    answer_started = self._print_answer_separator(
                                        reasoning_printed=reasoning_printed,
                                        answer_started=answer_started,
                                    )
                                    self._print(display_text)
                            printed_anything = True
                    for tool_call in delta.get("tool_calls") or []:
                        index = int(_field(tool_call, "index", 0) or 0)
                        state = tool_states.setdefault(
                            index, _ToolCallState(index=index)
                        )
                        state.update(tool_call)
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(
                content=parser.answer,
                reasoning="".join(reasoning_parts).strip() or parser.reasoning,
                finish_reason="cancelled",
                cancelled=True,
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"\n[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(usage_snapshot)

        for segment in parser.finish():
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_printed = True
            else:
                self._set_status("answering")
                answer_parts.append(segment.text)
                display_text = deferred_answer.feed(segment.text)
                if display_text:
                    answer_started = self._print_answer_separator(
                        reasoning_printed=reasoning_printed,
                        answer_started=answer_started,
                    )
                    self._print(display_text)
            printed_anything = True

        content = "".join(answer_parts).strip() or parser.answer
        reasoning = "".join(reasoning_parts).strip() or parser.reasoning
        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
        if not tool_calls and tools:
            tool_calls = _extract_tool_calls_from_text(content)
            if tool_calls:
                finish_reason = "tool_calls"
                content = ""

        deferred_tail = deferred_answer.finalize()
        if deferred_tail and not tool_calls:
            self._set_status("answering")
            answer_started = self._print_answer_separator(
                reasoning_printed=reasoning_printed,
                answer_started=answer_started,
            )
            self._print(deferred_tail)
            printed_anything = True

        if printed_anything and not content.endswith("\n"):
            self._print("\n")
        if finish_reason == "length" and not content:
            self._print("[truncated response without text]\n")

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason or "stop",
            reasoning=reasoning,
            tool_calls=tool_calls,
        )

    def _send_non_streaming(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        try:
            response = self.client.chat.complete(
                **self._request_kwargs(stream=False, tools=tools)
            )
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(content="", finish_reason="cancelled", cancelled=True)
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(getattr(response, "usage", None))
        choice = response.choices[0]
        message = choice.message
        if message is None:
            self._print("[error] empty response message\n")
            return _ModelTurn(content="", finish_reason="error", error=True)
        content_value = message.content
        finish_reason = choice.finish_reason or "stop"
        tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None))
        segments = _content_segments_from_value(content_value)
        content = _join_segments(segments, kind="answer")
        reasoning = _join_segments(segments, kind="reasoning")
        if not tool_calls and tools:
            if isinstance(content_value, str):
                tool_calls = _extract_tool_calls_from_text(content_value)
            if tool_calls:
                finish_reason = "tool_calls"
                content = ""
        reasoning_printed = False
        answer_started = False

        if tool_calls:
            return _ModelTurn(
                content=content,
                finish_reason=finish_reason,
                reasoning=reasoning,
                tool_calls=tool_calls,
            )

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
        if segments and not content.endswith("\n"):
            self._print("\n")
        elif finish_reason == "length":
            self._print("[truncated response without text]\n")
        self._finalize_remote_reasoning(
            reasoning=reasoning,
            finish_reason=finish_reason,
            has_answer_text=bool(content),
        )

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason,
            reasoning=reasoning,
        )

    def _send_streaming(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        finish_reason = ""
        tool_states: dict[int, _ToolCallState] = {}
        parser = _ReasoningParser()
        deferred_answer = _DeferredAnswerBuffer(enabled=bool(tools))
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_snapshot: Any = None

        try:
            stream = self.client.chat.stream(
                **self._request_kwargs(stream=True, tools=tools)
            )
            with stream as active_stream:
                for event in active_stream:
                    data = getattr(event, "data", None)
                    if not data or not getattr(data, "choices", None):
                        usage = getattr(data, "usage", None)
                        if usage is not None:
                            usage_snapshot = usage
                        continue
                    choice = data.choices[0]
                    usage = getattr(data, "usage", None)
                    if usage is not None:
                        usage_snapshot = usage
                    finish_reason = choice.finish_reason or finish_reason
                    delta = choice.delta
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        for segment in parser.feed(content):
                            if segment.kind == "reasoning":
                                self._set_status("answering")
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                                reasoning_parts.append(segment.text)
                            else:
                                self._set_status("answering")
                                answer_parts.append(segment.text)
                                display_text = deferred_answer.feed(segment.text)
                                if display_text:
                                    answer_started = self._print_answer_separator(
                                        reasoning_printed=reasoning_printed,
                                        answer_started=answer_started,
                                    )
                                    self._print(display_text)
                            printed_anything = True
                    elif isinstance(content, list):
                        for segment in _content_segments_from_value(content):
                            if segment.kind == "reasoning":
                                self._set_status("answering")
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                                reasoning_parts.append(segment.text)
                            else:
                                self._set_status("answering")
                                answer_parts.append(segment.text)
                                display_text = deferred_answer.feed(segment.text)
                                if display_text:
                                    answer_started = self._print_answer_separator(
                                        reasoning_printed=reasoning_printed,
                                        answer_started=answer_started,
                                    )
                                    self._print(display_text)
                            printed_anything = True
                    for tool_call in getattr(delta, "tool_calls", None) or []:
                        index = int(getattr(tool_call, "index", 0) or 0)
                        state = tool_states.setdefault(
                            index, _ToolCallState(index=index)
                        )
                        state.update(tool_call)
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return _ModelTurn(
                content=("".join(answer_parts).strip() or parser.answer),
                reasoning=("".join(reasoning_parts).strip() or parser.reasoning),
                finish_reason="cancelled",
                cancelled=True,
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"\n[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        self._record_usage(usage_snapshot)

        for segment in parser.finish():
            if segment.kind == "reasoning":
                self._set_status("answering")
                self._print_reasoning(segment.text)
                reasoning_printed = True
                reasoning_parts.append(segment.text)
            else:
                self._set_status("answering")
                answer_parts.append(segment.text)
                display_text = deferred_answer.feed(segment.text)
                if display_text:
                    answer_started = self._print_answer_separator(
                        reasoning_printed=reasoning_printed,
                        answer_started=answer_started,
                    )
                    self._print(display_text)
            printed_anything = True

        content = "".join(answer_parts).strip()
        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
        if not tool_calls and tools:
            tool_calls = _extract_tool_calls_from_text(content)
            if tool_calls:
                finish_reason = "tool_calls"
                content = ""

        deferred_tail = deferred_answer.finalize()
        if deferred_tail and not tool_calls:
            self._set_status("answering")
            answer_started = self._print_answer_separator(
                reasoning_printed=reasoning_printed,
                answer_started=answer_started,
            )
            self._print(deferred_tail)
            printed_anything = True

        if printed_anything and not content.endswith("\n"):
            self._print("\n")
        if finish_reason == "length" and not content:
            self._print("[truncated response without text]\n")
        self._finalize_remote_reasoning(
            reasoning="".join(reasoning_parts).strip(),
            finish_reason=finish_reason or "stop",
            has_answer_text=bool(content),
        )

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason or "stop",
            reasoning="".join(reasoning_parts).strip(),
            tool_calls=tool_calls,
        )
