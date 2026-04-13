"""Interactive session management for the local Mistral Small 4 CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, TextIO, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mistralai.client import Mistral

from mistral4cli.local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    BackendKind,
    LocalGenerationConfig,
)
from mistral4cli.mcp_bridge import MCPBridgeError, MCPToolResult
from mistral4cli.tooling import ToolBridge
from mistral4cli.ui import render_reasoning_chunk, render_runtime_summary

DEFAULT_SYSTEM_PROMPT = (
    "You are a local coding assistant for Mistral Small 4. Respond directly, "
    "focus on action, and include concrete commands or examples when they help. "
    "You always have access to these local tools: shell, read_file, write_file, "
    "list_dir, and search_text. Use shell for system commands, read_file and "
    "write_file to inspect or edit files, and list_dir or search_text to explore "
    "the project tree. You can also use MCP when you need external information "
    "or FireCrawl. Before asserting anything about the repository, filesystem, "
    "or system, verify it with tools whenever possible. If context is missing, "
    "ask for the minimum needed before guessing. If the conversation includes "
    "attached images or documents, analyze them carefully before replying."
)

REASONING_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("[THINK]", "[/THINK]"),
    ("[think]", "[/think]"),
)


def render_defaults_summary(
    *,
    backend_kind: BackendKind,
    model_id: str,
    server_url: str | None,
    generation: LocalGenerationConfig,
    stream_enabled: bool,
    reasoning_visible: bool,
    tool_summary: str,
) -> str:
    """Render the active runtime defaults as human-readable text."""

    return render_runtime_summary(
        backend_kind=backend_kind,
        model_id=model_id,
        server_url=server_url,
        generation=generation,
        stream_enabled=stream_enabled,
        reasoning_visible=reasoning_visible,
        tool_summary=tool_summary,
    )


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Result of a single user turn."""

    content: str
    finish_reason: str
    reasoning: str = ""
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class _ModelTurn:
    """Intermediate result from one model call."""

    content: str
    finish_reason: str
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False
    error: bool = False


@dataclass(slots=True)
class _RenderedSegment:
    kind: str
    text: str


@dataclass(slots=True)
class _ReasoningParser:
    in_reasoning: bool = False
    pending: str = ""
    answer_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)

    def feed(self, text: str) -> list[_RenderedSegment]:
        segments: list[_RenderedSegment] = []
        if not text:
            return segments

        self.pending += text
        while self.pending:
            if self.in_reasoning:
                close_match = _find_first_tag(self.pending, _close_tags())
                if close_match is None:
                    emit, keep = _split_possible_tag_suffix(self.pending, _close_tags())
                    if emit:
                        self.reasoning_parts.append(emit)
                        segments.append(_RenderedSegment(kind="reasoning", text=emit))
                    self.pending = keep
                    break

                close_index, close_tag = close_match
                if close_index > 0:
                    emit = self.pending[:close_index]
                    self.reasoning_parts.append(emit)
                    segments.append(_RenderedSegment(kind="reasoning", text=emit))
                self.pending = self.pending[close_index + len(close_tag) :]
                self.in_reasoning = False
                continue

            open_match = _find_first_tag(self.pending, _open_tags())
            close_match = _find_first_tag(self.pending, _close_tags())
            if close_match is not None and (
                open_match is None or close_match[0] < open_match[0]
            ):
                close_index, close_tag = close_match
                if close_index > 0:
                    emit = self.pending[:close_index]
                    self.answer_parts.append(emit)
                    segments.append(_RenderedSegment(kind="answer", text=emit))
                self.pending = self.pending[close_index + len(close_tag) :]
                continue
            if open_match is None:
                emit, keep = _split_possible_tag_suffix(
                    self.pending, [*_open_tags(), *_close_tags()]
                )
                if emit:
                    self.answer_parts.append(emit)
                    segments.append(_RenderedSegment(kind="answer", text=emit))
                self.pending = keep
                break

            open_index, open_tag = open_match
            if open_index > 0:
                emit = self.pending[:open_index]
                self.answer_parts.append(emit)
                segments.append(_RenderedSegment(kind="answer", text=emit))
            self.pending = self.pending[open_index + len(open_tag) :]
            self.in_reasoning = True

        return segments

    def finish(self) -> list[_RenderedSegment]:
        if not self.pending:
            return []
        segment = _RenderedSegment(
            kind="reasoning" if self.in_reasoning else "answer",
            text=self.pending,
        )
        if self.in_reasoning:
            self.reasoning_parts.append(self.pending)
        else:
            self.answer_parts.append(self.pending)
        self.pending = ""
        return [segment]

    @property
    def answer(self) -> str:
        return "".join(self.answer_parts).strip()

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts).strip()


@dataclass(slots=True)
class _ToolCallState:
    """Accumulator for streamed tool-call deltas."""

    index: int
    call_id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)

    def update(self, tool_call: Any) -> None:
        call_id = _field(tool_call, "id")
        if call_id and call_id != "null":
            self.call_id = str(call_id)

        function = _field(tool_call, "function")
        if function is None:
            return

        name = _field(function, "name")
        if name:
            self.name = str(name)

        arguments = _field(function, "arguments")
        if arguments is None:
            return
        if isinstance(arguments, str):
            self.arguments_parts.append(arguments)
        else:
            self.arguments_parts.append(json.dumps(arguments, ensure_ascii=False))

    def to_tool_call(self) -> dict[str, Any]:
        arguments = "".join(self.arguments_parts).strip() or "{}"
        call_id = self.call_id or f"tool_call_{self.index}"
        name = self.name or f"tool_{self.index}"
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }


@dataclass(slots=True)
class MistralCodingSession:
    """Stateful conversation helper for the local Mistral CLI."""

    client: Mistral
    backend_kind: BackendKind = BackendKind.LOCAL
    model_id: str = DEFAULT_MODEL_ID
    server_url: str | None = DEFAULT_SERVER_URL
    generation: LocalGenerationConfig = field(default_factory=LocalGenerationConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_bridge: ToolBridge | None = None
    stdout: TextIO | None = None
    stream_enabled: bool = True
    show_reasoning: bool = True
    max_tool_rounds: int = 4
    messages: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _mcp_warning_shown: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Normalize the session state after dataclass initialization."""

        if self.stdout is None:
            import sys

            self.stdout = sys.stdout
        self.system_prompt = self.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def reset(self) -> None:
        """Reset the conversation to the system prompt."""

        self.messages = [{"role": "system", "content": self.system_prompt}]

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

        return render_defaults_summary(
            backend_kind=self.backend_kind,
            model_id=self.model_id,
            server_url=self.server_url,
            generation=self._display_generation(),
            stream_enabled=self.stream_enabled,
            reasoning_visible=self.show_reasoning,
            tool_summary=self.describe_tool_status(),
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
        self.reset()

    def visible_reasoning_supported(self) -> bool:
        """Return whether the active backend can render visible reasoning."""

        return True

    def reasoning_status_text(self) -> str:
        """Return a user-facing visible-reasoning status string."""

        state = "on" if self.show_reasoning else "off"
        if self.backend_kind is BackendKind.REMOTE:
            return f"Visible reasoning: {state} (remote SDK)"
        return f"Visible reasoning: {state} (local raw endpoint)"

    def set_reasoning_visibility(self, visible: bool) -> None:
        """Enable or disable visible reasoning output."""

        self.show_reasoning = visible

    def toggle_reasoning_visibility(self) -> bool:
        """Toggle visible reasoning output and return the new state."""

        self.show_reasoning = not self.show_reasoning
        return self.show_reasoning

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute a tool through the active bridge."""

        return self._call_tool_bridge(public_name, arguments)

    def send_content(
        self,
        content: str | list[dict[str, Any]],
        *,
        stream: bool = True,
    ) -> TurnResult:
        """Send a text or multimodal user turn and update the conversation."""

        normalized = self._normalize_user_content(content)
        if normalized is None:
            return TurnResult(content="", finish_reason="empty", cancelled=False)

        message_start = len(self.messages)
        self.messages.append({"role": "user", "content": normalized})
        try:
            tools = self._resolve_tools()
            if not tools:
                turn = self._send_single_turn(stream=stream, tools=None)
                if turn.error:
                    self._rollback_to(message_start)
                if not turn.cancelled and not turn.error:
                    self._commit_assistant_message(turn)
                return TurnResult(
                    content=turn.content,
                    finish_reason=turn.finish_reason,
                    reasoning=turn.reasoning,
                    cancelled=turn.cancelled,
                )

            for _ in range(self.max_tool_rounds):
                turn = self._send_single_turn(stream=stream, tools=tools)
                if turn.cancelled or turn.error:
                    if turn.error:
                        self._rollback_to(message_start)
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        reasoning=turn.reasoning,
                        cancelled=turn.cancelled,
                    )

                if turn.finish_reason != "tool_calls" or not turn.tool_calls:
                    self._commit_assistant_message(turn)
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
                            text=f"[tool-error] {exc}", is_error=True
                        )
                    else:
                        result = self._call_tool_bridge(name, arguments)
                    self.messages.append(self._tool_message(call=call, result=result))

            self._rollback_to(message_start)
            self._print("[error] tool loop limit reached\n")
            return TurnResult(content="", finish_reason="error", cancelled=False)
        except KeyboardInterrupt:
            self._print("\n[interrupted]\n")
            return TurnResult(content="", finish_reason="cancelled", cancelled=True)

    def send(self, user_text: str, *, stream: bool = True) -> TurnResult:
        """Send one text user turn and update the conversation history."""

        return self.send_content(user_text, stream=stream)

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
        timeout_s = max(1.0, getattr(self.client, "timeout_ms", 120_000) / 1000)
        try:
            return urlopen(request, timeout=timeout_s)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"raw chat request failed with HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"raw chat request failed: {exc.reason}") from exc

    def _print(self, text: str) -> None:
        assert self.stdout is not None
        self.stdout.write(text)
        self.stdout.flush()

    def _print_reasoning(self, text: str) -> None:
        if not text or not self.show_reasoning:
            return
        assert self.stdout is not None
        self.stdout.write(render_reasoning_chunk(text, stream=self.stdout))
        self.stdout.flush()

    def _print_answer_separator(
        self, *, reasoning_printed: bool, answer_started: bool
    ) -> bool:
        """Separate visible reasoning from the final answer once per turn."""

        if reasoning_printed and not answer_started:
            self._print("\n\n")
            return True
        return answer_started

    def _rollback_to(self, message_count: int) -> None:
        self.messages = self.messages[:message_count]

    def _commit_assistant_message(self, turn: _ModelTurn) -> None:
        if turn.tool_calls:
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": turn.tool_calls,
            }
            if turn.content:
                assistant_message["content"] = turn.content
            self.messages.append(assistant_message)
            return

        if turn.content:
            self.messages.append({"role": "assistant", "content": turn.content})

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
        self, *, call: dict[str, Any], result: MCPToolResult
    ) -> dict[str, Any]:
        function = call["function"]
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": function["name"],
            "content": result.text,
        }

    def _call_tool_bridge(
        self, public_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        assert self.tool_bridge is not None
        try:
            return self.tool_bridge.call_tool(public_name, arguments)
        except MCPBridgeError as exc:
            return MCPToolResult(text=f"[tool-error] {exc}", is_error=True)

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

    def _send_single_turn(
        self,
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None,
    ) -> _ModelTurn:
        if self._should_use_raw_chat():
            if stream:
                return self._send_streaming_raw(tools=tools)
            return self._send_non_streaming_raw(tools=tools)
        if stream:
            return self._send_streaming(tools=tools)
        return self._send_non_streaming(tools=tools)

    def _send_non_streaming_raw(
        self, *, tools: list[dict[str, Any]] | None
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

        choice = raw["choices"][0]
        message = choice.get("message", {})
        raw_content = message.get("content") or ""
        raw_reasoning = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        parsed = _parse_reasoning_text(raw_content)
        content = parsed.answer
        reasoning = str(raw_reasoning).strip() or parsed.reasoning

        if tool_calls:
            return _ModelTurn(
                content=content,
                finish_reason=finish_reason,
                reasoning=reasoning,
                tool_calls=tool_calls,
            )

        if reasoning:
            self._print_reasoning(reasoning)
            printed_anything = True
            reasoning_printed = True
        for segment in parsed.segments:
            if segment.kind == "answer":
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

    def _send_streaming_raw(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        payload = self._request_kwargs(stream=True, tools=tools)
        finish_reason = ""
        tool_states: dict[int, _ToolCallState] = {}
        parser = _ReasoningParser()
        reasoning_parts: list[str] = []
        printed_anything = False
        reasoning_printed = False
        answer_started = False

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
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    reasoning_delta = delta.get("reasoning_content")
                    if isinstance(reasoning_delta, str) and reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                        self._print_reasoning(reasoning_delta)
                        printed_anything = True
                        reasoning_printed = True
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        for segment in parser.feed(content):
                            if segment.kind == "reasoning":
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                            else:
                                answer_started = self._print_answer_separator(
                                    reasoning_printed=reasoning_printed,
                                    answer_started=answer_started,
                                )
                                self._print(segment.text)
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

        for segment in parser.finish():
            if segment.kind == "reasoning":
                self._print_reasoning(segment.text)
                reasoning_printed = True
            else:
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
            printed_anything = True

        content = parser.answer
        reasoning = "".join(reasoning_parts).strip() or parser.reasoning
        if printed_anything and not content.endswith("\n"):
            self._print("\n")
        if finish_reason == "length" and not content:
            self._print("[truncated response without text]\n")

        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
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
                self._print_reasoning(segment.text)
                reasoning_printed = True
            else:
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
        if segments and not content.endswith("\n"):
            self._print("\n")
        elif finish_reason == "length":
            self._print("[truncated response without text]\n")

        return _ModelTurn(
            content=content,
            finish_reason=finish_reason,
            reasoning=reasoning,
        )

    def _send_streaming(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        finish_reason = ""
        tool_states: dict[int, _ToolCallState] = {}
        parser = _ReasoningParser()
        printed_anything = False
        reasoning_printed = False
        answer_started = False
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []

        try:
            stream = self.client.chat.stream(
                **self._request_kwargs(stream=True, tools=tools)
            )
            with stream as active_stream:
                for event in active_stream:
                    data = getattr(event, "data", None)
                    if not data or not getattr(data, "choices", None):
                        continue
                    choice = data.choices[0]
                    finish_reason = choice.finish_reason or finish_reason
                    delta = choice.delta
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        for segment in parser.feed(content):
                            if segment.kind == "reasoning":
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                                reasoning_parts.append(segment.text)
                            else:
                                answer_started = self._print_answer_separator(
                                    reasoning_printed=reasoning_printed,
                                    answer_started=answer_started,
                                )
                                self._print(segment.text)
                                answer_parts.append(segment.text)
                            printed_anything = True
                    elif isinstance(content, list):
                        for segment in _content_segments_from_value(content):
                            if segment.kind == "reasoning":
                                self._print_reasoning(segment.text)
                                reasoning_printed = True
                                reasoning_parts.append(segment.text)
                            else:
                                answer_started = self._print_answer_separator(
                                    reasoning_printed=reasoning_printed,
                                    answer_started=answer_started,
                                )
                                self._print(segment.text)
                                answer_parts.append(segment.text)
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

        for segment in parser.finish():
            if segment.kind == "reasoning":
                self._print_reasoning(segment.text)
                reasoning_printed = True
                reasoning_parts.append(segment.text)
            else:
                answer_started = self._print_answer_separator(
                    reasoning_printed=reasoning_printed,
                    answer_started=answer_started,
                )
                self._print(segment.text)
                answer_parts.append(segment.text)
            printed_anything = True

        content = "".join(answer_parts).strip()
        if printed_anything and not content.endswith("\n"):
            self._print("\n")
        if finish_reason == "length" and not content:
            self._print("[truncated response without text]\n")

        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
        return _ModelTurn(
            content=content,
            finish_reason=finish_reason or "stop",
            reasoning="".join(reasoning_parts).strip(),
            tool_calls=tool_calls,
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
        return self.generation.prompt_mode

    def _display_generation(self) -> LocalGenerationConfig:
        return replace(self.generation, prompt_mode=self._effective_prompt_mode())


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    normalized: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = _field(tool_call, "function")
        if function is None:
            continue
        arguments = _field(function, "arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append(
            {
                "id": _field(tool_call, "id", f"tool_call_{index}"),
                "type": _field(tool_call, "type", "function"),
                "function": {
                    "name": _field(function, "name", f"tool_{index}"),
                    "arguments": arguments or "{}",
                },
            }
        )
    return normalized


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _split_possible_tag_suffix(text: str, tags: list[str]) -> tuple[str, str]:
    keep = 0
    for tag in tags:
        max_prefix = min(len(tag) - 1, len(text))
        for prefix_length in range(max_prefix, 0, -1):
            if text.endswith(tag[:prefix_length]):
                keep = max(keep, prefix_length)
                break
    if keep == 0:
        return text, ""
    return text[:-keep], text[-keep:]


def _find_first_tag(text: str, tags: list[str]) -> tuple[int, str] | None:
    matches = [(text.find(tag), tag) for tag in tags]
    present = [(index, tag) for index, tag in matches if index != -1]
    if not present:
        return None
    return min(present, key=lambda item: item[0])


def _open_tags() -> list[str]:
    return [open_tag for open_tag, _close_tag in REASONING_TAG_PAIRS]


def _close_tags() -> list[str]:
    return [close_tag for _open_tag, close_tag in REASONING_TAG_PAIRS]


@dataclass(frozen=True, slots=True)
class _ParsedReasoningText:
    segments: list[_RenderedSegment]
    answer: str
    reasoning: str


def _parse_reasoning_text(text: str) -> _ParsedReasoningText:
    parser = _ReasoningParser()
    segments = parser.feed(text)
    segments.extend(parser.finish())
    return _ParsedReasoningText(
        segments=segments,
        answer=parser.answer,
        reasoning=parser.reasoning,
    )


def _content_segments_from_value(content: Any) -> list[_RenderedSegment]:
    if isinstance(content, str):
        return _parse_reasoning_text(content).segments
    if not isinstance(content, list):
        return []

    segments: list[_RenderedSegment] = []
    for block in content:
        block_type = _field(block, "type")
        if block_type == "text":
            text = _field(block, "text")
            if isinstance(text, str) and text:
                segments.append(_RenderedSegment(kind="answer", text=text))
        elif block_type == "thinking":
            for item in _field(block, "thinking", []) or []:
                text = _field(item, "text")
                if isinstance(text, str) and text:
                    segments.append(_RenderedSegment(kind="reasoning", text=text))
    return segments


def _join_segments(segments: list[_RenderedSegment], *, kind: str) -> str:
    return "".join(segment.text for segment in segments if segment.kind == kind).strip()
