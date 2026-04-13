"""Interactive session management for the local Mistral Small 4 CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TextIO, cast

from mistralai import Mistral

from mistral4cli.local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    LocalGenerationConfig,
)
from mistral4cli.mcp_bridge import MCPBridgeError, MCPToolResult
from mistral4cli.tooling import ToolBridge
from mistral4cli.ui import render_runtime_summary

DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente de codigo local para Mistral Small 4. Responde de forma "
    "directa, orientada a acciones y con comandos/ejemplos concretos cuando ayuden. "
    "Tienes acceso permanente a estas herramientas locales: shell, read_file, "
    "write_file, list_dir y search_text. Usa shell para comandos del sistema, "
    "read_file y write_file para inspeccionar o editar ficheros, y list_dir o "
    "search_text para explorar el arbol del proyecto. Tambien puedes usar MCP "
    "cuando necesites informacion externa o FireCrawl. Antes de afirmar algo "
    "sobre el repositorio, el filesystem o el sistema, verifica con herramientas "
    "siempre que sea posible. Si falta contexto, pregunta lo minimo necesario "
    "antes de inventar. Si la conversacion incluye imagenes o documentos "
    "adjuntos, analizalos con cuidado antes de responder."
)


def render_defaults_summary(
    *,
    model_id: str,
    server_url: str,
    generation: LocalGenerationConfig,
    stream_enabled: bool,
    tool_summary: str,
) -> str:
    """Render the active runtime defaults as human-readable text."""

    return render_runtime_summary(
        model_id=model_id,
        server_url=server_url,
        generation=generation,
        stream_enabled=stream_enabled,
        tool_summary=tool_summary,
    )


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Result of a single user turn."""

    content: str
    finish_reason: str
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class _ModelTurn:
    """Intermediate result from one model call."""

    content: str
    finish_reason: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False
    error: bool = False


@dataclass(slots=True)
class _ToolCallState:
    """Accumulator for streamed tool-call deltas."""

    index: int
    call_id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)

    def update(self, tool_call: Any) -> None:
        call_id = getattr(tool_call, "id", None)
        if call_id and call_id != "null":
            self.call_id = str(call_id)

        function = getattr(tool_call, "function", None)
        if function is None:
            return

        name = getattr(function, "name", None)
        if name:
            self.name = str(name)

        arguments = getattr(function, "arguments", None)
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
    model_id: str = DEFAULT_MODEL_ID
    server_url: str = DEFAULT_SERVER_URL
    generation: LocalGenerationConfig = field(default_factory=LocalGenerationConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_bridge: ToolBridge | None = None
    stdout: TextIO | None = None
    stream_enabled: bool = True
    max_tool_rounds: int = 4
    messages: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _mcp_warning_shown: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
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
            model_id=self.model_id,
            server_url=self.server_url,
            generation=self.generation,
            stream_enabled=self.stream_enabled,
            tool_summary=self.describe_tool_status(),
        )

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

        self.messages.append({"role": "user", "content": normalized})
        try:
            tools = self._resolve_tools()
            if not tools:
                turn = self._send_single_turn(stream=stream, tools=None)
                if not turn.cancelled and not turn.error:
                    self._commit_assistant_message(turn)
                return TurnResult(
                    content=turn.content,
                    finish_reason=turn.finish_reason,
                    cancelled=turn.cancelled,
                )

            for _ in range(self.max_tool_rounds):
                turn = self._send_single_turn(stream=stream, tools=tools)
                if turn.cancelled or turn.error:
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
                        cancelled=turn.cancelled,
                    )

                if turn.finish_reason != "tool_calls" or not turn.tool_calls:
                    self._commit_assistant_message(turn)
                    return TurnResult(
                        content=turn.content,
                        finish_reason=turn.finish_reason,
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

            self._print("[error] tool loop limit reached\n")
            return TurnResult(content="", finish_reason="error", cancelled=False)
        except KeyboardInterrupt:
            self._print("\n[interrumpido]\n")
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
        if self.generation.prompt_mode is not None:
            kwargs["prompt_mode"] = self.generation.prompt_mode
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _print(self, text: str) -> None:
        assert self.stdout is not None
        self.stdout.write(text)
        self.stdout.flush()

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
        if stream:
            return self._send_streaming(tools=tools)
        return self._send_non_streaming(tools=tools)

    def _send_non_streaming(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        try:
            response = self.client.chat.complete(
                **self._request_kwargs(stream=False, tools=tools)
            )
        except KeyboardInterrupt:
            self._print("\n[interrumpido]\n")
            return _ModelTurn(content="", finish_reason="cancelled", cancelled=True)
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        choice = response.choices[0]
        content_value = choice.message.content
        content = content_value if isinstance(content_value, str) else ""
        finish_reason = choice.finish_reason or "stop"
        tool_calls = _normalize_tool_calls(getattr(choice.message, "tool_calls", None))

        if tool_calls:
            return _ModelTurn(
                content=content,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )

        if content:
            self._print(content)
            if not content.endswith("\n"):
                self._print("\n")
        elif finish_reason == "length":
            self._print("[respuesta truncada sin texto]\n")

        return _ModelTurn(content=content, finish_reason=finish_reason)

    def _send_streaming(self, *, tools: list[dict[str, Any]] | None) -> _ModelTurn:
        chunks: list[str] = []
        finish_reason = ""
        tool_states: dict[int, _ToolCallState] = {}

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
                        chunks.append(content)
                        self._print(content)
                    for tool_call in getattr(delta, "tool_calls", None) or []:
                        index = int(getattr(tool_call, "index", 0) or 0)
                        state = tool_states.setdefault(
                            index, _ToolCallState(index=index)
                        )
                        state.update(tool_call)
        except KeyboardInterrupt:
            self._print("\n[interrumpido]\n")
            return _ModelTurn(
                content="".join(chunks),
                finish_reason="cancelled",
                cancelled=True,
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"\n[error] {exc}\n")
            return _ModelTurn(content="", finish_reason="error", error=True)

        content = "".join(chunks)
        if content and not content.endswith("\n"):
            self._print("\n")
        if finish_reason == "length" and not content:
            self._print("[respuesta truncada sin texto]\n")

        tool_calls = [state.to_tool_call() for _, state in sorted(tool_states.items())]
        return _ModelTurn(
            content=content,
            finish_reason=finish_reason or "stop",
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


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    normalized: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        arguments = getattr(function, "arguments", None)
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append(
            {
                "id": getattr(tool_call, "id", f"tool_call_{index}"),
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": getattr(function, "name", f"tool_{index}"),
                    "arguments": arguments or "{}",
                },
            }
        )
    return normalized
