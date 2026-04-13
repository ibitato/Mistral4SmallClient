"""Interactive session management for the local Mistral Small 4 CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TextIO

from mistralai import Mistral

from mistral4cli.local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    LocalGenerationConfig,
)

DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente de codigo local para Mistral Small 4. Responde de forma "
    "directa, orientada a acciones y con comandos/ejemplos concretos cuando ayuden. "
    "Si falta contexto, pregunta lo minimo necesario antes de inventar."
)


def render_defaults_summary(
    *,
    model_id: str,
    server_url: str,
    generation: LocalGenerationConfig,
    stream_enabled: bool,
) -> str:
    """Render the active runtime defaults as human-readable text."""

    max_tokens = (
        "unset" if generation.max_tokens is None else str(generation.max_tokens)
    )
    prompt_mode = generation.prompt_mode or "unset"
    stream_mode = "on" if stream_enabled else "off"
    return "\n".join(
        [
            "Mistral Small 4 local CLI",
            f"Server: {server_url}",
            f"Model: {model_id}",
            (
                "Defaults: "
                f"temperature={generation.temperature} "
                f"top_p={generation.top_p} "
                f"prompt_mode={prompt_mode} "
                f"max_tokens={max_tokens} "
                f"stream={stream_mode}"
            ),
        ]
    )


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Result of a single user turn."""

    content: str
    finish_reason: str
    cancelled: bool = False


@dataclass(slots=True)
class MistralCodingSession:
    """Stateful conversation helper for the local Mistral CLI."""

    client: Mistral
    model_id: str = DEFAULT_MODEL_ID
    server_url: str = DEFAULT_SERVER_URL
    generation: LocalGenerationConfig = field(default_factory=LocalGenerationConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    stdout: TextIO | None = None
    stream_enabled: bool = True
    messages: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.stdout is None:
            import sys

            self.stdout = sys.stdout
        self.system_prompt = self.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset(self) -> None:
        """Reset the conversation to the system prompt."""

        self.messages = [{"role": "system", "content": self.system_prompt}]

    def set_system_prompt(self, system_prompt: str) -> None:
        """Replace the active system prompt and reset the conversation."""

        self.system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.reset()

    def describe_defaults(self) -> str:
        """Render the active runtime defaults as human-readable text."""

        return "\n".join(
            [
                render_defaults_summary(
                    model_id=self.model_id,
                    server_url=self.server_url,
                    generation=self.generation,
                    stream_enabled=self.stream_enabled,
                ),
                "Commands: /help /defaults /reset /system /exit",
            ]
        )

    def _request_kwargs(self, *, stream: bool) -> dict[str, Any]:
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
        return kwargs

    def send(self, user_text: str, *, stream: bool = True) -> TurnResult:
        """Send one user turn and update the conversation history."""

        clean_text = user_text.strip()
        if not clean_text:
            return TurnResult(content="", finish_reason="empty", cancelled=False)

        self.messages.append({"role": "user", "content": clean_text})
        if stream:
            return self._send_streaming(clean_text)
        return self._send_non_streaming(clean_text)

    def _print(self, text: str) -> None:
        assert self.stdout is not None
        self.stdout.write(text)
        self.stdout.flush()

    def _send_non_streaming(self, user_text: str) -> TurnResult:
        try:
            response = self.client.chat.complete(
                **self._request_kwargs(stream=False)
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"[error] {exc}\n")
            return TurnResult(content="", finish_reason="error", cancelled=False)

        choice = response.choices[0]
        content_value = choice.message.content
        content = content_value if isinstance(content_value, str) else ""
        finish_reason = choice.finish_reason or "stop"

        if content:
            self._print(content)
            if not content.endswith("\n"):
                self._print("\n")
            self.messages.append({"role": "assistant", "content": content})
        elif finish_reason == "length":
            self._print("[respuesta truncada sin texto]\n")

        return TurnResult(content=content, finish_reason=finish_reason, cancelled=False)

    def _send_streaming(self, user_text: str) -> TurnResult:
        chunks: list[str] = []
        finish_reason = ""

        try:
            stream = self.client.chat.stream(**self._request_kwargs(stream=True))
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
        except KeyboardInterrupt:
            self._print("\n[interrumpido]\n")
            return TurnResult(
                content="".join(chunks),
                finish_reason="cancelled",
                cancelled=True,
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI smoke
            self._print(f"\n[error] {exc}\n")
            return TurnResult(content="", finish_reason="error", cancelled=False)

        content = "".join(chunks)
        if content and not content.endswith("\n"):
            self._print("\n")
        if content:
            self.messages.append({"role": "assistant", "content": content})
        elif finish_reason == "length":
            self._print("[respuesta truncada sin texto]\n")

        return TurnResult(
            content=content,
            finish_reason=finish_reason or "stop",
            cancelled=False,
        )
