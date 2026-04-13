from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from mistral4cli.cli import main
from mistral4cli.local_mistral import LocalGenerationConfig
from mistral4cli.session import MistralCodingSession


class FakeStdin(io.StringIO):
    def __init__(self, value: str = "", *, tty: bool = False) -> None:
        super().__init__(value)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@dataclass(slots=True)
class FakeMessage:
    content: str | None = None


@dataclass(slots=True)
class FakeDelta:
    content: str | None = None


@dataclass(slots=True)
class FakeChoice:
    message: FakeMessage = field(default_factory=FakeMessage)
    delta: FakeDelta = field(default_factory=FakeDelta)
    finish_reason: str | None = None


@dataclass(slots=True)
class FakeResponse:
    choices: list[FakeChoice]


@dataclass(slots=True)
class FakeEvent:
    data: FakeResponse


class FakeStream:
    def __init__(
        self, events: list[FakeEvent], interrupt_after: int | None = None
    ) -> None:
        self.events = events
        self.interrupt_after = interrupt_after
        self.closed = False

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.closed = True

    def __iter__(self):
        for index, event in enumerate(self.events):
            if self.interrupt_after is not None and index >= self.interrupt_after:
                raise KeyboardInterrupt
            yield event


class FakeChat:
    def __init__(
        self,
        *,
        complete_text: str = "ok",
        stream_chunks: list[str] | None = None,
        interrupt_after: int | None = None,
    ) -> None:
        self.complete_text = complete_text
        self.stream_chunks = stream_chunks or ["ok"]
        self.interrupt_after = interrupt_after
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.last_stream: FakeStream | None = None

    def complete(self, **kwargs: Any) -> FakeResponse:
        self.complete_calls.append(kwargs)
        return FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(content=self.complete_text),
                    finish_reason="stop",
                )
            ]
        )

    def stream(self, **kwargs: Any) -> FakeStream:
        self.stream_calls.append(kwargs)
        events = [
            FakeEvent(
                data=FakeResponse(
                    choices=[
                        FakeChoice(
                            delta=FakeDelta(content=chunk),
                            finish_reason="stop"
                            if index == len(self.stream_chunks) - 1
                            else None,
                        )
                    ]
                )
            )
            for index, chunk in enumerate(self.stream_chunks)
        ]
        self.last_stream = FakeStream(events, interrupt_after=self.interrupt_after)
        return self.last_stream


class FakeClient:
    def __init__(
        self,
        *,
        complete_text: str = "ok",
        stream_chunks: list[str] | None = None,
        interrupt_after: int | None = None,
    ) -> None:
        self.chat = FakeChat(
            complete_text=complete_text,
            stream_chunks=stream_chunks,
            interrupt_after=interrupt_after,
        )


def test_main_returns_zero_without_interactive_input() -> None:
    output = io.StringIO()
    exit_code = main(
        [],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: FakeClient(),
    )

    assert exit_code == 0
    assert output.getvalue() == ""


def test_print_defaults_shows_mistral_small_4_defaults() -> None:
    output = io.StringIO()
    client_factory_called = False

    def client_factory(_config: Any) -> FakeClient:
        nonlocal client_factory_called
        client_factory_called = True
        return FakeClient()

    exit_code = main(
        ["--print-defaults"],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=client_factory,
    )

    assert exit_code == 0
    assert client_factory_called is False
    rendered = output.getvalue()
    assert "Mistral Small 4 local CLI" in rendered
    assert "temperature=0.7" in rendered
    assert "top_p=0.95" in rendered
    assert "prompt_mode=reasoning" in rendered
    assert "stream=on" in rendered


def test_once_uses_effective_defaults_and_prints_answer() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")

    exit_code = main(
        ["--once", "Devuelve solo la palabra ok.", "--no-stream"],
        stdin=FakeStdin(""),
        stdout=output,
        client_factory=lambda _config: fake_client,
    )

    assert exit_code == 0
    assert output.getvalue() == "ok\n"
    assert len(fake_client.chat.complete_calls) == 1
    call = fake_client.chat.complete_calls[0]
    assert call["temperature"] == 0.7
    assert call["top_p"] == 0.95
    assert call["prompt_mode"] == "reasoning"
    assert "max_tokens" not in call


def test_stream_cancel_does_not_commit_partial_assistant_turn() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["hola", " mundo"], interrupt_after=1)
    session = MistralCodingSession(
        client=fake_client, generation=LocalGenerationConfig(), stdout=output
    )

    result = session.send("Haz una respuesta larga.", stream=True)

    assert result.cancelled is True
    assert result.content == "hola"
    assert fake_client.chat.last_stream is not None
    assert fake_client.chat.last_stream.closed is True
    assert session.messages == [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "Haz una respuesta larga."},
    ]
    assert "[interrumpido]" in output.getvalue()


def test_parse_command_supports_system_and_reset() -> None:
    from mistral4cli.cli import _parse_command

    assert _parse_command("/system cambia el tono") == (
        "system",
        "cambia el tono",
    )
    assert _parse_command(":reset") == ("reset", "")
