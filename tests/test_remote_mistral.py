from __future__ import annotations

import os

import pytest

from mistralcli.local_mistral import RemoteMistralConfig, build_client

pytestmark = [pytest.mark.integration, pytest.mark.remote]


def _field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name)


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="Set MISTRAL_API_KEY to enable remote Mistral tests.",
)
def test_remote_chat_completion_smoke() -> None:
    config = RemoteMistralConfig.from_env()
    client = build_client(config)

    response = client.chat.complete(
        model=config.model_id,
        messages=[{"role": "user", "content": "Return only the word ok."}],
        temperature=0,
        top_p=1.0,
        max_tokens=64,
        reasoning_effort="high",
        response_format={"type": "text"},
    )

    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert isinstance(choice.message.content, list)
    first_chunk = choice.message.content[0]
    last_chunk = choice.message.content[-1]
    assert _field(first_chunk, "type") == "thinking"
    assert _field(last_chunk, "type") == "text"
    assert str(_field(last_chunk, "text")).strip().lower() == "ok"


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="Set MISTRAL_API_KEY to enable remote Mistral tests.",
)
def test_remote_chat_completion_without_reasoning_smoke() -> None:
    config = RemoteMistralConfig.from_env()
    client = build_client(config)

    response = client.chat.complete(
        model=config.model_id,
        messages=[{"role": "user", "content": "Return only the word ok."}],
        temperature=0,
        top_p=1.0,
        max_tokens=64,
        reasoning_effort="none",
        response_format={"type": "text"},
    )

    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    content = choice.message.content
    if isinstance(content, list):
        assert all(_field(chunk, "type") != "thinking" for chunk in content)
        last_chunk = content[-1]
        assert _field(last_chunk, "type") == "text"
        assert str(_field(last_chunk, "text")).strip().lower() == "ok"
    else:
        assert str(content).strip().lower() == "ok"
