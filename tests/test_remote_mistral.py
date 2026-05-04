from __future__ import annotations

import asyncio
import os

import pytest

from mistralcli.local_mistral import RemoteMistralConfig, build_client, close_client

pytestmark = [pytest.mark.integration, pytest.mark.remote]


def _field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name)


class _FakeSyncHttpClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeAsyncHttpClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeSDKConfiguration:
    def __init__(self) -> None:
        self.client = _FakeSyncHttpClient()
        self.client_supplied = False
        self.async_client = _FakeAsyncHttpClient()
        self.async_client_supplied = False


class _FakeSDKClient:
    def __init__(self) -> None:
        self.sdk_configuration = _FakeSDKConfiguration()


def test_close_client_closes_sdk_managed_http_clients() -> None:
    client = _FakeSDKClient()
    sync_client = client.sdk_configuration.client
    async_client = client.sdk_configuration.async_client

    close_client(client)

    assert sync_client.close_calls == 1
    assert async_client.close_calls == 1
    assert client.sdk_configuration.client is None
    assert client.sdk_configuration.async_client is None
    sync_client.close()
    asyncio.run(async_client.aclose())
    assert sync_client.close_calls == 1
    assert async_client.close_calls == 1


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
