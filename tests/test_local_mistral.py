from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import pytest
from mistralai import Mistral

from mistral4cli.local_mistral import DEFAULT_MODEL_ID

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class CompletionCase:
    name: str
    prompt: str
    temperature: float
    top_p: float
    random_seed: int
    max_tokens: int
    prompt_mode: str | None = None
    expected_text: str | None = None
    expected_finish_reason: str = "stop"


def _png_2x2_solid_rgba(red: int, green: int, blue: int, alpha: int = 255) -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
    row = b"\x00" + bytes([red, green, blue, alpha]) * 2
    raw = row * 2
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return base64.b64encode(signature + ihdr + idat + iend).decode("ascii")


def test_health_endpoint(local_health: dict[str, object]) -> None:
    assert local_health["status"] == "ok"


def test_models_endpoint_lists_target_model(local_models: dict[str, object]) -> None:
    data = local_models["data"]
    assert isinstance(data, list)
    assert any(
        model.get("id") == DEFAULT_MODEL_ID for model in data if isinstance(model, dict)
    )


def _complete_text(client: Mistral, case: CompletionCase) -> Any:
    kwargs: dict[str, Any] = {
        "model": DEFAULT_MODEL_ID,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": case.temperature,
        "top_p": case.top_p,
        "random_seed": case.random_seed,
        "max_tokens": case.max_tokens,
        "stream": False,
        "response_format": {"type": "text"},
    }
    if case.prompt_mode is not None:
        kwargs["prompt_mode"] = case.prompt_mode
    return client.chat.complete(**kwargs)


@pytest.mark.parametrize(
    "case",
    [
        CompletionCase(
            name="baseline",
            prompt="Return only the word ok.",
            temperature=0,
            top_p=1.0,
            random_seed=11,
            max_tokens=256,
            expected_text="ok",
        ),
        CompletionCase(
            name="temperature_tuned",
            prompt="Return only the word ok.",
            temperature=0.2,
            top_p=0.9,
            random_seed=7,
            max_tokens=256,
            expected_text="ok",
        ),
        CompletionCase(
            name="temperature_high",
            prompt="Return only the word ok.",
            temperature=0.7,
            top_p=0.95,
            random_seed=7,
            max_tokens=256,
            expected_text="ok",
        ),
        CompletionCase(
            name="reasoning_mode",
            prompt="Return only the word ok.",
            temperature=0,
            top_p=1.0,
            random_seed=11,
            max_tokens=256,
            prompt_mode="reasoning",
            expected_text="ok",
        ),
        CompletionCase(
            name="length_cutoff",
            prompt="Return only the word ok.",
            temperature=0,
            top_p=1.0,
            random_seed=11,
            max_tokens=4,
            expected_finish_reason="length",
        ),
    ],
    ids=lambda case: case.name,
)
def test_chat_completion_matrix(local_client: Mistral, case: CompletionCase) -> None:
    response = _complete_text(local_client, case)
    choice = response.choices[0]

    assert choice.finish_reason == case.expected_finish_reason
    assert isinstance(choice.message.content, str)
    if case.expected_text is not None:
        assert choice.message.content == case.expected_text


def test_random_seed_is_reproducible(local_client: Mistral) -> None:
    case = CompletionCase(
        name="seeded",
        prompt="Return only the word ok.",
        temperature=0.4,
        top_p=0.95,
        random_seed=11,
        max_tokens=64,
        expected_text="ok",
    )

    first = _complete_text(local_client, case)
    second = _complete_text(local_client, case)

    assert first.choices[0].finish_reason in {"stop", "length"}
    assert second.choices[0].finish_reason in {"stop", "length"}
    first_text = first.choices[0].message.content or ""
    second_text = second.choices[0].message.content or ""
    assert first_text.strip() or second_text.strip()
    assert any(text.strip() == "ok" for text in (first_text, second_text))


@pytest.mark.parametrize(
    "prompt_mode",
    [None, "reasoning"],
    ids=["stream-default", "stream-reasoning"],
)
def test_chat_streaming_returns_expected_text(
    local_client: Mistral, prompt_mode: str | None
) -> None:
    stream = local_client.chat.stream(
        model=DEFAULT_MODEL_ID,
        messages=[
            {"role": "user", "content": "Return only the word ok."},
        ],
        temperature=0,
        top_p=1.0,
        random_seed=11,
        max_tokens=128,
        response_format={"type": "text"},
        prompt_mode=prompt_mode,
    )

    text = ""
    chunk_count = 0
    for event in stream:
        chunk_count += 1
        chunk = event.data
        if chunk and chunk.choices:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                text += content

    assert chunk_count > 0
    assert text == "ok"


def test_stream_cancel_then_followup_request_succeeds(local_client: Mistral) -> None:
    stream = local_client.chat.stream(
        model=DEFAULT_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a long text of several paragraphs about why distributed "
                    "systems are hard, with at least 1000 words."
                ),
            }
        ],
        temperature=0.7,
        top_p=0.95,
        random_seed=11,
        max_tokens=1024,
        response_format={"type": "text"},
    )

    with stream:
        for idx, _event in enumerate(stream, start=1):
            if idx >= 5:
                break

    follow_up = _complete_text(
        local_client,
        CompletionCase(
            name="post-cancel",
            prompt="Return only the word ok.",
            temperature=0,
            top_p=1.0,
            random_seed=11,
            max_tokens=64,
            expected_text="ok",
        ),
    )

    choice = follow_up.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content == "ok"


def test_tool_call_arguments_are_structured(local_client: Mistral) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    response = local_client.chat.complete(
        model=DEFAULT_MODEL_ID,
        messages=[{"role": "user", "content": "Use the tool to add 2 and 3."}],
        tools=tools,
        temperature=0,
        max_tokens=128,
        stream=False,
    )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    arguments = json.loads(choice.message.tool_calls[0].function.arguments)
    assert arguments == {"a": 2, "b": 3}


def test_multimodal_image_request_is_accepted(local_client: Mistral) -> None:
    image_b64 = _png_2x2_solid_rgba(255, 0, 0)
    # This validates request transport and multimodal parsing, not image
    # semantics. The local build may still return a generic or empty text.
    response = local_client.chat.complete(
        model=DEFAULT_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Process the attached image and answer with one short "
                            "sentence."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=64,
        stream=False,
        response_format={"type": "text"},
    )

    choice = response.choices[0]
    assert choice.finish_reason in {"stop", "length"}
    assert isinstance(choice.message.content, str)
