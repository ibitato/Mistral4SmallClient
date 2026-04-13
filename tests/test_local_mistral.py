from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

import pytest

from mistral4cli.local_mistral import DEFAULT_MODEL_ID

pytestmark = pytest.mark.integration


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


def test_chat_completion_returns_expected_text(local_client: Any) -> None:
    response = local_client.chat.complete(
        model=DEFAULT_MODEL_ID,
        messages=[
            {"role": "system", "content": "Responde solo con ok."},
            {"role": "user", "content": "Devuelve solo: ok"},
        ],
        temperature=0,
        max_tokens=128,
        stream=False,
        response_format={"type": "text"},
    )

    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content == "ok"


def test_chat_streaming_returns_expected_text(local_client: Any) -> None:
    stream = local_client.chat.stream(
        model=DEFAULT_MODEL_ID,
        messages=[
            {"role": "system", "content": "Responde solo con ok."},
            {"role": "user", "content": "Devuelve solo: ok"},
        ],
        temperature=0,
        max_tokens=128,
        response_format={"type": "text"},
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


def test_tool_call_arguments_are_structured(local_client: Any) -> None:
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
        messages=[{"role": "user", "content": "Usa la herramienta para sumar 2 y 3."}],
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


def test_multimodal_image_request_is_accepted(local_client: Any) -> None:
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
