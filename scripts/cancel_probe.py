from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass

from mistralai import Mistral

from mistral4cli.local_mistral import DEFAULT_MODEL_ID, DEFAULT_SERVER_URL


@dataclass(frozen=True, slots=True)
class ProbeResult:
    label: str
    finish_reason: str
    content: str


def build_client() -> Mistral:
    return Mistral(
        api_key=os.environ.get("MISTRAL_LOCAL_API_KEY", "local-test"),
        server_url=os.environ.get("MISTRAL_LOCAL_SERVER_URL", DEFAULT_SERVER_URL),
        timeout_ms=int(os.environ.get("MISTRAL_LOCAL_TIMEOUT_MS", "120000")),
    )


def ask_ok(client: Mistral) -> ProbeResult:
    response = client.chat.complete(
        model=os.environ.get("MISTRAL_LOCAL_MODEL_ID", DEFAULT_MODEL_ID),
        messages=[{"role": "user", "content": "Devuelve solo la palabra ok."}],
        temperature=0,
        top_p=1.0,
        random_seed=11,
        max_tokens=64,
        stream=False,
        response_format={"type": "text"},
    )
    choice = response.choices[0]
    return ProbeResult(
        label="follow_up",
        finish_reason=choice.finish_reason or "",
        content=choice.message.content or "",
    )


def graceful_cancel_probe(client: Mistral) -> ProbeResult:
    stream = client.chat.stream(
        model=os.environ.get("MISTRAL_LOCAL_MODEL_ID", DEFAULT_MODEL_ID),
        messages=[
            {
                "role": "user",
                "content": (
                    "Escribe un texto largo de varias parrafos sobre por que los "
                    "sistemas distribuidos son dificiles, con al menos 1000 palabras."
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

    return ask_ok(client)


def abrupt_cancel_probe() -> ProbeResult:
    api_key = os.environ.get("MISTRAL_LOCAL_API_KEY", "local-test")
    server_url = os.environ.get("MISTRAL_LOCAL_SERVER_URL", DEFAULT_SERVER_URL)
    model_id = os.environ.get("MISTRAL_LOCAL_MODEL_ID", DEFAULT_MODEL_ID)
    timeout_ms = int(os.environ.get("MISTRAL_LOCAL_TIMEOUT_MS", "120000"))
    child_code = (
        textwrap.dedent(
            """
        import time
        from mistralai import Mistral
        client = Mistral(
            api_key=__API_KEY__,
            server_url=__SERVER_URL__,
            timeout_ms=__TIMEOUT_MS__,
        )
        stream = client.chat.stream(
            model=__MODEL_ID__,
            messages=[{
                'role': 'user',
                'content': (
                    'Escribe un texto largo de varias parrafos sobre por que los '
                    'sistemas distribuidos son dificiles, con al menos 1000 palabras.'
                ),
            }],
            temperature=0.7,
            top_p=0.95,
            random_seed=11,
            max_tokens=1024,
            response_format={'type': 'text'},
        )
        with stream:
            for event in stream:
                data = event.data
                if data and data.choices:
                    delta = data.choices[0].delta
                    content = getattr(delta, 'content', None)
                    if content:
                        print(content, end='', flush=True)
                time.sleep(0.05)
        """
        )
        .replace("__API_KEY__", repr(api_key))
        .replace("__SERVER_URL__", repr(server_url))
        .replace("__TIMEOUT_MS__", repr(timeout_ms))
        .replace("__MODEL_ID__", repr(model_id))
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    client = build_client()
    return ask_ok(client)


def main() -> int:
    client = build_client()

    graceful = graceful_cancel_probe(client)
    abrupt = abrupt_cancel_probe()

    print(
        f"graceful cancel -> finish_reason={graceful.finish_reason!r} "
        f"content={graceful.content!r}"
    )
    print(
        f"abrupt cancel   -> finish_reason={abrupt.finish_reason!r} "
        f"content={abrupt.content!r}"
    )

    ok = (
        graceful.finish_reason == "stop"
        and graceful.content == "ok"
        and abrupt.finish_reason == "stop"
        and abrupt.content == "ok"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
