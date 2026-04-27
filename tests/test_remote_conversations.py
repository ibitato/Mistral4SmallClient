from __future__ import annotations

import os

import pytest

from mistral4cli.local_mistral import RemoteMistralConfig, build_client

pytestmark = pytest.mark.remote


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="Set MISTRAL_API_KEY to enable remote Mistral Conversations tests.",
)
def test_remote_conversations_start_and_append_smoke() -> None:
    config = RemoteMistralConfig.from_env()
    client = build_client(config)

    conversation_id = ""
    try:
        started = client.beta.conversations.start(
            model=config.model_id,
            instructions="Return short literal answers.",
            inputs="Return only the word ok.",
            store=True,
            completion_args={"max_tokens": 16, "temperature": 0.0},
        )

        assert started.conversation_id
        assert started.outputs
        conversation_id = started.conversation_id

        followup = client.beta.conversations.append(
            conversation_id=conversation_id,
            inputs="Return only the word done.",
            store=True,
            completion_args={"max_tokens": 16, "temperature": 0.0},
        )

        assert followup.conversation_id == conversation_id
        assert followup.outputs
    finally:
        if conversation_id:
            client.beta.conversations.delete(conversation_id=conversation_id)
