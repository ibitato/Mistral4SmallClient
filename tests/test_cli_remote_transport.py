# ruff: noqa: F403, F405
from typing import cast

from tests.cli_support import *


def test_remote_request_uses_reasoning_effort_and_omits_prompt_mode() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return only ok.", stream=False)

    assert result.content == "ok"
    assert "prompt_mode" not in fake_client.chat.complete_calls[0]
    assert fake_client.chat.complete_calls[0]["reasoning_effort"] == "high"


def test_remote_non_stream_usage_is_tracked_in_status_snapshot() -> None:
    output = io.StringIO()
    fake_client = FakeClient(
        complete_responses=[
            FakeResponse(
                choices=[
                    FakeChoice(
                        message=FakeMessage(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=FakeUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        ]
    )
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return only ok.", stream=False)

    snapshot = session.status_snapshot()
    assert result.content == "ok"
    assert snapshot.last_usage is not None
    assert snapshot.last_usage.total_tokens == 15
    assert snapshot.last_usage.max_context_tokens == 256_000
    assert snapshot.cumulative_usage is not None
    assert snapshot.cumulative_usage.total_tokens == 15


def test_remote_stream_usage_is_tracked_in_status_snapshot() -> None:
    output = io.StringIO()

    class UsageStreamChat:
        def __init__(self) -> None:
            self.complete_calls: list[dict[str, Any]] = []
            self.stream_calls: list[dict[str, Any]] = []
            self.last_stream = FakeStream(
                [
                    FakeEvent(
                        data=FakeResponse(
                            choices=[
                                FakeChoice(
                                    delta=FakeDelta(content="ok"),
                                    finish_reason="stop",
                                )
                            ],
                            usage=FakeUsage(
                                prompt_tokens=20,
                                completion_tokens=10,
                                total_tokens=30,
                            ),
                        )
                    )
                ]
            )

        def stream(self, **kwargs: Any) -> FakeStream:
            self.stream_calls.append(kwargs)
            return self.last_stream

    class UsageStreamClient:
        def __init__(self) -> None:
            self.chat = UsageStreamChat()

    session = MistralSession(
        client=cast(Any, UsageStreamClient()),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return only ok.", stream=True)

    snapshot = session.status_snapshot()
    assert result.content == "ok"
    assert snapshot.last_usage is not None
    assert snapshot.last_usage.total_tokens == 30
    assert snapshot.cumulative_usage is not None
    assert snapshot.cumulative_usage.total_tokens == 30


def test_remote_request_disables_reasoning_effort_when_hidden() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=False,
    )

    result = session.send("Return only ok.", stream=False)

    assert result.content == "ok"
    assert "prompt_mode" not in fake_client.chat.complete_calls[0]
    assert fake_client.chat.complete_calls[0]["reasoning_effort"] == "none"


def test_remote_request_keeps_reasoning_when_thinking_is_hidden() -> None:
    output = io.StringIO()
    fake_client = FakeClient(complete_text="ok")
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=True,
        show_thinking=False,
    )

    result = session.send("Return only ok.", stream=False)

    assert result.content == "ok"
    assert fake_client.chat.complete_calls[0]["reasoning_effort"] == "high"


def test_remote_stream_request_uses_reasoning_effort_and_omits_prompt_mode() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["ok"])
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
    )

    result = session.send("Return only ok.", stream=True)

    assert result.content == "ok"
    assert len(fake_client.chat.stream_calls) == 1
    call = fake_client.chat.stream_calls[0]
    assert "prompt_mode" not in call
    assert call["reasoning_effort"] == "high"


def test_remote_stream_request_disables_reasoning_effort_when_hidden() -> None:
    output = io.StringIO()
    fake_client = FakeClient(stream_chunks=["ok"])
    session = MistralSession(
        client=fake_client,
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        generation=LocalGenerationConfig(),
        stdout=output,
        show_reasoning=False,
    )

    result = session.send("Return only ok.", stream=True)

    assert result.content == "ok"
    assert len(fake_client.chat.stream_calls) == 1
    call = fake_client.chat.stream_calls[0]
    assert "prompt_mode" not in call
    assert call["reasoning_effort"] == "none"


def test_conversations_session_starts_then_appends() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "first"}],
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "second"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    first = session.send("hello", stream=False)
    second = session.send("again", stream=False)

    assert first.content == "first"
    assert second.content == "second"
    assert session.conversation_id == "conv_1"
    assert conversations.start_calls[0]["inputs"] == "hello"
    assert conversations.start_calls[0]["completion_args"]["reasoning_effort"] == "high"
    assert conversations.append_calls[0]["conversation_id"] == "conv_1"
    assert conversations.append_calls[0]["inputs"] == "again"
    assert (
        conversations.append_calls[0]["completion_args"]["reasoning_effort"] == "high"
    )


def test_conversations_session_start_and_append_disable_reasoning_effort() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "first"}],
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_1",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "second"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        show_reasoning=False,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.set_reasoning_visibility(False)

    first = session.send("hello", stream=False)
    second = session.send("again", stream=False)

    assert first.content == "first"
    assert second.content == "second"
    assert conversations.start_calls[0]["completion_args"]["reasoning_effort"] == "none"
    assert (
        conversations.append_calls[0]["completion_args"]["reasoning_effort"] == "none"
    )


def test_conversations_keep_reasoning_effort_when_thinking_is_hidden() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_hidden_thinking",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "ok"}],
                    )
                ],
            )
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        show_reasoning=True,
        show_thinking=False,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("hello", stream=False)

    assert result.content == "ok"
    assert conversations.start_calls[0]["completion_args"]["reasoning_effort"] == "high"
    assert "Mistral Conversations returned no thinking blocks" not in output.getvalue()


def test_conversations_streaming_records_usage_and_text() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        stream_events=[
            FakeConversationEvent(
                event="conversation.response.started",
                data=FakeConversationStarted(conversation_id="conv_stream"),
            ),
            FakeConversationEvent(
                event="message.output.delta",
                data=FakeConversationMessageDelta(
                    content={"type": "text", "text": "ok"}
                ),
            ),
            FakeConversationEvent(
                event="conversation.response.done",
                data=FakeConversationDone(
                    usage=FakeUsage(
                        prompt_tokens=10,
                        completion_tokens=2,
                        total_tokens=12,
                    )
                ),
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("hello", stream=True)

    assert result.content == "ok"
    assert "ok\n" in output.getvalue()
    assert "Mistral Conversations returned no thinking blocks" in output.getvalue()
    assert session.conversation_id == "conv_stream"
    assert (
        conversations.start_stream_calls[0]["completion_args"]["reasoning_effort"]
        == "high"
    )
    last_usage = session.status_snapshot().last_usage
    assert last_usage is not None
    assert last_usage.total_tokens == 12


def test_conversations_append_stream_disables_reasoning_effort_when_hidden() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        stream_events=[
            FakeConversationEvent(
                event="message.output.delta",
                data=FakeConversationMessageDelta(
                    content={"type": "text", "text": "ok"}
                ),
            ),
            FakeConversationEvent(
                event="conversation.response.done",
                data=FakeConversationDone(
                    usage=FakeUsage(
                        prompt_tokens=8,
                        completion_tokens=2,
                        total_tokens=10,
                    )
                ),
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        stdout=output,
        show_reasoning=False,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )
    session.set_reasoning_visibility(False)
    session.conversation_id = "conv_stream"

    result = session.send("hello", stream=True)

    assert result.content == "ok"
    assert conversations.append_stream_calls[0]["conversation_id"] == "conv_stream"
    assert (
        conversations.append_stream_calls[0]["completion_args"]["reasoning_effort"]
        == "none"
    )


def test_conversations_tool_call_executes_bridge_and_appends_result() -> None:
    output = io.StringIO()
    tool_bridge = FakeToolBridge()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="function.call",
                        tool_call_id="tool_call_1",
                        name="web_search",
                        arguments='{"query":"mistral"}',
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "done"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        tool_bridge=tool_bridge,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("search", stream=False)

    assert result.content == "done"
    assert tool_bridge.calls == [("web_search", {"query": "mistral"})]
    assert conversations.append_calls[0]["inputs"][0]["type"] == "function.result"
    assert conversations.append_calls[0]["inputs"][0]["tool_call_id"] == "tool_call_1"


def test_conversations_tool_interrupt_appends_cancel_result_and_recovers() -> None:
    output = io.StringIO()
    tool_bridge = InterruptingToolBridge()
    conversations = FakeConversations(
        responses=[
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="function.call",
                        tool_call_id="tool_call_1",
                        name="web_search",
                        arguments='{"query":"mistral"}',
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "cancelled cleanup"}],
                    )
                ],
            ),
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "recovered"}],
                    )
                ],
            ),
        ]
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        tool_bridge=tool_bridge,
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    first = session.send("search", stream=False)
    second = session.send("try again", stream=False)

    assert first.cancelled is True
    assert first.finish_reason == "cancelled"
    assert second.cancelled is False
    assert second.content == "recovered"
    assert conversations.append_calls[0]["inputs"][0]["type"] == "function.result"
    assert conversations.append_calls[0]["inputs"][0]["tool_call_id"] == "tool_call_1"
    assert "cancelled_by_user" in conversations.append_calls[0]["inputs"][0]["result"]
    assert conversations.append_calls[1]["inputs"] == "try again"


def test_conversations_stream_interrupt_with_tool_call_appends_cancel_result() -> None:
    output = io.StringIO()
    conversations = FakeConversations(
        stream_events=[
            FakeConversationEvent(
                "conversation.response.started",
                FakeConversationStarted("conv_tools"),
            ),
            FakeConversationEvent(
                "function.call.delta",
                FakeConversationFunctionDelta(
                    tool_call_id="tool_call_1",
                    name="web_search",
                    arguments='{"query":"mistral"}',
                ),
            ),
            FakeConversationEvent(
                "conversation.response.done",
                FakeConversationDone(),
            ),
        ],
        responses=[
            FakeConversationResponse(
                conversation_id="conv_tools",
                outputs=[
                    FakeConversationOutput(
                        type="message.output",
                        content=[{"type": "text", "text": "cleanup"}],
                    )
                ],
            )
        ],
        stream_interrupt_after=2,
    )
    session = MistralSession(
        client=FakeConversationClient(conversations),
        backend_kind=BackendKind.REMOTE,
        model_id="mistral-small-latest",
        server_url=None,
        tool_bridge=FakeToolBridge(),
        stdout=output,
    )
    session.enable_conversations(
        client=session.client,
        model_id="mistral-small-latest",
        store=True,
    )

    result = session.send("search", stream=True)

    assert result.cancelled is True
    assert result.finish_reason == "cancelled"
    assert conversations.append_calls[0]["inputs"][0]["type"] == "function.result"
    assert conversations.append_calls[0]["inputs"][0]["tool_call_id"] == "tool_call_1"
    assert "cancelled_by_user" in conversations.append_calls[0]["inputs"][0]["result"]
