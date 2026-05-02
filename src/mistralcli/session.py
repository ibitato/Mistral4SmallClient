# mypy: disable-error-code="assignment"
"""Interactive session facade for the dual-model Mistral CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TextIO

from mistralcli.conversation_registry import ConversationRegistry
from mistralcli.local_mistral import (
    DEFAULT_MODEL_ID,
    DEFAULT_SERVER_URL,
    BackendKind,
    ContextConfig,
    ConversationConfig,
    LocalGenerationConfig,
)
from mistralcli.mistral_client import MistralClientProtocol
from mistralcli.session_context import SessionContextMixin
from mistralcli.session_conversations import SessionConversationsMixin
from mistralcli.session_primitives import (
    DEFAULT_SYSTEM_PROMPT,
    ContextStatus,
    PendingConversationSettings,
    SessionStatusSnapshot,
    TurnResult,
    UsageSnapshot,
    _BackendState,
    render_defaults_summary,
)
from mistralcli.session_runtime import SessionRuntimeMixin
from mistralcli.session_tools import SessionToolsMixin
from mistralcli.session_transport import SessionTransportMixin
from mistralcli.tooling import ToolBridge


@dataclass(slots=True)
class MistralSession(
    SessionRuntimeMixin,
    SessionContextMixin,
    SessionToolsMixin,
    SessionConversationsMixin,
    SessionTransportMixin,
):
    """Stateful conversation helper for the dual-model Mistral CLI."""

    client: MistralClientProtocol
    backend_kind: BackendKind = BackendKind.LOCAL
    model_id: str = DEFAULT_MODEL_ID
    server_url: str | None = DEFAULT_SERVER_URL
    generation: LocalGenerationConfig = field(default_factory=LocalGenerationConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_bridge: ToolBridge | None = None
    stdout: TextIO | None = None
    stream_enabled: bool = True
    show_reasoning: bool = True
    show_thinking: bool = True
    conversations: ConversationConfig = field(default_factory=ConversationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    logging_summary: str = "debug=on level=DEBUG rotate=daily retention=2d"
    max_tool_rounds: int = 20
    conversation_registry: ConversationRegistry | None = None
    answer_writer: Callable[[str], None] | None = None
    reasoning_writer: Callable[[str], None] | None = None
    status_callback: Callable[[], None] | None = None
    messages: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _mcp_warning_shown: bool = field(init=False, repr=False, default=False)
    _status_phase: str = field(init=False, repr=False, default="idle")
    _status_detail: str | None = field(init=False, repr=False, default=None)
    _last_usage: UsageSnapshot | None = field(init=False, repr=False, default=None)
    _cumulative_usage: UsageSnapshot | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _turn_usage_accumulator: UsageSnapshot | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _cached_context_status: ContextStatus | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _context_status_dirty: bool = field(init=False, repr=False, default=True)
    _missing_reasoning_notice_shown: bool = field(
        init=False,
        repr=False,
        default=False,
    )
    conversation_id: str | None = field(init=False, repr=False, default=None)
    pending_conversation: PendingConversationSettings = field(
        init=False,
        repr=False,
        default_factory=PendingConversationSettings,
    )
    conversation_resume_source: str = field(init=False, repr=False, default="new")
    _previous_backend_state: _BackendState | None = field(
        init=False,
        repr=False,
        default=None,
    )


MistralCodingSession = MistralSession

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ContextStatus",
    "MistralCodingSession",
    "MistralSession",
    "PendingConversationSettings",
    "SessionStatusSnapshot",
    "TurnResult",
    "UsageSnapshot",
    "render_defaults_summary",
]
