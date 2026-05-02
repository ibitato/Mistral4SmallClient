"""Slash-command dispatch and domain handlers for the interactive CLI."""

from __future__ import annotations

import logging
import shlex
import sys
from collections.abc import Callable
from typing import TextIO

from mistralcli.attachments import PathPicker
from mistralcli.cli_config import _parse_metadata_pairs
from mistralcli.cli_shortcuts import (
    _run_doc_shortcut,
    _run_edit_shortcut,
    _run_find_shortcut,
    _run_image_shortcut,
    _run_ls_shortcut,
    _run_shell_shortcut,
)
from mistralcli.cli_state import (
    _clear_attachments,
    _render_session_status,
    _ReplState,
)
from mistralcli.local_mistral import (
    DEFAULT_TIMEOUT_MS,
    REMOTE_MODEL_ID,
    BackendKind,
    LocalMistralConfig,
    MistralConfig,
    RemoteAPIKeyError,
    RemoteMistralConfig,
    get_client_timeout_ms,
    remote_api_key_available,
)
from mistralcli.mistral_client import MistralClientProtocol
from mistralcli.session import MistralSession
from mistralcli.ui import render_help_screen, render_status_snapshot

logger = logging.getLogger("mistralcli.cli")


def _run_command(
    command: str,
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState | None = None,
    local_config: LocalMistralConfig | None = None,
    client_factory: Callable[[MistralConfig], MistralClientProtocol],
    remote_model_id: str = REMOTE_MODEL_ID,
    input_func: Callable[[str], str] = input,
    stdin: TextIO = sys.stdin,
    path_picker: PathPicker | None = None,
    print_paginated_text: Callable[..., None],
    refresh_repl_screen: Callable[[TextIO, MistralSession], None],
) -> bool:
    if repl_state is None:
        repl_state = _ReplState()
    logger.debug("Running command name=%s has_argument=%s", command, bool(argument))
    if command in {"help", "h", "?"}:
        print_paginated_text(
            text=render_help_screen(
                summary=session.describe_defaults(),
                tools=session.describe_tools().splitlines(),
                stream=stdout,
            ),
            stdout=stdout,
            stdin=stdin,
            prompt_label="help",
        )
        return False
    if command == "tools":
        print_paginated_text(
            text=session.describe_tools(),
            stdout=stdout,
            stdin=stdin,
            prompt_label="tools",
        )
        return False
    if command == "status":
        stdout.write(
            render_status_snapshot(
                _render_session_status(session, repl_state),
                stream=stdout,
            )
            + "\n"
        )
        stdout.flush()
        return False
    if command in {"conversations", "conv"}:
        return _run_conversations_command(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            local_config=local_config,
            client_factory=client_factory,
            remote_model_id=remote_model_id,
            stdin=stdin,
            input_func=input_func,
            print_paginated_text=print_paginated_text,
            refresh_repl_screen=refresh_repl_screen,
        )
    if command == "compact":
        return _run_compact_command(argument, session, stdout)
    if command == "remote":
        return _run_remote_command(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            local_config=local_config,
            client_factory=client_factory,
            remote_model_id=remote_model_id,
            refresh_repl_screen=refresh_repl_screen,
        )
    if command == "timeout":
        return _run_timeout_command(argument, session, stdout)
    if command == "reasoning":
        normalized = argument.strip().lower()
        if not normalized:
            stdout.write(session.reasoning_status_text() + "\n")
        elif normalized in {"on", "true", "1"}:
            session.set_reasoning_visibility(True)
            stdout.write("Reasoning request enabled.\n")
        elif normalized in {"off", "false", "0"}:
            session.set_reasoning_visibility(False)
            stdout.write("Reasoning request disabled.\n")
        elif normalized == "toggle":
            session.toggle_reasoning_visibility()
            stdout.write(session.reasoning_status_text() + "\n")
        else:
            stdout.write("Usage: /reasoning [on|off|toggle]\n")
        stdout.flush()
        return False
    if command == "thinking":
        normalized = argument.strip().lower()
        if not normalized:
            stdout.write(session.thinking_status_text() + "\n")
        elif normalized in {"on", "true", "1"}:
            session.set_thinking_visibility(True)
            stdout.write("Thinking display enabled.\n")
        elif normalized in {"off", "false", "0"}:
            session.set_thinking_visibility(False)
            stdout.write("Thinking display disabled.\n")
        elif normalized == "toggle":
            session.toggle_thinking_visibility()
            stdout.write(session.thinking_status_text() + "\n")
        else:
            stdout.write("Usage: /thinking [on|off|toggle]\n")
        stdout.flush()
        return False
    if command == "run":
        return _run_shell_shortcut(argument, session, stdout)
    if command == "ls":
        return _run_ls_shortcut(argument, session, stdout)
    if command == "find":
        return _run_find_shortcut(argument, session, stdout)
    if command == "edit":
        return _run_edit_shortcut(argument, session, stdout)
    if command == "image":
        return _run_image_shortcut(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            input_func=input_func,
            stdin=stdin,
            path_picker=path_picker,
        )
    if command in {"doc", "docs"}:
        return _run_doc_shortcut(
            argument,
            session,
            stdout,
            repl_state=repl_state,
            input_func=input_func,
            stdin=stdin,
            path_picker=path_picker,
        )
    if command == "drop":
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        stdout.write("Active attachments cleared.\n")
        stdout.flush()
        return False
    if command == "dropimage":
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=False,
            clear_pending=True,
        )
        stdout.write("Active image attachments cleared.\n")
        stdout.flush()
        return False
    if command == "dropdoc":
        _clear_attachments(
            repl_state,
            clear_images=False,
            clear_documents=True,
            clear_pending=True,
        )
        stdout.write("Active document attachments cleared.\n")
        stdout.flush()
        return False
    if command in {"exit", "quit", "q"}:
        return True
    if command in {"reset", "new"}:
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        session.reset()
        refresh_repl_screen(stdout, session)
        stdout.write("Conversation reset.\n")
        stdout.flush()
        return False
    if command == "defaults":
        stdout.write(session.describe_defaults() + "\n")
        stdout.flush()
        return False
    if command == "system":
        if argument:
            _clear_attachments(
                repl_state,
                clear_images=True,
                clear_documents=True,
                clear_pending=True,
            )
            session.set_system_prompt(argument)
            refresh_repl_screen(stdout, session)
            stdout.write("System prompt updated and conversation reset.\n")
        else:
            stdout.write("Current system prompt:\n")
            stdout.write(session.system_prompt + "\n")
        stdout.flush()
        return False

    stdout.write(f"Unknown command: /{command}\n")
    stdout.flush()
    return False


def _run_conversations_command(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState,
    local_config: LocalMistralConfig | None,
    client_factory: Callable[[MistralConfig], MistralClientProtocol],
    remote_model_id: str,
    stdin: TextIO,
    input_func: Callable[[str], str],
    print_paginated_text: Callable[..., None],
    refresh_repl_screen: Callable[[TextIO, MistralSession], None],
) -> bool:
    def ensure_enabled(*, allow_resume: bool) -> bool:
        if session.conversations.enabled:
            return True
        try:
            remote_config = RemoteMistralConfig.from_env(
                timeout_ms=get_client_timeout_ms(session.client, DEFAULT_TIMEOUT_MS),
                model_id=remote_model_id,
            )
        except RemoteAPIKeyError as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        session.enable_conversations(
            client=client_factory(remote_config),
            model_id=remote_config.model_id,
            store=session.conversations.store,
        )
        if allow_resume:
            _maybe_resume_conversation(
                session,
                stdin=stdin,
                stdout=stdout,
                input_func=input_func,
                interactive=stdin.isatty(),
            )
        return True

    def parse_metadata_filters(
        raw_tokens: list[str],
    ) -> tuple[int, int, dict[str, str]]:
        page = 0
        page_size = 20
        metadata: dict[str, str] = {}
        index = 0
        while index < len(raw_tokens):
            token = raw_tokens[index]
            if token == "--page" and index + 1 < len(raw_tokens):
                page = int(raw_tokens[index + 1])
                index += 2
                continue
            if token in {"--size", "--page-size"} and index + 1 < len(raw_tokens):
                page_size = int(raw_tokens[index + 1])
                index += 2
                continue
            if token == "--meta" and index + 1 < len(raw_tokens):
                metadata.update(_parse_metadata_pairs([raw_tokens[index + 1]]))
                index += 2
                continue
            raise ValueError(
                "Usage: /conversations list [--page N] [--size N] [--meta KEY=VALUE]"
            )
        return page, page_size, metadata

    tokens = shlex.split(argument)
    action = tokens[0].lower() if tokens else ""
    if not action:
        stdout.write(session.current_conversation_text() + "\n")
        stdout.flush()
        return False

    if action == "on":
        if not ensure_enabled(allow_resume=True):
            return False
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        refresh_repl_screen(stdout, session)
        if session.conversation_id is not None:
            stdout.write(f"Conversations enabled. Resumed {session.conversation_id}.\n")
        else:
            stdout.write("Conversations enabled. Conversation reset.\n")
        stdout.flush()
        return False

    if action == "off":
        session.disable_conversations()
        if (
            local_config is not None
            and session.backend_kind is BackendKind.LOCAL
            and session.server_url == local_config.server_url
        ):
            session.client = client_factory(local_config)
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        refresh_repl_screen(stdout, session)
        stdout.write("Conversations disabled. Conversation reset.\n")
        stdout.flush()
        return False

    if action == "new":
        if not session.conversations.enabled:
            stdout.write("Conversations mode is off.\n")
        else:
            session.reset()
            if session.conversation_registry is not None:
                session.conversation_registry.clear_last_active()
            _clear_attachments(
                repl_state,
                clear_images=True,
                clear_documents=True,
                clear_pending=True,
            )
            refresh_repl_screen(stdout, session)
            stdout.write("New Conversation will start on the next turn.\n")
        stdout.flush()
        return False

    if action == "store":
        value = tokens[1].lower() if len(tokens) > 1 else ""
        if value not in {"on", "off"}:
            stdout.write("Usage: /conversations store [on|off]\n")
        else:
            session.set_conversation_store(value == "on")
            stdout.write(f"Conversation store set to {value}. Conversation reset.\n")
        stdout.flush()
        return False

    if action in {"current", "status"}:
        stdout.write(session.current_conversation_text() + "\n")
        stdout.flush()
        return False

    if action == "id":
        stdout.write(f"{session.conversation_id or 'No active conversation id yet.'}\n")
        stdout.flush()
        return False

    if action == "list":
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            page, page_size, metadata = parse_metadata_filters(tokens[1:])
            text = session.list_remote_conversations(
                page=page,
                page_size=page_size,
                metadata=metadata,
            )
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        print_paginated_text(
            text=text,
            stdout=stdout,
            stdin=stdin,
            prompt_label="conversations list",
        )
        return False

    if action == "bookmarks":
        print_paginated_text(
            text=session.list_remote_conversations(bookmarks_only=True),
            stdout=stdout,
            stdin=stdin,
            prompt_label="conversations bookmarks",
        )
        return False

    if action == "show":
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            text = session.show_remote_conversation(
                tokens[1] if len(tokens) > 1 else None
            )
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        print_paginated_text(
            text=text,
            stdout=stdout,
            stdin=stdin,
            prompt_label="conversations show",
        )
        return False

    if action == "use":
        reference = tokens[1] if len(tokens) > 1 else ""
        if not reference:
            stdout.write("Usage: /conversations use <conversation_id>\n")
            stdout.flush()
            return False
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            conversation_id = session.resolve_conversation_reference(reference)
            rendered = session.attach_remote_conversation(
                conversation_id,
                source="manual",
            )
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        refresh_repl_screen(stdout, session)
        stdout.write(rendered + "\n")
        stdout.flush()
        return False

    if action in {"history", "messages"}:
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            text = session.conversation_history_text(
                messages_only=action == "messages",
                conversation_id=tokens[1] if len(tokens) > 1 else None,
            )
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
            stdout.flush()
            return False
        print_paginated_text(
            text=text,
            stdout=stdout,
            stdin=stdin,
            prompt_label=f"conversations {action}",
        )
        return False

    if action == "delete":
        try:
            rendered = session.delete_remote_conversation(
                tokens[1] if len(tokens) > 1 else None
            )
        except Exception as exc:
            stdout.write(f"[conversations] delete failed: {exc}\n")
            stdout.flush()
            return False
        refresh_repl_screen(stdout, session)
        stdout.write(rendered + "\n")
        stdout.flush()
        return False

    if action == "restart":
        from_entry_id = tokens[1] if len(tokens) > 1 else ""
        restart_reference: str | None = tokens[2] if len(tokens) > 2 else None
        if not from_entry_id:
            stdout.write("Usage: /conversations restart <entry_id> [conversation_id]\n")
            stdout.flush()
            return False
        if not ensure_enabled(allow_resume=False):
            return False
        try:
            rendered = session.restart_remote_conversation(
                from_entry_id=from_entry_id,
                conversation_id=restart_reference,
            )
        except Exception as exc:
            stdout.write(f"[conversations] restart failed: {exc}\n")
            stdout.flush()
            return False
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        refresh_repl_screen(stdout, session)
        stdout.write(rendered + "\n")
        stdout.flush()
        return False

    if action == "set":
        field = tokens[1].lower() if len(tokens) > 1 else ""
        if field == "name" and len(tokens) > 2:
            session.set_pending_conversation_name(" ".join(tokens[2:]))
            stdout.write("Pending conversation name updated.\n")
        elif field == "description" and len(tokens) > 2:
            session.set_pending_conversation_description(" ".join(tokens[2:]))
            stdout.write("Pending conversation description updated.\n")
        elif field == "meta" and len(tokens) > 2:
            try:
                metadata = _parse_metadata_pairs(tokens[2:])
            except ValueError as exc:
                stdout.write(f"[conversations] {exc}\n")
            else:
                for key, value in metadata.items():
                    session.set_pending_conversation_metadata(key, value)
                stdout.write("Pending conversation metadata updated.\n")
        else:
            stdout.write(
                "Usage: /conversations set "
                "[name TEXT|description TEXT|meta KEY=VALUE ...]\n"
            )
        stdout.flush()
        return False

    if action == "unset":
        field = tokens[1].lower() if len(tokens) > 1 else ""
        if field == "name":
            session.clear_pending_conversation_name()
            stdout.write("Pending conversation name cleared.\n")
        elif field == "description":
            session.clear_pending_conversation_description()
            stdout.write("Pending conversation description cleared.\n")
        elif field == "meta":
            if len(tokens) > 2:
                session.clear_pending_conversation_metadata(tokens[2])
                stdout.write(f"Pending conversation metadata {tokens[2]} cleared.\n")
            else:
                session.clear_pending_conversation_metadata()
                stdout.write("Pending conversation metadata cleared.\n")
        elif field == "all":
            session.clear_pending_conversation_name()
            session.clear_pending_conversation_description()
            session.clear_pending_conversation_metadata()
            stdout.write("All pending conversation settings cleared.\n")
        else:
            stdout.write(
                "Usage: /conversations unset [name|description|meta <key>|all]\n"
            )
        stdout.flush()
        return False

    if action == "alias":
        if len(tokens) < 2:
            stdout.write("Usage: /conversations alias [<conversation_id>] <text>\n")
        else:
            try:
                if len(tokens) == 2:
                    if not session.conversation_id:
                        raise ValueError(
                            "No active conversation id yet. Use "
                            '"/conv alias <conversation_id> <text>" or start one first.'
                        )
                    reference = session.conversation_id
                    alias = tokens[1]
                else:
                    reference = tokens[1]
                    alias = " ".join(tokens[2:])
                stdout.write(
                    session.set_local_conversation_alias(reference, alias) + "\n"
                )
            except Exception as exc:
                stdout.write(f"[conversations] {exc}\n")
        stdout.flush()
        return False

    if action == "note":
        if len(tokens) < 3:
            stdout.write("Usage: /conversations note <conversation_id> <text>\n")
        else:
            try:
                stdout.write(
                    session.set_local_conversation_note(tokens[1], " ".join(tokens[2:]))
                    + "\n"
                )
            except Exception as exc:
                stdout.write(f"[conversations] {exc}\n")
        stdout.flush()
        return False

    if action == "tag":
        mode = tokens[1].lower() if len(tokens) > 1 else ""
        if mode not in {"add", "remove"} or len(tokens) < 4:
            stdout.write(
                "Usage: /conversations tag [add|remove] <conversation_id> <tag>\n"
            )
            stdout.flush()
            return False
        try:
            if mode == "add":
                rendered = session.add_local_conversation_tag(tokens[2], tokens[3])
            else:
                rendered = session.remove_local_conversation_tag(tokens[2], tokens[3])
            stdout.write(rendered + "\n")
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
        stdout.flush()
        return False

    if action == "forget":
        if len(tokens) < 2:
            stdout.write("Usage: /conversations forget <conversation_id>\n")
            stdout.flush()
            return False
        try:
            stdout.write(session.forget_local_conversation(tokens[1]) + "\n")
        except Exception as exc:
            stdout.write(f"[conversations] {exc}\n")
        stdout.flush()
        return False

    stdout.write(
        "Usage: /conversations "
        "[on|off|new|store on|store off|current|id|list|bookmarks|show|use|"
        "history|messages|delete|restart|set|unset|alias|note|tag|forget]\n"
    )
    stdout.flush()
    return False


def _maybe_resume_conversation(
    session: MistralSession,
    *,
    stdin: TextIO,
    stdout: TextIO,
    input_func: Callable[[str], str],
    interactive: bool,
) -> None:
    """Resume the last known stored conversation when configured to do so."""

    if not session.conversations.enabled or not session.conversations.store:
        return
    registry = session.conversation_registry
    if registry is None or not registry.last_active_conversation_id:
        return
    if session.conversation_id:
        return

    should_resume = False
    policy = session.conversations.resume_policy
    if policy == "last":
        should_resume = True
    elif policy == "prompt" and interactive:
        try:
            answer = input_func(
                "Resume the last Mistral Conversation "
                f"({registry.last_active_conversation_id})? [Y/n] "
            )
        except EOFError:
            answer = "n"
        should_resume = answer.strip().lower() not in {"n", "no"}

    if not should_resume:
        return

    try:
        session.attach_remote_conversation(
            registry.last_active_conversation_id,
            source="resume",
        )
    except Exception as exc:
        logger.warning(
            "Could not resume conversation id=%s error=%s",
            registry.last_active_conversation_id,
            exc,
        )
        registry.clear_last_active(registry.last_active_conversation_id)
        if interactive:
            stdout.write(
                "[conversations] Could not resume the last conversation; "
                "starting a new one instead.\n"
            )
            stdout.flush()


def _run_compact_command(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
) -> bool:
    tokens = shlex.split(argument)
    action = tokens[0].lower() if tokens else "now"
    if action in {"now", "run"}:
        try:
            stdout.write(session.compact_context().summary() + "\n")
        except Exception as exc:
            stdout.write(f"[compact] failed: {exc}\n")
        stdout.flush()
        return False
    if action in {"status", "show"}:
        stdout.write(session.context_status_text() + "\n")
        stdout.flush()
        return False
    if action == "auto":
        value = tokens[1].lower() if len(tokens) > 1 else ""
        if value not in {"on", "off"}:
            stdout.write("Usage: /compact auto [on|off]\n")
        else:
            session.configure_context(auto_compact=value == "on")
            stdout.write(f"Auto compact set to {value}.\n")
        stdout.flush()
        return False
    if action == "threshold":
        value = tokens[1] if len(tokens) > 1 else ""
        try:
            threshold = float(value.rstrip("%"))
        except ValueError:
            stdout.write("Usage: /compact threshold [0.1-0.99|10-99]\n")
        else:
            session.configure_context(threshold=threshold)
            stdout.write(
                f"Compact threshold set to {round(session.context.threshold * 100)}%.\n"
            )
        stdout.flush()
        return False
    if action == "reserve":
        value = tokens[1] if len(tokens) > 1 else ""
        try:
            reserve_tokens = int(value)
        except ValueError:
            stdout.write("Usage: /compact reserve [TOKENS]\n")
        else:
            session.configure_context(reserve_tokens=reserve_tokens)
            stdout.write(
                f"Context reserve set to {session.context.reserve_tokens} tokens.\n"
            )
        stdout.flush()
        return False
    if action in {"keep", "keep-turns"}:
        value = tokens[1] if len(tokens) > 1 else ""
        try:
            keep_turns = int(value)
        except ValueError:
            stdout.write("Usage: /compact keep [TURNS]\n")
        else:
            session.configure_context(keep_recent_turns=keep_turns)
            stdout.write(
                f"Compact keep turns set to {session.context.keep_recent_turns}.\n"
            )
        stdout.flush()
        return False

    stdout.write(
        "Usage: /compact [status|now|auto on|auto off|threshold N|reserve N|keep N]\n"
    )
    stdout.flush()
    return False


def _run_remote_command(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
    *,
    repl_state: _ReplState,
    local_config: LocalMistralConfig | None,
    client_factory: Callable[[MistralConfig], MistralClientProtocol],
    remote_model_id: str,
    refresh_repl_screen: Callable[[TextIO, MistralSession], None],
) -> bool:
    normalized = argument.strip().lower()
    if not normalized:
        availability = (
            "available" if remote_api_key_available() else "missing MISTRAL_API_KEY"
        )
        stdout.write(f"Backend: {session.backend_kind.value}\n")
        stdout.write(f"Remote mode: {availability}\n")
        stdout.write(f"Remote model: {remote_model_id}\n")
        stdout.flush()
        return False

    if normalized == "on":
        try:
            remote_config = RemoteMistralConfig.from_env(
                timeout_ms=get_client_timeout_ms(session.client, DEFAULT_TIMEOUT_MS),
                model_id=remote_model_id,
            )
        except RemoteAPIKeyError as exc:
            stdout.write(f"[remote] {exc}\n")
            stdout.flush()
            return False

        session.switch_backend(
            client=client_factory(remote_config),
            backend_kind=BackendKind.REMOTE,
            model_id=remote_config.model_id,
            server_url=None,
        )
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        refresh_repl_screen(stdout, session)
        stdout.write(
            f"Remote backend enabled ({remote_config.model_id}). Conversation reset.\n"
        )
        stdout.flush()
        return False

    if normalized == "off":
        if local_config is None:
            stdout.write("[remote] Local configuration is unavailable.\n")
            stdout.flush()
            return False
        session.switch_backend(
            client=client_factory(local_config),
            backend_kind=BackendKind.LOCAL,
            model_id=local_config.model_id,
            server_url=local_config.server_url,
        )
        _clear_attachments(
            repl_state,
            clear_images=True,
            clear_documents=True,
            clear_pending=True,
        )
        refresh_repl_screen(stdout, session)
        stdout.write("Local backend enabled. Conversation reset.\n")
        stdout.flush()
        return False

    stdout.write("Usage: /remote [on|off]\n")
    stdout.flush()
    return False


def _run_timeout_command(
    argument: str,
    session: MistralSession,
    stdout: TextIO,
) -> bool:
    normalized = argument.strip().lower()
    if not normalized:
        stdout.write(f"Timeout: {session.timeout_ms} ms\n")
        stdout.flush()
        return False

    try:
        timeout_ms = _parse_timeout_ms(normalized)
    except ValueError as exc:
        stdout.write(f"[timeout] {exc}\n")
        stdout.flush()
        return False

    session.set_timeout_ms(timeout_ms)
    stdout.write(f"Timeout set to {timeout_ms} ms.\n")
    stdout.flush()
    return False


def _parse_timeout_ms(value: str) -> int:
    text = value.strip().lower()
    if not text:
        raise ValueError("Usage: /timeout [MILLISECONDS|SECONDSs|MINUTESm]")
    if text.endswith("ms"):
        amount = int(text[:-2].strip())
        multiplier = 1
    elif text.endswith("s"):
        amount = int(text[:-1].strip())
        multiplier = 1_000
    elif text.endswith("m"):
        amount = int(text[:-1].strip())
        multiplier = 60_000
    else:
        amount = int(text)
        multiplier = 1

    timeout_ms = amount * multiplier
    if timeout_ms < 1_000:
        raise ValueError("Timeout must be at least 1000 ms.")
    return timeout_ms
