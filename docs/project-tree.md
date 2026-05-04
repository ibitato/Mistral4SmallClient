# Project Tree

This document records the checked-in project structure at a high level.

Keep it updated whenever files or directories are added, removed, renamed, or
reorganized in the repository.

```text
.
|-- .github
|   |-- workflows
|   |   |-- ci.yml
|   |   `-- release.yml
|   `-- PULL_REQUEST_TEMPLATE.md
|-- docs
|   |-- backends-and-runtime.md
|   |-- project-tree.md
|   |-- reference.md
|   `-- user-guide.md
|-- scripts
|   |-- cancel_probe.py
|   |-- check_docs.py
|   |-- generate_reference.py
|   `-- install.sh
|-- src
|   `-- mistralcli
|       |-- __init__.py
|       |-- __main__.py
|       |-- attachments.py
|       |-- cli.py
|       |-- cli_commands.py
|       |-- cli_config.py
|       |-- cli_repl.py
|       |-- cli_shortcuts.py
|       |-- cli_state.py
|       |-- conversation_registry.py
|       |-- local_mistral.py
|       |-- local_tools.py
|       |-- logging_config.py
|       |-- mcp_bridge.py
|       |-- session.py
|       |-- session_context.py
|       |-- session_conversations.py
|       |-- session_primitives.py
|       |-- session_runtime.py
|       |-- session_tools.py
|       |-- session_transport.py
|       |-- terminal_picker.py
|       |-- tooling.py
|       `-- ui.py
|-- tests
|   |-- fixtures
|   |   `-- internet
|   |       `-- README.md
|   |-- __init__.py
|   |-- cli_support.py
|   |-- conftest.py
|   |-- test_attachments.py
|   |-- test_cli.py
|   |-- test_cli_conversations_commands.py
|   |-- test_cli_reasoning.py
|   |-- test_cli_remote_transport.py
|   |-- test_cli_runtime_attachments.py
|   |-- test_cli_tty_status.py
|   |-- test_cli_ui_tools.py
|   |-- test_conversation_registry.py
|   |-- test_local_mistral.py
|   |-- test_local_tools.py
|   |-- test_logging_config.py
|   |-- test_mcp_bridge.py
|   |-- test_remote_conversations.py
|   |-- test_remote_mistral.py
|   `-- test_tooling.py
|-- AGENTS.md
|-- CHANGELOG.md
|-- LICENSE
|-- Makefile
|-- mcp.json
|-- mistral-small-4-reasoning.jinja
|-- pyproject.toml
|-- README.md
`-- uv.lock
```
