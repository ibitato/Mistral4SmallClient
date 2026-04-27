# Changelog

All notable changes to this project are documented here.

## 1.3.0 - 2026-04-27

### Added

- Optional Mistral Cloud Conversations mode via `client.beta.conversations`.
- CLI flags `--conversations`, `--no-conversations`, and `--conversation-store`.
- REPL commands `/conv` and `/conversations` with `on`, `off`, `new`, `store`,
  `id`, `history`, `messages`, and `delete` actions.
- Conversation-aware status/defaults rendering, including storage mode and active
  conversation id.
- Unit coverage for Conversations startup, append, streaming, tool calls, and
  slash commands.
- Remote smoke coverage for Conversations start/append.

### Changed

- Bumped package version to `1.3.0`.
- Updated README, local runtime docs, and generated API reference for
  Conversations mode.
- Treat `store=off` Conversations turns as stateless because Mistral does not
  allow appending to un-stored conversations.

### Unchanged

- Chat completions remain the default mode.
- Local `llama.cpp` behavior remains unchanged.
