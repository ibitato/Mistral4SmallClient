# Changelog

All notable changes to this project are documented here.

## Unreleased

## 1.5.1 - 2026-04-28

### Added

- Standard CLI version flags `--version` and `-v`.
- Package-level `mistral4cli.__version__` export for CLI and test coverage.

### Changed

- Bumped package version to `1.5.1`.
- Updated README and user-facing runtime guides to document version checks
  before smoke-testing or remote installs.

## 1.5.0 - 2026-04-28

### Added

- Full remote Conversations manager inside the CLI with `/conv list`, `show`,
  `use`, `restart`, `current`, `bookmarks`, `alias`, `note`, `tag`, and
  `forget`.
- Persistent local conversation registry for aliases, tags, notes, branch
  lineage, and last-active auto-resume state.
- Startup flags for Conversations resume policy, pending conversation name,
  description, metadata, and registry path override.
- Coverage for auto-resume, pending metadata, restart, local bookmarks, and
  conversation-id migration.

### Changed

- Bumped package version to `1.5.0`.
- Conversations mode can now resume the last known stored remote conversation
  automatically with `resume=last` by default.
- `/conv history` now prints remote entry ids so restart/branching is usable
  from the CLI.
- Documentation was expanded and rewritten around Conversations management,
  especially the user guide.

## 1.4.1 - 2026-04-27

### Added

- CLI flags `--reasoning` and `--no-reasoning` to control visible reasoning at
  startup for local, remote chat-completions, and Conversations runs.
- Coverage for Conversations reasoning requests, disabled reasoning, and
  best-effort reasoning notices.

### Changed

- Bumped package version to `1.4.1`.
- Remote Conversations now report visible reasoning as a best-effort request in
  user-facing status text.
- When reasoning is requested but Mistral Conversations returns no `thinking`
  blocks, the CLI prints an explicit notice instead of failing silently.
- Updated README, local runtime notes, and the user guide for reasoning flags
  and Conversations reasoning behavior.

## 1.4.0 - 2026-04-27

### Added

- Client-side context overflow management for the default chat-completions mode.
- Automatic context compaction before a request crosses the configured threshold,
  enabled by default at `90%`.
- Manual `/compact` REPL command with `status`, `auto`, `threshold`, `reserve`,
  and `keep` actions.
- CLI and environment configuration for context windows, reserve tokens, compact
  threshold, retained recent turns, and summary budget.
- Unit and functional coverage for manual compaction, automatic compaction,
  overflow blocking, CLI flags, and slash command behavior.

### Changed

- Bumped package version to `1.4.0`.
- Local context status now uses the configured 262144-token Mistral Small 4
  window instead of showing an unknown limit.
- Normal non-Conversations chat completions keep tools enabled even when
  `conversation_store=off` is configured for Conversations mode.
- Updated README, local runtime docs, and generated API reference for context
  management.

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
