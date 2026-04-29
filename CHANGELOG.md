# Changelog

All notable changes to this project are documented here.

## 2.0.3 - 2026-04-29

### Fixed

- Hardened local `llama.cpp` raw-chat sessions after tool use so a missing final
  assistant message no longer leaves the next prompt vulnerable to
  `After the optional system message, conversation roles must alternate user and assistant roles`.
- Added defensive repair for previously broken local raw-chat histories before
  the next prompt, including stale trailing tool and orphan user turns.

## 2.0.2 - 2026-04-28

### Fixed

- Normalized FireCrawl MCP `firecrawl_search.sources` arguments when the model
  emits a legacy list of strings instead of the current object-based schema,
  restoring searches such as PDF lookups through the installed CLI.
- Preserved nested MCP validation errors in user-facing tool failures so schema
  mismatches no longer look like generic MCP connection problems.

## 2.0.1 - 2026-04-28

### Changed

- Added `pyright` as a second static type checker alongside `mypy`, wired both
  into `make typecheck`, and extended static type validation to cover the test
  suite with documented integration-test exceptions.
- Introduced a structural Mistral client typing boundary so the CLI keeps using
  lightweight test doubles without accidentally switching runtime transport
  paths.
- Improved the terminal-native `/image` and `/doc` picker flow so directory
  navigation stays usable with long paths, including explicit parent-directory
  traversal and current-directory selection.
- Added `docs/project-tree.md` and updated repository instructions so the
  checked-in project tree stays aligned with future layout changes.

## 2.0.0 - 2026-04-28

### Changed

- Refactored the session layer into domain-focused modules for runtime state,
  local and SDK transports, Conversations handling, tool execution, context
  management, and shared parsing primitives while keeping the public
  `MistralSession` facade stable.
- Refactored the CLI layer into smaller modules for parser/config resolution,
  REPL orchestration, slash-command handling, shortcuts, and session state
  rendering while keeping `mistral4cli.cli:main` and the existing REPL UX
  intact.
- Split the previous monolithic CLI test suite into focused test files by
  domain, with shared fixtures and fakes moved into `tests/cli_support.py`.
- Regenerated the checked-in API reference and aligned the user-facing
  documentation with the refactored code layout and maintained CLI behavior.

### Added

- Module docstrings and targeted inline comments across the new session and CLI
  modules so the refactored code remains navigable without reintroducing
  monolithic files.
- A brief architecture overview in the README describing the new internal code
  layout for contributors.

## 1.5.14 - 2026-04-28

### Added

- `/status` now renders in a dedicated red TTY accent so live session snapshots
  stand out from normal assistant output.
- Regression coverage for local `reasoning=on` with `thinking=off` to ensure
  the CLI keeps the local raw transport selected.

### Fixed

- Local `llama.cpp` sessions now keep the raw chat transport enabled when
  backend reasoning is still requested but `/thinking off` hides terminal
  thinking output. This avoids SDK streaming tool-call unmarshalling failures
  on the local backend.

## 1.5.13 - 2026-04-28

### Added

- New `/thinking on|off|toggle` command plus `--thinking` and `--no-thinking`
  startup flags to control whether returned `thinking` blocks are rendered in
  the terminal independently from backend reasoning requests.

### Changed

- `/reasoning` now controls backend reasoning requests only, while `/thinking`
  controls local rendering. This keeps `reasoning_effort` enabled for remote
  chat completions and Conversations even when terminal thinking display is
  hidden.
- Updated README, user guide, and local runtime notes to distinguish backend
  reasoning requests from local thinking display.

## 1.5.12 - 2026-04-28

### Added

- New `/status` REPL command to print the live session snapshot on demand
  between turns, including phase, backend, model, Conversations state, context
  estimate, backend usage, and active attachments.

### Changed

- Help text and user documentation now distinguish `/defaults` as static
  runtime configuration and `/status` as dynamic session state.

## 1.5.11 - 2026-04-28

### Fixed

- Interrupted Mistral Conversations tool turns now send synthetic
  `function.result` cancellation payloads back to the backend before rolling
  the turn back locally. This prevents remote conversations from getting stuck
  with `Cannot append other inputs while function results are still missing`
  after `Ctrl-C`.

### Added

- Regression coverage for interrupted Conversations tool execution and
  interrupted Conversations streaming after a remote `function.call` has been
  emitted.

## 1.5.10 - 2026-04-28

### Fixed

- `/conv use <alias>` now resolves local conversation aliases before calling the
  Mistral Conversations API, so bookmarked conversations can be resumed through
  aliases instead of only raw `conversation_id` values.

### Added

- Regression coverage for `/conv use <alias>` to ensure local alias resolution
  stays wired through the REPL command handler.

## 1.5.9 - 2026-04-28

### Changed

- Updated the pinned `mistralai` dependency floor to `2.4.3` to keep the CLI
  aligned with the current official SDK surface for beta Conversations.

### Added

- Local regression coverage for remote reasoning request wiring across remote
  chat completions and beta Conversations in `complete`, `stream`, `start`,
  `append`, `start_stream`, and `append_stream` paths.
- Remote smoke coverage for `reasoning_effort=\"none\"` in chat completions and
  beta Conversations so the live API behavior stays checked against the current
  SDK.

## 1.5.8 - 2026-04-28

### Added

- TTY Markdown-lite rendering now converts standalone horizontal rules such as
  `---`, `***`, and `___` into real terminal divider lines outside fenced code
  blocks.

## 1.5.7 - 2026-04-28

### Added

- TTY answer rendering now highlights fenced code blocks and other literal code
  lines in a dedicated cyan code style so source snippets stand out from normal
  assistant prose.

## 1.5.6 - 2026-04-28

### Changed

- The interactive status line now separates the live client-side context
  estimate from backend-reported token accounting. `est:` shows the current
  prompt estimate for non-Conversations chat sessions, `last:` shows the most
  recent backend `usage.total_tokens`, and `usage:` shows the cumulative
  backend-reported usage when available.
- Conversations status lines now mark context estimation as `est:backend`
  instead of implying that the CLI can measure the remote server-side context.

### Fixed

- Local streaming sessions no longer make the rightmost status counter look
  frozen when `llama.cpp` skips `usage` updates. The growing context is now
  visible through the cached `est:` value.

## 1.5.5 - 2026-04-28

### Added

- `/conv alias <text>` now assigns an alias directly to the active conversation
  without requiring its `conversation_id`.

### Changed

- `/conv alias <conversation_id> <text>` remains supported for explicit remote
  conversation references.
- Conversations help text and user documentation now document the active
  conversation alias shortcut.

## 1.5.4 - 2026-04-28

### Fixed

- Conversations registry now normalizes remote `datetime` timestamps returned by
  the SDK instead of assuming plain strings.
- `/conv restart` now sends an empty `inputs` field because the live Mistral
  Conversations API rejects restart requests with `inputs=None`.
- `/conv list --meta ...` no longer depends on unreliable backend metadata
  filtering. The CLI enriches listed conversations with `get()` details and
  falls back to its local registry metadata cache for conversations created from
  this CLI.

### Added

- Local regression coverage for timestamp normalization, status/id/store/new,
  `note`, `tag remove`, `unset`, `forget`, metadata-filtered listing, and other
  Conversations management commands.
- A broader remote Conversations smoke test that exercises start, list, filtered
  list, show, history, messages, restart, and delete through the same session
  management path used by the CLI.

## 1.5.3 - 2026-04-28

### Fixed

- `/conv list` now sends `metadata={}` when no metadata filter is requested.
  This works around the current `mistralai` SDK serialization bug where an
  unset `metadata` query field is sent as the SDK sentinel and rejected by the
  Mistral API with `3001 invalid_request_error`.

## 1.5.2 - 2026-04-28

### Fixed

- `/conv list` in Conversations mode now omits the `metadata` query argument
  entirely when no metadata filter is requested, which fixes Mistral API error
  `3001 invalid_request_error` for the default listing path.

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
