# MistralClient

[![CI](https://github.com/ibitato/MistralClient/actions/workflows/ci.yml/badge.svg)](https://github.com/ibitato/MistralClient/actions/workflows/ci.yml)
[![Mistral](https://img.shields.io/badge/models-Mistral-ff6f00)](https://docs.mistral.ai/)
[![llama.cpp](https://img.shields.io/badge/runtime-llama.cpp-00a000)](https://github.com/ggerganov/llama.cpp)
[![Mistral Cloud](https://img.shields.io/badge/backend-Mistral%20Cloud-2457ff)](https://docs.mistral.ai/)
[![Docs](https://img.shields.io/badge/docs-generated-blue)](docs/reference.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python: >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-orange.svg)](https://www.python.org/downloads/)

Retro terminal CLI for working with **Mistral models** through two backends:
local `llama.cpp` deployments and remote **Mistral Cloud** through the
official `mistralai` Python SDK.

The client is currently supported on **Linux only**.

The repository is intentionally focused on one product surface:

- use local `llama.cpp` and remote Mistral Cloud from one consistent REPL
- handle image and document turns through backend-appropriate multimodal flows
- keep local OS tools and MCP tools available when the task needs them
- support coding, document work, research, OCR, and general assistant workflows

## What this project includes

- a dedicated interactive CLI with retro green/orange presentation
- a general-purpose multimodal assistant experience for Mistral models
- always-on local tools: `shell`, `read_file`, `write_file`, `list_dir`, `search_text`
- optional FireCrawl MCP tools loaded from `mcp.json` using
  `FIRECRAWL_API_KEY` from your environment
- `/image` and `/doc` attachment commands with a terminal-native picker
- `/remote on|off` to switch between local `llama.cpp` and Mistral cloud
- `/remote model [small|medium]` to change the remote model (only when remote is on)
- `/compact` to inspect, tune, or manually compact chat-completions context
- tests for completion, streaming, cancellation recovery and multimodal payloads

## Quick start

```bash
make sync
uv run python -m mistralcli
```

For a complete command-by-command walkthrough, see the
[user guide](docs/user-guide.md).

## Requirements

- Python `>=3.10` (tested on 3.10, 3.11, 3.12, 3.13, 3.14)
- Linux
- `uv`
- a local `llama.cpp` server at `http://127.0.0.1:8080` if you want local mode
- `MISTRAL_API_KEY` in your shell if you want remote mode
- `FIRECRAWL_API_KEY` in your shell if you want FireCrawl MCP tools
- `pdftoppm` available in `PATH` if you want PDF document rasterization via `/doc`
- a real terminal with TTY support if you want the interactive attachment picker

Useful one-shot smoke test:

```bash
uv run python -m mistralcli --version
uv run python -m mistralcli --once "Return only the word ok." --no-stream
```

Reasoning can be requested or disabled at startup:

```bash
uv run python -m mistralcli --reasoning
uv run python -m mistralcli --no-reasoning
```

## Install without cloning the repo

The supported install path for workstations and servers is a built wheel plus
`uv tool install`.

Build distributable artifacts on one machine:

```bash
make build
```

This creates:

- `dist/mistralcli-<version>-py3-none-any.whl`
- `dist/mistralcli-<version>.tar.gz`

Version tags such as `v3.1.0` also trigger a GitHub Actions release build that
publishes the wheel and source archive as GitHub release assets after passing
the normal repo checks and a wheel-install smoke test.

Install from a local wheel:

```bash
uv tool install ./mistralcli-<version>-py3-none-any.whl
```

Reinstall or upgrade from a newer wheel:

```bash
uv tool install --force ./mistralcli-<version>-py3-none-any.whl
```

Install directly from a GitHub release asset without cloning the repo:

```bash
uv tool install \
  "https://github.com/ibitato/MistralClient/releases/download/v3.1.0/mistralcli-3.1.0-py3-none-any.whl"
```

An optional convenience wrapper is available in [`scripts/install.sh`](scripts/install.sh).
It still uses `uv tool install` under the hood, and its job is only to help
with local wheel discovery, optional release URLs, and cleanup of legacy
repo-local installs.

After installation:

```bash
mistralcli --version
mistralcli --print-defaults
mistralcli
```

## Code layout

The public CLI behavior stays centered around `mistralcli`, but the internal
implementation is now split into smaller domain modules:

- `src/mistralcli/session.py` is the thin `MistralSession` facade
- `session_runtime.py`, `session_transport.py`, `session_conversations.py`,
  `session_tools.py`, `session_context.py`, and `session_primitives.py` own the
  main session domains
- `src/mistralcli/cli.py` is the thin CLI entrypoint facade
- `cli_config.py`, `cli_repl.py`, `cli_commands.py`, `cli_shortcuts.py`, and
  `cli_state.py` own CLI-specific runtime responsibilities
- `tests/cli_support.py` contains shared CLI fixtures and fakes, while the CLI
  behavior tests are split by domain under `tests/test_cli_*.py`

This keeps individual Python units easier to navigate without changing the
user-facing command surface.

Inside the REPL:

- `/help` for actionable usage
- `/defaults` to inspect runtime parameters
- `/status` to inspect the current live session snapshot
- `--version` or `-v` to print the installed CLI version
- `/tools` to inspect loaded tools
- `/timeout [VALUE]` to inspect or change the active request timeout
- `/run -- ...` to execute a shell command
- `/ls [PATH]` to inspect the tree
- `/find -- ...` to search text in the workspace
- `/edit PATH -- ...` to write text files
- `/image` to pick and analyze images in the terminal
- `/doc` to pick and analyze documents in the terminal
- `/remote on|off` to switch cloud mode
- `/remote model [small|medium]` to change the remote model
- `/conv ...` to manage Mistral Cloud Conversations and local bookmarks
- `/reasoning [on|off|toggle]` to request or suppress backend reasoning
- `/thinking [on|off|toggle]` to show or hide returned thinking blocks
- `/compact [status|now|auto on|auto off|threshold N|reserve N|keep N]` to manage context
- `/reset`, `/system ...`, `/exit`

Interactive TTY behavior:

- the prompt is rendered as a retro green `MC>` composer in TTY sessions
- long prompts wrap in the composer instead of overflowing one raw line
- multiline paste in the TTY composer is flattened into one editable text
  buffer; nothing is sent until you press Enter
- a bottom status bar appears during active turns and shows live phase, backend,
  attachments, live context estimate, and backend token accounting
- `/status` prints that same live session state on demand between turns
- fenced code blocks in assistant answers are highlighted in a dedicated cyan
  code style so snippets stand out from normal prose
- standalone Markdown separators such as `---` are rendered as terminal divider
  lines outside code fences
- assistant reasoning and answer text stream with a fast typewriter-style
  cadence in TTY mode
- assistant prose wraps cleanly without splitting words in the middle

Typical tasks include:

- general chat and question answering
- image and document analysis
- OCR and extraction from attached files
- summaries, comparisons, translations, and drafting
- local workspace automation with tools
- programming and debugging when you want it

## Local tool semantics

The model always sees these local tools, but they are intentionally specialized:

- `shell` is the primary tool for Linux and OS inspection: `rg`, `grep`, `find`,
  `git`, `ps`, `systemctl`, package managers, logs, env vars, permissions, and
  system-level discovery.
- `search_text` is only for searching text inside files under a workspace path.
  It is for repo/source lookup and returns one matching line per file.
- `list_dir` is for directory orientation before reading or searching deeper.
- `read_file` is for reading one specific known text file.
- `write_file` is for saving or updating text on disk when the task requires it.

Examples:

- "Find files mentioning timeout in `src/`" -> `search_text`
- "Check running nginx processes" -> `shell`
- "Search the OS for docker service files" -> `shell`
- "Show what is in `/etc/systemd`" -> `list_dir` or `shell`
- "Read `pyproject.toml`" -> `read_file`

Remote mode requirements:

- export `MISTRAL_API_KEY` in your shell
- remote mode defaults to `mistral-small-latest` and `--remote-model` can switch to `mistral-medium-3.5`
- backend switching resets the active conversation
- optional Conversations mode uses `client.beta.conversations` and is off by default
- `--conversations` starts in Conversations mode; `--conversation-store on|off`
  controls server-side persistence and defaults to `on`
- `--conversation-resume {last,new,prompt}` controls whether Conversations mode
  resumes the last known stored remote conversation; the default is `last`
- `--conversation-name`, `--conversation-description`, and repeated
  `--conversation-meta KEY=VALUE` set pending metadata for the next remote
  conversation start
- `--reasoning` and `--no-reasoning` control whether reasoning is requested
- `--thinking` and `--no-thinking` control whether returned thinking is rendered
- `store=off` runs stateless one-shot Conversation calls, so it does not preserve
  `conversation_id` across turns
- the CLI keeps a local registry at `~/.local/state/mistralcli/conversations.json`
  (or `$XDG_STATE_HOME/...`) for aliases, tags, notes, and last-active resume state
- the default request timeout is `300000 ms` (5 minutes)

Conversations management:

- `/conv on` enables Conversations mode and resumes the last stored conversation
  when the resume policy is `last`
- `/conv list --page 0 --size 20 --meta owner=dlopez` lists remote conversations;
  metadata filters are applied by the CLI using remote details plus its local
  registry cache
- `/conv show <id>` inspects remote metadata for one conversation
- `/conv use <id>` reattaches the current session to an existing remote conversation
- `/conv history [id]` and `/conv messages [id]` inspect remote history
- `/conv restart <entry_id> [id]` branches from a specific remote history entry
- `/conv delete [id]` deletes a remote conversation
- `/conv alias`, `/conv note`, `/conv tag`, `/conv bookmarks`, and `/conv forget`
  manage the CLI-side local overlay
- `/conv alias release-review` assigns an alias directly to the active
  conversation; `/conv alias <id> release-review` still works for any known id
- Mistral does not expose a remote update API for existing conversation
  `name`/`metadata`, so aliases and bookmarks are stored locally by the CLI
- Mistral model Conversations currently preserve `name` and `description`, but
  may not return custom `metadata` in `get/list`; when a conversation is created
  from this CLI, the requested metadata is preserved in the local registry so
  `/conv list --meta ...` remains useful for those sessions

Context management:

- default chat completions are client-managed and send the full local history
- the CLI estimates context before each non-Conversations request because the
  SDK does not expose a backend tokenizer for this path
- local mode defaults to the configured local model window of `262144`
  tokens; remote chat completions default to `256000` tokens
- auto-compaction is enabled by default at `90%` of the configured window and
  reserves `8192` tokens for the next response
- `/compact` summarizes older turns into one compact assistant message and keeps
  the most recent 6 user turns verbatim
- if compaction cannot bring a request under the hard window, the CLI blocks the
  turn before sending it to the backend
- Conversations mode is not compacted locally; its server-side context handling
  remains backend-managed by Mistral

Typical environment setup:

```bash
export MISTRAL_API_KEY=...
export FIRECRAWL_API_KEY=...
```

Attachment picker flow:

- `/image` and `/doc` use a pure terminal picker with no GUI requirements
- first browse directories in the terminal picker
- use `[use]` to keep the current directory or `[..]` to move to the parent
- then use a fuzzy list to pick one matching file
- `Enter` selects the highlighted entry and `Ctrl-C` cancels
- if the picker cannot run, the CLI falls back to manual path entry

## Local llama.cpp setup

The local runtime is expected to be running outside this repo with `llama.cpp`.
The validated launch profile for the current test stack is:

```bash
llama-server \
  -hf unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL \
  --host 0.0.0.0 --port 8080 \
  --jinja --flash-attn off --no-mmap \
  --chat-template-file ./mistral-small-4-reasoning.jinja \
  --ctx-size 262144 \
  -ngl 99 \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.0 \
  --parallel 1 --ctx-checkpoints 32 --cache-prompt \
  --threads "$(nproc)"
```

Recommended runtime defaults used by the CLI:

- `temperature=0.7`
- `top_p=0.95`
- `prompt_mode=reasoning`
- `timeout_ms=300000`
- streaming on by default
- `max_tokens` unset unless you override it
- Conversations mode off by default; `store=on` when enabled
- auto context compaction on at `90%`, preserving 6 recent turns

Remote mode keeps the same sampling defaults, but it does not send
`prompt_mode=reasoning`. The live Mistral cloud API rejects that setting for
`mistral-small-latest` or `mistral-medium-3.5`, so the CLI uses the official SDK with
`reasoning_effort=high` when reasoning is enabled, and `reasoning_effort=none`
when it is disabled. `/thinking` only affects terminal rendering.

Conversations mode is an optional Mistral Cloud path. It resets the current chat
when enabled, starts a fresh remote `conversation_id` on the next user turn, and
keeps the normal chat-completions path as the default. When reasoning is
enabled, the CLI requests thinking traces for Conversations too, but the
backend may still omit them on some turns; the CLI reports that explicitly
when thinking display is on.

Attachment handling is backend-aware:

- local `/image` sends `image_url` blocks to `llama.cpp`
- local `/doc` rasterizes supported documents into page images for OCR/vision
- remote `/image` uses the official SDK vision flow with `image_url`
- remote `/doc` uses the official SDK document flow with `document_url` for `pdf` and `docx`
- remote plain-text documents are embedded directly as text because they are already machine-readable

The repository now includes the exact reasoning template at
[`mistral-small-4-reasoning.jinja`](mistral-small-4-reasoning.jinja). In this
local setup it is effectively required if you want reasoning requested by
default, because it sets `reasoning_effort=high` in the llama.cpp chat
template.

For the detailed backend and runtime runbook, see
[docs/backends-and-runtime.md](docs/backends-and-runtime.md).
For day-to-day CLI usage, see [docs/user-guide.md](docs/user-guide.md).

## Testing

```bash
make check
make test
make docs
```

`make check` runs formatting, lint, `mypy`, `pyright`, and docs checks.
`make test` runs the full `pytest` suite, including local integration tests
that require the `llama.cpp` server.
Remote cloud integration runs automatically when `MISTRAL_API_KEY` is present
in the environment and skips cleanly when it is absent.
`make docs` regenerates the checked-in API reference from public docstrings.
`make typecheck` runs both static type checkers, with `mypy` covering `src`
and `tests` and `pyright` providing a second pass over the configured project
tree.

For the intended retro palette in the interactive REPL, prefer:

```bash
export TERM=xterm-256color
```

The interactive REPL clears the screen on startup and after conversation reset
actions so the app stays pinned to the top of the terminal.

## Security

- Secrets are not stored in the repository.
- `mcp.json` uses `${FIRECRAWL_API_KEY}` interpolation instead of a checked-in token.
- Remote cloud mode reads `MISTRAL_API_KEY` from the user environment at runtime.
- The CLI exposes powerful local tools:
  `shell`, `read_file`, `write_file`, `list_dir`, and `search_text`.
- Run it in a workspace and environment you trust.
- Keep `.venv`, `.env`, and any ad hoc credential files out of version control.

## Repository layout

- `src/mistralcli/` - CLI, session, tools and attachment handling
- `tests/` - unit and integration tests
- `docs/user-guide.md` - practical end-user guide for the CLI
- `docs/backends-and-runtime.md` - backend-specific deployment and runtime notes
- `docs/reference.md` - generated API reference from public docstrings
- `mistral-small-4-reasoning.jinja` - versioned llama.cpp reasoning template
- `mcp.json` - optional FireCrawl MCP config that expands
  `FIRECRAWL_API_KEY` at runtime

## License

MIT. See [LICENSE](LICENSE).
