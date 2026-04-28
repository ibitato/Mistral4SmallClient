# Mistral Small 4 Local and Remote CLI

This repository provides a general Mistral Small 4 CLI built on top of the
official `mistralai` Python SDK against two backends:

- a local `llama.cpp` deployment
- the hosted Mistral cloud model exposed as `mistral-small-latest`

The CLI is designed to switch between those backends without changing the main
REPL workflow, attachments, or tool availability.

The client is currently supported on Linux only.

## Runtime requirements

- Python `3.10`
- Linux
- `uv`
- local mode: a running `llama.cpp` server at `http://127.0.0.1:8080`
- remote mode: `MISTRAL_API_KEY` in the environment
- FireCrawl MCP: `FIRECRAWL_API_KEY` in the environment
- `/doc` with PDF inputs: `pdftoppm` available in `PATH`

## Local runtime under test

- Model: `unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL`
- Server URL: `http://127.0.0.1:8080`
- Health endpoint: `GET /health`
- Models endpoint: `GET /v1/models`

## Local start command

The local deployment is started outside this repository with:

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

The chat template in use sets `reasoning_effort=high` by default. The exact
template is versioned in this repository at
[`mistral-small-4-reasoning.jinja`](../mistral-small-4-reasoning.jinja).
For this local stack it is the practical switch that makes reasoning active by
default in llama.cpp.

## Known local constraint

Images smaller than `2x2` pixels crash in `llama.cpp` preprocessing:

```text
GGML_ASSERT(src.nx >= 2 && src.ny >= 2) failed
```

The integration test uses a `2x2` PNG fixture to avoid that crash.

## CLI

The repository ships a dedicated `mistral4cli` REPL for Mistral Small 4. It
uses the official `mistralai` SDK directly and can target either the local or
remote backend at runtime.

The REPL has a retro green/orange presentation, an ASCII welcome banner, and
an actionable help system:

- `--version` or `-v` prints the installed CLI version and exits
- `/help` shows commands, examples, and MCP status
- `/defaults` prints the active runtime defaults
- `/status` prints the current live session snapshot
- `/timeout [value]` shows or updates the active request timeout
- `/remote on|off` switches between local `llama.cpp` and Mistral cloud
- `/conv ...` manages optional Mistral Cloud Conversations mode, bookmarks,
  aliases, and restart/branching
- `/compact [status|now|auto on|auto off|threshold N|reserve N|keep N]` manages
  chat-completions context compaction
- `--reasoning` and `--no-reasoning` control whether reasoning is requested at
  startup
- `--thinking` and `--no-thinking` control whether returned thinking is
  rendered at startup
- `/tools` shows the loaded local and FireCrawl MCP tool catalog
- `/run -- ...` runs a shell command through the local shell tool
- `/ls [PATH]` lists files and directories
- `/find -- ...` searches text in the project tree
- `/edit PATH -- ...` writes text to a file
- `/image --prompt ...` opens a terminal image picker and sends the selected images as a multimodal turn
- `/doc --prompt ...` opens a terminal document picker and uses the backend-appropriate document flow
- `/reset` clears the conversation but keeps the system prompt
- `/system <text>` replaces the system prompt and resets the chat
- `/exit` or `/quit` leaves the REPL

TTY usability details:

- the prompt is rendered as a retro green `M4S>` composer in TTY sessions
- long prompts wrap in the interactive composer instead of overflowing one line
- a bottom status bar appears during active turns and shows phase, backend,
  attachments, the cached context estimate, and backend token accounting
- fenced code blocks are tinted in a dedicated cyan code style so generated
  source snippets are easier to scan than surrounding prose
- standalone Markdown separators such as `---` render as real divider lines
  outside fenced code blocks
- assistant reasoning and answer text stream with a fast typewriter-style
  cadence in TTY mode
- assistant prose wraps cleanly without splitting words in the middle

Status bar token fields:

- `est:` is the CLI's current client-side estimate of the prompt/context size
- `last:` is the most recent backend `usage.total_tokens`
- `usage:` is the cumulative backend-reported usage for the session

Because local streaming through `llama.cpp` can skip `usage` in some events,
`last:` and `usage:` are not guaranteed to move on every turn. `est:` is the
field to watch when you want to know whether the live local context is still
growing.

FireCrawl MCP is configured in [`mcp.json`](../mcp.json) and loaded
automatically when present. The checked-in config expands
`FIRECRAWL_API_KEY` at runtime, so the secret stays out of the repository. The
current setup uses the official MCP Python SDK against FireCrawl's Streamable
HTTP endpoint, while local inference continues to go through the official
`mistralai` client pointed at `llama.cpp`.
Use `--mcp-config <path>` to point to a different config file or `--no-mcp` to
disable tool loading for a run.

The CLI also exposes always-on local OS tools, so the model can inspect files,
run shell commands, and save output without extra setup:

- `shell`
- `read_file`
- `write_file`
- `list_dir`
- `search_text`

When FireCrawl is available, those remote tools are added on top of the local
OS tool set.

Tool semantics are intentionally narrow:

- `shell` is the primary tool for Linux and OS inspection, command execution,
  `rg`/`grep`/`find`, `git`, processes, services, packages, env vars, logs, and
  other system-level discovery.
- `search_text` is only for searching text inside files under a workspace path.
  It is for repo/source lookup and returns one matching line per file.
- `list_dir` is for directory orientation.
- `read_file` is for reading one specific known file.
- `write_file` is for saving or updating text when the task requires it.

Examples:

- "Find files mentioning timeout in `src/`" -> `search_text`
- "Check running nginx processes" -> `shell`
- "Search the OS for docker service files" -> `shell`
- "Show what is in `/etc/systemd`" -> `list_dir` or `shell`
- "Read `pyproject.toml`" -> `read_file`

The attachment commands are designed for multimodal turns:

- `/image` uses a terminal-native picker and builds a multimodal message with
  one selected image. If the picker cannot run, it falls back to a
  manual terminal path prompt.
- `/doc` uses a terminal-native picker for supported document types:
  `txt`, `md`, `rst`, `json`, `yaml`, `yml`, `toml`, `csv`, `pdf`, and `docx`.
- Both commands accept an optional `--prompt`/`-p` argument so you can steer
  the analysis without retyping the default instruction.
- The picker stays fully inside the terminal: first select a root directory,
  then use the fuzzy picker to choose one matching file.
- In local mode, `/doc` rasterizes the selected document into page images before
  sending it to `llama.cpp`.
- In remote mode, `/image` uses the official SDK `image_url` chat flow, and
  `/doc` uses the official SDK `document_url` chat flow for `pdf` and `docx`.
- In remote mode, plain-text documents are embedded directly as text because
  they are already machine-readable and do not need OCR.

For long outputs the local shell and search tools are paginated, so you can use
`offset`/`lines` style arguments to continue from a later page instead of
dumping everything at once.

Runtime defaults:

- temperature: `0.7`
- top-p: `0.95`
- prompt mode: `reasoning`
- timeout: `300000 ms` (`5m`)
- max tokens: unset, so the server can decide
- streaming: on
- Conversations: off by default; `store=on` when enabled
- MCP: FireCrawl auto-tools on when `mcp.json` is present

Remote mode:

- reads `MISTRAL_API_KEY` from the environment
- uses `mistral-small-latest`
- resets the conversation when switching backend
- stays on the official Python SDK path
- uses `reasoning_effort=high` when reasoning is enabled
- uses `reasoning_effort=none` when reasoning is disabled
- `/thinking` only affects whether returned thinking is rendered locally

Conversations mode:

- is optional and uses `client.beta.conversations` against Mistral Cloud
- can start with `--conversations` or inside the REPL with `/conv on`
- requires `MISTRAL_API_KEY`
- resets the current chat when enabled, disabled, or restarted with `/conv new`
- uses `store=on` by default and can switch with `/conv store on|off`
- uses `resume=last` by default and can switch startup policy with
  `--conversation-resume last|new|prompt`
- with `store=off`, each user turn is stateless and no `conversation_id` is kept
- supports `/conv list`, `/conv show`, `/conv use`, `/conv current`,
  `/conv history`, `/conv messages`, `/conv restart`, and `/conv delete`
- supports local bookmarks and aliases with `/conv alias`, `/conv note`,
  `/conv tag`, `/conv bookmarks`, and `/conv forget`
- `/conv alias release-review` is a shortcut for aliasing the active
  conversation without repeating its id
- keeps a local registry under `~/.local/state/mistral4cli/conversations.json`
  or `$XDG_STATE_HOME/mistral4cli/conversations.json`
- accepts pending remote creation metadata with `--conversation-name`,
  `--conversation-description`, `--conversation-meta KEY=VALUE`, and
  `/conv set ...`
- thinking display remains best-effort in Conversations mode; when reasoning is
  requested but the backend omits `thinking`, the CLI reports that explicitly
- remote `name`/`description`/`metadata` are creation-time fields only; the
  CLI cannot update them after creation because Mistral exposes no update API
- on the current model Conversations path, Mistral may not return custom
  `metadata` on later `get/list` calls; the CLI keeps the requested metadata in
  its local registry and uses that cache for `/conv list --meta ...`

Context management:

- applies to the default non-Conversations chat-completions mode
- local mode defaults to the documented `--ctx-size 262144`
- remote chat completions default to a `256000` token window
- auto-compaction is enabled by default at `90%` with an `8192` token response
  reserve
- `/compact` manually summarizes older turns into a compact assistant message
  while keeping the most recent 6 user turns verbatim
- `/compact status` shows the estimated context, configured window, threshold,
  reserve, retained turns, and auto mode
- `/compact threshold 85`, `/compact reserve 4096`, `/compact keep 4`, and
  `/compact auto off` tune the policy inside the REPL
- if the estimated prompt is still above the hard window after compaction, the
  CLI blocks the turn before sending it to the backend
- Conversations mode leaves context handling to the Mistral Conversations
  backend and is not compacted locally

Session commands:

- `/help`
- `/defaults`
- `/status`
- `/remote [on|off]`
- `/conv [on|off|new|store on|store off|current|id|list|bookmarks|show|use|history|messages|delete|restart|set|unset|alias|note|tag|forget]`
- `/compact [status|now|auto on|auto off|threshold N|reserve N|keep N]`
- `/reset`
- `/system <text>`
- `/exit` or `/quit`

Operational notes:

- `Ctrl-C` cancels the current generation without dropping the whole session.
- `Ctrl-D` exits the CLI.
- The interactive REPL clears the screen on startup and after reset actions.
- `TERM=xterm-256color` is recommended for the intended retro palette.
- `uv run python -m mistral4cli --once "..." --no-stream` runs a one-shot smoke
  prompt against the local server.
- The CLI prefers `MISTRAL_LOCAL_MCP_CONFIG` when set; otherwise it falls back
  to `./mcp.json` if the file exists. The FireCrawl URL in that file expands
  `FIRECRAWL_API_KEY` from the environment.
- `uv run python -m mistral4cli` plus `/image` or `/doc` can be used to inspect,
  summarize, compare, or extract from attached files without leaving the session.

## Validation targets

- Chat completions
- Parameter matrix across `temperature`, `top_p`, `max_tokens`, `random_seed`,
  and `prompt_mode`
- Streaming completions
- Cancellation recovery after closing a stream early
- Tool calling
- MCP tool loading and tool-call execution through FireCrawl
- Multimodal request acceptance
- The current local build may return generic or empty text for image prompts, so
  the multimodal test only asserts that image requests are accepted without
  crashing.
- Seeded calls should remain reproducible when the same prompt and sampling
  settings are reused.
- Server health and model listing

## Reasoning visibility

In the local deployment, visible reasoning does not reliably arrive inside the
official `mistralai` SDK models. Direct calls to the llama.cpp-compatible
`/v1/chat/completions` endpoint show that the server emits reasoning in the
separate `reasoning_content` field.

Implications:

- the versioned template is needed so the server enables reasoning by default
- the CLI uses the raw local chat endpoint when reasoning is requested and
  thinking display is enabled
- `/reasoning on|off|toggle` controls whether the CLI requests backend
  reasoning
- `/thinking on|off|toggle` controls whether returned thinking is rendered
  locally
- the remote cloud backend does not use this raw fallback; it relies on the
  official SDK structured `thinking` content instead

## Remote Mistral cloud notes

The CLI can switch to the remote Mistral cloud backend with `/remote on`.

Current constraints from the official SDK and live API:

- `mistral-small-latest` works through normal chat completions
- `prompt_mode="reasoning"` is rejected by the live API for that model
- `reasoning_effort` works through the current official Python SDK and returns
  structured `thinking` content

Because of that, remote mode is implemented through the official SDK with
`reasoning_effort`, while the local backend still needs the raw fallback for
llama.cpp reasoning visibility.

Remote cloud tests are enabled automatically whenever `MISTRAL_API_KEY` exists
in the environment. When the variable is absent, the remote test module skips
cleanly.

## Security notes

- `MISTRAL_API_KEY` and `FIRECRAWL_API_KEY` must come from the user environment.
- `mcp.json` only stores `${FIRECRAWL_API_KEY}` interpolation, never a literal key.
- No cloud secret is required for local mode.
- The CLI intentionally exposes powerful local OS tools, so it should be run in
  a workspace you trust.

## Cancellation behavior

The official `mistralai` SDK does not expose a public `cancel()` method for an
in-flight request. For streaming calls, the supported way to stop consumption is
to exit the stream context so the underlying HTTP response is closed.

I probed two cases locally:

- closing an active stream after a few chunks and then sending another request
- terminating the streaming process abruptly and then sending another request

In both cases the follow-up request completed normally against the local
`llama.cpp` server. That means the server itself does not appear to stay stuck
after a cancel.

Inference: if a CLI starts failing with template or prompt errors after cancel,
the more likely cause is the CLI's conversation state handling. Typical failure
modes are:

- reusing a partially built assistant message after abort
- leaving an unfinished tool-call payload in history
- appending malformed content after an interrupted stream

Use `make cancel-probe` to reproduce the local recovery checks from this repo.
