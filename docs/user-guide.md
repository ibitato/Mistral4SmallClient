# MistralClient User Guide

This guide explains how to run and test the `mistralcli` command-line
assistant as an end user. It focuses on day-to-day CLI usage rather than
developer internals.

## What The CLI Does

`mistralcli` is an interactive Linux terminal client for working with Mistral
models across local `llama.cpp` and remote Mistral Cloud backends. It
can use:

- a local `llama.cpp` server, which is the default backend
- Mistral Cloud chat completions through the official Python SDK
- Mistral Cloud Conversations mode through `client.beta.conversations`
- local OS tools for shell, file reads, file writes, directory listing, and text
  search
- optional FireCrawl MCP tools when `mcp.json` and `FIRECRAWL_API_KEY` are
  available

The normal mode is chat completions. Conversations mode is optional and must be
enabled explicitly.

## Requirements

- Linux
- Python `>=3.10` (tested on 3.10, 3.11, 3.12, 3.13, 3.14)
- `uv`
- local mode: a running `llama.cpp` server at `http://127.0.0.1:8080`
- remote mode: `MISTRAL_API_KEY` in your environment
- FireCrawl tools: `FIRECRAWL_API_KEY` in your environment
- PDF document support: `pdftoppm` available in `PATH`
- interactive file picker: a real TTY terminal

For the intended terminal palette, use:

```bash
export TERM=xterm-256color
```

## Install A Released Wheel

For non-development machines, the supported install path is `uv tool install`
against a released wheel.

Install from a local wheel:

```bash
uv tool install ./mistralcli-<version>-py3-none-any.whl
```

Install directly from a GitHub release asset:

```bash
uv tool install \
  "https://github.com/ibitato/MistralClient/releases/download/v3.2.1/mistralcli-3.2.1-py3-none-any.whl"
```

Upgrade or reinstall from a newer wheel:

```bash
uv tool install --force ./mistralcli-<version>-py3-none-any.whl
```

If you have an older repo-local install or wrapper script, remove it before
switching to the supported `uv tool` flow.

## First Run

From a source checkout:

```bash
make sync
uv run python -m mistralcli --version
uv run python -m mistralcli --print-defaults
uv run python -m mistralcli
```

Installed as a tool:

```bash
mistralcli --version
mistralcli --print-defaults
mistralcli
```

One-shot smoke test:

```bash
uv run python -m mistralcli --once "Return only the word ok." --no-stream
```

Reasoning flags:

```bash
uv run python -m mistralcli --reasoning
uv run python -m mistralcli --no-reasoning
```

If thinking display is enabled, the local backend may print reasoning before
the final answer. Disable display when you want cleaner smoke output without
changing backend reasoning requests:

```bash
uv run python -m mistralcli --once "Return only ok." --no-stream --no-thinking --system-prompt "Answer exactly as requested."
```

Then inside the REPL:

```text
/help
/defaults
/status
/tools
```

Use `--version` or `-v` whenever you want to confirm which installed build you
are testing before running a smoke check or switching servers.

## Reading The Startup Summary

`--print-defaults`, `/defaults`, and the welcome banner show the active runtime
configuration:

- `Backend`: `local` or `remote`
- `Server`: local server URL or Mistral Cloud
- `Model`: active model id
- `Sampling`: temperature, top-p, prompt mode, max tokens, streaming, reasoning
- `Conversations`: whether Mistral Cloud Conversations mode is active
- `Context`: compact policy, threshold, reserve, and configured windows
- `Tools`: local OS tools and optional MCP tools
- `Logging`: active log file and retention policy

Use this output first when debugging a surprising runtime behavior.

Use `/status` when you want the live session snapshot instead of the static
runtime configuration. `/status` reports the current phase, active
backend/server/model, current Conversations state, context estimate, most
recent backend usage, cumulative usage, and active attachments.

During active turns, the bottom TTY status bar exposes three different token
signals:

- `est:` is the CLI's cached estimate of the current prompt/context size for
  normal chat-completions mode
- `last:` is the backend-reported `usage.total_tokens` for the most recent turn
- `usage:` is the cumulative backend-reported usage for the current session

In local streaming mode, `llama.cpp` may omit `usage` on some turns. When that
happens, `last:` and `usage:` may stay flat while `est:` continues to grow.

TTY rendering also distinguishes source snippets from prose: fenced code blocks
are shown in a dedicated cyan code style instead of the normal assistant text
color.

Standalone Markdown separators such as `---`, `***`, and `___` are rendered as
real terminal divider lines outside fenced code blocks.

In the interactive TTY composer, multiline paste is normalized into one text
buffer. Pasted line breaks and tabs are flattened to spaces, the full text
stays editable in the prompt, and nothing is sent until you press Enter.

## Basic REPL Commands

Common commands:

```text
/help
/defaults
/status
/tools
/reset
/system You are a concise assistant.
/exit
```

Local workspace shortcuts:

```text
/run -- git status
/run --cwd . -- rg "ContextConfig" src tests
/ls .
/find --path src -- compact_context
/edit notes.txt -- This text is written to disk.
```

Backend and runtime commands:

```text
/remote
/remote on
/remote off
/remote model small
/remote model medium
/timeout
/timeout 300000
/reasoning
/reasoning off
/reasoning on
/thinking
/thinking off
/thinking on
```

Attachment commands:

```text
/image
/image --prompt Describe this image and list visible text.
/doc
/doc --prompt Summarize this document and extract action items.
/drop
/dropimage
/dropdoc
```

## Local Mode

Local mode is the default. It expects an OpenAI-compatible `llama.cpp` server at
`http://127.0.0.1:8080`.

Recommended local smoke tests:

```bash
uv run python -m mistralcli --print-defaults
uv run python -m mistralcli --once "Return only ok." --no-stream
```

Useful REPL checks:

```text
/defaults
/run -- pwd
/run -- git status --short
/find --path docs -- mistral
```

The local setup uses `prompt_mode=reasoning` by default. `/reasoning` controls
whether the CLI requests backend reasoning. `/thinking` controls whether any
returned thinking is rendered in the terminal:

```text
/reasoning off
/reasoning on
/thinking off
/thinking on
```

## Remote Chat Completions

Remote chat completions use Mistral Cloud through the official SDK. They are
not the same as Conversations mode.

Prepare your shell:

```bash
export MISTRAL_API_KEY=...
```

Start local first, then switch in the REPL:

```text
/remote on
/defaults
Ask a remote test question.
/remote off
```

Or start a one-shot remote-style test by enabling remote mode inside an
interactive session. Backend switching resets the active conversation.

Remote mode uses:

- model `mistral-small-latest` by default
- `--remote-model medium` or `--remote-model mistral-medium-3.5` to switch to Medium 3.5
- `reasoning_effort=high` when reasoning is on

You can also change the remote model from within the REPL while remote mode is active:

```text
/remote on
/remote model medium
```

Supported model names: `small`, `small4`, `small-4`, `mistral-small-latest`, `medium`, `medium3.5`, `medium-3.5`, `mistral-medium-3.5`.
Changing the model resets the current conversation.
- `reasoning_effort=none` when reasoning is off
- `/thinking` only affects local rendering of returned `thinking` blocks
- no local `prompt_mode=reasoning`, because Mistral Cloud rejects that setting

For the backend-specific runtime profile and the validated local `llama.cpp`
launch command, see [backends-and-runtime.md](backends-and-runtime.md).

## Conversations Mode

Conversations mode is optional and only works against Mistral Cloud. The default
CLI behavior remains normal chat completions.

Start directly in Conversations mode:

```bash
export MISTRAL_API_KEY=...
uv run python -m mistralcli --conversations
```

Enable or disable inside the REPL:

```text
/conv on
/conv off
```

Startup flags:

```text
--conversations
--conversation-store on|off
--conversation-resume last|new|prompt
--conversation-name "Release review"
--conversation-description "Track rollout notes"
--conversation-meta ticket=OPS-42
--conversation-meta owner=dlopez
--conversation-index /custom/path/conversations.json
```

Conversations persistence model:

- with `store=on`, Mistral stores the conversation history server-side and the
  CLI keeps the current `conversation_id`
- with `store=off`, each turn is stateless and no persistent `conversation_id`
  is kept locally
- Mistral may return a new `conversation_id` on append or restart; the CLI
  follows that id automatically and migrates its local bookmark metadata
- local aliases, tags, notes, and bookmarks are **not** stored on Mistral; they
  live in the local registry file
- Mistral model Conversations currently keep `name` and `description`, but may
  not return custom `metadata` on later `get/list` calls

Local registry:

- default path: `~/.local/state/mistralcli/conversations.json`
- if `XDG_STATE_HOME` is set, the registry lives under
  `$XDG_STATE_HOME/mistralcli/conversations.json`
- `--conversation-index` overrides that path for one run

Remote management commands:

```text
/conv
/conv current
/conv on
/conv off
/conv new
/conv id
/conv list --page 0 --size 20 --meta owner=dlopez
/conv show conv_123
/conv use conv_123
/conv history
/conv history conv_123
/conv messages
/conv delete
/conv delete conv_123
/conv restart entry_123
/conv restart entry_123 conv_123
/conv store on
/conv store off
```

Local organization commands:

```text
/conv alias release-review
/conv alias conv_123 release-review
/conv note conv_123 Track rollout blockers
/conv tag add conv_123 ops
/conv tag remove conv_123 ops
/conv bookmarks
/conv forget conv_123
```

Pending remote creation settings:

```text
/conv set name Release review
/conv set description Track rollout notes
/conv set meta ticket=OPS-42
/conv set meta owner=dlopez
/conv unset name
/conv unset description
/conv unset meta ticket
/conv unset all
```

How to think about those settings:

- `name`, `description`, and `metadata` are applied to the **next** remote
  conversation start
- Mistral does not expose a remote update API for an existing conversation
  name/description/metadata, so the CLI cannot retroactively edit those fields
- local aliases and notes are the CLI-side workaround for post-creation
  organization
- when the backend does not return custom metadata later, the CLI preserves the
  metadata requested at creation time in the local registry and reuses it for
  `/conv list --meta ...` filtering

Important behavior:

- enabling Conversations switches to the remote backend and resets the local chat
- disabling Conversations resets the chat and restores the prior backend if known
- `/conv new` drops the active remote id so the next user turn starts a fresh
  conversation
- `/conv use <id>` reattaches the current session to an existing remote
  conversation
- `/conv alias release-review` applies the alias directly to the active
  conversation; use `/conv alias <id> release-review` when you want to label a
  different one
- `/conv history` prints remote entry ids; use those ids with `/conv restart`
  to branch from a specific point
- `/conv restart <entry_id>` creates a new remote conversation and switches the
  session to that branch
- the current API requires a follow-up `inputs` field on restart; the CLI sends
  an empty input automatically so `/conv restart <entry_id>` works without
  extra arguments
- `/reasoning on|off|toggle` still applies in Conversations mode and controls
  whether the CLI requests reasoning from the backend
- `/thinking on|off|toggle` only controls whether returned `thinking` blocks
  are rendered locally
- local `/compact` does not compact Conversations mode; Mistral handles that
  context server-side
- if Mistral Conversations does not return any `thinking` blocks for a turn, the
  CLI prints a one-line best-effort notice instead of silently pretending
  reasoning was unavailable locally

Auto-resume:

- `resume=last` is the default
- when starting directly in Conversations mode, the CLI reattaches to the last
  stored remote conversation recorded in the local registry
- `resume=new` always starts from a fresh conversation
- `resume=prompt` asks before resuming when the session is interactive
- if the last saved id no longer exists remotely, the CLI clears the stale
  pointer and starts fresh

## Context Management And Compacting

Normal chat-completions mode sends local message history to the backend. The
CLI now has a client-side context policy to avoid sending prompts beyond the
configured window.

Defaults:

- local window: `262144` tokens
- remote fallback window: `256000` tokens
- auto compact: on
- threshold: `90%`
- response reserve: `8192` tokens
- retained recent turns: 6 user turns
- summary budget: `2048` tokens

Inspect the current estimate:

```text
/compact status
```

Manual compaction:

```text
/compact
/compact now
```

Tune the policy inside the REPL:

```text
/compact threshold 85
/compact reserve 4096
/compact keep 4
/compact auto off
/compact auto on
```

Start with custom settings:

```bash
uv run python -m mistralcli --compact-threshold 85
uv run python -m mistralcli --no-auto-compact
uv run python -m mistralcli --context-reserve-tokens 4096
uv run python -m mistralcli --context-keep-turns 4
```

How compaction works:

- old messages are summarized into one assistant message
- recent turns remain verbatim
- the system prompt is preserved
- tool schemas are included in context estimates
- if the request still exceeds the hard window after compaction, the CLI blocks
  the turn before sending it

Because the SDK does not expose a token counter for this path, estimates use a
conservative local heuristic. Treat them as safety estimates, not exact backend
token accounting.

## Images And Documents

Image flow:

```text
/image
/image --prompt Describe this image and list all visible text.
```

Document flow:

```text
/doc
/doc --prompt Summarize this document and extract action items.
```

Behavior by backend:

- local `/image` sends image blocks to `llama.cpp`
- local `/doc` rasterizes documents into page images for OCR and vision
- remote `/image` uses the official SDK image flow
- remote `/doc` uses the official SDK document flow for `pdf` and `docx`
- remote plain-text documents are embedded directly as text

Attachment picker flow:

- first browse directories in the terminal picker
- use `[use]` to keep the current directory or `[..]` to move to the parent
- then use the fuzzy list to choose one matching file
- `Enter` selects the highlighted entry and `Ctrl-C` cancels
- if the picker cannot run, the CLI falls back to manual path entry

Attachment lifecycle:

- selected images or documents stay active until dropped or replaced
- `/drop` clears all active and staged attachments
- `/dropimage` clears image attachments
- `/dropdoc` clears document attachments

## Tools

The CLI always exposes these local OS tools to the model:

- `shell`
- `read_file`
- `write_file`
- `list_dir`
- `search_text`

FireCrawl MCP tools are added when `mcp.json` exists and `FIRECRAWL_API_KEY` is
available.

Check the live tool catalog:

```text
/tools
```

Tool-oriented examples:

```text
Use shell to inspect the current git status and summarize it.
Find files mentioning timeout in src/.
Read README.md and tell me the main usage flow.
Search docs for compact and summarize the behavior.
```

For direct commands, prefer slash shortcuts:

```text
/run -- git status --short
/ls src
/find --path tests -- conversations
```

## Interrupts And Resets

Keyboard behavior:

- `Ctrl-C` cancels the current response without exiting the CLI
- `Ctrl-D` exits the REPL
- `/reset` clears conversation history but keeps the system prompt
- `/system TEXT` replaces the system prompt and resets history

After a cancelled response, the next turn should continue cleanly. If the model
or backend behaves unexpectedly, use:

```text
/reset
/defaults
```

## Logging

The CLI enables debug logging by default with date-based rotation and short
retention. The default location is shown by `/defaults`.

Useful flags:

```bash
uv run python -m mistralcli --log-dir /tmp/mistralcli-logs
uv run python -m mistralcli --no-debug
uv run python -m mistralcli --log-retention-days 7
```

Use logs when checking backend errors, tool execution, cancellation behavior, or
context compaction decisions.

## Troubleshooting

Local server is not reachable:

```bash
uv run python -m mistralcli --once "Return only ok." --no-stream
```

Then verify the server separately:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

Remote mode does not start:

```bash
echo "$MISTRAL_API_KEY"
```

Then retry:

```text
/remote on
```

Conversations mode does not start:

```text
/conv on
/conv
/conv current
```

If `store=off` is active, remember that it is stateless and will not preserve a
conversation id.

If the wrong conversation is resumed automatically, either start a fresh one or
switch explicitly:

```text
/conv new
/conv use conv_123
/conv bookmarks
```

Context is too large:

```text
/compact status
/compact
/compact threshold 85
```

If auto compact is disabled:

```text
/compact auto on
```

Attachment picker cannot run:

- use a real TTY terminal
- ensure the file exists and is readable
- when the picker fallback appears, paste the absolute path manually

Thinking output is noisy:

```text
/thinking off
```

Terminal colors look wrong:

```bash
export TERM=xterm-256color
```

## Recommended Test Script

Use this sequence after installing or updating the CLI:

```bash
uv run python -m mistralcli --print-defaults
uv run python -m mistralcli --once "Return only ok." --no-stream
uv run python -m mistralcli --no-auto-compact --print-defaults
uv run python -m mistralcli --compact-threshold 85 --print-defaults
```

Then open the REPL and run:

```text
/help
/defaults
/tools
/compact status
/compact threshold 85
/compact auto off
/compact auto on
/run -- git status --short
/reset
/exit
```

If `MISTRAL_API_KEY` is available, also test:

```text
/remote on
/defaults
/remote off
/conv on
/conv current
/conv list --page 0 --size 5
/conv bookmarks
/conv new
/conv off
/remote off
```
