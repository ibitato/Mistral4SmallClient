# Mistral4Cli User Guide

This guide explains how to run and test the `mistral4cli` command-line
assistant as an end user. It focuses on day-to-day CLI usage rather than
developer internals.

## What The CLI Does

`mistral4cli` is an interactive Linux terminal client for Mistral Small 4. It
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
- Python `3.10`
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

## First Run

From a source checkout:

```bash
make sync
uv run python -m mistral4cli --print-defaults
uv run python -m mistral4cli
```

Installed as a tool:

```bash
mistral4cli --print-defaults
mistral4cli
```

One-shot smoke test:

```bash
uv run python -m mistral4cli --once "Return only the word ok." --no-stream
```

Reasoning flags:

```bash
uv run python -m mistral4cli --reasoning
uv run python -m mistral4cli --no-reasoning
```

If visible reasoning is enabled, the local backend may print reasoning before
the final answer. Disable reasoning when you want cleaner smoke output:

```bash
uv run python -m mistral4cli --once "Return only ok." --no-stream --system-prompt "Answer exactly as requested."
```

Then inside the REPL:

```text
/help
/defaults
/tools
```

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

## Basic REPL Commands

Common commands:

```text
/help
/defaults
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
/timeout
/timeout 300000
/reasoning
/reasoning off
/reasoning on
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
uv run python -m mistral4cli --print-defaults
uv run python -m mistral4cli --once "Return only ok." --no-stream
```

Useful REPL checks:

```text
/defaults
/run -- pwd
/run -- git status --short
/find --path docs -- mistral
```

The local setup uses `prompt_mode=reasoning` by default. Visible reasoning is
rendered when the backend emits it. Toggle display with:

```text
/reasoning off
/reasoning on
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

- model `mistral-small-latest`
- `reasoning_effort=high` when visible reasoning is on
- `reasoning_effort=none` when visible reasoning is off
- no local `prompt_mode=reasoning`, because Mistral Cloud rejects that setting

## Conversations Mode

Conversations mode is optional and only works against Mistral Cloud. The default
CLI behavior remains normal chat completions.

Start directly in Conversations mode:

```bash
export MISTRAL_API_KEY=...
uv run python -m mistral4cli --conversations
```

Enable or disable inside the REPL:

```text
/conv on
/conv off
```

Useful Conversations commands:

```text
/conv
/conv new
/conv id
/conv history
/conv messages
/conv store on
/conv store off
/conv delete
```

Important behavior:

- enabling Conversations resets the local chat
- disabling Conversations resets the chat and restores the prior backend
- `/conv new` starts a fresh Conversation on the next user turn
- `store=on` keeps a server-side `conversation_id`
- `store=off` is stateless and does not preserve a `conversation_id`
- `/reasoning on|off|toggle` still applies in Conversations mode and controls
  whether the CLI requests visible reasoning
- local `/compact` does not compact Conversations mode; Mistral handles that
  context server-side
- if Mistral Conversations does not return any `thinking` blocks for a turn, the
  CLI prints a one-line best-effort notice instead of silently pretending
  reasoning was unavailable locally

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
uv run python -m mistral4cli --compact-threshold 85
uv run python -m mistral4cli --no-auto-compact
uv run python -m mistral4cli --context-reserve-tokens 4096
uv run python -m mistral4cli --context-keep-turns 4
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
uv run python -m mistral4cli --log-dir /tmp/mistral4cli-logs
uv run python -m mistral4cli --no-debug
uv run python -m mistral4cli --log-retention-days 7
```

Use logs when checking backend errors, tool execution, cancellation behavior, or
context compaction decisions.

## Troubleshooting

Local server is not reachable:

```bash
uv run python -m mistral4cli --once "Return only ok." --no-stream
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
```

If `store=off` is active, remember that it is stateless and will not preserve a
conversation id.

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

Reasoning output is noisy:

```text
/reasoning off
```

Terminal colors look wrong:

```bash
export TERM=xterm-256color
```

## Recommended Test Script

Use this sequence after installing or updating the CLI:

```bash
uv run python -m mistral4cli --print-defaults
uv run python -m mistral4cli --once "Return only ok." --no-stream
uv run python -m mistral4cli --no-auto-compact --print-defaults
uv run python -m mistral4cli --compact-threshold 85 --print-defaults
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
/conv on
/conv
/conv new
/conv off
/remote off
```
