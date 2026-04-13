# Local Mistral Small 4 Harness

This repository validates the official `mistralai` Python SDK against a local `llama.cpp` deployment of Mistral Small 4.

## Runtime under test

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

The chat template in use sets `reasoning_effort=high` by default.

## Known local constraint

Images smaller than `2x2` pixels crash in `llama.cpp` preprocessing:

```text
GGML_ASSERT(src.nx >= 2 && src.ny >= 2) failed
```

The integration test uses a `2x2` PNG fixture to avoid that crash.

## CLI

The repository now ships a dedicated `mistral4cli` REPL for this local model.
It uses the official `mistralai` SDK directly and inherits the local runtime
defaults from the same configuration layer used by the tests.

The REPL now has a retro green/orange presentation, an ASCII welcome banner and
an actionable help system:

- `/help` shows commands, examples and MCP status
- `/defaults` prints the active runtime defaults
- `/tools` shows the loaded FireCrawl MCP catalogue
- `/run -- ...` runs a shell command through the local shell tool
- `/ls [PATH]` lists files and directories
- `/find -- ...` searches text in the project tree
- `/edit PATH -- ...` writes text to a file
- `/reset` clears the conversation but keeps the system prompt
- `/system <texto>` replaces the system prompt and resets the chat
- `/exit` or `/quit` leaves the REPL

FireCrawl MCP is configured in [`mcp.json`](../mcp.json) and loaded
automatically when present. The current setup uses the official MCP Python SDK
against the SSE endpoint provided by FireCrawl, while local inference continues
to go through the official `mistralai` client pointed at `llama.cpp`.
Use `--mcp-config <path>` to point at a different config file or `--no-mcp` to
disable tool loading for a run.

The CLI also exposes always-on local OS tools, so the model can inspect and edit
the workspace without asking for extra setup:

- `shell`
- `read_file`
- `write_file`
- `list_dir`
- `search_text`

When FireCrawl is available, those remote tools are added on top of the local
OS tool set.

For long outputs the local shell and search tools are paginated, so you can use
`offset`/`lines` style arguments to continue from a later page instead of
dumping everything at once.

Runtime defaults:

- temperature: `0.7`
- top-p: `0.95`
- prompt mode: `reasoning`
- max tokens: unset, so the server can decide
- streaming: on
- MCP: FireCrawl auto-tools on when `mcp.json` is present

Session commands:

- `/help`
- `/defaults`
- `/reset`
- `/system <texto>`
- `/exit` or `/quit`

Operational notes:

- `Ctrl-C` cancels the current generation without dropping the whole session.
- `Ctrl-D` exits the CLI.
- `uv run python -m mistral4cli --once "..." --no-stream` runs a one-shot smoke
  prompt against the local server.
- The CLI prefers `MISTRAL_LOCAL_MCP_CONFIG` when set, otherwise it falls back
  to `./mcp.json` if the file exists.

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
