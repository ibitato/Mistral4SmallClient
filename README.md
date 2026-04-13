# Mistral4Cli

[![CI](https://github.com/ibitato/Mistral4SmallClient/actions/workflows/ci.yml/badge.svg)](https://github.com/ibitato/Mistral4SmallClient/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python: 3.10](https://img.shields.io/badge/python-3.10-orange.svg)](https://www.python.org/downloads/release/python-3100/)

Retro terminal CLI for testing and using **Mistral Small 4** locally against
`llama.cpp`, with the official `mistralai` Python SDK.

The repository is intentionally focused on one workflow:

- run Mistral Small 4 locally
- exercise the official SDK against that deployment
- validate image, document, tool and MCP flows
- keep the CLI useful for day-to-day experimentation

## What this project includes

- a dedicated interactive CLI with retro green/orange presentation
- always-on local tools: `shell`, `read_file`, `write_file`, `list_dir`, `search_text`
- optional FireCrawl MCP tools loaded from `mcp.json`
- `/image` and `/doc` attachment commands
- tests for completion, streaming, cancellation recovery and multimodal payloads

## Quick start

```bash
make sync
uv run python -m mistral4cli
```

Useful one-shot smoke test:

```bash
uv run python -m mistral4cli --once "Devuelve solo la palabra ok." --no-stream
```

Inside the REPL:

- `/help` for actionable usage
- `/defaults` to inspect runtime parameters
- `/tools` to inspect loaded tools
- `/run -- ...` to execute a shell command
- `/ls [PATH]` to inspect the tree
- `/find -- ...` to search text in the workspace
- `/edit PATH -- ...` to write text files
- `/image` to pick and analyze images
- `/doc` to pick and analyze documents
- `/reset`, `/system ...`, `/exit`

## Local Mistral Small 4 setup

The local model is expected to be running outside this repo with `llama.cpp`.
The documented launch profile is:

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
- streaming on by default
- `max_tokens` unset unless you override it

For the detailed local runbook, see
[docs/local-mistral-small-4.md](docs/local-mistral-small-4.md).

## Testing

```bash
make check
make test
```

`make check` runs formatting, lint and type checks.
`make test` runs the full `pytest` suite, including local integration tests
that require the `llama.cpp` server.

## Repository layout

- `src/mistral4cli/` - CLI, session, tools and attachment handling
- `tests/` - unit and integration tests
- `docs/local-mistral-small-4.md` - detailed local deployment notes
- `mcp.json` - optional FireCrawl MCP config

## License

MIT. See [LICENSE](LICENSE).
