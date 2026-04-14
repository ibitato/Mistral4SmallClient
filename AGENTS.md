# AGENTS.md

## Goal
This repository uses Python 3.10, `uv` for environment and dependency management, and `Makefile` as the single interface for development tasks.

## Operating rules
- Use `make sync` to create or update `.venv`.
- Use `make lock` when dependencies change.
- Use `make format`, `make lint`, and `make typecheck` before delivering changes.
- Use `make check` to validate without modifying files.
- Use `make test` to run the `pytest` suite.
- Use `make docs` to regenerate the checked-in API reference from public docstrings.
- Use `make docs-check` to verify that the generated documentation and language hygiene are up to date.
- Validate against the local server at `http://127.0.0.1:8080` with the model `unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL`.
- For CLI smoke tests, use `uv run python -m mistral4cli --print-defaults` and `uv run python -m mistral4cli --once "..." --no-stream`.
- When working with images, always use an image of at least `2x2` pixels.
- When working with MCP, use `mcp.json` at the repository root or `MISTRAL_LOCAL_MCP_CONFIG` to point to another configuration. Keep secrets out of the file and resolve them from environment variables such as `FIRECRAWL_API_KEY`.
- The CLI must always expose local OS tools: `shell`, `read_file`, `write_file`, `list_dir`, and `search_text`.
- The REPL must keep help clear and actionable: `/help`, `/defaults`, `/tools`, `/run`, `/ls`, `/find`, `/edit`, `/image`, `/doc`, `/reset`, `/system`, `/exit`.
- `/image` and `/doc` must use a terminal-native file picker in TTY environments and fall back to a manual terminal path prompt when the picker cannot run.
- `/doc` must use the backend-appropriate document flow: rasterized page images for local `llama.cpp`, and the official remote document flow for Mistral cloud.
- For long outputs, `shell` and `search_text` must support pagination or truncation with clear continuation indicators.
- The CLI UI must preserve the retro green/orange style and an ASCII banner that remains readable in TTY terminals.
- The interactive REPL must clear the screen on startup and conversation resets, and recommend `TERM=xterm-256color` when the terminal is not already configured for the intended palette.
- All repository text, comments, prompts, and documentation should remain in English.
- Do not use `pip`, `poetry`, `pipenv`, or global installations for the normal project workflow.
- Always execute code within the `uv` environment with `uv run ...` or via `make ...`.
- Keep the code compatible with Python 3.10.

## Development style
- Prefer the `src/` layout for application code.
- Keep functions small and with a single responsibility.
- Type public functions and module boundaries.
- Avoid complex logic in `__init__.py`; use explicit modules for the CLI entrypoint.
- Use `pathlib` for paths, `dataclasses` for simple data structures, and `logging` for operational output.

## Lint and formatting
- The canonical formatter is `ruff format`.
- Import hygiene and lint are validated with `ruff check`.
- If a change requires a lint exception, document why and keep the scope narrow.
- Do not disable rules globally unless there is a clear and stable reason.

## Typing
- `mypy` must keep passing on `src/`.
- Add annotations to new or modified functions.
- Prefer concrete types over `Any`.
- If you need a `# type: ignore`, justify it and revisit it later.

## Dependencies
- Add dependencies only in `pyproject.toml`.
- Regenerate `uv.lock` after changing dependencies.
- Keep the dependency tree small and justified.

## Recommended workflow
1. `make sync`
2. Implement the change
3. `make format`
4. `make lint`
5. `make typecheck`
6. `make test`
7. If everything passes, deliver the change

## Quality bar
- Accompany behavior changes with tests when applicable.
- Avoid breaking Python 3.10 compatibility.
- Keep messages and APIs simple if the project is still small.
