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

## Validation targets

- Chat completions
- Streaming completions
- Tool calling
- Multimodal request acceptance
- The current local build may return generic or empty text for image prompts, so
  the multimodal test only asserts that image requests are accepted without
  crashing.
- Server health and model listing
