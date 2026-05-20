# MistralCLI Configuration System

This document describes the unified configuration system for MistralCLI introduced in version 3.4.0.

## Overview

MistralCLI now supports file-based configuration in addition to command-line arguments and environment variables. This provides a more convenient way to manage settings across multiple sessions and projects.

## Configuration Precedence

When determining the value of a configuration option, MistralCLI uses the following precedence order (highest to lowest):

1. **Command-line arguments** - Explicit arguments passed to the CLI (e.g., `--server-url`, `--model`)
2. **Environment variables** - Variables set in the shell environment (e.g., `MISTRAL_LOCAL_MODEL_ID`)
3. **Configuration file** - Settings defined in a configuration file (YAML, JSON, or TOML)
4. **Default values** - Hardcoded default values in the application

This means that command-line arguments override environment variables, which in turn override file-based configuration, which finally overrides defaults.

## Configuration File Locations

MistralCLI searches for configuration files in the following order:

1. Explicit path provided via `--config-path <path>` command-line argument
2. Path specified in the `$MISTRAL_CONFIG_PATH` environment variable
3. `~/.config/mistralcli/config.yaml`
4. `~/.mistralcli.yaml`
5. `~/.mistralcli.json`
6. `~/.mistralcli.toml`
7. `./mistralcli.yaml` (current directory)
8. `./mistralcli.json` (current directory)
9. `./mistralcli.toml` (current directory)

The first file found in this search order is used. If no configuration file is found, the CLI uses default values combined with environment variables and command-line arguments.

## Supported Formats

Configuration files can be written in three formats:

- **YAML** (recommended) - Most readable and commonly used
- **JSON** - Standard format, good for programmatic generation
- **TOML** - Alternative format, requires Python 3.11+ or `tomli` package

### YAML Example

```yaml
# ~/.config/mistralcli/config.yaml
version: "3.4"

# Backend configuration
backend: local

# Local backend (llama.cpp)
local:
  api_key: "local-test"
  model_id: "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
  server_url: "http://127.0.0.1:8080"
  timeout_ms: 300000

# Generation parameters
generation:
  temperature: 0.3
  top_p: 0.95
  prompt_mode: "reasoning"
  max_tokens: null

# Remote backend (Mistral Cloud)
remote:
  model_id: "mistral-small-latest"
  timeout_ms: 300000
  # Note: api_key is NOT stored in config file for security
  # Use MISTRAL_API_KEY environment variable instead

# Conversations
conversations:
  enabled: false
  store: true
  resume_policy: "last"  # Options: "last", "new", "prompt"

# Context management
context:
  auto_compact: true
  threshold: 0.9  # 0.1-0.99 or 10-99 as percentage
  reserve_tokens: 8192
  local_window_tokens: 262144
  remote_window_tokens: 256000
  keep_recent_turns: 6
  summary_max_tokens: 2048

# Logging
logging:
  directory: "~/.local/state/mistralcli/logs"
  debug_enabled: true
  retention_days: 2
  file_name: "mistralcli.log"

# MCP (Model Context Protocol)
mcp:
  enabled: true
  config_path: "~/mcp.json"

# UI/REPL
ui:
  stream_enabled: true
  show_reasoning: true
  show_thinking: true
  system_prompt: "You are a helpful assistant. Follow the user's instructions carefully."
  max_tool_rounds: 20

# Conversation registry
registry:
  conversation_index_path: "~/.local/state/mistralcli/conversations.json"
```

### JSON Example

```json
{
  "version": "3.4",
  "backend": "local",
  "local": {
    "api_key": "local-test",
    "model_id": "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL",
    "server_url": "http://127.0.0.1:8080",
    "timeout_ms": 300000
  },
  "generation": {
    "temperature": 0.3,
    "top_p": 0.95,
    "prompt_mode": "reasoning",
    "max_tokens": null
  },
  "remote": {
    "model_id": "mistral-small-latest",
    "timeout_ms": 300000
  },
  "conversations": {
    "enabled": false,
    "store": true,
    "resume_policy": "last"
  },
  "context": {
    "auto_compact": true,
    "threshold": 0.9,
    "reserve_tokens": 8192,
    "local_window_tokens": 262144,
    "remote_window_tokens": 256000,
    "keep_recent_turns": 6,
    "summary_max_tokens": 2048
  },
  "logging": {
    "directory": "~/.local/state/mistralcli/logs",
    "debug_enabled": true,
    "retention_days": 2,
    "file_name": "mistralcli.log"
  },
  "mcp": {
    "enabled": true,
    "config_path": "~/mcp.json"
  },
  "ui": {
    "stream_enabled": true,
    "show_reasoning": true,
    "show_thinking": true,
    "system_prompt": "You are a helpful assistant.",
    "max_tool_rounds": 20
  },
  "registry": {
    "conversation_index_path": "~/.local/state/mistralcli/conversations.json"
  }
}
```

### TOML Example

```toml
# ~/.config/mistralcli/config.toml
version = "3.4"
backend = "local"

[local]
api_key = "local-test"
model_id = "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
server_url = "http://127.0.0.1:8080"
timeout_ms = 300000

[generation]
temperature = 0.3
top_p = 0.95
prompt_mode = "reasoning"
max_tokens = null

[remote]
model_id = "mistral-small-latest"
timeout_ms = 300000

[conversations]
enabled = false
store = true
resume_policy = "last"

[context]
auto_compact = true
threshold = 0.9
reserve_tokens = 8192
local_window_tokens = 262144
remote_window_tokens = 256000
keep_recent_turns = 6
summary_max_tokens = 2048

[logging]
directory = "~/.local/state/mistralcli/logs"
debug_enabled = true
retention_days = 2
file_name = "mistralcli.log"

[mcp]
enabled = true
config_path = "~/mcp.json"

[ui]
stream_enabled = true
show_reasoning = true
show_thinking = true
system_prompt = "You are a helpful assistant."
max_tool_rounds = 20

[registry]
conversation_index_path = "~/.local/state/mistralcli/conversations.json"
```

## Command-Line Arguments

### Configuration Management

#### `--config-path <path>`

Specify an explicit path to a configuration file. This takes precedence over all other configuration file locations.

```bash
# Use a specific configuration file
mistralcli --config-path /path/to/my/config.yaml

# Use a project-specific configuration
mistralcli --config-path ./mistralcli-prod.yaml
```

#### `--generate-config [PATH]`

Generate a sample configuration file with all default values and exit. If `PATH` is `-`, the configuration is printed to stdout. If `PATH` is not provided, the configuration is saved to `~/.config/mistralcli/config.yaml`.

```bash
# Print configuration to stdout
mistralcli --generate-config -

# Save to default location
mistralcli --generate-config

# Save to specific path
mistralcli --generate-config /path/to/config.yaml
```

## Environment Variables

MistralCLI recognizes the following environment variables:

### Configuration File Path

- `MISTRAL_CONFIG_PATH` - Path to the configuration file (same as `--config-path`)

### Local Backend

- `MISTRAL_LOCAL_API_KEY` - API key for local server
- `MISTRAL_LOCAL_MODEL_ID` - Model identifier for local server
- `MISTRAL_LOCAL_SERVER_URL` - Server URL for local endpoint
- `MISTRAL_LOCAL_TIMEOUT_MS` - Request timeout in milliseconds
- `MISTRAL_LOCAL_TEMPERATURE` - Sampling temperature
- `MISTRAL_LOCAL_TOP_P` - Nucleus sampling top-p
- `MISTRAL_LOCAL_PROMPT_MODE` - Prompt mode for local template
- `MISTRAL_LOCAL_MAX_TOKENS` - Maximum generated tokens

### Remote Backend

- `MISTRAL_API_KEY` - API key for Mistral Cloud (required for remote mode)
- `MISTRAL_REMOTE_MODEL_ID` - Default remote model identifier

### Conversations

- `MISTRAL_CONVERSATIONS` - Enable/disable Conversations mode
- `MISTRAL_CONVERSATION_STORE` - Enable/disable conversation storage
- `MISTRAL_CONVERSATION_RESUME` - Resume policy (`last`, `new`, or `prompt`)

### Context Management

- `MISTRAL_CONTEXT_AUTO_COMPACT` - Enable/disable automatic context compaction
- `MISTRAL_CONTEXT_THRESHOLD` - Context compaction threshold
- `MISTRAL_CONTEXT_RESERVE_TOKENS` - Tokens reserved for model response
- `MISTRAL_CONTEXT_LOCAL_WINDOW_TOKENS` - Local context window tokens
- `MISTRAL_CONTEXT_REMOTE_WINDOW_TOKENS` - Remote context window tokens
- `MISTRAL_CONTEXT_KEEP_RECENT_TURNS` - Recent user turns to preserve
- `MISTRAL_CONTEXT_SUMMARY_MAX_TOKENS` - Maximum tokens for compact summaries

### Logging

- `MISTRAL_LOCAL_LOG_DIR` - Directory for log files
- `MISTRAL_LOCAL_LOG_RETENTION_DAYS` - Days to keep rotated logs
- `MISTRAL_LOCAL_DEBUG` - Enable/disable debug logging

### MCP

- `MISTRAL_LOCAL_MCP_CONFIG` - Path to MCP configuration file

## Configuration Reference

### `version` (string)

Configuration file version. Currently `"3.4"`. Used for future compatibility.

### `backend` (string)

Default backend to use. Options: `"local"` or `"remote"`.

- **Type**: string
- **Default**: `"local"`
- **CLI override**: N/A (use `/remote on|off` in REPL or switch via environment)

### `local` (object)

Configuration for the local llama.cpp backend.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `api_key` | string | `"local-test"` | API key for local server authentication |
| `model_id` | string | `"unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"` | Model identifier for local server |
| `server_url` | string | `"http://127.0.0.1:8080"` | URL of the local llama.cpp server |
| `timeout_ms` | integer | 300000 | Request timeout in milliseconds |

**CLI overrides**: `--api-key`, `--model`, `--server-url`, `--timeout-ms`

### `generation` (object)

Sampling and generation parameters for the local backend.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `temperature` | float | 0.3 | Sampling temperature (0.0-1.0) |
| `top_p` | float | 0.95 | Nucleus sampling top-p (0.0-1.0) |
| `prompt_mode` | string/null | `"reasoning"` | Prompt mode for local template |
| `max_tokens` | integer/null | null | Maximum generated tokens. When null: 16384 for local, server default for remote |

**CLI overrides**: `--temperature`, `--top-p`, `--max-tokens`, `--prompt-mode`

### `remote` (object)

Configuration for the Mistral Cloud backend.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model_id` | string | `"mistral-small-latest"` | Model identifier for remote endpoint |
| `timeout_ms` | integer | 300000 | Request timeout in milliseconds |

**Note**: The API key is NOT stored in the configuration file for security reasons. Use the `MISTRAL_API_KEY` environment variable.

**CLI overrides**: `--remote-model`

### `conversations` (object)

Configuration for Mistral Cloud Conversations mode.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `enabled` | boolean | false | Enable Conversations mode |
| `store` | boolean | true | Store conversations server-side |
| `resume_policy` | string | `"last"` | Policy for resuming conversations (`"last"`, `"new"`, `"prompt"`) |

**CLI overrides**: `--conversations`/`--no-conversations`, `--conversation-store`, `--conversation-resume`

### `context` (object)

Context overflow and compaction policy.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `auto_compact` | boolean | true | Enable automatic context compaction |
| `threshold` | float | 0.9 | Threshold for compaction (0.1-0.99 or 10-99%) |
| `reserve_tokens` | integer | 8192 | Tokens reserved for model response |
| `local_window_tokens` | integer | 262144 | Local context window size in tokens |
| `remote_window_tokens` | integer | 256000 | Remote context window size in tokens |
| `keep_recent_turns` | integer | 6 | Recent user turns to preserve when compacting |
| `summary_max_tokens` | integer | 2048 | Maximum tokens for compact summaries |

**CLI overrides**: `--auto-compact`/`--no-auto-compact`, `--compact-threshold`, `--context-reserve-tokens`, `--context-local-window-tokens`, `--context-remote-window-tokens`, `--context-keep-turns`, `--context-summary-max-tokens`

### `logging` (object)

Logging configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `directory` | string | `"~/.local/state/mistralcli/logs"` | Directory for rotated log files |
| `debug_enabled` | boolean | true | Enable debug logging |
| `retention_days` | integer | 2 | Days to keep rotated log files |
| `file_name` | string | `"mistralcli.log"` | Log file name |

**CLI overrides**: `--log-dir`, `--log-retention-days`, `--debug`/`--no-debug`

### `mcp` (object)

Model Context Protocol (MCP) configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `enabled` | boolean | true | Enable MCP backend |
| `config_path` | string/null | null | Path to MCP configuration file |

**CLI overrides**: `--mcp-config`, `--no-mcp`

### `ui` (object)

User interface and REPL configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `stream_enabled` | boolean | true | Enable token streaming |
| `show_reasoning` | boolean | true | Show backend reasoning |
| `show_thinking` | boolean | true | Show returned thinking blocks |
| `system_prompt` | string | Default system prompt | System prompt for the assistant |
| `max_tool_rounds` | integer | 20 | Maximum tool invocation rounds |

**CLI overrides**: `--no-stream`, `--reasoning`/`--no-reasoning`, `--thinking`/`--no-thinking`, `--system-prompt`

### `registry` (object)

Conversation registry configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `conversation_index_path` | string/null | null | Path to conversation registry JSON file |

**CLI overrides**: `--conversation-index`

## Migration Guide

### From Environment Variables

If you're currently using environment variables, you can generate a configuration file and then modify it:

```bash
# Generate configuration with current environment variables
MISTRAL_LOCAL_MODEL_ID=my-model mistralcli --generate-config - > config.yaml

# Or save directly
MISTRAL_LOCAL_MODEL_ID=my-model mistralcli --generate-config ~/my-config.yaml
```

### From Command-Line Arguments

If you're using many command-line arguments, you can create a configuration file and use `--config-path`:

```bash
# Create a configuration file with your preferred settings
cat > ~/.config/mistralcli/config.yaml << EOF
version: "3.4"
local:
  model_id: "my-custom-model"
  server_url: "http://192.168.1.100:8080"
generation:
  temperature: 0.7
  top_p: 0.9
ui:
  stream_enabled: false
EOF

# Use the CLI with your configuration
mistralcli
```

## Best Practices

1. **Security**: Never commit configuration files containing API keys to version control. The `remote.api_key` field is intentionally NOT supported in configuration files. Always use environment variables for secrets.

2. **Environment-Specific Configuration**: Use separate configuration files for different environments:
   - `~/.config/mistralcli/config-dev.yaml` for development
   - `~/.config/mistralcli/config-prod.yaml` for production
   - Use `--config-path` to select the appropriate file

3. **Project-Specific Configuration**: For project-specific settings, create a `mistralcli.yaml` file in your project directory. This allows different projects to have different configurations.

4. **Version Control**: It's safe to commit configuration files to version control if they don't contain secrets. This ensures all team members use consistent settings.

5. **Backup**: Configuration files are valuable. Consider backing up your configuration directory (`~/.config/mistralcli/`).

## Troubleshooting

### Configuration Not Loading

If your configuration file isn't being loaded:

1. Verify the file exists at one of the search locations
2. Check file permissions
3. Ensure the file has valid YAML/JSON/TOML syntax
4. Use `--config-path` to specify the exact path
5. Set `MISTRAL_CONFIG_PATH` environment variable
6. Enable debug logging with `--debug` to see configuration loading messages

### Invalid Configuration Values

If you get validation errors:

1. Check that all numeric values are valid (positive timeouts, percentages between 0-1, etc.)
2. Ensure enum values match expected options (e.g., `resume_policy` must be `last`, `new`, or `prompt`)
3. Review the schema reference above for valid ranges and options

### Missing Dependencies

If you get errors about missing dependencies when loading YAML or TOML files:

- For YAML: Install PyYAML with `pip install pyyaml` or `uv pip install pyyaml`
- For TOML with Python < 3.11: Install tomli with `pip install tomli`

JSON files require no additional dependencies as they use Python's built-in `json` module.

## Examples

### Minimal Configuration

```yaml
# ~/.config/mistralcli/config.yaml
version: "3.4"
local:
  model_id: "my-custom-model"
  server_url: "http://localhost:8080"
```

### Production Configuration

```yaml
# ~/.config/mistralcli/config-prod.yaml
version: "3.4"
backend: remote
remote:
  model_id: "mistral-medium-3.5"
generation:
  temperature: 0.1
  top_p: 0.9
context:
  auto_compact: true
  threshold: 0.85
ui:
  stream_enabled: false
  show_reasoning: true
```

### Development Configuration

```yaml
# ~/.config/mistralcli/config-dev.yaml
version: "3.4"
backend: local
local:
  model_id: "unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL"
  server_url: "http://localhost:8080"
generation:
  temperature: 0.7
  top_p: 0.9
  prompt_mode: "reasoning"
ui:
  debug_enabled: true
  show_thinking: true
```
