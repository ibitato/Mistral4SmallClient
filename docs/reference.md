# Mistral4Cli API Reference

This file is generated from public docstrings.

## `mistral4cli.attachments`

Attachment helpers for image and document turns.

### Class `LoadedDocument`

One document loaded as text for rendering.

### Class `PathPicker`

Callable path picker used by the REPL.

### Class `RenderedDocument`

One document converted into model-ready image blocks.

#### `build_document_message(paths: 'Sequence[Path]', prompt: 'str | None' = None) -> 'list[dict[str, Any]]'`

Build a multimodal user message by rasterizing documents to images.

#### `build_image_message(paths: 'Sequence[Path]', prompt: 'str | None' = None) -> 'list[dict[str, Any]]'`

Build a multimodal user message for one or more images.

#### `build_remote_document_message(paths: 'Sequence[Path]', prompt: 'str | None' = None) -> 'list[dict[str, Any]]'`

Build a cloud-native document turn for the official Mistral SDK.

#### `build_remote_image_message(paths: 'Sequence[Path]', prompt: 'str | None' = None) -> 'list[dict[str, Any]]'`

Build a cloud-native multimodal message for one or more images.

#### `choose_paths(kind: 'str', input_func: 'Callable[[str], str]', stdin: 'TextIO', stdout: 'TextIO', path_picker: 'PathPicker | None', filetypes: 'Sequence[tuple[str, str]]', suffixes: 'set[str]', multiple: 'bool' = True) -> 'list[Path]'`

Choose one or more files, preferring a terminal picker in TTY sessions.

#### `format_selection_summary(paths: 'Sequence[Path]') -> 'str'`

Return a compact human-readable summary for selected files.

#### `load_document(path: 'Path') -> 'LoadedDocument'`

Load a supported document from disk as text.

#### `render_document(path: 'Path') -> 'RenderedDocument'`

Render a supported document into image blocks for the model.

## `mistral4cli.cli`

Linux-only command-line entrypoint for the general Mistral Small 4 + Medium 3.5 CLI.

#### `main(argv: 'Sequence[str] | None' = None, input_func: 'Callable[[str], str]' = input, stdin: 'TextIO' = <TextIOWrapper>, stdout: 'TextIO' = <TextIOWrapper>, stderr: 'TextIO' = <TextIOWrapper>, client_factory: 'Callable[[MistralConfig], MistralClientProtocol]' = build_client, path_picker: 'PathPicker | None' = None) -> 'int'`

Run the CLI.

## `mistral4cli.local_mistral`

Configuration and client helpers for local and remote Mistral backends.

### Class `BackendKind`

Runtime backend modes supported by the CLI.

### Class `ContextConfig`

Client-side context overflow and compaction policy.

#### Methods

  #### `from_env(cls) -> 'ContextConfig'`

  Build context defaults from environment variables.

  #### `normalized(self) -> 'ContextConfig'`

  Return a sanitized copy with bounded operational values.

### Class `ConversationConfig`

Runtime defaults for Mistral Cloud Conversations mode.

#### Methods

  #### `from_env(cls) -> 'ConversationConfig'`

  Build conversation defaults from environment variables.

### Class `LocalGenerationConfig`

Sampling defaults for the local Mistral Small 4 deployment.

#### Methods

  #### `from_env(cls) -> 'LocalGenerationConfig'`

  Build generation defaults from environment variables.

### Class `LocalMistralConfig`

Configuration for the local llama.cpp-backed Mistral endpoint.

#### Methods

  #### `from_env(cls) -> 'LocalMistralConfig'`

  Build a config from environment variables with safe defaults.

### Class `RemoteAPIKeyError`

Raised when remote Mistral cloud mode is requested without an API key.

### Class `RemoteMistralConfig`

Configuration for the remote Mistral cloud endpoint.

#### Methods

  #### `from_env(cls, timeout_ms: 'int' = 300000) -> 'RemoteMistralConfig'`

  Build a remote config from environment variables.

#### `build_client(config: 'MistralConfig | None' = None) -> 'Mistral'`

Construct an official `mistralai` client for the selected backend.

#### `get_client_timeout_ms(client: 'MistralClientProtocol | object', default: 'int' = 300000) -> 'int'`

Return the effective timeout configured on a Mistral client.

#### `get_health(server_url: 'str | None' = None) -> 'dict[str, Any]'`

Return the `/health` payload from the local server.

#### `get_json(url: 'str', timeout_s: 'float' = 2.0) -> 'dict[str, Any]'`

Fetch and decode a JSON document from the local server.

#### `list_models(server_url: 'str | None' = None) -> 'dict[str, Any]'`

Return the `/v1/models` payload from the local server.

#### `remote_api_key_available() -> 'bool'`

Return whether the remote Mistral cloud API key is available.

#### `set_client_timeout_ms(client: 'MistralClientProtocol | object', timeout_ms: 'int') -> 'None'`

Update the effective timeout on a Mistral client in place.

## `mistral4cli.local_tools`

Always-on local Linux shell and workspace tools for the Mistral Small 4 + Medium 3.5 CLI.

### Class `LocalToolBridge`

Local Linux filesystem and shell tools that are always available.

#### Methods

  #### `call_tool(self, public_name: 'str', arguments: 'dict[str, Any]') -> 'MCPToolResult'`

  Execute one local tool call by public name.

  #### `describe_tools(self) -> 'str'`

  Render the local tool catalog.

  #### `runtime_summary(self) -> 'str'`

  Summarize the local OS tool backend.

  #### `to_mistral_tools(self) -> 'list[dict[str, Any]]'`

  Return local tools in the Mistral SDK shape.

### Class `LocalToolSpec`

Shape of one local tool exposed to the model.

## `mistral4cli.mcp_bridge`

MCP bridge helpers for FireCrawl-style remote tools.

### Class `MCPBridgeError`

Raised when the MCP bridge cannot load or execute tools.

### Class `MCPConfig`

Parsed MCP configuration file.

#### Methods

  #### `configured(self) -> 'bool'`

  Return whether at least one MCP server is configured.

  #### `load(cls, path: 'Path') -> 'MCPConfig'`

  Load an MCP config file from disk.

### Class `MCPServerConfig`

Definition of one MCP server entry.

### Class `MCPToolBridge`

Sync wrapper around one or more MCP SSE servers.

#### Methods

  #### `call_tool(self, public_name: 'str', arguments: 'dict[str, Any]') -> 'MCPToolResult'`

  Execute one remote tool call.

  #### `configured(self) -> 'bool'`

  Return whether the underlying MCP configuration is usable.

  #### `describe_tools(self) -> 'str'`

  Render a detailed catalog of the available MCP tools.

  #### `load_tools(self) -> 'list[MCPToolSpec]'`

  Load and cache remote tool metadata.

  #### `runtime_summary(self) -> 'str'`

  Summarize the remote MCP tool backend.

  #### `to_mistral_tools(self) -> 'list[dict[str, Any]]'`

  Return MCP tools shaped for the official Mistral SDK.

  #### `tools_summary(self) -> 'str'`

  Summarize whether remote tools are available.

### Class `MCPToolResult`

Normalized result of one MCP tool invocation.

### Class `MCPToolSpec`

A tool exposed to the Mistral chat-completion API.

#### `discover_mcp_config_path(explicit_path: 'str | Path | None' = None) -> 'Path | None'`

Resolve the MCP config path from CLI, env or repo defaults.

## `mistral4cli.session`

Interactive session facade for the Mistral Small 4 + Medium 3.5 CLI.

### Class `MistralSession`

Stateful conversation helper for the Mistral Small 4 + Medium 3.5 CLI.

## `mistral4cli.tooling`

Tool bridge composition for the Mistral Small 4 + Medium 3.5 CLI.

### Class `CompositeToolBridge`

Combine multiple tool bridges into one namespace.

#### Methods

  #### `call_tool(self, public_name: 'str', arguments: 'dict[str, Any]') -> 'MCPToolResult'`

  Dispatch a tool call to the bridge that owns it.

  #### `describe_tools(self) -> 'str'`

  Render a combined tool catalog for all backends.

  #### `last_error(self) -> 'str | None'`

  Return the last load or dispatch error, if any.

  #### `runtime_summary(self) -> 'str'`

  Summarize the active tool backends.

  #### `to_mistral_tools(self) -> 'list[dict[str, Any]]'`

  Return all tools normalized for the official Mistral SDK.

### Class `ToolBridge`

Minimal interface shared by local and MCP tool bridges.

#### Methods

  #### `call_tool(self, public_name: 'str', arguments: 'dict[str, Any]') -> 'MCPToolResult'`

  Execute a single tool call.

  #### `describe_tools(self) -> 'str'`

  Render a human-readable tool catalogue.

  #### `runtime_summary(self) -> 'str'`

  Summarize the active tool backend.

  #### `to_mistral_tools(self) -> 'list[dict[str, Any]]'`

  Return tools shaped for the official Mistral SDK.

## `mistral4cli.ui`

Terminal rendering helpers for the Mistral Small 4 + Medium 3.5 CLI.

### Class `InteractiveTTYRenderer`

Manage the wrapped REPL composer, status bar, and streamed output.

#### Methods

  #### `clear_overlay(self) -> 'None'`

  Remove the currently rendered composer and status-bar overlay.

  #### `commit_input(self, prompt: 'str', buffer: 'str') -> 'None'`

  Replace the overlay with the committed input before a turn starts.

  #### `finalize_output(self) -> 'None'`

  Flush any pending output fragments before the turn fully settles.

  #### `render_input(self, prompt: 'str', buffer: 'str') -> 'None'`

  Draw only the wrapped input composer while the user is typing.

  #### `render_status_bar(self) -> 'str'`

  Render the one-line bottom status bar for the current turn state.

  #### `show_status(self) -> 'None'`

  Render only the bottom status bar while a turn is active.

  #### `write_answer(self, text: 'str') -> 'None'`

  Write wrapped assistant answer text with a fast TTY typewriter feel.

  #### `write_reasoning(self, text: 'str') -> 'None'`

  Write visible reasoning text with the same TTY streaming cadence.

### Class `SmartOutputWriter`

Incremental terminal writer that wraps normal prose safely.

#### Methods

  #### `feed(self, text: 'str') -> 'str'`

  Consume streamed text and return the wrapped terminal output.

  #### `finish(self) -> 'str'`

  Flush any pending text that was held for wrapping decisions.

#### `iter_typewriter_chunks(text: 'str', visible_chars: 'int') -> 'list[str]'`

Split ANSI-decorated text into small visible chunks for TTY playback.

#### `paint_prompt_lines(lines: 'Sequence[str]', prompt: 'str', stream: 'TextIO') -> 'list[str]'`

Paint the REPL prompt prefixes without affecting wrap calculations.

#### `render_help_screen(summary: 'str', tools: 'Sequence[str] | None', stream: 'TextIO') -> 'str'`

Render a concise but actionable help screen.

#### `render_reasoning_chunk(text: 'str', stream: 'TextIO') -> 'str'`

Render one visible reasoning fragment for the terminal.

#### `render_runtime_summary(backend_kind: 'BackendKind', model_id: 'str', server_url: 'str | None', timeout_ms: 'int', generation: 'LocalGenerationConfig', stream_enabled: 'bool', reasoning_enabled: 'bool', thinking_visible: 'bool', conversations: 'ConversationConfig', context: 'ContextConfig', conversation_id: 'str | None', tool_summary: 'str', logging_summary: 'str', stream: 'TextIO') -> 'str'`

Render a formatted runtime summary.

#### `render_status_snapshot(text: 'str', stream: 'TextIO') -> 'str'`

Render the `/status` snapshot with the dedicated terminal accent.

#### `render_tools_screen(tool_lines: 'Sequence[str]', stream: 'TextIO') -> 'str'`

Render a detailed tool status screen.

#### `render_welcome_banner(summary: 'str', stream: 'TextIO') -> 'str'`

Render the startup banner with retro colors when supported.

#### `supports_full_terminal_ui(stream: 'TextIO') -> 'bool'`

Return whether the current output stream supports the interactive TUI.

#### `terminal_recommendation(stream: 'TextIO') -> 'str'`

Return a short terminal recommendation when colors may degrade.

#### `wrap_prompt_buffer(prompt: 'str', buffer: 'str', width: 'int') -> 'list[str]'`

Wrap one logical REPL buffer into prompt-display lines.
