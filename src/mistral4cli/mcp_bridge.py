"""MCP bridge helpers for FireCrawl-style remote tools."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, ListToolsResult

ENV_MCP_CONFIG = "MISTRAL_LOCAL_MCP_CONFIG"


class MCPBridgeError(RuntimeError):
    """Raised when the MCP bridge cannot load or execute tools."""


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Definition of one MCP server entry."""

    name: str
    type: str
    url: str


@dataclass(frozen=True, slots=True)
class MCPConfig:
    """Parsed MCP configuration file."""

    path: Path
    servers: tuple[MCPServerConfig, ...]

    @property
    def configured(self) -> bool:
        return bool(self.servers)

    @classmethod
    def load(cls, path: Path) -> MCPConfig:
        """Load an MCP config file from disk."""

        resolved_path = path.expanduser()
        if not resolved_path.exists():
            raise FileNotFoundError(resolved_path)

        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("mcp.json must contain a JSON object at the top level")
        raw_servers = payload.get("mcpServers")
        if not isinstance(raw_servers, dict):
            raise ValueError("mcp.json is missing the 'mcpServers' object")

        servers: list[MCPServerConfig] = []
        for server_name, raw_config in raw_servers.items():
            if not isinstance(raw_config, dict):
                raise ValueError(f"MCP server {server_name!r} must be an object")
            raw_type = raw_config.get("type")
            raw_url = raw_config.get("url")
            server_type = str(raw_type).strip().lower() if raw_type is not None else ""
            url = str(raw_url).strip() if raw_url is not None else ""
            if not server_type:
                raise ValueError(f"MCP server {server_name!r} is missing a type")
            if not url:
                raise ValueError(f"MCP server {server_name!r} is missing a url")
            servers.append(
                MCPServerConfig(
                    name=str(server_name),
                    type=server_type,
                    url=url,
                )
            )

        return cls(path=resolved_path, servers=tuple(servers))


@dataclass(frozen=True, slots=True)
class MCPToolSpec:
    """A tool exposed to the Mistral chat-completion API."""

    public_name: str
    server_name: str
    remote_name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MCPToolResult:
    """Normalized result of one MCP tool invocation."""

    text: str
    is_error: bool
    structured_content: dict[str, Any] | None = None


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "mcp.json"


def discover_mcp_config_path(explicit_path: str | Path | None = None) -> Path | None:
    """Resolve the MCP config path from CLI, env or repo defaults."""

    if explicit_path is not None:
        return Path(explicit_path).expanduser()

    env_path = os.environ.get(ENV_MCP_CONFIG)
    if env_path:
        return Path(env_path).expanduser()

    default_path = _default_config_path()
    return default_path if default_path.exists() else None


def _format_tool_result_content(result: CallToolResult) -> str:
    pieces: list[str] = []
    if result.structuredContent is not None:
        pieces.append(
            json.dumps(result.structuredContent, ensure_ascii=False, sort_keys=True)
        )

    for item in result.content:
        item_type = getattr(item, "type", "content")
        if item_type == "text":
            text = getattr(item, "text", "")
            if text:
                pieces.append(text)
            continue
        if item_type == "image":
            mime_type = getattr(item, "mimeType", "application/octet-stream")
            data = getattr(item, "data", "")
            pieces.append(f"[image:{mime_type}] {len(data)} bytes")
            continue
        if item_type == "resource":
            uri = getattr(item, "uri", "")
            pieces.append(f"[resource] {uri}")
            continue
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            pieces.append(json.dumps(dump(), ensure_ascii=False, sort_keys=True))
        else:
            pieces.append(str(item))

    text = "\n".join(piece for piece in pieces if piece).strip()
    if not text:
        text = "{}"
    if result.isError:
        return f"[tool-error] {text}"
    return text


def _tool_to_public_name(
    server_name: str, remote_name: str, used_names: set[str]
) -> str:
    candidate = remote_name
    if candidate in used_names:
        candidate = f"{server_name}__{remote_name}"
    used_names.add(candidate)
    return candidate


@dataclass(slots=True)
class MCPToolBridge:
    """Sync wrapper around one or more MCP SSE servers."""

    config: MCPConfig
    _tools_loaded: bool = field(init=False, repr=False, default=False)
    _tool_specs: list[MCPToolSpec] = field(init=False, repr=False, default_factory=list)
    _tool_lookup: dict[str, MCPToolSpec] = field(
        init=False, repr=False, default_factory=dict
    )
    _last_error: str | None = field(init=False, repr=False, default=None)

    @property
    def configured(self) -> bool:
        return self.config.configured

    def runtime_summary(self) -> str:
        if not self.configured:
            return "FireCrawl MCP: disabled"
        server_names = ", ".join(server.name for server in self.config.servers)
        return (
            "FireCrawl MCP: auto-tools on "
            f"({server_names} from {self.config.path.name})"
        )

    def tools_summary(self) -> str:
        if self._tools_loaded and self._tool_specs:
            return f"{len(self._tool_specs)} tool(s) ready"
        if self._last_error is not None:
            return f"unavailable: {self._last_error}"
        if not self.configured:
            return "disabled"
        return "configured, not loaded yet"

    def describe_tools(self) -> str:
        """Render a detailed catalog of the available MCP tools."""

        try:
            tool_specs = self.load_tools()
        except Exception as exc:  # pragma: no cover - surfaced in CLI smoke
            return "\n".join(
                [
                    self.runtime_summary(),
                    f"Status: unavailable ({exc})",
                ]
            )

        if not tool_specs:
            return "\n".join([self.runtime_summary(), "Status: no tools exposed."])

        lines = [self.runtime_summary(), "Tools:"]
        for spec in tool_specs:
            description = spec.description or "(no description)"
            lines.append(f"  - {spec.public_name}: {description}")
        return "\n".join(lines)

    def to_mistral_tools(self) -> list[dict[str, Any]]:
        """Return MCP tools shaped for the official Mistral SDK."""

        tool_specs = self.load_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.public_name,
                    "description": spec.description or spec.remote_name,
                    "parameters": spec.input_schema,
                },
            }
            for spec in tool_specs
        ]

    def load_tools(self) -> list[MCPToolSpec]:
        """Load and cache remote tool metadata."""

        if self._tools_loaded:
            return list(self._tool_specs)

        try:
            tool_specs = anyio.run(self._load_tools_async)
        except Exception as exc:  # pragma: no cover - surfaced in CLI smoke
            self._last_error = str(exc)
            raise MCPBridgeError(f"Could not load MCP tools: {exc}") from exc

        self._tool_specs = tool_specs
        self._tool_lookup = {spec.public_name: spec for spec in tool_specs}
        self._tools_loaded = True
        self._last_error = None
        return list(self._tool_specs)

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute one remote tool call."""

        if not self._tools_loaded:
            self.load_tools()
        spec = self._tool_lookup.get(public_name)
        if spec is None:
            raise MCPBridgeError(f"Unknown MCP tool: {public_name}")

        try:
            result = anyio.run(self._call_tool_async, spec, arguments)
        except Exception as exc:  # pragma: no cover - surfaced in CLI smoke
            raise MCPBridgeError(f"Tool {public_name!r} failed: {exc}") from exc

        return result

    async def _load_tools_async(self) -> list[MCPToolSpec]:
        tool_specs: list[MCPToolSpec] = []
        used_names: set[str] = set()

        for server in self.config.servers:
            async with (
                self._open_server_streams(server) as streams,
                ClientSession(*streams) as session,
            ):
                await session.initialize()
                result = await session.list_tools()
                tool_specs.extend(
                    self._server_tools_to_specs(server.name, result, used_names)
                )

        return tool_specs

    async def _call_tool_async(
        self, spec: MCPToolSpec, arguments: dict[str, Any]
    ) -> MCPToolResult:
        server = self._server_by_name(spec.server_name)
        async with (
            self._open_server_streams(server) as streams,
            ClientSession(*streams) as session,
        ):
            await session.initialize()
            result = await session.call_tool(spec.remote_name, arguments=arguments)
            return self._normalize_tool_result(result)

    def _server_by_name(self, server_name: str) -> MCPServerConfig:
        for server in self.config.servers:
            if server.name == server_name:
                return server
        raise MCPBridgeError(f"Unknown MCP server: {server_name}")

    def _server_tools_to_specs(
        self,
        server_name: str,
        result: ListToolsResult,
        used_names: set[str],
    ) -> list[MCPToolSpec]:
        specs: list[MCPToolSpec] = []
        for tool in result.tools:
            public_name = _tool_to_public_name(server_name, tool.name, used_names)
            specs.append(
                MCPToolSpec(
                    public_name=public_name,
                    server_name=server_name,
                    remote_name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                )
            )
        return specs

    def _normalize_tool_result(self, result: CallToolResult) -> MCPToolResult:
        return MCPToolResult(
            text=_format_tool_result_content(result),
            is_error=result.isError,
            structured_content=result.structuredContent,
        )

    @asynccontextmanager
    async def _open_server_streams(
        self, server: MCPServerConfig
    ) -> AsyncIterator[tuple[Any, Any]]:
        transport = self._transport_for_server(server)
        try:
            if transport == "streamable-http":
                async with streamable_http_client(server.url) as streams:
                    yield streams[0], streams[1]
                return

            async with sse_client(server.url) as streams:
                yield streams
        except Exception as exc:  # pragma: no cover - surfaced in CLI smoke
            raise MCPBridgeError(
                f"Could not connect to MCP server {server.name!r} at {server.url}: "
                f"{self._friendly_transport_error(server, exc)}"
            ) from exc

    def _transport_for_server(self, server: MCPServerConfig) -> str:
        if server.type in {"streamable-http", "streamablehttp", "http"}:
            return "streamable-http"
        if "/v2/mcp" in server.url:
            return "streamable-http"
        return "sse"

    def _friendly_transport_error(
        self, server: MCPServerConfig, exc: BaseException
    ) -> str:
        if (
            "400 Bad Request" in str(exc)
            and self._transport_for_server(server) == "sse"
        ):
            return (
                "the endpoint returned HTTP 400 when opened as SSE. "
                "This FireCrawl URL looks like Streamable HTTP; set type to "
                "'streamable-http' or keep /v2/mcp and let the bridge auto-detect it."
            )
        return str(exc)
