from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from mistral4cli.mcp_bridge import (
    MCPBridgeError,
    MCPConfig,
    MCPServerConfig,
    MCPToolBridge,
    MCPToolSpec,
    _normalize_firecrawl_arguments,
    discover_mcp_config_path,
)


def test_load_mcp_config_parses_firecrawl_server(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "FireCrawl": {
                        "type": "sse",
                        "url": "https://example.test/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = MCPConfig.load(config_path)

    assert config.configured is True
    assert config.path == config_path
    assert config.servers[0].name == "FireCrawl"
    assert config.servers[0].type == "sse"
    assert config.servers[0].url == "https://example.test/mcp"


def test_load_mcp_config_expands_firecrawl_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "mcp.json"
    monkeypatch.setenv("FIRECRAWL_API_KEY", "example-firecrawl-token")
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "FireCrawl": {
                        "type": "streamable-http",
                        "url": "https://mcp.firecrawl.dev/${FIRECRAWL_API_KEY}/v2/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = MCPConfig.load(config_path)

    assert config.configured is True
    assert (
        config.servers[0].url
        == "https://mcp.firecrawl.dev/example-firecrawl-token/v2/mcp"
    )


def test_load_mcp_config_skips_unresolved_env_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "FireCrawl": {
                        "type": "streamable-http",
                        "url": "https://mcp.firecrawl.dev/${FIRECRAWL_API_KEY}/v2/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = MCPConfig.load(config_path)

    assert config.configured is False
    assert config.servers == ()


def test_discover_mcp_config_path_prefers_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "custom-mcp.json"
    config_path.write_text('{"mcpServers": {}}', encoding="utf-8")
    monkeypatch.setenv("MISTRAL_LOCAL_MCP_CONFIG", str(config_path))

    discovered = discover_mcp_config_path()

    assert discovered == config_path


def test_bridge_runtime_summary_mentions_mcp_json(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text('{"mcpServers": {}}', encoding="utf-8")
    bridge = MCPToolBridge(MCPConfig.load(config_path))

    assert bridge.runtime_summary() == "FireCrawl MCP: disabled"


def test_bridge_autodetects_streamable_http_for_v2_mcp(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "FireCrawl": {
                        "type": "sse",
                        "url": "https://mcp.firecrawl.dev/example/v2/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    bridge = MCPToolBridge(MCPConfig.load(config_path))

    assert bridge._transport_for_server(bridge.config.servers[0]) == "streamable-http"


def test_normalize_firecrawl_arguments_wraps_string_sources() -> None:
    normalized = _normalize_firecrawl_arguments(
        "firecrawl_search",
        {
            "query": "pruebas filetype:pdf",
            "sources": ["web", "images"],
        },
    )

    assert normalized["sources"] == [{"type": "web"}, {"type": "images"}]


def test_call_tool_surfaces_nested_mcp_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = MCPToolBridge(
        MCPConfig(
            path=Path("mcp.json"),
            servers=(
                MCPServerConfig(
                    name="FireCrawl", type="streamable-http", url="https://example.test"
                ),
            ),
        )
    )
    spec = MCPToolSpec(
        public_name="firecrawl_search",
        server_name="FireCrawl",
        remote_name="firecrawl_search",
        description="Search the web",
        input_schema={"type": "object"},
    )
    bridge._tools_loaded = True
    bridge._tool_lookup = {spec.public_name: spec}

    class NestedToolError(Exception):
        def __init__(self, error: BaseException) -> None:
            super().__init__("nested")
            self.exceptions = (error,)

    def fake_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise NestedToolError(
            McpError(
                ErrorData(
                    code=-32602,
                    message=(
                        "Tool 'firecrawl_search' parameter validation failed: "
                        "sources.0: Invalid input: expected object, received string."
                    ),
                )
            )
        )

    monkeypatch.setattr(anyio, "run", fake_run)

    with pytest.raises(MCPBridgeError, match="parameter validation failed"):
        bridge.call_tool(
            "firecrawl_search",
            {"query": "pruebas filetype:pdf", "sources": ["web"]},
        )
