from __future__ import annotations

import json
from pathlib import Path

import pytest

from mistral4cli.mcp_bridge import MCPConfig, MCPToolBridge, discover_mcp_config_path


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
