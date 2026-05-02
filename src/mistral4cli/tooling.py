"""Tool bridge composition for the dual-model Mistral CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from mistral4cli.mcp_bridge import MCPToolResult


class ToolBridge(Protocol):
    """Minimal interface shared by local and MCP tool bridges."""

    def runtime_summary(self) -> str:
        """Summarize the active tool backend."""
        ...

    def describe_tools(self) -> str:
        """Render a human-readable tool catalogue."""
        ...

    def to_mistral_tools(self) -> list[dict[str, Any]]:
        """Return tools shaped for the official Mistral SDK."""
        ...

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute a single tool call."""
        ...


@dataclass(slots=True)
class CompositeToolBridge:
    """Combine multiple tool bridges into one namespace."""

    bridges: list[ToolBridge]
    _catalog_loaded: bool = field(init=False, repr=False, default=False)
    _tool_to_bridge: dict[str, ToolBridge] = field(
        init=False, repr=False, default_factory=dict
    )
    _last_error: str | None = field(init=False, repr=False, default=None)

    def runtime_summary(self) -> str:
        """Summarize the active tool backends."""

        summaries = [bridge.runtime_summary() for bridge in self.bridges]
        return " | ".join(summaries) if summaries else "Tools: disabled"

    def describe_tools(self) -> str:
        """Render a combined tool catalog for all backends."""

        lines: list[str] = []
        for bridge in self.bridges:
            lines.append(bridge.runtime_summary())
            lines.append(bridge.describe_tools())
        return "\n".join(lines).strip()

    def to_mistral_tools(self) -> list[dict[str, Any]]:
        """Return all tools normalized for the official Mistral SDK."""

        tools: list[dict[str, Any]] = []
        self._tool_to_bridge.clear()
        seen: set[str] = set()
        for bridge in self.bridges:
            bridge_label = self._bridge_label(bridge)
            for tool in bridge.to_mistral_tools():
                function = tool.get("function", {})
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    continue
                public_name = name
                if public_name in seen:
                    public_name = f"{bridge_label}__{name}"
                seen.add(public_name)
                self._tool_to_bridge[public_name] = bridge
                normalized_tool = {
                    **tool,
                    "function": {
                        **function,
                        "name": public_name,
                    },
                }
                tools.append(normalized_tool)
        self._catalog_loaded = True
        self._last_error = None
        return tools

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Dispatch a tool call to the bridge that owns it."""

        if not self._catalog_loaded:
            self.to_mistral_tools()
        bridge = self._tool_to_bridge.get(public_name)
        if bridge is None:
            raise KeyError(f"Unknown tool: {public_name}")
        return bridge.call_tool(public_name, arguments)

    @property
    def last_error(self) -> str | None:
        """Return the last load or dispatch error, if any."""

        return self._last_error

    def _bridge_label(self, bridge: ToolBridge) -> str:
        name = type(bridge).__name__
        lowered = name.removesuffix("ToolBridge").removesuffix("Bridge").lower()
        return lowered or "tool"
