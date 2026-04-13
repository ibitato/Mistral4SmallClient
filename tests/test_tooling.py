from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mistral4cli.tooling import CompositeToolBridge


@dataclass(slots=True)
class FakeBridge:
    prefix: str

    def runtime_summary(self) -> str:
        return f"{self.prefix} bridge"

    def describe_tools(self) -> str:
        return f"{self.prefix} tools"

    def to_mistral_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "shared",
                    "description": f"{self.prefix} tool",
                    "parameters": {"type": "object"},
                },
            }
        ]

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> tuple[str, str]:
        return self.prefix, public_name


def test_composite_tool_bridge_prefixes_collisions() -> None:
    bridge = CompositeToolBridge([FakeBridge("local"), FakeBridge("remote")])

    tools = bridge.to_mistral_tools()

    assert [tool["function"]["name"] for tool in tools] == [
        "shared",
        "fake__shared",
    ]
    assert bridge.call_tool("shared", {}) == ("local", "shared")
    assert bridge.call_tool("fake__shared", {}) == ("remote", "fake__shared")
