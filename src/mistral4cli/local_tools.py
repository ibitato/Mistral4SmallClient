"""Always-on local OS tools for the Mistral Small 4 CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mistral4cli.mcp_bridge import MCPToolResult


@dataclass(frozen=True, slots=True)
class LocalToolSpec:
    """Shape of one local tool exposed to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class LocalToolBridge:
    """Local filesystem and shell tools that are always available."""

    root: Path = field(default_factory=Path.cwd)

    def runtime_summary(self) -> str:
        return f"Local OS tools: ready ({self.root})"

    def describe_tools(self) -> str:
        lines = ["Tools:"]
        for spec in self._tool_specs():
            lines.append(f"  - {spec.name}: {spec.description}")
        return "\n".join(lines)

    def to_mistral_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in self._tool_specs()
        ]

    def call_tool(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        match public_name:
            case "shell":
                return self._shell(arguments)
            case "read_file":
                return self._read_file(arguments)
            case "write_file":
                return self._write_file(arguments)
            case "list_dir":
                return self._list_dir(arguments)
            case "search_text":
                return self._search_text(arguments)
            case _:
                raise KeyError(f"Unknown local tool: {public_name}")

    def _tool_specs(self) -> list[LocalToolSpec]:
        return [
            LocalToolSpec(
                name="shell",
                description="Run a shell command under the current user.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout_seconds": {"type": "integer", "minimum": 1},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
            LocalToolSpec(
                name="read_file",
                description="Read a UTF-8 text file from disk.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_bytes": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            LocalToolSpec(
                name="write_file",
                description="Write UTF-8 text to disk, creating parents if needed.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            ),
            LocalToolSpec(
                name="list_dir",
                description="List files and directories.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_entries": {"type": "integer", "minimum": 1},
                    },
                    "additionalProperties": False,
                },
            ),
            LocalToolSpec(
                name="search_text",
                description="Search for text in project files.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ]

    def _resolve_path(self, raw_path: str | None) -> Path:
        candidate = Path(raw_path or ".").expanduser()
        if not candidate.is_absolute():
            candidate = (self.root / candidate).resolve()
        return candidate

    def _shell(self, arguments: dict[str, Any]) -> MCPToolResult:
        command = str(arguments.get("command", "")).strip()
        cwd = self._resolve_path(str(arguments.get("cwd", ".")))
        timeout_seconds = int(arguments.get("timeout_seconds", 30))

        if not command:
            return MCPToolResult(text="[tool-error] command is required", is_error=True)
        if not cwd.exists():
            return MCPToolResult(
                text=f"[tool-error] cwd not found: {cwd}",
                is_error=True,
            )

        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return MCPToolResult(
                text=(
                    "[tool-error] shell command timed out after "
                    f"{timeout_seconds}s: {exc}"
                ),
                is_error=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return MCPToolResult(
                text=f"[tool-error] shell failed: {exc}",
                is_error=True,
            )

        output = "\n".join(
            [
                f"exit_code={completed.returncode}",
                f"cwd={cwd}",
                "--- stdout ---",
                completed.stdout.rstrip(),
                "--- stderr ---",
                completed.stderr.rstrip(),
            ]
        ).strip()
        return MCPToolResult(text=output, is_error=completed.returncode != 0)

    def _read_file(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = self._resolve_path(str(arguments.get("path", "")))
        max_bytes = int(arguments.get("max_bytes", 1024 * 1024))
        try:
            if not path.exists():
                return MCPToolResult(
                    text=f"[tool-error] file not found: {path}", is_error=True
                )
            if not path.is_file():
                return MCPToolResult(
                    text=f"[tool-error] not a file: {path}", is_error=True
                )
            data = path.read_bytes()[:max_bytes]
            text = data.decode("utf-8", errors="replace")
            return MCPToolResult(text=text, is_error=False)
        except Exception as exc:  # pragma: no cover - defensive
            return MCPToolResult(text=f"[tool-error] read failed: {exc}", is_error=True)

    def _write_file(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = self._resolve_path(str(arguments.get("path", "")))
        content = str(arguments.get("content", ""))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return MCPToolResult(
                text=f"wrote {len(content.encode('utf-8'))} bytes to {path}",
                is_error=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return MCPToolResult(
                text=f"[tool-error] write failed: {exc}",
                is_error=True,
            )

    def _list_dir(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = self._resolve_path(str(arguments.get("path", ".")))
        max_entries = int(arguments.get("max_entries", 200))
        try:
            if not path.exists():
                return MCPToolResult(
                    text=f"[tool-error] path not found: {path}", is_error=True
                )
            entries: list[str] = []
            for child in sorted(path.iterdir(), key=lambda item: item.name.lower())[
                :max_entries
            ]:
                suffix = "/" if child.is_dir() else ""
                entries.append(child.name + suffix)
            return MCPToolResult(text="\n".join(entries), is_error=False)
        except Exception as exc:  # pragma: no cover - defensive
            return MCPToolResult(
                text=f"[tool-error] list_dir failed: {exc}",
                is_error=True,
            )

    def _search_text(self, arguments: dict[str, Any]) -> MCPToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return MCPToolResult(text="[tool-error] query is required", is_error=True)

        root = self._resolve_path(str(arguments.get("path", ".")))
        max_results = int(arguments.get("max_results", 25))
        try:
            if not root.exists():
                return MCPToolResult(
                    text=f"[tool-error] path not found: {root}", is_error=True
                )

            query_lower = query.lower()
            results: list[str] = []
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if len(results) >= max_results:
                    break
                if path.is_dir():
                    continue
                if not self._looks_textual(path):
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if query_lower in line.lower():
                        excerpt = line.strip()
                        results.append(f"{path}:{line_number}: {excerpt}")
                        break
            return MCPToolResult(
                text="\n".join(results) if results else "No matches found.",
                is_error=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return MCPToolResult(
                text=f"[tool-error] search_text failed: {exc}",
                is_error=True,
            )

    def _looks_textual(self, path: Path) -> bool:
        return path.suffix not in {
            ".pyc",
            ".so",
            ".bin",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
        }
