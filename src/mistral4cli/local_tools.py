"""Always-on local Linux shell and workspace tools for the Mistral Small 4 CLI."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mistral4cli.mcp_bridge import MCPToolResult

DEFAULT_SHELL_MAX_LINES = 120
DEFAULT_SEARCH_MAX_RESULTS = 25

logger = logging.getLogger("mistral4cli.local_tools")


@dataclass(frozen=True, slots=True)
class LocalToolSpec:
    """Shape of one local tool exposed to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class LocalToolBridge:
    """Local Linux filesystem and shell tools that are always available."""

    root: Path = field(default_factory=Path.cwd)

    def runtime_summary(self) -> str:
        """Summarize the local OS tool backend."""

        return f"Local OS tools: ready ({self.root})"

    def describe_tools(self) -> str:
        """Render the local tool catalog."""

        lines = ["Tools:"]
        for spec in self._tool_specs():
            lines.append(f"  - {spec.name}: {spec.description}")
        return "\n".join(lines)

    def to_mistral_tools(self) -> list[dict[str, Any]]:
        """Return local tools in the Mistral SDK shape."""

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
        """Execute one local tool call by public name."""

        logger.debug(
            "Dispatching local tool name=%s argument_keys=%s",
            public_name,
            sorted(arguments),
        )
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
                description=(
                    "Run a Linux shell command under the current user. Use this "
                    "for OS inspection, rg/grep/find, git, processes, services, "
                    "packages, env vars, logs, and system-level searches. Output "
                    "is paginated; use offset_lines/max_lines to page."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout_seconds": {"type": "integer", "minimum": 1},
                        "offset_lines": {"type": "integer", "minimum": 0},
                        "max_lines": {"type": "integer", "minimum": 1},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
            LocalToolSpec(
                name="read_file",
                description=(
                    "Read one known UTF-8 text file from disk after you have "
                    "identified the target path."
                ),
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
                description=(
                    "Write UTF-8 text to a known file path, creating parents if "
                    "needed. Use only when the task requires saving or updating "
                    "text on disk."
                ),
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
                description=(
                    "List files and directories at one path for directory "
                    "orientation within the workspace before reading or "
                    "searching deeper."
                ),
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
                description=(
                    "Search text inside workspace files under a specific path. "
                    "Use this for repo or source searches such as finding files "
                    "that mention a symbol. It returns one matching line per "
                    "file and is not a replacement for Linux shell grep/find, "
                    "process inspection, package lookup, logs, or OS-wide search. "
                    "Results are paginated; use offset/max_results to page."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1},
                        "offset": {"type": "integer", "minimum": 0},
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
        offset_lines = max(0, int(arguments.get("offset_lines", 0)))
        max_lines = max(1, int(arguments.get("max_lines", DEFAULT_SHELL_MAX_LINES)))

        if not command:
            return self._tool_error("shell", "command is required")
        if not cwd.exists():
            return self._tool_error("shell", f"cwd not found: {cwd}", cwd=str(cwd))

        logger.info(
            "Running shell command cwd=%s timeout_seconds=%s command=%s",
            cwd,
            timeout_seconds,
            command,
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
            return self._tool_error(
                "shell",
                f"shell command timed out after {timeout_seconds}s: {exc}",
                cwd=str(cwd),
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._tool_error("shell", f"shell failed: {exc}", cwd=str(cwd))

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
        paginated = self._paginate_text(
            output,
            label="shell output",
            offset=offset_lines,
            max_lines=max_lines,
        )
        logger.debug(
            "Shell command finished exit_code=%s stdout_len=%s stderr_len=%s",
            completed.returncode,
            len(completed.stdout),
            len(completed.stderr),
        )
        return MCPToolResult(
            text=paginated,
            is_error=completed.returncode != 0,
            structured_content={
                "status": "error" if completed.returncode != 0 else "ok",
                "tool": "shell",
                "command": command,
                "cwd": str(cwd),
                "exit_code": completed.returncode,
                "offset_lines": offset_lines,
                "max_lines": max_lines,
            },
        )

    def _read_file(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = self._resolve_path(str(arguments.get("path", "")))
        max_bytes = int(arguments.get("max_bytes", 1024 * 1024))
        logger.debug("Reading file path=%s max_bytes=%s", path, max_bytes)
        try:
            if not path.exists():
                return self._tool_error(
                    "read_file", f"file not found: {path}", path=str(path)
                )
            if not path.is_file():
                return self._tool_error(
                    "read_file", f"not a file: {path}", path=str(path)
                )
            data = path.read_bytes()[:max_bytes]
            text = data.decode("utf-8", errors="replace")
            return MCPToolResult(
                text=text,
                is_error=False,
                structured_content={
                    "status": "ok",
                    "tool": "read_file",
                    "path": str(path),
                    "bytes_read": len(data),
                    "max_bytes": max_bytes,
                    "truncated": path.stat().st_size > len(data),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._tool_error("read_file", f"read failed: {exc}", path=str(path))

    def _write_file(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = self._resolve_path(str(arguments.get("path", "")))
        content = str(arguments.get("content", ""))
        logger.info("Writing file path=%s content_len=%s", path, len(content))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            bytes_written = len(content.encode("utf-8"))
            return MCPToolResult(
                text=f"wrote {bytes_written} bytes to {path}",
                is_error=False,
                structured_content={
                    "status": "ok",
                    "tool": "write_file",
                    "path": str(path),
                    "bytes_written": bytes_written,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._tool_error(
                "write_file", f"write failed: {exc}", path=str(path)
            )

    def _list_dir(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = self._resolve_path(str(arguments.get("path", ".")))
        max_entries = int(arguments.get("max_entries", 200))
        logger.debug("Listing directory path=%s max_entries=%s", path, max_entries)
        try:
            if not path.exists():
                return self._tool_error(
                    "list_dir", f"path not found: {path}", path=str(path)
                )
            entries: list[str] = []
            all_children = sorted(path.iterdir(), key=lambda item: item.name.lower())
            for child in all_children[:max_entries]:
                suffix = "/" if child.is_dir() else ""
                entries.append(child.name + suffix)
            return MCPToolResult(
                text="\n".join(entries),
                is_error=False,
                structured_content={
                    "status": "ok",
                    "tool": "list_dir",
                    "path": str(path),
                    "count": len(entries),
                    "max_entries": max_entries,
                    "truncated": len(all_children) > len(entries),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._tool_error(
                "list_dir", f"list_dir failed: {exc}", path=str(path)
            )

    def _search_text(self, arguments: dict[str, Any]) -> MCPToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return self._tool_error("search_text", "query is required")

        root = self._resolve_path(str(arguments.get("path", ".")))
        max_results = max(
            1, int(arguments.get("max_results", DEFAULT_SEARCH_MAX_RESULTS))
        )
        offset = max(0, int(arguments.get("offset", 0)))
        logger.debug(
            "Searching text root=%s query=%s max_results=%s offset=%s",
            root,
            query,
            max_results,
            offset,
        )
        try:
            if not root.exists():
                return self._tool_error(
                    "search_text",
                    f"path not found: {root}",
                    path=str(root),
                    query=query,
                )

            query_lower = query.lower()
            results: list[str] = []
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if path.is_dir():
                    continue
                if not self._looks_textual(path):
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                # Keep this tool focused on quick workspace discovery. Exhaustive
                # or OS-level grep belongs to the Linux shell tool instead.
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if query_lower in line.lower():
                        excerpt = line.strip()
                        results.append(f"{path}:{line_number}: {excerpt}")
                        break
            if not results:
                return MCPToolResult(
                    text="No matches found.",
                    is_error=False,
                    structured_content={
                        "status": "ok",
                        "tool": "search_text",
                        "query": query,
                        "path": str(root),
                        "total_matches": 0,
                        "offset": offset,
                        "max_results": max_results,
                    },
                )

            total_results = len(results)
            page = results[offset : offset + max_results]
            if not page:
                return MCPToolResult(
                    text=(
                        f"No more matches. total_matches={total_results} "
                        f"offset={offset} max_results={max_results}"
                    ),
                    is_error=False,
                    structured_content={
                        "status": "ok",
                        "tool": "search_text",
                        "query": query,
                        "path": str(root),
                        "total_matches": total_results,
                        "offset": offset,
                        "max_results": max_results,
                        "returned": 0,
                    },
                )

            footer: list[str] = []
            end = offset + len(page)
            if offset > 0:
                footer.append(
                    f"[page offset={offset} max_results={max_results} "
                    f"total_matches={total_results}]"
                )
            if end < total_results:
                footer.append(
                    f"[more results available: use offset={end} "
                    f"max_results={max_results}]"
                )

            text = "\n".join(page)
            if footer:
                text = "\n".join([text, *footer])
            return MCPToolResult(
                text=text,
                is_error=False,
                structured_content={
                    "status": "ok",
                    "tool": "search_text",
                    "query": query,
                    "path": str(root),
                    "total_matches": total_results,
                    "offset": offset,
                    "max_results": max_results,
                    "returned": len(page),
                    "next_offset": end if end < total_results else None,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._tool_error(
                "search_text",
                f"search_text failed: {exc}",
                path=str(root),
                query=query,
            )

    def _tool_error(self, tool: str, message: str, **structured: Any) -> MCPToolResult:
        logger.warning("Local tool error tool=%s message=%s", tool, message)
        return MCPToolResult(
            text=f"[tool-error] {message}",
            is_error=True,
            structured_content={
                "status": "error",
                "tool": tool,
                "message": message,
                **structured,
            },
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

    def _paginate_text(
        self,
        text: str,
        *,
        label: str,
        offset: int,
        max_lines: int,
    ) -> str:
        lines = text.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return f"[{label}] no output"

        start = min(offset, total_lines)
        if start >= total_lines:
            return (
                f"[{label}] no more output. total_lines={total_lines} "
                f"offset_lines={offset} max_lines={max_lines}"
            )
        end = min(start + max_lines, total_lines)
        page = lines[start:end]
        footer: list[str] = []
        if start > 0:
            footer.append(
                f"[page offset_lines={start} max_lines={max_lines} "
                f"total_lines={total_lines}]"
            )
        if end < total_lines:
            footer.append(
                f"[more output available: use offset_lines={end} max_lines={max_lines}]"
            )

        rendered = "\n".join(page)
        if footer:
            rendered = "\n".join([rendered, *footer])
        return rendered
