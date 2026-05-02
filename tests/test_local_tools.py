from __future__ import annotations

from pathlib import Path

from mistralcli.local_tools import LocalToolBridge


def test_local_tool_descriptions_harden_shell_vs_search_text() -> None:
    bridge = LocalToolBridge()

    tools = {
        entry["function"]["name"]: entry["function"]["description"]
        for entry in bridge.to_mistral_tools()
    }

    assert "Linux shell command" in tools["shell"]
    assert "processes" in tools["shell"]
    assert "workspace files" in tools["search_text"]
    assert "not a replacement for Linux shell grep/find" in tools["search_text"]
    assert "known UTF-8 text file" in tools["read_file"]
    assert "directory orientation" in tools["list_dir"]


def test_local_tools_can_read_write_list_and_search(tmp_path: Path) -> None:
    bridge = LocalToolBridge(root=tmp_path)

    write_result = bridge.call_tool(
        "write_file",
        {"path": "notes/hello.txt", "content": "hello world\nsecond line"},
    )
    assert write_result.is_error is False
    assert "wrote" in write_result.text
    assert write_result.structured_content is not None
    assert write_result.structured_content["status"] == "ok"
    assert write_result.structured_content["tool"] == "write_file"
    assert write_result.structured_content["path"].endswith("notes/hello.txt")

    read_result = bridge.call_tool("read_file", {"path": "notes/hello.txt"})
    assert read_result.is_error is False
    assert "hello world" in read_result.text
    assert read_result.structured_content is not None
    assert read_result.structured_content["status"] == "ok"
    assert read_result.structured_content["tool"] == "read_file"
    assert read_result.structured_content["bytes_read"] > 0

    list_result = bridge.call_tool("list_dir", {"path": "notes"})
    assert list_result.is_error is False
    assert "hello.txt" in list_result.text
    assert list_result.structured_content is not None
    assert list_result.structured_content["status"] == "ok"
    assert list_result.structured_content["tool"] == "list_dir"
    assert list_result.structured_content["count"] == 1

    search_result = bridge.call_tool(
        "search_text",
        {"query": "hello", "path": ".", "max_results": 10},
    )
    assert search_result.is_error is False
    assert "hello.txt" in search_result.text
    assert search_result.structured_content is not None
    assert search_result.structured_content["status"] == "ok"
    assert search_result.structured_content["tool"] == "search_text"
    assert search_result.structured_content["total_matches"] >= 1


def test_local_shell_runs_commands(tmp_path: Path) -> None:
    bridge = LocalToolBridge(root=tmp_path)

    result = bridge.call_tool("shell", {"command": "printf 'ok'"})

    assert result.is_error is False
    assert "exit_code=0" in result.text
    assert "ok" in result.text
    assert result.structured_content is not None
    assert result.structured_content["status"] == "ok"
    assert result.structured_content["tool"] == "shell"
    assert result.structured_content["exit_code"] == 0


def test_shell_output_can_be_paginated(tmp_path: Path) -> None:
    bridge = LocalToolBridge(root=tmp_path)

    result = bridge.call_tool(
        "shell",
        {
            "command": "printf 'one\\ntwo\\nthree\\nfour\\n'",
            "max_lines": 4,
            "offset_lines": 2,
        },
    )

    assert result.is_error is False
    assert "[page offset_lines=2" in result.text
    assert "[more output available" in result.text


def test_search_text_can_be_paginated(tmp_path: Path) -> None:
    bridge = LocalToolBridge(root=tmp_path)
    for index in range(4):
        bridge.call_tool(
            "write_file",
            {
                "path": f"notes/file{index}.txt",
                "content": f"needle {index}\n",
            },
        )

    result = bridge.call_tool(
        "search_text",
        {"query": "needle", "path": ".", "max_results": 2, "offset": 1},
    )

    assert result.is_error is False
    assert "[page offset=1" in result.text
    assert "[more results available" in result.text
    assert result.structured_content is not None
    assert result.structured_content["next_offset"] is not None


def test_local_tool_errors_are_structured(tmp_path: Path) -> None:
    bridge = LocalToolBridge(root=tmp_path)

    read_result = bridge.call_tool("read_file", {"path": "missing.txt"})
    search_result = bridge.call_tool("search_text", {"query": "", "path": "."})

    assert read_result.is_error is True
    assert read_result.structured_content is not None
    assert read_result.structured_content["status"] == "error"
    assert read_result.structured_content["tool"] == "read_file"

    assert search_result.is_error is True
    assert search_result.structured_content is not None
    assert search_result.structured_content["status"] == "error"
    assert search_result.structured_content["tool"] == "search_text"
