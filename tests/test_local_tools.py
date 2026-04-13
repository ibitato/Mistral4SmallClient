from __future__ import annotations

from pathlib import Path

from mistral4cli.local_tools import LocalToolBridge


def test_local_tools_can_read_write_list_and_search(tmp_path: Path) -> None:
    bridge = LocalToolBridge(root=tmp_path)

    write_result = bridge.call_tool(
        "write_file",
        {"path": "notes/hello.txt", "content": "hola mundo\nsecond line"},
    )
    assert write_result.is_error is False
    assert "wrote" in write_result.text

    read_result = bridge.call_tool("read_file", {"path": "notes/hello.txt"})
    assert read_result.is_error is False
    assert "hola mundo" in read_result.text

    list_result = bridge.call_tool("list_dir", {"path": "notes"})
    assert list_result.is_error is False
    assert "hello.txt" in list_result.text

    search_result = bridge.call_tool(
        "search_text",
        {"query": "hola", "path": ".", "max_results": 10},
    )
    assert search_result.is_error is False
    assert "hello.txt" in search_result.text


def test_local_shell_runs_commands(tmp_path: Path) -> None:
    bridge = LocalToolBridge(root=tmp_path)

    result = bridge.call_tool("shell", {"command": "printf 'ok'"})

    assert result.is_error is False
    assert "exit_code=0" in result.text
    assert "ok" in result.text
