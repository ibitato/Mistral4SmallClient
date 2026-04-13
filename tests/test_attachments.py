from __future__ import annotations

import io
from pathlib import Path

import pytest

from mistral4cli.attachments import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_IMAGE_PROMPT,
    build_document_message,
    build_image_message,
    choose_paths,
    format_selection_summary,
    load_document,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "internet"


def test_build_image_message_uses_prompt_and_data_url() -> None:
    image = FIXTURE_DIR / "wikimedia-demo.png"

    message = build_image_message([image], prompt="Analyze the image.")

    assert message[0]["type"] == "text"
    assert "Analyze the image." in message[0]["text"]
    assert "wikimedia-demo.png" in message[0]["text"]
    assert message[1]["type"] == "image_url"
    assert message[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_format_selection_summary_limits_preview(tmp_path: Path) -> None:
    paths = [tmp_path / f"file{index}.txt" for index in range(4)]

    assert format_selection_summary(paths) == "file0.txt, file1.txt, file2.txt, +1 more"


def test_choose_paths_falls_back_to_terminal_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mistral4cli.attachments.build_tk_path_picker", lambda: None)
    output = io.StringIO()

    paths = choose_paths(
        kind="image",
        input_func=lambda _prompt: 'one.png "two files.png"',
        stdout=output,
        path_picker=None,
        filetypes=(),
    )

    assert [path.name for path in paths] == ["one.png", "two files.png"]
    assert output.getvalue() == ""


def test_load_document_supports_text_docx_and_pdf(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello world", encoding="utf-8")

    docx_file = FIXTURE_DIR / "pywordform-sample_form.docx"
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"

    assert load_document(text_file).text == "hello world"
    assert "Sample MS Word form" in load_document(docx_file).text
    assert "Dummy PDF file" in load_document(pdf_file).text


def test_build_document_message_combines_prompt_and_docs(
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("test content", encoding="utf-8")

    docx_file = FIXTURE_DIR / "pywordform-sample_form.docx"
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"

    message = build_document_message(
        [text_file, docx_file, pdf_file], prompt="Summarize the attachments."
    )

    image_blocks = [block for block in message if block["type"] == "image_url"]

    assert message[0]["type"] == "text"
    assert "Summarize the attachments." in message[0]["text"]
    assert "[Document 1: notes.txt]" in [
        block["text"] for block in message if block["type"] == "text"
    ]
    assert "[Document 2: pywordform-sample_form.docx]" in [
        block["text"] for block in message if block["type"] == "text"
    ]
    assert "[Document 3: w3c-dummy.pdf]" in [
        block["text"] for block in message if block["type"] == "text"
    ]
    assert image_blocks
    assert all(
        block["image_url"]["url"].startswith("data:image/png;base64,")
        for block in image_blocks
    )


def test_document_defaults_are_available() -> None:
    assert DEFAULT_IMAGE_PROMPT
    assert DEFAULT_DOCUMENT_PROMPT
