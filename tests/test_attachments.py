from __future__ import annotations

import io
from pathlib import Path

import pytest

from mistral4cli.attachments import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_IMAGE_PROMPT,
    DOCUMENT_SUFFIXES,
    IMAGE_SUFFIXES,
    MAX_DOCUMENT_PAGES,
    build_document_message,
    build_image_message,
    build_remote_document_message,
    build_remote_image_message,
    choose_paths,
    format_selection_summary,
    load_document,
    render_document,
)
from mistral4cli.terminal_picker import (
    FINISH_SELECTION_VALUE,
    GO_TO_PARENT_VALUE,
    SELECT_CURRENT_DIRECTORY_VALUE,
    TerminalPickerUnavailableError,
    _discover_matching_files,
    _prompt_candidate_selection,
    _prompt_root_directory,
    pick_paths_in_terminal,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "internet"


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakePrompt:
    def __init__(self, result: object) -> None:
        self._result = result

    def execute(self) -> object:
        return self._result


class _FakeInquirer:
    def __init__(self, *, filepath_result: object, fuzzy_results: list[object]) -> None:
        self._filepath_result = filepath_result
        self._fuzzy_results = list(fuzzy_results)

    def filepath(self, **_kwargs: object) -> _FakePrompt:
        return _FakePrompt(self._filepath_result)

    def fuzzy(self, **_kwargs: object) -> _FakePrompt:
        return _FakePrompt(self._fuzzy_results.pop(0))


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
    output = io.StringIO()

    paths = choose_paths(
        kind="image",
        input_func=lambda _prompt: 'one.png "two files.png"',
        stdin=io.StringIO(),
        stdout=output,
        path_picker=None,
        filetypes=(),
        suffixes=IMAGE_SUFFIXES,
    )

    assert [path.name for path in paths] == ["one.png", "two files.png"]
    assert output.getvalue() == ""


def test_choose_paths_prefers_terminal_picker_in_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdin = _FakeTTY()
    stdout = _FakeTTY()
    expected = [Path("one.png"), Path("two.png")]
    monkeypatch.setattr(
        "mistral4cli.attachments.pick_paths_in_terminal",
        lambda **_kwargs: expected,
    )

    paths = choose_paths(
        kind="image",
        input_func=lambda _prompt: "",
        stdin=stdin,
        stdout=stdout,
        path_picker=None,
        filetypes=(),
        suffixes=IMAGE_SUFFIXES,
    )

    assert paths == expected


def test_load_inquirer_components_uses_real_submodule_exports() -> None:
    from mistral4cli.terminal_picker import _load_inquirer_components

    inquirer, get_style, path_validator = _load_inquirer_components()

    assert hasattr(inquirer, "filepath")
    assert hasattr(inquirer, "fuzzy")
    assert callable(get_style)
    assert path_validator.__name__ == "PathValidator"


def test_prompt_candidate_selection_returns_parent_action(tmp_path: Path) -> None:
    candidate = tmp_path / "nested" / "file.png"
    candidate.parent.mkdir()
    candidate.write_text("ok", encoding="utf-8")
    fake_inquirer = _FakeInquirer(
        filepath_result=tmp_path,
        fuzzy_results=[GO_TO_PARENT_VALUE],
    )

    action, selected = _prompt_candidate_selection(
        kind="image",
        root=tmp_path,
        candidates=[candidate],
        multiple=True,
        selected_paths=[],
        inquirer=fake_inquirer,
        style=None,
    )

    assert action == "parent"
    assert selected == []


def test_prompt_root_directory_can_descend_and_select_current(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    fake_inquirer = _FakeInquirer(
        filepath_result=tmp_path,
        fuzzy_results=[str(child), SELECT_CURRENT_DIRECTORY_VALUE],
    )

    selected = _prompt_root_directory(
        kind="image",
        start_dir=tmp_path,
        inquirer=fake_inquirer,
        path_validator=None,
        style=None,
    )

    assert selected == child


def test_pick_paths_in_terminal_can_go_to_parent_before_selecting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    selected = root / "keep.png"
    selected.write_text("ok", encoding="utf-8")
    (child / "skip.png").write_text("ok", encoding="utf-8")

    fake_inquirer = _FakeInquirer(
        filepath_result=child,
        fuzzy_results=[
            SELECT_CURRENT_DIRECTORY_VALUE,
            GO_TO_PARENT_VALUE,
            str(selected),
            FINISH_SELECTION_VALUE,
        ],
    )

    monkeypatch.setattr(
        "mistral4cli.terminal_picker._load_inquirer_components",
        lambda: (
            fake_inquirer,
            lambda _style, style_override=False: None,
            lambda **_kwargs: None,
        ),
    )

    paths = pick_paths_in_terminal(
        kind="image",
        suffixes=IMAGE_SUFFIXES,
        stdout=io.StringIO(),
        multiple=True,
        start_dir=child,
    )

    assert paths == [selected]


def test_prompt_candidate_selection_can_finish_after_some_selections(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "file.png"
    candidate.write_text("ok", encoding="utf-8")
    fake_inquirer = _FakeInquirer(
        filepath_result=tmp_path,
        fuzzy_results=[FINISH_SELECTION_VALUE],
    )

    action, selected = _prompt_candidate_selection(
        kind="image",
        root=tmp_path,
        candidates=[candidate],
        multiple=True,
        selected_paths=[candidate],
        inquirer=fake_inquirer,
        style=None,
    )

    assert action == "finish"
    assert selected == []


def test_choose_paths_falls_back_to_manual_prompt_when_terminal_picker_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdin = _FakeTTY()
    stdout = _FakeTTY()

    def raise_unavailable(**_kwargs: object) -> list[Path]:
        raise TerminalPickerUnavailableError("no picker")

    monkeypatch.setattr(
        "mistral4cli.attachments.pick_paths_in_terminal",
        raise_unavailable,
    )

    paths = choose_paths(
        kind="document",
        input_func=lambda _prompt: '"report.pdf" notes.docx',
        stdin=stdin,
        stdout=stdout,
        path_picker=None,
        filetypes=(),
        suffixes=DOCUMENT_SUFFIXES,
    )

    assert [path.name for path in paths] == ["report.pdf", "notes.docx"]


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


def test_build_remote_image_message_uses_string_image_url() -> None:
    image = FIXTURE_DIR / "wikimedia-demo.png"

    message = build_remote_image_message([image], prompt="Analyze the image.")

    assert message[0]["type"] == "text"
    assert message[1]["type"] == "image_url"
    assert isinstance(message[1]["image_url"], str)
    assert message[1]["image_url"].startswith("data:image/png;base64,")


def test_build_remote_document_message_uses_document_url_for_binary_docs(
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("plain text", encoding="utf-8")
    pdf_file = FIXTURE_DIR / "w3c-dummy.pdf"

    message = build_remote_document_message(
        [text_file, pdf_file], prompt="Analyze the documents."
    )

    assert message[0]["type"] == "text"
    assert message[1]["type"] == "text"
    assert "notes.txt" in message[1]["text"]
    assert "plain text" in message[1]["text"]
    assert message[2]["type"] == "document_url"
    assert message[2]["document_url"].startswith("data:application/pdf;base64,")
    assert message[2]["document_name"] == "w3c-dummy.pdf"


def test_document_defaults_are_available() -> None:
    assert DEFAULT_IMAGE_PROMPT
    assert DEFAULT_DOCUMENT_PROMPT


def test_render_text_document_does_not_flag_exact_page_limit_as_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    text_file = tmp_path / "exact-pages.txt"
    text_file.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "mistral4cli.attachments._wrap_text_for_document",
        lambda _text: ["line"] * (42 * MAX_DOCUMENT_PAGES),
    )
    monkeypatch.setattr(
        "mistral4cli.attachments._render_text_page",
        lambda *, title, lines: b"png",
    )

    rendered = render_document(text_file)

    assert rendered.truncated is False
    assert len(rendered.content_blocks) == MAX_DOCUMENT_PAGES


def test_discover_matching_files_filters_hidden_entries_and_limits_results(
    tmp_path: Path,
) -> None:
    visible = tmp_path / "visible"
    visible.mkdir()
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (visible / "keep.png").write_text("ok", encoding="utf-8")
    (visible / ".skip.png").write_text("skip", encoding="utf-8")
    (hidden / "also_skip.png").write_text("skip", encoding="utf-8")

    matches, truncated = _discover_matching_files(
        root=tmp_path,
        suffixes=IMAGE_SUFFIXES,
        max_candidates=10,
    )

    assert truncated is False
    assert matches == [visible / "keep.png"]


def test_discover_matching_files_reports_overflow(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file{index}.pdf").write_text("x", encoding="utf-8")

    matches, truncated = _discover_matching_files(
        root=tmp_path,
        suffixes=DOCUMENT_SUFFIXES,
        max_candidates=2,
    )

    assert truncated is True
    assert len(matches) == 2
