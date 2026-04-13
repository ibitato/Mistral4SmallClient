"""Attachment helpers for image and document turns."""

from __future__ import annotations

import base64
import mimetypes
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

DEFAULT_IMAGE_PROMPT = (
    "Analiza las imagenes adjuntas y responde de forma breve, util y concreta."
)
DEFAULT_DOCUMENT_PROMPT = (
    "Analiza los documentos adjuntos y resume lo importante de forma concreta."
)

IMAGE_FILETYPES: tuple[tuple[str, str], ...] = (
    ("Images", "*.png *.jpg *.jpeg *.webp *.gif *.bmp *.tif *.tiff"),
    ("All files", "*.*"),
)
DOCUMENT_FILETYPES: tuple[tuple[str, str], ...] = (
    ("Documents", "*.txt *.md *.rst *.json *.yaml *.yml *.toml *.csv *.pdf *.docx"),
    ("All files", "*.*"),
)

TEXT_DOCUMENT_SUFFIXES = {
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
}
PDF_SUFFIX = ".pdf"
DOCX_SUFFIX = ".docx"

MAX_DOCUMENT_CHARS = 120_000


class PathPicker(Protocol):
    """Callable path picker used by the REPL."""

    def __call__(
        self,
        *,
        title: str,
        filetypes: Sequence[tuple[str, str]],
        multiple: bool,
    ) -> list[Path]:
        """Return the selected files, or an empty list when canceled."""


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """One document loaded from disk for analysis."""

    path: Path
    text: str
    truncated: bool = False


def format_selection_summary(paths: Sequence[Path]) -> str:
    """Return a compact human-readable summary for selected files."""

    if not paths:
        return "No files selected."
    names = [path.name for path in paths]
    preview = ", ".join(names[:3])
    if len(names) > 3:
        preview = f"{preview}, +{len(names) - 3} more"
    return preview


def build_tk_path_picker() -> PathPicker | None:
    """Build a tkinter-based file picker when GUI support is available."""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    def picker(
        *,
        title: str,
        filetypes: Sequence[tuple[str, str]],
        multiple: bool,
    ) -> list[Path]:
        try:
            root = tk.Tk()
            root.withdraw()
            try:
                if multiple:
                    selection: list[str] = list(
                        filedialog.askopenfilenames(
                            title=title,
                            filetypes=filetypes,
                        )
                    )
                else:
                    single = filedialog.askopenfilename(
                        title=title,
                        filetypes=filetypes,
                    )
                    selection = [single] if single else []
            finally:
                root.destroy()
        except Exception:
            return []
        return [Path(item).expanduser() for item in selection if item]

    return picker


def choose_paths(
    *,
    kind: str,
    input_func: Callable[[str], str],
    stdout: TextIO,
    path_picker: PathPicker | None,
    filetypes: Sequence[tuple[str, str]],
    multiple: bool = True,
) -> list[Path]:
    """Choose one or more files, preferring a GUI picker when available."""

    picker = path_picker or build_tk_path_picker()
    if picker is not None:
        selection = picker(
            title=f"Select {kind} files",
            filetypes=filetypes,
            multiple=multiple,
        )
        return [path.expanduser() for path in selection]

    prompt = f"Enter one or more {kind} paths separated by spaces (blank to cancel): "
    raw = input_func(prompt).strip()
    if not raw:
        return []
    try:
        return [Path(token).expanduser() for token in shlex.split(raw)]
    except ValueError as exc:
        stdout.write(f"[{kind}] invalid paths: {exc}\n")
        stdout.flush()
        return []


def build_image_message(
    paths: Sequence[Path], *, prompt: str | None = None
) -> list[dict[str, Any]]:
    """Build a multimodal user message for one or more images."""

    image_paths = [path.expanduser() for path in paths]
    if not image_paths:
        raise ValueError("At least one image is required")

    message = (prompt or DEFAULT_IMAGE_PROMPT).strip() or DEFAULT_IMAGE_PROMPT
    message_lines = [message, "", "Imagenes adjuntas:"]
    message_lines.extend(f"- {path.name}" for path in image_paths)
    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(message_lines)}]

    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(path)},
            }
        )
    return content


def build_document_message(paths: Sequence[Path], *, prompt: str | None = None) -> str:
    """Build a plain-text user message from one or more documents."""

    document_paths = [path.expanduser() for path in paths]
    if not document_paths:
        raise ValueError("At least one document is required")

    message = (prompt or DEFAULT_DOCUMENT_PROMPT).strip() or DEFAULT_DOCUMENT_PROMPT
    sections = [message, ""]
    for index, path in enumerate(document_paths, start=1):
        loaded = load_document(path)
        header = f"[Documento {index}: {path.name}]"
        sections.append(header)
        sections.append(loaded.text.strip() or "[Documento vacio]")
        if loaded.truncated:
            sections.append(
                f"[nota: contenido truncado a {MAX_DOCUMENT_CHARS} caracteres]"
            )
        sections.append("")
    return "\n".join(sections).strip()


def load_document(path: Path) -> LoadedDocument:
    """Load a supported document from disk."""

    normalized = path.expanduser()
    if not normalized.exists():
        raise FileNotFoundError(f"Document not found: {normalized}")
    if not normalized.is_file():
        raise IsADirectoryError(f"Document path is not a file: {normalized}")

    suffix = normalized.suffix.lower()
    if suffix in TEXT_DOCUMENT_SUFFIXES:
        return _load_text_document(normalized)
    if suffix == PDF_SUFFIX:
        return _load_pdf_document(normalized)
    if suffix == DOCX_SUFFIX:
        return _load_docx_document(normalized)
    raise ValueError(
        f"Unsupported document type: {normalized.suffix or normalized.name}"
    )


def _load_text_document(path: Path) -> LoadedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    return _maybe_truncate(LoadedDocument(path=path, text=text, truncated=False))


def _load_pdf_document(path: Path) -> LoadedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError(
            "PDF support requires the pypdf package. Run `uv sync` first."
        ) from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        text = extracted.strip()
        if text:
            parts.append(f"[page {page_number}]\n{text}")
    return _maybe_truncate(
        LoadedDocument(path=path, text="\n\n".join(parts).strip(), truncated=False)
    )


def _load_docx_document(path: Path) -> LoadedDocument:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError(
            "DOCX support requires python-docx. Run `uv sync` first."
        ) from exc

    document = Document(str(path))
    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return _maybe_truncate(
        LoadedDocument(path=path, text="\n".join(parts).strip(), truncated=False)
    )


def _maybe_truncate(document: LoadedDocument) -> LoadedDocument:
    if len(document.text) <= MAX_DOCUMENT_CHARS:
        return document
    return LoadedDocument(
        path=document.path,
        text=document.text[:MAX_DOCUMENT_CHARS],
        truncated=True,
    )


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    if not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {path.suffix or path.name}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"
