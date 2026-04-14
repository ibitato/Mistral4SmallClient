from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

from generate_reference import MODULE_NAMES, build_reference_markdown

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILE = ROOT / "docs" / "reference.md"
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]")


def main(argv: list[str] | None = None) -> int:
    """Validate generated docs and language hygiene."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite docs/reference.md from the current docstrings.",
    )
    args = parser.parse_args(argv)

    if args.fix:
        REFERENCE_FILE.write_text(build_reference_markdown(), encoding="utf-8")
        return 0

    errors = validate_reference() + validate_language_hygiene()
    if errors:
        for error in errors:
            print(error)
        return 1
    return 0


def validate_reference() -> list[str]:
    """Check that the generated reference matches the checked-in file."""

    expected = build_reference_markdown()
    if not REFERENCE_FILE.exists():
        return [f"{REFERENCE_FILE} is missing; run `make docs`."]
    current = REFERENCE_FILE.read_text(encoding="utf-8")
    if current != expected:
        return [f"{REFERENCE_FILE} is out of date; run `make docs`."]
    return []


def validate_language_hygiene() -> list[str]:
    """Check for obvious non-English text in tracked source and docs files."""

    errors: list[str] = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if NON_ASCII_PATTERN.search(text):
            errors.append(f"{path}: contains non-English text markers")

    errors.extend(validate_docstrings())
    return errors


def validate_docstrings() -> list[str]:
    """Check that public package objects are documented."""

    errors: list[str] = []
    sys.path.insert(0, str(ROOT / "src"))
    for module_name in MODULE_NAMES:
        module = __import__(module_name, fromlist=["*"])
        if not inspect.getdoc(module):
            errors.append(f"{module_name}: missing module docstring")
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if getattr(obj, "__name__", name) != name:
                continue
            if inspect.isclass(obj) and getattr(obj, "__module__", None) == module_name:
                if not inspect.getdoc(obj):
                    errors.append(f"{module_name}.{name}: missing class docstring")
                for member_name, descriptor in vars(obj).items():
                    if member_name.startswith("_"):
                        continue
                    if not isinstance(
                        descriptor, property | classmethod | staticmethod
                    ) and not inspect.isfunction(descriptor):
                        continue
                    method = unwrap_descriptor(descriptor)
                    qualname = getattr(method, "__qualname__", "")
                    if not qualname.startswith(f"{name}."):
                        continue
                    if not inspect.getdoc(method):
                        errors.append(
                            f"{module_name}.{name}.{member_name}: "
                            "missing method docstring"
                        )
            elif (
                inspect.isfunction(obj)
                and getattr(obj, "__module__", None) == module_name
            ):
                if not inspect.getdoc(obj):
                    errors.append(f"{module_name}.{name}: missing function docstring")
    return errors


def unwrap_descriptor(descriptor: object) -> object:
    """Return the callable behind a classmethod, staticmethod, or property."""

    if isinstance(descriptor, classmethod | staticmethod):
        return descriptor.__func__
    if isinstance(descriptor, property):
        return descriptor.fget or descriptor
    return descriptor


def iter_text_files() -> list[Path]:
    """Yield tracked text files that should be checked for language hygiene."""

    files: list[Path] = []
    roots = [
        ROOT / "src",
        ROOT / "docs",
        ROOT / ".github",
        ROOT / "scripts",
    ]
    for root in roots:
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.name != "check_docs.py"
                and path.suffix.lower()
                in {".py", ".md", ".yml", ".yaml", ".toml", ".json"}
            )
    files.extend(
        path
        for path in [
            ROOT / "README.md",
            ROOT / "AGENTS.md",
            ROOT / "Makefile",
            ROOT / "pyproject.toml",
        ]
        if path.exists()
    )
    return sorted(dict.fromkeys(files))


if __name__ == "__main__":
    raise SystemExit(main())
