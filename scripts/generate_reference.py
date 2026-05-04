from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_NAMES = (
    "mistralcli.attachments",
    "mistralcli.cli",
    "mistralcli.local_mistral",
    "mistralcli.local_tools",
    "mistralcli.mcp_bridge",
    "mistralcli.session",
    "mistralcli.tooling",
    "mistralcli.ui",
)


def build_reference_markdown() -> str:
    """Build Markdown API reference for the public package surface."""

    lines: list[str] = [
        "# MistralClient API Reference",
        "",
        "This file is generated from public docstrings.",
        "",
    ]
    for module_name in MODULE_NAMES:
        module = importlib.import_module(module_name)
        lines.extend(render_module(module))
    return "\n".join(lines).rstrip() + "\n"


def render_module(module: ModuleType) -> list[str]:
    """Render one module section."""

    lines = [f"## `{module.__name__}`", ""]
    module_doc = inspect.getdoc(module)
    if module_doc:
        lines.append(module_doc)
        lines.append("")

    public_members = sorted(_iter_public_members(module), key=lambda item: item[0])
    if not public_members:
        lines.append("_No public members._")
        lines.append("")
        return lines

    for name, obj in public_members:
        if inspect.isclass(obj):
            lines.extend(render_class(name, obj))
        elif inspect.isfunction(obj):
            lines.extend(render_function(name, obj))
    return lines


def render_class(name: str, cls: type[Any]) -> list[str]:
    """Render one class section."""

    lines = [f"### Class `{name}`", ""]
    cls_doc = inspect.getdoc(cls)
    if cls_doc:
        lines.append(cls_doc)
        lines.append("")

    methods = [
        (method_name, _unwrap_descriptor(descriptor))
        for method_name, descriptor in vars(cls).items()
        if _is_public_class_member(name, method_name, descriptor)
    ]
    if methods:
        lines.append("#### Methods")
        lines.append("")
        for method_name, method in sorted(methods, key=lambda item: item[0]):
            lines.extend(render_function(method_name, method, indent="  "))
    return lines


def render_function(name: str, func: Any, *, indent: str = "") -> list[str]:
    """Render one function or method section."""

    signature = _format_signature(func)
    lines = [f"{indent}#### `{name}{signature}`", ""]
    doc = inspect.getdoc(func)
    if doc:
        lines.extend(
            f"{indent}{line}" if line else ""
            for line in textwrap.dedent(doc).splitlines()
        )
    else:
        lines.append(f"{indent}_No docstring available._")
    lines.append("")
    return lines


def _iter_public_members(module: ModuleType) -> Iterable[tuple[str, Any]]:
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if getattr(obj, "__module__", module.__name__) != module.__name__:
            continue
        if getattr(obj, "__name__", name) != name:
            continue
        yield name, obj


def _is_public_class_member(class_name: str, member_name: str, descriptor: Any) -> bool:
    if member_name.startswith("_"):
        return False
    if not isinstance(
        descriptor, property | classmethod | staticmethod
    ) and not inspect.isfunction(descriptor):
        return False
    method = _unwrap_descriptor(descriptor)
    qualname = getattr(method, "__qualname__", "")
    return qualname.startswith(f"{class_name}.")


def _unwrap_descriptor(descriptor: Any) -> Any:
    if isinstance(descriptor, classmethod | staticmethod):
        return descriptor.__func__
    if isinstance(descriptor, property):
        return descriptor.fget or descriptor
    return descriptor


def _format_signature(obj: Any) -> str:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return "()"
    rendered = []
    for parameter in signature.parameters.values():
        text = parameter.name
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            text = f"*{text}"
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            text = f"**{text}"
        if parameter.annotation is not inspect._empty:
            text += f": {parameter.annotation!r}"
        if parameter.default is not inspect._empty:
            text += f" = {_format_default(parameter.default)}"
        rendered.append(text)
    result = ", ".join(rendered)
    if signature.return_annotation is not inspect._empty:
        return f"({result}) -> {signature.return_annotation!r}"
    return f"({result})"


def _format_default(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, int | float | bool) or value is None:
        return repr(value)
    if inspect.isfunction(value) or inspect.ismethod(value):
        return value.__name__
    name = getattr(value, "__name__", None)
    if isinstance(name, str):
        return name
    return f"<{type(value).__name__}>"


def main(argv: list[str] | None = None) -> int:
    """Write the generated API reference to stdout or a file."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Defaults to stdout.",
    )
    args = parser.parse_args(argv)
    output = build_reference_markdown()
    if args.output is None:
        sys.stdout.write(output)
        return 0
    args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
