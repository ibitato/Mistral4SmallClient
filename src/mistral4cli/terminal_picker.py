"""Terminal-native file picker helpers for attachment commands."""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any, TextIO

TERMINAL_PICKER_MAX_CANDIDATES = 1_500
TERMINAL_PICKER_HEIGHT = "70%"
GO_TO_PARENT_VALUE = "__mistral4cli_go_to_parent__"
FINISH_SELECTION_VALUE = "__mistral4cli_finish_selection__"
SELECT_CURRENT_DIRECTORY_VALUE = "__mistral4cli_select_current_directory__"


class TerminalPickerUnavailableError(RuntimeError):
    """Raised when the terminal picker cannot run in the current environment."""


def pick_paths_in_terminal(
    *,
    kind: str,
    suffixes: Collection[str],
    stdout: TextIO,
    multiple: bool = True,
    start_dir: Path | None = None,
) -> list[Path]:
    """Pick one or more files in a pure terminal flow."""

    inquirer, get_style, path_validator = _load_inquirer_components()
    style = get_style(
        {
            "questionmark": "#ff8700 bold",
            "question": "#5fff00 bold",
            "pointer": "#ff8700 bold",
            "marker": "#ff8700 bold",
            "answer": "#5fff00",
            "instruction": "#ffaf00",
            "long_instruction": "#ffaf00",
            "input": "#5fff00",
            "fuzzy_prompt": "#ff8700 bold",
            "fuzzy_info": "#ffaf00",
            "separator": "#ff8700",
        },
        style_override=False,
    )
    root = _prompt_root_directory(
        kind=kind,
        start_dir=start_dir or Path.cwd(),
        inquirer=inquirer,
        path_validator=path_validator,
        style=style,
    )
    if root is None:
        return []

    current_root = root
    selected_paths: list[Path] = []
    while True:
        candidates, truncated = _discover_matching_files(
            root=current_root,
            suffixes=suffixes,
            max_candidates=TERMINAL_PICKER_MAX_CANDIDATES,
        )
        if truncated:
            stdout.write(
                f"[{kind}] too many matching files under {current_root} "
                f"(>{TERMINAL_PICKER_MAX_CANDIDATES}); choose a narrower directory.\n"
            )
            stdout.flush()
            return []
        available_candidates = [
            path for path in candidates if path not in selected_paths
        ]
        if not available_candidates:
            if selected_paths:
                return selected_paths
            stdout.write(f"[{kind}] no supported files found under {current_root}.\n")
            stdout.flush()
            return []

        action, selected = _prompt_candidate_selection(
            kind=kind,
            root=current_root,
            candidates=available_candidates,
            multiple=multiple,
            selected_paths=selected_paths,
            inquirer=inquirer,
            style=style,
        )
        if action == "cancel":
            return []
        if action == "finish":
            return selected_paths
        if action == "parent":
            parent = current_root.parent
            if parent == current_root:
                stdout.write(f"[{kind}] already at the filesystem root.\n")
                stdout.flush()
                continue
            current_root = parent
            continue
        if selected:
            if multiple:
                selected_paths.extend(
                    path for path in selected if path not in selected_paths
                )
                continue
            return selected


def _load_inquirer_components() -> tuple[Any, Any, Any]:
    try:
        from InquirerPy import inquirer as inquirer_module
        from InquirerPy.utils import get_style
        from InquirerPy.validator import PathValidator
    except Exception as exc:  # pragma: no cover - exercised through fallback path
        raise TerminalPickerUnavailableError(
            "terminal picker dependencies are unavailable"
        ) from exc
    return inquirer_module, get_style, PathValidator


def _prompt_root_directory(
    *,
    kind: str,
    start_dir: Path,
    inquirer: Any,
    path_validator: Any,
    style: Any,
) -> Path | None:
    del path_validator
    current_dir = start_dir.expanduser().resolve()
    while True:
        child_directories = [
            path
            for path in sorted(
                current_dir.iterdir(), key=lambda path: path.name.lower()
            )
            if path.is_dir() and not path.name.startswith(".")
        ]
        choices = [
            {
                "name": f"[use] Select this directory: {current_dir}",
                "value": SELECT_CURRENT_DIRECTORY_VALUE,
            }
        ]
        if current_dir.parent != current_dir:
            choices.append(
                {"name": "[..] Go to parent directory", "value": GO_TO_PARENT_VALUE}
            )
        choices.extend(
            {"name": f"[dir] {path.name}", "value": str(path)}
            for path in child_directories
        )
        try:
            selection = inquirer.fuzzy(
                message=f"Choose the {kind} search directory:",
                choices=choices,
                multiselect=False,
                instruction="Type to filter directories",
                long_instruction=(
                    "Enter selects the highlighted directory. Use [use] to keep "
                    "the current directory, [..] to go to the parent, and Ctrl-C "
                    "to cancel."
                ),
                style=style,
                pointer=">",
                marker="*",
                marker_pl=".",
                max_height=TERMINAL_PICKER_HEIGHT,
                cycle=True,
                match_exact=False,
                keybindings={"toggle-exact": [{"key": "c-t"}]},
                raise_keyboard_interrupt=True,
            ).execute()
        except KeyboardInterrupt:
            return None
        if not selection:
            return None
        if selection == SELECT_CURRENT_DIRECTORY_VALUE:
            return current_dir
        if selection == GO_TO_PARENT_VALUE:
            current_dir = current_dir.parent
            continue
        current_dir = Path(str(selection)).expanduser().resolve()


def _discover_matching_files(
    *,
    root: Path,
    suffixes: Collection[str],
    max_candidates: int,
) -> tuple[list[Path], bool]:
    normalized_suffixes = {suffix.lower() for suffix in suffixes}
    matches: list[Path] = []

    for current_root, dirs, files in os.walk(root):
        dirs[:] = sorted(
            directory for directory in dirs if not directory.startswith(".")
        )
        base = Path(current_root)
        for filename in sorted(files):
            if filename.startswith("."):
                continue
            path = base / filename
            if path.suffix.lower() not in normalized_suffixes:
                continue
            matches.append(path)
            if len(matches) > max_candidates:
                return matches[:max_candidates], True
    return matches, False


def _prompt_candidate_selection(
    *,
    kind: str,
    root: Path,
    candidates: list[Path],
    multiple: bool,
    selected_paths: list[Path],
    inquirer: Any,
    style: Any,
) -> tuple[str, list[Path]]:
    choices = [{"name": "[..] Go to parent directory", "value": GO_TO_PARENT_VALUE}]
    if multiple and selected_paths:
        choices.append(
            {
                "name": f"[done] Finish selection ({len(selected_paths)} selected)",
                "value": FINISH_SELECTION_VALUE,
            }
        )
    choices.extend(
        {"name": str(path.relative_to(root)), "value": str(path)} for path in candidates
    )
    try:
        selection = inquirer.fuzzy(
            message=f"Select {kind} files:",
            choices=choices,
            multiselect=False,
            instruction="Type to filter",
            long_instruction=(
                "Use arrows to move, Enter to select the highlighted file, Ctrl-C "
                "to cancel, and [..] to go to parent."
                if not multiple
                else "Use arrows to move, Enter to add the highlighted file, Ctrl-C "
                "to cancel, [..] to go to parent, and [done] to finish."
            ),
            style=style,
            pointer=">",
            marker="*",
            marker_pl=".",
            max_height=TERMINAL_PICKER_HEIGHT,
            cycle=True,
            match_exact=False,
            keybindings={"toggle-exact": [{"key": "c-t"}]},
            raise_keyboard_interrupt=True,
        ).execute()
    except KeyboardInterrupt:
        return "cancel", []

    if not selection:
        return "cancel", []
    if selection == GO_TO_PARENT_VALUE:
        return "parent", []
    if selection == FINISH_SELECTION_VALUE:
        return "finish", []
    return "selected", [Path(str(selection)).expanduser()]
