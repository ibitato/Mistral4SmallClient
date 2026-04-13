"""Module entrypoint for `python -m mistral4cli`."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
