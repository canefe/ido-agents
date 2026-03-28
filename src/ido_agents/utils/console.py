from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rich.console import Console

_console: Optional["Console"] = None


def get_console() -> Optional["Console"]:
    global _console
    if _console is not None:
        return _console
    try:
        from rich.console import Console
    except ImportError:
        return None

    _console = Console(stderr=True, force_terminal=True)
    return _console


def console_print(message: str) -> None:
    console = get_console()
    if console is None:
        print(message, file=sys.stderr, flush=True)
    else:
        console.print(message)
