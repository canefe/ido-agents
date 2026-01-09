from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rich.console import Console


def get_console() -> Optional["Console"]:
    try:
        from rich.console import Console
    except ImportError:
        return None

    return Console()


def console_print(message: str) -> None:
    console = get_console()
    if console is None:
        print(message)
    else:
        console.print(message)
