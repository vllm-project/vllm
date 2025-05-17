# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Callable


def _lazy_load_template(template_path: Path) -> str:
    with open(template_path) as f:
        return f.read()


CHAT_TEMPLATES: dict[str, Callable[[], str]] = {
    template.stem: lambda p=template: _lazy_load_template(p)
    for template in Path(__file__).parent.glob("*.jinja")
}


def __getattr__(name: str) -> str:
    if name in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[name]()
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return list(CHAT_TEMPLATES.keys())


__all__ = list(CHAT_TEMPLATES.keys())
