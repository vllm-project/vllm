import types
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class PatchApplicationError(Exception):
    """match text not found or not unique in source."""


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EditSpec(_StrictBase):
    """Specify one edit: replace, prepend before, or append after the matched text.

    Use ``replacement`` to substitute the matched text (empty string = delete).
    Use ``prepend`` to keep the matched text and add lines before it.
    Use ``append`` to keep the matched text and add lines after it.
    Only one of ``replacement``, ``prepend``, and ``append`` may be set.
    """

    match: str
    replacement: str = ""
    prepend: str = ""
    append: str = ""

    @model_validator(mode="after")
    def _check_modes_mutually_exclusive(self) -> "EditSpec":
        active: list[str] = [
            name
            for name in ("replacement", "prepend", "append")
            if getattr(self, name).strip()
        ]
        if len(active) > 1:
            raise ValueError(
                f"only one of 'replacement', 'prepend', 'append' may be set, "
                f"got: {', '.join(active)}"
            )
        return self


class PatchSpec(_StrictBase):
    target: str
    edits: list[EditSpec]
    preamble: str = ""


class PatchConfig(_StrictBase):
    patches: list[PatchSpec]


class PatchState:
    def __init__(
        self, *, target_fn: Callable[..., Any], original_code: types.CodeType
    ) -> None:
        self.target_fn = target_fn
        self.original_code = original_code

    def restore(self) -> None:
        self.target_fn.__code__ = self.original_code
