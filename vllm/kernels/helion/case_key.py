# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Structured key for identifying kernel config/autotune/benchmark cases.
"""

from __future__ import annotations

import json
from typing import Any


class CaseKey(dict[str, Any]):
    """Immutable, hashable dict for identifying kernel cases.

    Used as the key for config lookup, autotuning, benchmarking, and
    input generation.  Behaves like a read-only dict and can be used
    as a dict key or in sets.

    The canonical string form (``__str__``) is stable JSON with sorted
    keys.  Use ``CaseKey.default()`` for the default/fallback key.
    The regular constructor requires at least one key-value pair::

        CaseKey({"intermediate": 2048, "numtokens": 256})
        CaseKey.default()  # default/fallback
    """

    def __init__(self, *args: Any, _allow_empty: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not self and not _allow_empty:
            raise TypeError(
                "CaseKey requires at least one key-value pair. "
                "Use CaseKey.default() for the default config key."
            )
        self._str: str | None = None
        self._hash: int | None = None

    @classmethod
    def default(cls) -> CaseKey:
        """Create a default case key (empty)."""
        return cls(_allow_empty=True)

    def __hash__(self) -> int:  # type: ignore[override]
        if self._hash is None:
            self._hash = hash(str(self))
        return self._hash

    def __str__(self) -> str:
        if self._str is None:
            self._str = json.dumps(dict(self), sort_keys=True, separators=(",", ":"))
        return self._str

    def __repr__(self) -> str:
        if not self:
            return "CaseKey.default()"
        return f"CaseKey({dict(self)})"

    def is_default(self) -> bool:
        """Return True if this is the default case key (empty)."""
        return not self

    def _readonly(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("CaseKey is immutable")

    __setitem__ = _readonly  # type: ignore[assignment]
    __delitem__ = _readonly  # type: ignore[assignment]
    __ior__ = _readonly  # type: ignore[assignment]
    update = _readonly  # type: ignore[assignment]
    pop = _readonly  # type: ignore[assignment]
    popitem = _readonly  # type: ignore[assignment]
    setdefault = _readonly  # type: ignore[assignment]
    clear = _readonly  # type: ignore[assignment]
