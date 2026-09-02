# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether a recorded frame counts as use, and under which reading.

The record's evidence is "did this job enter this function". Module and class
bodies are recorded for every file a job imports, and `<module>` is in every
(row, file) pair there is, so the plain reading degenerates to "did this job
import the file". That is true of nearly everything, and it is why the drop
side refuses two thirds of its decisions.

Requiring a real call frame instead is `PhaseMode`. It lives in its own module
because the keep check and `look_up` re-make the same match one check apart:
changing one alone moves zero decisions and reads as "the stricter mode buys
nothing". Both import from here so they cannot drift.

The predicate takes a `FileQuery`, never the whole `Query`. The keep check reads
the whole diff and `look_up` reads a scoped subset, so a predicate that depends
only on per-file state keeps the two in agreement and one that depends on the
query would not.
"""

from __future__ import annotations

import os
from enum import Enum

from .changed_funcs import FileQuery
from .model import Row

ENV_VAR = "CI_SELECTOR_PHASE_DROP"


class PhaseMode(str, Enum):
    """How much a recorded frame has to be before it blocks a drop.

    OFF     an import counts as use. The original reading, kept as a
            measurement baseline. NOT the default since 2026-08-29.
    CARVED  the SHIPPED DEFAULT. A call frame is required, except on a file
            whose changed names are
            ALL import-time. Nothing in such a file could have been called,
            and an importer genuinely breaks on import errors, class-body
            attributes, decorator evaluation and registry side effects.
    STRICT  a call frame is always required. Measurement only: on a file with
            no callable changed name no row can match, so the verdict is a
            constant and every step drops. A ceiling, not a rule.

    Per FILE, not per name, and that distinction is the whole design. A
    per-name carve-out is a no-op: `import_time_names` always yields
    `<module>`, so `<module>` is in `FileQuery.import_time` whenever it is in
    `names`, and the carve-out could never deny the one name that matters.
    """

    OFF = "off"
    CARVED = "carved"
    STRICT = "strict"


#: What every entry point uses when the caller says nothing. One constant, so
#: "the default" cannot mean two different things in two different signatures.
DEFAULT_MODE = PhaseMode.CARVED


def row_shows_use(row: Row, changed: FileQuery, name: str, mode: PhaseMode) -> bool:
    """Whether this row's record of `changed.path` counts as running `name`."""
    if mode is PhaseMode.OFF:
        return row.contains(changed.path, name)
    if mode is PhaseMode.CARVED and not changed.function_names:
        return row.contains(changed.path, name)
    return row.contains_call(changed.path, name)


def mode_from_env() -> PhaseMode:
    """The mode this process runs in. Unset is `DEFAULT_MODE`.

    Raises on anything else rather than falling back. `decide` catches broadly
    and degrades to map-only, so a misspelled value that fell back would run
    every PR through the map alone and read as "the mode does nothing".
    """
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return DEFAULT_MODE
    try:
        return PhaseMode(raw)
    except ValueError:
        allowed = ", ".join(m.value for m in PhaseMode)
        raise ValueError(f"{ENV_VAR}={raw!r}, expected one of: {allowed}") from None
