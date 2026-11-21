# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Publish committed expert placements."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_placement_snapshot(path: str, snapshot: dict[str, Any]) -> None:
    """Atomically replace a JSON snapshot on the destination filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(snapshot, stream)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
