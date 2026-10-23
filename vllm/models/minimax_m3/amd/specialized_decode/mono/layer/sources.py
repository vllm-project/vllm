# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All full-layer kernel sources participate in its compiled build key."""

import hashlib
from pathlib import Path


def _digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        if source.name in ("step.py", "weights.py"):
            continue
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


SOURCES = _digest()
