# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source digest for the Qwen3.8-27B dense mono kernels' JIT cache key."""

import functools
import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


@functools.cache
def digest() -> str:
    h = hashlib.sha256()
    for f in sorted(_ROOT.rglob("*.py")):
        h.update(str(f.relative_to(_ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
