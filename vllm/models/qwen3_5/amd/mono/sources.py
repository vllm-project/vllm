# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8 mono kernels' source digest: every ``.py`` under this package
and the Kimi-K3 mono framework files they are built from, so an edit to
either reaches every build's JIT cache key."""

import functools
import hashlib
from pathlib import Path

from vllm.models.kimi_k3.amd.mono.common.plan import source_digest

_ROOT = Path(__file__).resolve().parent


@functools.cache
def digest() -> str:
    h = hashlib.sha256(source_digest("common", "stages/gemv.py").encode())
    for f in sorted(_ROOT.rglob("*.py")):
        h.update(str(f.relative_to(_ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
