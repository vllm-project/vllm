# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Flash-Next mono kernels' source digest: every ``.py`` under this package,
the Qwen3.8 mono files they import stages from, and the Kimi-K3 mono framework
files, so an edit to any reaches every build's JIT cache key."""

import functools
import hashlib
from pathlib import Path

from vllm.models.kimi_k3.amd.mono.common.plan import source_digest
from vllm.models.qwen3_5.amd.mono.sources import digest as qwen38_digest

_ROOT = Path(__file__).resolve().parent


@functools.cache
def digest() -> str:
    h = hashlib.sha256(source_digest("common", "stages/gemv.py").encode())
    h.update(qwen38_digest().encode())
    for f in sorted(_ROOT.rglob("*.py")):
        h.update(str(f.relative_to(_ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
