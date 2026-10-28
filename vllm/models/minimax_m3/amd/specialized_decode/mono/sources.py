# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This package's kernel sources and the DS V4.1 primitives they import, in
every build's JIT cache key: an edit to either rebuilds."""

import hashlib
from pathlib import Path

import vllm.models.deepseek_v41.amd.mono.common as ds_common


def _digest() -> str:
    h = hashlib.sha256()
    roots = (Path(__file__).parent, Path(ds_common.__file__).parent)
    for root in roots:
        for f in sorted(root.glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


SOURCES = _digest()
