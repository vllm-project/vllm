# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode tests need SM100 multi-GPU nodes and real-rank setups
that CI does not provide. They are collected only when `specialized_decode` (or
a path inside it) is passed to pytest explicitly, e.g.
`pytest tests/models/kimi_k3/nvidia/specialized_decode`; a run over
`tests/models/kimi_k3` skips them."""

from pathlib import Path

import pytest

_SPECIALIZED_DECODE = Path(__file__).parent / "specialized_decode"


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if collection_path != _SPECIALIZED_DECODE:
        return None
    root = config.invocation_params.dir
    for arg in config.args:
        target = (root / arg.split("::")[0]).resolve()
        if target == _SPECIALIZED_DECODE or _SPECIALIZED_DECODE in target.parents:
            return None
    return True
