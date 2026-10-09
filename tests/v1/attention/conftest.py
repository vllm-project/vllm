# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed import parallel_state


@pytest.fixture
def single_rank_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide TP rank metadata without initializing distributed."""
    monkeypatch.setattr(
        parallel_state,
        "_TP",
        SimpleNamespace(rank_in_group=0, world_size=1),
    )
