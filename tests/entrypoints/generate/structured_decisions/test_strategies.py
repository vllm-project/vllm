# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

from vllm.entrypoints.generate.structured_decisions.strategies import (
    NextTokenStrategy,
    select_read_strategy,
)


def model(is_diffusion: bool) -> Any:
    return SimpleNamespace(is_diffusion=is_diffusion)


def test_strategy_selection():
    assert select_read_strategy(model(False)) is NextTokenStrategy
    assert select_read_strategy(model(True)) is None
