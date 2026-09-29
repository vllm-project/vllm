# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.entrypoints.generate.structured_decisions.strategies import (
    NextTokenStrategy,
    select_read_strategy,
)


def test_strategy_selection():
    assert select_read_strategy(SimpleNamespace(is_diffusion=False)) is (
        NextTokenStrategy
    )
    assert select_read_strategy(SimpleNamespace(is_diffusion=True)) is None
