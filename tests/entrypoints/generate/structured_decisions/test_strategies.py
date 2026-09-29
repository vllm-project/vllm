# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

from vllm.entrypoints.generate.structured_decisions.strategies import (
    NextTokenStrategy,
    select_read_strategy,
)


def model_naming(strategy: str) -> Any:
    return SimpleNamespace(_model_info=SimpleNamespace(decision_read_strategy=strategy))


def test_strategy_selection():
    assert select_read_strategy(model_naming("next_token")) is NextTokenStrategy
    assert select_read_strategy(model_naming("not_registered")) is None
