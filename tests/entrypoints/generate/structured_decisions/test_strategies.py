# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.entrypoints.generate.structured_decisions.strategies import (
    NextTokenStrategy,
    select_read_strategy,
)
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS


def test_autoregressive_models_read_next_token():
    assert select_read_strategy(SimpleNamespace(is_diffusion=False)) is (
        NextTokenStrategy
    )


def test_diffusion_models_get_no_strategy_yet():
    assert select_read_strategy(SimpleNamespace(is_diffusion=True)) is None


def test_next_token_limits():
    limits = NextTokenStrategy.limits(SimpleNamespace(is_diffusion=False))
    assert limits.max_choices == MAX_LOGPROB_TOKEN_IDS
    assert limits.max_questions == 64
