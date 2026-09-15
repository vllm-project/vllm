# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A dense drafter must not inherit the target's ``enable_expert_parallel``.

``create_draft_parallel_config`` copies the target's EP flag so that a MoE
drafter which requires expert parallelism (DeepSeek-V4 MegaMoE, #55914) gets
it. Copied unconditionally it also reaches drafters with no experts at all,
and ``SpeculativeConfig.__post_init__`` then runs
``draft_model_config.verify_with_parallel_config(draft_parallel_config)``,
which rejects them with "Number of experts in the model must be greater than
0 when expert parallelism is enabled" before any weight is loaded.
"""

from types import SimpleNamespace

import pytest

from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig


def _draft_ep(target_ep: bool, draft_model_config) -> bool:
    target = ParallelConfig(tensor_parallel_size=2, enable_expert_parallel=target_ep)
    draft = SpeculativeConfig.create_draft_parallel_config(
        target, 2, draft_model_config
    )
    return draft.enable_expert_parallel


@pytest.mark.parametrize("target_ep", [True, False])
def test_dense_draft_never_inherits_expert_parallel(target_ep: bool):
    """A drafter with no experts has nothing to shard."""
    assert _draft_ep(target_ep, SimpleNamespace(is_moe=False)) is False


def test_moe_draft_still_inherits_expert_parallel():
    """Preserves #55914: a MoE drafter must keep the target's EP flag."""
    assert _draft_ep(True, SimpleNamespace(is_moe=True)) is True


def test_moe_draft_does_not_gain_expert_parallel():
    """EP is inherited, never invented, when the target has it disabled."""
    assert _draft_ep(False, SimpleNamespace(is_moe=True)) is False


def test_unknown_draft_model_config_inherits():
    """Without a draft model config the previous behaviour is kept."""
    assert _draft_ep(True, None) is True
