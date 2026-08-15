# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for https://github.com/vllm-project/vllm/issues/52435.

DeepseekV4MoE crashes deep inside every worker process with a bare
``AssertionError`` when the number of physical experts (routed + redundant)
isn't evenly divisible by the expert/tensor parallel size — e.g. 256 routed
experts with ``--tensor-parallel-size 3 --enable-expert-parallel`` and no
redundant experts configured. Bare asserts are stripped under ``python -O``
and don't match the ``ValueError``-based validation convention used
elsewhere for user-facing config errors (see vllm/config/parallel.py), so
this should raise ``ValueError`` instead.

These tests exercise the private ``_init_mega_moe_experts`` /
``_init_fused_moe_experts`` helpers directly (constructing the module via
``__new__`` to avoid needing a real distributed process group), since the
existing tests/models/test_deepseek_v4_mega_moe.py suite is CUDA-only.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import model as deepseek_v4_model


class _FakeEpGroup:
    def __init__(self, world_size: int, rank_in_group: int = 0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group


def _make_moe() -> deepseek_v4_model.DeepseekV4MoE:
    moe = deepseek_v4_model.DeepseekV4MoE.__new__(deepseek_v4_model.DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    return moe


@pytest.mark.cpu_test
def test_mega_moe_experts_raises_value_error_when_not_divisible_by_ep_size(
    monkeypatch,
):
    monkeypatch.setattr(
        deepseek_v4_model, "get_ep_group", lambda: _FakeEpGroup(world_size=3)
    )

    moe = _make_moe()
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        )
    )
    config = SimpleNamespace(n_routed_experts=256, n_shared_experts=None)

    with pytest.raises(ValueError, match="n_physical_experts=256"):
        moe._init_mega_moe_experts(vllm_config, config, prefix="layers.0.ffn")


@pytest.mark.cpu_test
def test_fused_moe_experts_raises_value_error_when_not_divisible_by_tp_size(
    monkeypatch,
):
    monkeypatch.setattr(
        deepseek_v4_model, "get_tensor_model_parallel_rank", lambda: 0
    )

    moe = _make_moe()
    moe.tp_size = 3
    moe.n_routed_experts = 256

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        )
    )
    config = SimpleNamespace(n_shared_experts=None)

    with pytest.raises(ValueError, match="n_physical_experts=256"):
        moe._init_fused_moe_experts(
            vllm_config, config, quant_config=None, prefix="layers.0.ffn"
        )
