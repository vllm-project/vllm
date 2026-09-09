# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In pool mode the NVFP4 method never advertises shared-expert overlap, so
the runner runs the shared experts itself and the pool ignores the wrapper
it is handed (the synchronous modular path does the same)."""

from types import SimpleNamespace

from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE


def _method(pool_mode: bool):
    m = ModelOptNvFp4FusedMoE.__new__(ModelOptNvFp4FusedMoE)
    m._pool_mode = pool_mode
    m.moe_kernel = SimpleNamespace(can_overlap_shared_experts=True)
    return m


def test_pool_mode_disables_shared_expert_overlap():
    assert _method(pool_mode=True).mk_can_overlap_shared_experts is False
    assert _method(pool_mode=False).mk_can_overlap_shared_experts is True
