# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
def test_deepseek_v41_dspark_does_not_inherit_target_eplb():
    from vllm.models.deepseek_v4_1.nvidia.dspark import _get_dspark_vllm_config

    target_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_eplb=True,
            eplb_config=SimpleNamespace(num_redundant_experts=32),
        )
    )

    draft_config = _get_dspark_vllm_config(target_config)

    assert target_config.parallel_config.enable_eplb
    assert target_config.parallel_config.eplb_config.num_redundant_experts == 32
    assert draft_config is not target_config
    assert not draft_config.parallel_config.enable_eplb
    assert draft_config.parallel_config.eplb_config.num_redundant_experts == 0
