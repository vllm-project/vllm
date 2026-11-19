# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ParallelConfig, VllmConfig
from vllm.models.deepseek_v41.nvidia.specialized_decode import (
    decode_backend,
    specialized_decode_config,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.dep2.config import (
    supports_layout,
)


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("explicit_size", [False, True])
def test_dep_layout(size, explicit_size):
    parallel = ParallelConfig(
        data_parallel_size=size,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
    )
    config = VllmConfig(
        parallel_config=parallel,
        additional_config={
            "decode_backend": f"latency_specialized_dep{size}"
            if explicit_size
            else "latency_specialized_dep"
        },
    )
    assert decode_backend(config) == f"latency_specialized_dep{size}"
    assert supports_layout(parallel, size)
    assert not supports_layout(parallel, 4 if size == 2 else 2)
    parallel.enable_expert_parallel = False
    assert not supports_layout(parallel, size)


def test_dep_is_opt_in():
    config = VllmConfig()
    assert specialized_decode_config(config) is None
    config.additional_config = {"decode_backend": "latency_specialized_dep3"}
    with pytest.raises(ValueError, match="decode_backend"):
        specialized_decode_config(config)
    config.additional_config = {"decode_backend": "latency_specialized_dep"}
    with pytest.raises(ValueError, match="requires DP2, DP4 or DP8"):
        specialized_decode_config(config)
