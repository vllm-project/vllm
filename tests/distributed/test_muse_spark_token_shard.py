# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x ROCm token-sharded residual stream: row ownership and collectives."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tensor_model_parallel_rank,
)
from vllm.platforms import current_platform

from ..utils import multi_gpu_test
from .eplb_utils import distributed_run, set_env_vars_and_device

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MuseSpark1x ROCm model path"
)


def token_shard_round_trip(env: dict[str, str], world_size: int, num_tokens: int):
    set_env_vars_and_device(env)
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(world_size, 1)
    from vllm.models.muse_spark.amd.model import TokenShard

    rank = get_tensor_model_parallel_rank()
    full = torch.arange(num_tokens * 64, device="cuda") % 61
    full = full.view(num_tokens, 64).to(torch.bfloat16)
    shard = TokenShard(num_tokens)
    local = shard.local(full)
    assert local.shape == (shard.rows, 64)
    owned = full[shard.start : shard.start + shard.rows]
    torch.testing.assert_close(local[: owned.shape[0]], owned, atol=0, rtol=0)
    assert not local[owned.shape[0] :].any()
    torch.testing.assert_close(shard.gather(local), full, atol=0, rtol=0)
    summed = shard.reduce_scatter(full * (rank + 1))
    expected = shard.local(full * sum(range(1, world_size + 1)))
    torch.testing.assert_close(summed, expected)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("num_tokens", [1, 5, 16, 1027])
def test_muse_spark_token_shard(num_tokens: int) -> None:
    distributed_run(token_shard_round_trip, 2, num_tokens)
