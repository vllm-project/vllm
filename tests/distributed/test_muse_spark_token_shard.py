# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x ROCm token-sharded residual stream: row ownership and collectives."""

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MuseSpark1x ROCm model path"
)


@ray.remote(num_gpus=1, max_calls=1)
def token_shard_round_trip(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, distributed_init_port
) -> None:
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        from vllm.models.muse_spark.amd.model import TokenShard

        for num_tokens in (1, 5, 16, 1027):
            full = torch.arange(num_tokens * 64, device=device) % 61
            full = full.view(num_tokens, 64).to(torch.bfloat16)
            shard = TokenShard(num_tokens)
            local = shard.local(full)
            assert local.shape == (shard.rows, 64)
            owned = full[shard.start : shard.start + shard.rows]
            torch.testing.assert_close(local[: owned.shape[0]], owned, atol=0, rtol=0)
            assert not local[owned.shape[0] :].any()
            torch.testing.assert_close(shard.gather(local), full, atol=0, rtol=0)
            summed = shard.reduce_scatter(full * (rank + 1))
            expected = shard.local(full * sum(range(1, tp_size + 1)))
            torch.testing.assert_close(summed, expected)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("tp_size", [2])
def test_muse_spark_token_shard(monkeypatch: pytest.MonkeyPatch, tp_size: int) -> None:
    multi_process_parallel(monkeypatch, tp_size, 1, token_shard_round_trip)
