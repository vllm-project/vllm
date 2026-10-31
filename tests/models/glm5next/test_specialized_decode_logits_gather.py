# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The multicast LM head logits all-gather against the concatenated shards."""

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ...utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

COLUMNS = 38720  # GLM-5.3-Flash's vocabulary shard at TP4


@ray.remote(num_gpus=1, max_calls=1)
def _gather(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.models.glm5next.nvidia.specialized_decode.logits_gather import (
        LogitsGather,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        op = LogitsGather(columns=COLUMNS, capacity=64, device=device)
        g = torch.Generator(device=device).manual_seed(0)

        def logits(rows):
            full = torch.randn(rows, tp_size * COLUMNS, generator=g, device=device)
            full = full.bfloat16()
            # Bit patterns pass through untouched.
            full[0, :4] = torch.tensor([float("nan"), float("-inf"), -0.0, 0.0])
            shard = full[:, rank * COLUMNS : (rank + 1) * COLUMNS].contiguous()
            return full, shard

        for rows in (1, 3, 64, 16, 7, 64):
            full, shard = logits(rows)
            out = op(shard)
            assert out.shape == full.shape
            assert torch.equal(out.view(torch.int16), full.view(torch.int16))
        # Two gathers in flight: the second takes the other buffer.
        (a, sa), (b, sb) = logits(64), logits(32)
        out_a, out_b = op(sa), op(sb)
        assert torch.equal(out_a.view(torch.int16), a.view(torch.int16))
        assert torch.equal(out_b.view(torch.int16), b.view(torch.int16))


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)
def test_logits_gather_matches_concatenated_shards(monkeypatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _gather)
