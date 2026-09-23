# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
import torch

from tests.utils import (
    ensure_current_vllm_config,
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.config.lora import LoRAConfig
from vllm.lora.utils import from_layer
from vllm.model_executor.layers.linear import ColumnParallelLinear


@ray.remote(num_gpus=1, max_calls=1)
@torch.inference_mode()
def check_mla_lora_shards(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    torch.accelerator.set_device_index(rank)
    with ensure_current_vllm_config(), torch.device(f"cuda:{rank}"):
        init_test_distributed_environment(
            tp_size, pp_size, rank, distributed_init_port, local_rank=rank
        )
        a = (
            (torch.arange(1, 9, dtype=torch.bfloat16)[:, None] / 8)
            .expand(8, 64)
            .contiguous()
        )
        blocks = torch.arange(1, 9, dtype=torch.bfloat16).repeat_interleave(32)
        x = torch.ones((6, 1, 64), dtype=torch.bfloat16)
        mapping = torch.tensor([0, 1, -1, 1, 0, -1])
        for fully_sharded in (False, True):
            config = LoRAConfig(
                max_loras=2,
                max_lora_rank=8,
                lora_dtype=torch.bfloat16,
                fully_sharded_loras=fully_sharded,
            )
            base = ColumnParallelLinear(
                64, 256, bias=False, gather_output=False, params_dtype=torch.bfloat16
            )
            base.weight.zero_()
            layer = from_layer(base, 2, config, [])
            assert layer.lora_a_stacked[0].shape[2] == (1 if fully_sharded else 8)
            for value in (0.125, 0.25):
                expected = torch.zeros((6, 32), dtype=torch.bfloat16)
                for slot in range(2):
                    b = (
                        blocks[:, None]
                        * torch.arange(1, 9, dtype=torch.bfloat16)[None, :]
                        * value
                        * (slot + 1)
                    ).contiguous()
                    layer.set_lora(slot, a, b)
                    local_b = b[rank * 32 : (rank + 1) * 32]
                    expected[mapping == slot] = (
                        x[mapping == slot, 0].float() @ a.float().T @ local_b.float().T
                    ).to(torch.bfloat16)
                output = base(x.squeeze(1))[0]
                layer.apply_mla_kv_b_lora_linear(x, output, mapping)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)


@multi_gpu_test(num_gpus=8)
def test_mla_lora_tp8_adapter_replacement(monkeypatch: pytest.MonkeyPatch):
    """Cached-token projection uses the correct A/B shards after adapter replacement."""
    multi_process_parallel(monkeypatch, 8, 1, check_mla_lora_shards)
