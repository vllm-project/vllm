# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused vocab-parallel logits gather equals a last-dim all-gather, the
vocab slice and the dtype cast, bitwise."""

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables


def _worker(local_rank: int, world_size: int, port: int):
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.model_executor.layers.logits_processor import (
        all_gather_vocab_parallel_logits,
    )

    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(config):
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
            }
        )
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        shard_size = 3001
        for in_dtype in (torch.bfloat16, torch.float16, torch.float32):
            for num_rows in (0, 1, 7, 64, 300):
                # The last shard ends in vocab padding.
                for vocab_size in (
                    world_size * shard_size,
                    world_size * shard_size - 37,
                ):
                    g = torch.Generator(device=device).manual_seed(
                        local_rank + num_rows
                    )
                    logits = torch.randn(
                        num_rows, shard_size, device=device, generator=g
                    ).to(in_dtype)
                    ref = tensor_model_parallel_all_gather(logits, dim=-1)
                    ref = ref[:, :vocab_size]
                    for out_dtype in (in_dtype, torch.float32):
                        out = all_gather_vocab_parallel_logits(
                            logits, vocab_size, out_dtype
                        )
                        assert out.is_contiguous()
                        assert out.dtype == out_dtype
                        assert torch.equal(out, ref.to(out_dtype))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("world_size", [2, 4])
def test_vocab_parallel_logits_gather(world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    try:
        mp.spawn(_worker, args=(world_size, get_open_port()), nprocs=world_size)
    finally:
        cleanup_dist_env_and_memory()
