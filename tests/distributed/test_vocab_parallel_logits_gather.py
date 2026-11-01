# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused vocab-parallel logits gather equals a last-dim all-gather, the
vocab slice and the dtype cast, bitwise, and LogitsProcessor uses it only
where it reproduces the unfused path."""

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

        with pytest.raises(ValueError):
            all_gather_vocab_parallel_logits(
                logits, world_size * shard_size + 1, torch.float32
            )

        _check_logits_processor(world_size, device)


def _check_logits_processor(world_size: int, device: torch.device):
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    class UnfusedLogitsProcessor(LogitsProcessor):
        # Overriding _gather_logits (as a custom all-gather does) keeps the
        # unfused path: gather, slice, then scale / soft cap in the head dtype.
        def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
            return super()._gather_logits(logits)

    vocab_size, hidden_size = 1000 * world_size - 37, 64
    with torch.device(device):
        lm_head = ParallelLMHead(vocab_size, hidden_size, params_dtype=torch.bfloat16)
    lm_head.weight.data.normal_(generator=torch.Generator(device).manual_seed(7))
    hidden_states = torch.randn(
        33, hidden_size, device=device, generator=torch.Generator(device).manual_seed(0)
    ).bfloat16()
    for scale, soft_cap in ((1.0, None), (0.5, None), (1.0, 30.0)):
        for gather_out_dtype in (None, torch.float32):
            processors = [
                cls(vocab_size, scale=scale, soft_cap=soft_cap)
                for cls in (LogitsProcessor, UnfusedLogitsProcessor)
            ]
            for processor in processors:
                processor.gather_out_dtype = gather_out_dtype
            fused, unfused = (p(lm_head, hidden_states) for p in processors)
            assert unfused.dtype == torch.bfloat16
            # FP32 only where nothing runs on the logits after the gather.
            upcast = gather_out_dtype is not None and scale == 1.0 and not soft_cap
            assert fused.dtype == (torch.float32 if upcast else torch.bfloat16)
            assert torch.equal(fused, unfused.to(fused.dtype))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("world_size", [2, 4])
def test_vocab_parallel_logits_gather(world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    try:
        mp.spawn(_worker, args=(world_size, get_open_port()), nprocs=world_size)
    finally:
        cleanup_dist_env_and_memory()
