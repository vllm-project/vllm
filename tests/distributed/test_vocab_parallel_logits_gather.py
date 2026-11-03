# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused vocab-parallel logits gather equals a last-dim all-gather, the
vocab slice and the dtype cast, bitwise, and LogitsProcessor uses it only
where it reproduces the unfused path. The model runner asks it for FP32 only
for a model that opts in with SupportsFP32LogitsGather, and only when the
sampler or rejection sampler wants FP32 logits for the step."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import (
    SupportsFP32LogitsGather,
    supports_fp32_logits_gather,
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
        _check_model_opt_in(world_size, device)


class UnfusedLogitsProcessor(LogitsProcessor):
    # Overriding _gather_logits (as a custom all-gather does) keeps the
    # unfused path: gather, slice, then scale / soft cap in the head dtype.
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return super()._gather_logits(logits)


class PassthroughModel(nn.Module):
    """compute_logits returns the logits processor's output unchanged."""

    def __init__(self, lm_head: nn.Module, processor: LogitsProcessor):
        super().__init__()
        self.lm_head = lm_head
        self.logits_processor = processor

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)


class OptedInModel(PassthroughModel, SupportsFP32LogitsGather):
    pass


class SoftCapModel(PassthroughModel):
    """Post-processes the processor's output, as HunYuan-V4 does."""

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(super().compute_logits(hidden_states) / 30.0) * 30.0


class OptedInSoftCapModel(SoftCapModel, SupportsFP32LogitsGather):
    pass


def _make_lm_head(world_size: int, device: torch.device):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    vocab_size, hidden_size = 1000 * world_size - 37, 64
    with torch.device(device):
        lm_head = ParallelLMHead(vocab_size, hidden_size, params_dtype=torch.bfloat16)
    lm_head.weight.data.normal_(generator=torch.Generator(device).manual_seed(7))
    hidden_states = torch.randn(
        33, hidden_size, device=device, generator=torch.Generator(device).manual_seed(0)
    ).bfloat16()
    return vocab_size, lm_head, hidden_states


def _check_logits_processor(world_size: int, device: torch.device):
    vocab_size, lm_head, hidden_states = _make_lm_head(world_size, device)
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


def _check_model_opt_in(world_size: int, device: torch.device):
    """The runner's target logits: FP32 only for a model that opts in and only
    when the logits' consumer (the rejection sampler on a step with drafts,
    the sampler otherwise) wants FP32; bitwise equal to main (head-dtype
    gather, then the sampler's FP32 copy where it makes one) in every case."""
    from vllm.v1.worker.gpu.model_runner import (
        GPUModelRunner,
        fp32_gather_logits_processors,
    )

    vocab_size, lm_head, hidden_states = _make_lm_head(world_size, device)
    for model_cls in (PassthroughModel, SoftCapModel):
        # main: no FP32 request reaches the gather.
        main = model_cls(lm_head, UnfusedLogitsProcessor(vocab_size)).compute_logits(
            hidden_states
        )
        assert main.dtype == torch.bfloat16
        opted_in_cls = {
            PassthroughModel: OptedInModel,
            SoftCapModel: OptedInSoftCapModel,
        }[model_cls]
        for cls in (model_cls, opted_in_cls):
            model = cls(lm_head, LogitsProcessor(vocab_size))
            processors = fp32_gather_logits_processors(model)
            opted_in = cls is opted_in_cls
            assert processors == ([model.logits_processor] if opted_in else [])
            # Which consumer is asked: the sampler without drafts, the
            # rejection sampler with them. Each says whether it wants FP32
            # (a verify path that reads bf16 logits directly says no).
            for num_draft_tokens, sampler_wants, rejection_wants in (
                (0, True, False),
                (0, False, True),
                (7, False, True),
                (7, True, False),
            ):
                consumer_wants = rejection_wants if num_draft_tokens else sampler_wants
                runner = SimpleNamespace(
                    sampler=SimpleNamespace(
                        wants_fp32_logits=lambda _, w=sampler_wants: w
                    ),
                    rejection_sampler=SimpleNamespace(
                        wants_fp32_logits=lambda _, w=rejection_wants: w
                    ),
                    fp32_logits_processors=processors,
                    vocab_size=vocab_size,
                    model_state=SimpleNamespace(
                        compute_sample_logits=model.compute_logits
                    ),
                )
                input_batch = SimpleNamespace(
                    idx_mapping_np=np.zeros(1, dtype=np.int32),
                    num_draft_tokens=num_draft_tokens,
                )
                logits = GPUModelRunner._compute_logits(
                    runner,  # type: ignore[arg-type]
                    hidden_states,
                    input_batch,  # type: ignore[arg-type]
                )
                assert model.logits_processor.gather_out_dtype is None
                fp32 = opted_in and consumer_wants
                assert logits.dtype == (torch.float32 if fp32 else torch.bfloat16)
                if not fp32:
                    # Head dtype: the same tensor as main, for the consumer
                    # to copy or read as on main.
                    assert torch.equal(logits, main)
                elif model_cls is PassthroughModel:
                    # FP32: exactly the sampler's FP32 copy of main's logits.
                    assert torch.equal(logits, main.float())
                else:
                    # Why post-processing models must not opt in: the soft
                    # cap then runs on FP32 logits and differs from main.
                    assert not torch.equal(logits, main.float())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
def test_fp32_logits_gather_opt_in_flags():
    from vllm.models.minimax_m3 import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )

    assert supports_fp32_logits_gather(MiniMaxM3SparseForCausalLM)
    assert supports_fp32_logits_gather(MiniMaxM3SparseForConditionalGeneration)
