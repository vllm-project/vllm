# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused vocab-parallel logits gather equals a last-dim all-gather, the
vocab slice and the dtype cast, bitwise. LogitsProcessor uses it only when the
forward context carries a gather dtype, which the model runner sets only for a
model that opts in with SupportsFP32LogitsGather: FP32 when the step's
consumer (the sampler, or the rejection sampler on a step with drafts) wants
FP32 logits, else the head's dtype."""

import dataclasses
from contextlib import AbstractContextManager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from vllm.config import (
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
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

        _check_logits_processor(config, world_size, device)
        _check_model_opt_in(config, world_size, device)


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


def _make_lm_head(world_size: int, device: torch.device):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    # The vocab is padded, so the unfused gather returns a non-contiguous
    # slice and the fused one a contiguous tensor: that tells them apart.
    vocab_size, hidden_size = 1000 * world_size - 37, 64
    with torch.device(device):
        lm_head = ParallelLMHead(vocab_size, hidden_size, params_dtype=torch.bfloat16)
    lm_head.weight.data.normal_(generator=torch.Generator(device).manual_seed(7))
    hidden_states = torch.randn(
        33, hidden_size, device=device, generator=torch.Generator(device).manual_seed(0)
    ).bfloat16()
    return vocab_size, lm_head, hidden_states


_NO_CONTEXT = object()


def _gather_context(config: VllmConfig, dtype) -> AbstractContextManager:
    if dtype is _NO_CONTEXT:
        return nullcontext()
    return set_forward_context(None, config, logits_gather_dtype=dtype)


def _check_logits_processor(config: VllmConfig, world_size: int, device: torch.device):
    """The forward context's gather dtype is honoured; without one (or with
    None) the processor takes the unfused path, in the head's dtype."""
    vocab_size, lm_head, hidden_states = _make_lm_head(world_size, device)
    for scale, soft_cap in ((1.0, None), (0.5, None), (1.0, 30.0)):
        for requested in (_NO_CONTEXT, None, torch.bfloat16, torch.float32):
            processors = [
                cls(vocab_size, scale=scale, soft_cap=soft_cap)
                for cls in (LogitsProcessor, UnfusedLogitsProcessor)
            ]
            with _gather_context(config, requested):
                out, unfused = (p(lm_head, hidden_states) for p in processors)
            assert unfused.dtype == torch.bfloat16
            # FP32 only where nothing runs on the logits after the gather.
            upcast = requested == torch.float32 and scale == 1.0 and not soft_cap
            assert out.dtype == (torch.float32 if upcast else torch.bfloat16)
            assert torch.equal(out, unfused.to(out.dtype))
            if soft_cap is None:
                fused = requested is not _NO_CONTEXT and requested is not None
                assert out.is_contiguous() == fused
                assert not unfused.is_contiguous()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("world_size", [2, 4])
def test_vocab_parallel_logits_gather(world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    try:
        mp.spawn(_worker, args=(world_size, get_open_port()), nprocs=world_size)
    finally:
        cleanup_dist_env_and_memory()


def _check_model_opt_in(config: VllmConfig, world_size: int, device: torch.device):
    """The runner's target logits: as on main for a model that does not opt
    in; for one that does, the fused gather, in FP32 exactly when the step's
    consumer wants FP32 (which may then process them in place)."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu.states import RequestState

    class Bf16VerifyRejectionSampler(RejectionSampler):
        # A verify path that reads head-dtype logits directly.
        def wants_fp32_logits(self, idx_mapping_np: np.ndarray) -> bool:
            return False

    vocab_size, lm_head, hidden_states = _make_lm_head(world_size, device)
    main = PassthroughModel(lm_head, UnfusedLogitsProcessor(vocab_size))
    main_logits = main.compute_logits(hidden_states)

    req_states = RequestState(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=vocab_size,
        device=device,
    )
    sampler = Sampler(
        vllm_config=config,
        max_num_reqs=2,
        vocab_size=vocab_size,
        device=device,
        req_states=req_states,
        logprobs_mode="raw_logprobs",
    )
    # Request 0 is greedy (no processing, no FP32 copy); request 1 is not.
    sampler.add_request(0, SamplingParams(temperature=0.0))
    sampler.add_request(1, SamplingParams(temperature=2.0))
    sampler.apply_staged_writes()
    spec_config = SpeculativeConfig(method="ngram", num_speculative_tokens=1)
    batch = InputBatch.make_dummy(1, 1, InputBuffers(2, 2, device))

    for model_cls in (PassthroughModel, OptedInModel):
        model = model_cls(lm_head, LogitsProcessor(vocab_size))
        opted_in = model_cls is OptedInModel
        for rejection_cls in (RejectionSampler, Bf16VerifyRejectionSampler):
            # A real runner needs a model config and weights; _compute_logits
            # reads only these attributes.
            runner = GPUModelRunner.__new__(GPUModelRunner)
            runner.vllm_config = config
            runner.vocab_size = vocab_size
            runner.sampler = sampler
            runner.rejection_sampler = rejection_cls(sampler, spec_config, device)
            runner.fused_logits_gather = supports_fp32_logits_gather(model)
            runner.model_state = SimpleNamespace(
                compute_sample_logits=model.compute_logits
            )
            for req_idx, num_draft_tokens in ((0, 0), (1, 0), (1, 1)):
                input_batch = dataclasses.replace(
                    batch,
                    idx_mapping_np=np.array([req_idx]),
                    num_draft_tokens=num_draft_tokens,
                )
                logits, writable = runner._compute_logits(hidden_states, input_batch)
                consumer_wants = req_idx == 1 and not (
                    num_draft_tokens and rejection_cls is Bf16VerifyRejectionSampler
                )
                fp32 = opted_in and consumer_wants
                assert writable == fp32
                assert logits.dtype == (torch.float32 if fp32 else torch.bfloat16)
                # Bitwise main's logits (or the sampler's FP32 copy of them).
                assert torch.equal(logits, main_logits.to(logits.dtype))
                # Fused only for the model that opts in.
                assert logits.is_contiguous() == opted_in


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
def test_fp32_logits_gather_opt_in_flags():
    from vllm.models.minimax_m3 import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )

    assert supports_fp32_logits_gather(MiniMaxM3SparseForCausalLM)
    assert supports_fp32_logits_gather(MiniMaxM3SparseForConditionalGeneration)
