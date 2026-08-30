# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end adaptive DSpark verification on a hybrid-GDN target."""

import asyncio
import math
from contextlib import AsyncExitStack
from itertools import accumulate
from typing import TypedDict

import pytest
import torch

from tests.utils import large_gpu_mark
from vllm import SamplingParams
from vllm.config.compilation import CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)

TARGET_MODEL = "Qwen/Qwen3.5-0.8B"
DRAFT_MODEL = "satgeze/Qwen3.5-0.8B-DSpark"
NUM_SPECULATIVE_TOKENS = 7
DP_TARGET_MODEL = "Qwen/Qwen3.6-35B-A3B"
DP_DRAFT_MODEL = "RedHatAI/Qwen3.6-35B-A3B-speculator.dspark"
DP_NUM_SPECULATIVE_TOKENS = 8


class _BatchObservation(TypedDict):
    request_rows: tuple[tuple[str, bool, bool], ...]
    has_prefill: bool
    has_spec_decode: bool


class _DispatchObservation(_BatchObservation):
    uniform_token_count: int | None
    cg_mode: CUDAGraphMode
    decode_only: bool


pytestmark = [
    pytest.mark.hybrid_model,
    pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="adaptive GDN CUDA-graph coverage requires CUDA",
    ),
]


@large_gpu_mark(min_gb=16)
@pytest.mark.parametrize("gdn_decode_kernel", ["cuda", "triton"])
def test_adaptive_dspark_replays_ragged_gdn_decode(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner,
    gdn_decode_kernel: str,
) -> None:
    """Ragged device query lengths must produce valid target outputs."""

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setenv("VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "64")
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", gdn_decode_kernel)

    trace_step = {"value": 0}
    pending_layouts: dict[int, tuple[tuple[str, ...], tuple[int, ...], int]] = {}
    observed_layouts: list[
        tuple[tuple[str, ...], tuple[int, ...], torch.Tensor, int]
    ] = []
    observed_dispatches: list[tuple[int | None, bool, CUDAGraphMode, bool]] = []

    original_get_num_tokens = AdaptiveVerificationManager.get_num_tokens
    original_reallocate_drafts = AdaptiveVerificationManager.reallocate_drafts
    original_dispatch = CudaGraphManager.dispatch

    def record_dispatch(
        manager: CudaGraphManager,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        has_prefill: bool = False,
    ):
        desc = original_dispatch(
            manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            max_query_len=max_query_len,
            has_prefill=has_prefill,
        )
        if manager.varlen_decode:
            observed_dispatches.append(
                (uniform_token_count, has_prefill, desc.cg_mode, desc.decode_only)
            )
        return desc

    def controlled_get_num_tokens(
        manager: AdaptiveVerificationManager,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        original_num_tokens = original_get_num_tokens(
            manager, num_tokens_per_req, draft_tokens
        )
        batch_budget = manager._batch_budget
        assert batch_budget is not None
        drafts_per_req, non_draft_per_req, _ = batch_budget
        req_ids = tuple(num_tokens_per_req)
        num_verification_reqs = sum(drafts_per_req[req_id] > 0 for req_id in req_ids)
        total_drafts = sum(drafts_per_req.values())

        if num_verification_reqs < 2:
            return original_num_tokens

        drafts_per_high_confidence_req = (4, 3)[trace_step["value"] % 2]
        draft_budget = min(
            drafts_per_high_confidence_req * num_verification_reqs,
            total_drafts - 1,
        )
        trace_step["value"] += 1

        manager._batch_budget = (
            drafts_per_req,
            non_draft_per_req,
            draft_budget,
        )

        if draft_budget == total_drafts:
            cpu_draft_lens = [drafts_per_req[req_id] for req_id in req_ids]
        else:
            cpu_draft_lens = [0] * len(req_ids)
            even_drafts, remainder = divmod(draft_budget, num_verification_reqs)
            for req_idx in range(num_verification_reqs):
                cpu_draft_lens[req_idx] = even_drafts + (req_idx < remainder)
        cpu_query_lens = tuple(
            non_draft_per_req[req_id] + cpu_draft_lens[req_idx]
            for req_idx, req_id in enumerate(req_ids)
        )
        pending_layouts[id(manager)] = (req_ids, cpu_query_lens, draft_budget)
        return sum(non_draft_per_req.values()) + draft_budget

    def record_reallocated_drafts(
        manager: AdaptiveVerificationManager,
        req_ids: list[str],
        idx_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        pending = pending_layouts.pop(id(manager), None)
        if pending is not None:
            # Duplicate prompts are adjacent. Make the first request in every
            # pair win the GPU ranking and the second lose it. The CPU splits
            # the same total budget evenly, guaranteeing a CPU/device mismatch
            # without changing draft tokens or target verification semantics.
            confidences = torch.full(
                (len(req_ids), manager.num_speculative_steps),
                0.01,
                dtype=torch.float32,
                device=idx_mapping.device,
            )
            confidences[::2].fill_(0.99)
            manager._confidence_probs[idx_mapping] = confidences

        result = original_reallocate_drafts(manager, req_ids, idx_mapping)
        if pending is not None:
            pending_req_ids, cpu_query_lens, draft_budget = pending
            observed_layouts.append(
                (
                    pending_req_ids,
                    cpu_query_lens,
                    result[1][: len(req_ids) + 1].clone(),
                    draft_budget,
                )
            )
        return result

    monkeypatch.setattr(
        AdaptiveVerificationManager,
        "get_num_tokens",
        controlled_get_num_tokens,
    )
    monkeypatch.setattr(
        AdaptiveVerificationManager,
        "reallocate_drafts",
        record_reallocated_drafts,
    )
    monkeypatch.setattr(CudaGraphManager, "dispatch", record_dispatch)

    runner_config = {
        "language_model_only": True,
        "dtype": "bfloat16",
        "max_model_len": 512,
        "max_num_seqs": 6,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 << 30,
        "block_size": None,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "gdn_prefill_backend": "triton",
        "ignore_patterns": [
            "model_v02.safetensors",
            "model_v03.safetensors",
        ],
        "compilation_config": {
            "cudagraph_capture_sizes": [8, 16, 32, 48],
        },
        "speculative_config": {
            "method": "dspark",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "enable_adaptive_verification": True,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": [0.0] * NUM_SPECULATIVE_TOKENS,
        },
    }
    unique_prompts = [
        "The capital of France is",
        "One plus one equals",
        "The color of grass is",
    ]
    prompts = [prompt for prompt in unique_prompts for _ in range(2)]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=12,
        ignore_eos=True,
        logprobs=1,
        seed=0,
    )

    with vllm_runner(TARGET_MODEL, **runner_config) as runner:
        llm = runner.get_llm()
        cudagraph_mode = llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
        assert cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE

        outputs = llm.generate(prompts, sampling_params)
        torch.accelerator.synchronize()

        assert len(outputs) == len(prompts)
        for output in outputs:
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            assert len(token_ids) == sampling_params.max_tokens
            assert completion.logprobs is not None
            assert len(completion.logprobs) == len(token_ids)
            for token_id, step_logprobs in zip(
                token_ids, completion.logprobs, strict=True
            ):
                assert token_id in step_logprobs
                assert math.isfinite(step_logprobs[token_id].logprob)

        ragged_layouts: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        for (
            req_ids,
            cpu_query_lens,
            device_query_start_loc,
            draft_budget,
        ) in observed_layouts:
            if len(req_ids) < 2:
                continue
            device_starts = device_query_start_loc.cpu().tolist()
            device_query_lens = tuple(
                end - start for start, end in zip(device_starts, device_starts[1:])
            )
            if len(set(device_query_lens)) > 1:
                ragged_layouts.append((cpu_query_lens, device_query_lens, draft_budget))

        assert ragged_layouts, "adaptive verification never produced a ragged batch"
        assert any(cpu != device for cpu, device, _ in ragged_layouts), (
            "device query lengths never differed from the CPU placeholder"
        )
        assert len({device for _, device, _ in ragged_layouts}) >= 2, (
            f"device query lengths did not change across decode steps: {ragged_layouts}"
        )
        assert any(
            uniform_token_count is None
            and not has_prefill
            and cg_mode == CUDAGraphMode.FULL
            and decode_only
            for (
                uniform_token_count,
                has_prefill,
                cg_mode,
                decode_only,
            ) in observed_dispatches
        ), "ragged decode never used a decode-only FULL graph"
        assert not any(
            has_prefill and decode_only
            for _, has_prefill, _, decode_only in observed_dispatches
        ), "a prefill batch used a decode-only FULL graph"
        for cpu_query_lens, device_query_lens, draft_budget in ragged_layouts:
            expected_total = len(device_query_lens) + draft_budget
            assert sum(cpu_query_lens) == expected_total
            assert sum(device_query_lens) == expected_total
            assert all(
                1 <= query_len <= NUM_SPECULATIVE_TOKENS + 1
                for query_len in device_query_lens
            )
            assert tuple(accumulate(device_query_lens, initial=0)) != tuple(
                accumulate(cpu_query_lens, initial=0)
            )


@large_gpu_mark(min_gb=16)
def test_adaptive_dspark_mixed_prefill_uses_piecewise(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner,
) -> None:
    """A real prefill+adaptive-decode batch must not use decode-only FULL."""

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setenv("VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "64")
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "cuda")

    current_batch: _BatchObservation | None = None
    observed_dispatches: list[_DispatchObservation] = []
    original_gather = GPUModelRunner.gather_batch_req_state
    original_dispatch = CudaGraphManager.dispatch

    def record_gather(
        runner: GPUModelRunner,
        scheduler_output,
        dummy_run: bool,
    ):
        nonlocal current_batch
        result = original_gather(runner, scheduler_output, dummy_run)
        batch_state, _ = result
        current_batch = None
        if not dummy_run and batch_state is not None:
            spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
            draft_req_ids = {
                req_id for req_id, tokens in spec_decode_tokens.items() if tokens
            }
            request_rows = tuple(
                (req_id, bool(is_prefilling), req_id in draft_req_ids)
                for req_id, is_prefilling in zip(
                    batch_state.req_ids,
                    batch_state.is_prefilling_np,
                    strict=True,
                )
            )
            current_batch = _BatchObservation(
                request_rows=request_rows,
                has_prefill=batch_state.has_prefill,
                has_spec_decode=any(
                    not is_prefilling and has_drafts
                    for _, is_prefilling, has_drafts in request_rows
                ),
            )
        return result

    def record_dispatch(
        manager: CudaGraphManager,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        has_prefill: bool = False,
    ):
        descriptor = original_dispatch(
            manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            max_query_len=max_query_len,
            has_prefill=has_prefill,
        )
        if manager.varlen_decode and current_batch is not None:
            observed_dispatches.append(
                _DispatchObservation(
                    **current_batch,
                    uniform_token_count=uniform_token_count,
                    cg_mode=descriptor.cg_mode,
                    decode_only=descriptor.decode_only,
                )
            )
        return descriptor

    monkeypatch.setattr(GPUModelRunner, "gather_batch_req_state", record_gather)
    monkeypatch.setattr(CudaGraphManager, "dispatch", record_dispatch)

    runner_config = {
        "language_model_only": True,
        "dtype": "bfloat16",
        "max_model_len": 256,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 256,
        "kv_cache_memory_bytes": 2 << 30,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "gdn_prefill_backend": "triton",
        "ignore_patterns": ["model_v02.safetensors", "model_v03.safetensors"],
        "compilation_config": {"cudagraph_capture_sizes": [8, 16, 24, 32]},
        "speculative_config": {
            "method": "dspark",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "enable_adaptive_verification": True,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": [0.0] * NUM_SPECULATIVE_TOKENS,
        },
    }
    decode_params = SamplingParams(
        temperature=0,
        max_tokens=16,
        ignore_eos=True,
        logprobs=1,
        seed=0,
    )
    prefill_params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        logprobs=1,
        seed=1,
    )

    with vllm_runner(TARGET_MODEL, **runner_config) as runner:
        llm = runner.get_llm()
        engine = llm.llm_engine
        for request_idx, prompt in enumerate(
            ["The capital of France is", "One plus one equals"]
        ):
            engine.add_request(f"decode-{request_idx}", prompt, decode_params)

        finished_outputs = {}
        for _ in range(32):
            for output in engine.step():
                if output.finished:
                    finished_outputs[output.request_id] = output
            if any(
                not dispatch["has_prefill"] and dispatch["has_spec_decode"]
                for dispatch in observed_dispatches
            ):
                break
        else:
            pytest.fail("initial requests never reached adaptive decode")

        engine.add_request("late-prefill", "Hello", prefill_params)
        for _ in range(64):
            for output in engine.step():
                if output.finished:
                    finished_outputs[output.request_id] = output
            if not engine.has_unfinished_requests():
                break
        else:
            pytest.fail("generation did not finish")

        torch.accelerator.synchronize()

    mixed_dispatches = [
        dispatch
        for dispatch in observed_dispatches
        if dispatch["has_prefill"]
        and dispatch["has_spec_decode"]
        and any(
            req_id.startswith("late-prefill-") and is_prefilling
            for req_id, is_prefilling, _ in dispatch["request_rows"]
        )
    ]
    assert mixed_dispatches, (
        "the late request was never scheduled with an adaptive decode request"
    )
    assert all(
        dispatch["cg_mode"] == CUDAGraphMode.PIECEWISE and not dispatch["decode_only"]
        for dispatch in mixed_dispatches
    ), mixed_dispatches

    assert len(finished_outputs) == 3
    assert set(finished_outputs) == {"decode-0", "decode-1", "late-prefill"}
    for output in finished_outputs.values():
        completion = output.outputs[0]
        expected_tokens = (
            prefill_params.max_tokens
            if output.request_id == "late-prefill"
            else decode_params.max_tokens
        )
        assert len(completion.token_ids) == expected_tokens
        assert completion.logprobs is not None
        assert len(completion.logprobs) == expected_tokens


@large_gpu_mark(min_gb=40)
@pytest.mark.distributed(num_gpus=2)
@pytest.mark.asyncio
async def test_adaptive_dspark_gdn_dp2_mixed_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DP=2/EP must finish when one rank adds prefill during GDN decode."""
    if torch.accelerator.device_count() < 2:
        pytest.skip("adaptive GDN DP=2 coverage requires two GPUs")

    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setenv("VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "64")
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "cuda")

    engine_args = AsyncEngineArgs(
        model=DP_TARGET_MODEL,
        language_model_only=True,
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.8,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gdn_prefill_backend="triton",
        data_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        compilation_config={"cudagraph_capture_sizes": [8, 16, 32, 64]},
        speculative_config={
            "method": "dspark",
            "model": DP_DRAFT_MODEL,
            "num_speculative_tokens": DP_NUM_SPECULATIVE_TOKENS,
            "enable_adaptive_verification": True,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": [0.0] * DP_NUM_SPECULATIVE_TOKENS,
        },
        trust_remote_code=True,
        disable_log_stats=True,
    )

    async def generate(
        engine: AsyncLLM,
        request_id: str,
        prompt: str,
        dp_rank: int,
        max_tokens: int,
        first_output: asyncio.Event,
    ) -> tuple[int, bool]:
        params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            ignore_eos=True,
            logprobs=1,
            seed=0,
            output_kind=RequestOutputKind.DELTA,
        )
        num_tokens = 0
        valid_logprobs = True
        async for output in engine.generate(
            request_id=request_id,
            prompt=prompt,
            sampling_params=params,
            data_parallel_rank=dp_rank,
        ):
            completion = output.outputs[0]
            num_tokens += len(completion.token_ids)
            assert completion.logprobs is not None
            for token_id, step_logprobs in zip(
                completion.token_ids, completion.logprobs, strict=True
            ):
                valid_logprobs &= math.isfinite(step_logprobs[token_id].logprob)
            if num_tokens:
                first_output.set()
            await asyncio.sleep(0)
        return num_tokens, valid_logprobs

    async with AsyncExitStack() as stack:
        engine = AsyncLLM.from_engine_args(engine_args)
        stack.callback(engine.shutdown)

        first_rank0 = asyncio.Event()
        first_rank1 = asyncio.Event()
        rank0_task = asyncio.create_task(
            generate(
                engine,
                "rank0-decode",
                "Explain why GPU synchronization can reduce throughput.",
                0,
                96,
                first_rank0,
            )
        )
        rank1_task = asyncio.create_task(
            generate(
                engine,
                "rank1-decode",
                "Explain why GPU synchronization can reduce throughput.",
                1,
                96,
                first_rank1,
            )
        )
        await asyncio.wait_for(
            asyncio.gather(first_rank0.wait(), first_rank1.wait()), timeout=300
        )
        assert not (rank0_task.done() and rank1_task.done())

        late_first_output = asyncio.Event()
        late_prefill_task = asyncio.create_task(
            generate(
                engine,
                "rank1-late-prefill",
                (
                    "Adaptive verification keeps request-specific query lengths "
                    "on the GPU while the CPU keeps fixed-size bookkeeping. "
                )
                * 24,
                1,
                16,
                late_first_output,
            )
        )
        await asyncio.wait_for(late_first_output.wait(), timeout=300)
        assert not (rank0_task.done() and rank1_task.done())

        results = await asyncio.wait_for(
            asyncio.gather(rank0_task, rank1_task, late_prefill_task), timeout=900
        )

    assert [num_tokens for num_tokens, _ in results] == [96, 96, 16]
    assert all(valid_logprobs for _, valid_logprobs in results)
