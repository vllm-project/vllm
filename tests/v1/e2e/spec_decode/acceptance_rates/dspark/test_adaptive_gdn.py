# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end adaptive DSpark verification on a hybrid-GDN target."""

import math
from itertools import accumulate

import pytest
import torch

from tests.utils import large_gpu_mark
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)

TARGET_MODEL = "Qwen/Qwen3.5-0.8B"
DRAFT_MODEL = "satgeze/Qwen3.5-0.8B-DSpark"
NUM_SPECULATIVE_TOKENS = 7

pytestmark = [
    pytest.mark.hybrid_model,
    pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="adaptive GDN CUDA-graph coverage requires CUDA",
    ),
]


@large_gpu_mark(min_gb=16)
def test_adaptive_dspark_replays_ragged_gdn_decode(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner,
) -> None:
    """Ragged device query lengths must produce valid target outputs.

    The capability overrides let this test audit the execution path before GDN
    advertises adaptive-verification support globally. Remove them when the
    production capability gates are enabled.
    """

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setenv("VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "64")

    def supports_device_query_lens(_cls) -> bool:
        return True

    monkeypatch.setattr(
        GDNAttentionBackend,
        "supports_device_cpu_query_lens_mismatch",
        classmethod(supports_device_query_lens),
    )
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder,
        "_cudagraph_support",
        AttentionCGSupport.ALWAYS,
    )

    trace_step = {"value": 0}
    pending_layouts: dict[int, tuple[tuple[str, ...], tuple[int, ...], int]] = {}
    observed_layouts: list[
        tuple[tuple[str, ...], tuple[int, ...], torch.Tensor, int]
    ] = []

    original_get_num_tokens = AdaptiveVerificationManager.get_num_tokens
    original_reallocate_drafts = AdaptiveVerificationManager.reallocate_drafts

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
        assert cudagraph_mode.has_full_cudagraphs()

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
        for cpu_query_lens, device_query_lens, draft_budget in ragged_layouts:
            expected_total = len(device_query_lens) + draft_budget
            assert sum(cpu_query_lens) == expected_total
            assert sum(device_query_lens) == expected_total
            assert tuple(accumulate(device_query_lens, initial=0)) != tuple(
                accumulate(cpu_query_lens, initial=0)
            )
