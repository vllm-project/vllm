# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy Uno parity with the original Qwen3-8B adapter on one NVIDIA GPU."""

from pathlib import Path

import pytest
import torch
from huggingface_hub.constants import HF_HUB_OFFLINE

import vllm.envs as envs
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.metrics.reader import Counter, Metric

from .utils import (
    assert_request_outputs_match,
    compute_acceptance_len,
    get_spec_decode_metric_value,
    get_test_prompts,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Uno requires an NVIDIA CUDA device"
)


@pytest.fixture(scope="module")
def uno_adapter_path() -> str:
    snapshot = hf_api().snapshot_download(
        repo_id="s-sahoo/uno-qwen3-8B",
        revision="8819e09ac901e7290d8d89d62c98b9f756c602fe",
        allow_patterns=["adapter/*"],
        local_files_only=HF_HUB_OFFLINE,
    )
    return str(Path(snapshot) / "adapter")


def _prefix_cache_hits(metrics: list[Metric]) -> int:
    return sum(
        metric.value
        for metric in metrics
        if isinstance(metric, Counter) and metric.name == "vllm:prefix_cache_hits"
    )


@pytest.mark.parametrize(
    ("k", "enable_prefix_caching"),
    [(1, False), (8, True)],
    ids=["seed_only", "eight_candidates_cached"],
)
def test_uno_greedy_matches_base_model(
    vllm_runner,
    monkeypatch: pytest.MonkeyPatch,
    uno_adapter_path: str,
    k: int,
    enable_prefix_caching: bool,
):
    """Verify unequal prefills, batched decoding, and reused prefixes against base."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    prompts = get_test_prompts(mm_enabled=False, num_prompts=4)
    batches = [prompts]
    if enable_prefix_caching:
        batches.append(list(reversed(prompts)))
    sampling = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True, seed=0)
    prefill_budget = 256
    # Bound the reservation on larger devices while keeping both engines matched.
    total_memory = torch.cuda.get_device_properties(0).total_memory
    common = dict(
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        dtype="bfloat16",
        trust_remote_code=False,
        enforce_eager=True,
        async_scheduling=False,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=prefill_budget,
        enable_chunked_prefill=True,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=min(0.9, 24 * 1024**3 / total_memory),
        kv_cache_memory_bytes=2 * 1024**3,
        enable_lora=True,
        max_lora_rank=128,
        max_loras=2,
        max_cpu_loras=2,
        disable_log_stats=False,
    )
    chat_kwargs = dict(
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )

    # Keep the same LoRA infrastructure in the reference without a target adapter.
    # Sequential runner contexts release the first engine before loading the second.
    with vllm_runner("Qwen/Qwen3-8B", **common) as reference:
        ref_outputs = [reference.llm.chat(batch, **chat_kwargs) for batch in batches]
    prompt_lengths = [len(output.prompt_token_ids) for output in ref_outputs[0]]
    assert len(set(prompt_lengths)) > 1
    assert max(prompt_lengths) > prefill_budget

    with vllm_runner(
        "Qwen/Qwen3-8B",
        **common,
        speculative_config={
            "method": "uno",
            "uno_lora_path": uno_adapter_path,
            "uno_mask_token_id": 151669,
            "num_speculative_tokens": k,
        },
    ) as speculative:
        spec_outputs = []
        previous_metrics = None
        for batch in batches:
            spec_outputs.append(speculative.llm.chat(batch, **chat_kwargs))
            metrics = speculative.llm.get_metrics()
            draft_tokens = get_spec_decode_metric_value(
                metrics, "vllm:spec_decode_num_draft_tokens"
            )
            if previous_metrics is not None:
                draft_tokens -= get_spec_decode_metric_value(
                    previous_metrics, "vllm:spec_decode_num_draft_tokens"
                )
                assert _prefix_cache_hits(metrics) > _prefix_cache_hits(
                    previous_metrics
                )
            assert draft_tokens > 0, "Uno did not verify any draft candidates"
            acceptance_len = compute_acceptance_len(metrics, previous_metrics)
            assert 1 < acceptance_len <= k + 1
            print(f"Uno K={k}: mean acceptance length={acceptance_len:.3f}")
            previous_metrics = metrics

    for batch_index, (ref_batch, spec_batch) in enumerate(
        zip(ref_outputs, spec_outputs)
    ):
        context = (
            f"Uno K={k}, prefix_cache={enable_prefix_caching}, batch={batch_index}"
        )
        assert_request_outputs_match(
            ref_batch,
            spec_batch,
            required_matches=len(ref_batch),
            context=context,
        )
        for index, (ref_output, spec_output) in enumerate(zip(ref_batch, spec_batch)):
            ref_ids = list(ref_output.outputs[0].token_ids)
            spec_ids = list(spec_output.outputs[0].token_ids)
            assert ref_ids == spec_ids, (
                f"{context}, request={index}: reference tokens={ref_ids}, "
                f"Uno tokens={spec_ids}"
            )
