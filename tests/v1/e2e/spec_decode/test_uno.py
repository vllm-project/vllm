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


def _uno_execution_state(worker) -> dict:
    from vllm.v1.worker.gpu.spec_decode.uno import UnoSpeculator

    runner = worker.model_runner
    proposer = runner.speculator
    assert isinstance(proposer, UnoSpeculator)
    return {
        "shared_model": proposer.model is runner.model,
        "draft_graph_replays": proposer.num_graph_replays,
        "draft_graphs": len(proposer.cudagraph_manager.graphs),
        "draft_eager_proposals": proposer.num_eager_proposals,
        "lora_plan_hits": runner.uno_lora_state.plan_cache.hits,
        "lora_plan_misses": runner.uno_lora_state.plan_cache.misses,
        "lora_plan_bypasses": runner.uno_lora_state.plan_cache.bypasses,
    }


def _disable_uno_adapter_for_control(worker) -> None:
    """Use identical noise and sampling, with base weights on every draft row."""
    runner = worker.model_runner
    proposer = runner.speculator

    def base_only(shape):
        if shape is not None:
            _num_reqs, num_tokens = shape
            runner.uno_lora_state.install_base(num_tokens, num_tokens)

    proposer.set_lora_hook(base_only)


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
    ("k", "enable_prefix_caching", "enforce_eager", "dual_stream"),
    [
        (1, False, True, False),
        (8, True, True, False),
        (8, True, False, False),
        (8, True, False, True),
    ],
    ids=[
        "seed_only",
        "eight_candidates_cached",
        "eight_candidates_graphs",
        "native_overlap_graphs",
    ],
)
def test_uno_greedy_matches_base_model(
    vllm_runner,
    monkeypatch: pytest.MonkeyPatch,
    uno_adapter_path: str,
    k: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    dual_stream: bool,
):
    """Verify unequal prefills, batched decoding, and reused prefixes against base."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_LORA_ENABLE_DUAL_STREAM", str(int(dual_stream)))
    monkeypatch.setattr(envs, "VLLM_LORA_ENABLE_DUAL_STREAM", dual_stream)

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
        enforce_eager=enforce_eager,
        compilation_config={"cudagraph_capture_sizes": [8, 16, 32, 64]},
        async_scheduling=True,
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
        trained_acceptance = []
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
            assert (1 if k == 1 else 3) < acceptance_len <= k + 1
            trained_acceptance.append(acceptance_len)
            print(f"Uno K={k}: mean acceptance length={acceptance_len:.3f}")
            previous_metrics = metrics

        # Exact greedy output alone cannot detect a disabled adapter: the
        # verifier corrects poor proposals, and the base seed already gives
        # acceptance length near two. This eager control must lose the trained
        # adapter's advantage while preserving the final target tokens.
        if k == 8 and enforce_eager:
            speculative.llm.llm_engine.collective_rpc(_disable_uno_adapter_for_control)
            control_outputs = speculative.llm.chat(batches[-1], **chat_kwargs)
            control_metrics = speculative.llm.get_metrics()
            control_acceptance = compute_acceptance_len(
                control_metrics, previous_metrics
            )
            print(f"Uno adapter-disabled acceptance={control_acceptance:.3f}")
            assert control_acceptance < min(trained_acceptance) - 1.0
            assert_request_outputs_match(
                ref_outputs[-1],
                control_outputs,
                required_matches=len(control_outputs),
                context="adapter-disabled",
            )
            assert [
                list(output.outputs[0].token_ids) for output in control_outputs
            ] == [list(output.outputs[0].token_ids) for output in ref_outputs[-1]]
        states = speculative.llm.llm_engine.collective_rpc(_uno_execution_state)
        assert all(state["shared_model"] for state in states)
        assert all(
            state["lora_plan_hits"]
            + state["lora_plan_misses"]
            + state["lora_plan_bypasses"]
            > 0
            for state in states
        ), f"Uno LoRA plan cache did not observe any installs: {states}"
        assert all(
            state["lora_plan_hits"]
            / (
                state["lora_plan_hits"]
                + state["lora_plan_misses"]
                + state["lora_plan_bypasses"]
            )
            >= 0.8
            for state in states
        ), f"Uno LoRA plans are rebuilding too often: {states}"
        assert all(state["lora_plan_bypasses"] == 0 for state in states), states
        if enforce_eager:
            assert all(state["draft_eager_proposals"] > 0 for state in states)
        if not enforce_eager:
            assert all(state["draft_graphs"] > 0 for state in states)
            assert all(state["draft_graph_replays"] > 0 for state in states)
        print(f"Uno execution state: {states}")

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
