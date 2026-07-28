# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import pytest
import torch

from tests.utils import single_gpu_only, wait_for_rocm_memory_to_settle
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

from ..utils import (
    _skip_if_insufficient_gpus_for_tp,
    evaluate_llm_for_gsm8k,
    get_test_prompts,
    greedy_sampling,
)


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (("mtp", "XiaomiMiMo/MiMo-7B-Base", 1), False, 0.5),  # ref: 65%-70%
        pytest.param(
            ("mtp", "ZixiQi/DeepSeek-V3-4layers-MTP-FP8", 1),
            False,
            0.0,
            marks=pytest.mark.skipif(
                current_platform.is_device_capability_family(100),
                reason="DeepSeek MTP: TRTLLM MoE top_k check fails on Blackwell",
            ),
        ),  # dummy model
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
        (
            ("mtp", "google/gemma-4-E4B-it", 1, "google/gemma-4-E4B-it-assistant"),
            False,
            0.50,
        ),  # gemma4 MTP with assistant model, ref: ~62%
    ],
    ids=["mimo", "deepseek", "qwen3_5-hybrid", "gemma4-e4b"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int] | tuple[str, str, int, str],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
):
    """
    Compare the outputs of a original LLM and a speculative LLM
    which should be the same when using MTP speculative decoding. Due to some variance
    in the engine, it is possible for some outputs to differ, so we expect that at least
    6/10 output tokens match exactly, and that the GSM8k accuracy is above a precomputed
    reference threshold for each model.
    """
    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        if len(model_setup) == 4:
            method, model_name, tp_size, draft_model = model_setup
        else:
            method, model_name, tp_size = model_setup
            draft_model = None
        _skip_if_insufficient_gpus_for_tp(tp_size)

        if "Qwen3.5" in model_name and os.environ.get("VLLM_USE_V2_MODEL_RUNNER"):
            pytest.skip(
                "Model Runner V2 does not yet support hybrid models "
                "(Qwen3.5 mixes Mamba-style GDN with attention layers)."
            )

        attn_backend = "TRITON_ATTN" if current_platform.is_rocm() else "auto"

        # Skip multimodal profiling for models that don't need it in this test.
        extra_kwargs: dict[str, Any] = {}
        if "Qwen3.5" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        elif "gemma-4" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}

        if draft_model is not None and "gemma-4" in draft_model:
            import transformers
            from packaging.version import Version

            if Version(transformers.__version__) < Version("5.8.0"):
                pytest.skip(
                    "Gemma4 MTP assistant requires transformers>=5.8.0, "
                    f"got {transformers.__version__}"
                )

        ref_llm = LLM(
            model=model_name,
            max_model_len=2048,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            attention_backend=attn_backend,
            **extra_kwargs,
        )
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        evaluate_llm_for_gsm8k(
            ref_llm, expected_accuracy_threshold=expected_accuracy_threshold
        )
        del ref_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()

        speculative_config: dict[str, Any] = {
            "method": method,
            "num_speculative_tokens": 1,
            "max_model_len": 2048,
        }
        if draft_model is not None:
            speculative_config["model"] = draft_model
            speculative_config["num_speculative_tokens"] = 2

        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config=speculative_config,
            max_model_len=2048,
            attention_backend=attn_backend,
            **extra_kwargs,
        )
        # MTP supports async scheduling; assert it is active by default.
        assert spec_llm.llm_engine.vllm_config.scheduler_config.async_scheduling
        evaluate_llm_for_gsm8k(
            spec_llm, expected_accuracy_threshold=expected_accuracy_threshold
        )
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)
        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        # Heuristic: expect at least 80% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.8 * len(ref_outputs))
        del spec_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


def _is_sharded_sampling_active(worker) -> bool:
    return worker.model_runner.batch_sharder is not None


def test_mtp_sharded_sampling_equivalence(monkeypatch: pytest.MonkeyPatch):
    """Batch-sharded sampling must be a bit-exact drop-in for replicated
    sampling under MTP spec decoding: the collectives move the same logits
    bytes, Gumbel keys derive from (request slot, position, seed), and slot
    assignment is rank-deterministic. Unlike the spec-vs-ref comparison
    above, both runs here are spec decode with identical math, so outputs
    must match exactly."""
    tp_size = 2
    _skip_if_insufficient_gpus_for_tp(tp_size)
    model_name = "Qwen/Qwen3.5-0.8B-Base"
    test_prompts = get_test_prompts(mm_enabled=False, num_prompts=20)

    with monkeypatch.context() as m:
        # Batch-sharded sampling only exists in the V2 model runner.
        m.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        # Required for the collective_rpc mode check below.
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        def run(disable_sharding: bool):
            llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                enable_batch_sharded_sampling=not disable_sharding,
                speculative_config={
                    "method": "mtp",
                    "num_speculative_tokens": 1,
                    "max_model_len": 2048,
                },
                max_model_len=2048,
                limit_mm_per_prompt={"image": 0, "video": 0},
                # More requests than ranks, so round-robin slot ownership
                # splits the batch across both ranks.
                max_num_seqs=len(test_prompts),
                # Timing-based kernel selection is the dominant source of
                # cross-boot numeric noise; disable for a tight comparison.
                kernel_config={"enable_flashinfer_autotune": False},
            )
            try:
                # Guard against a vacuous comparison: verify every worker is
                # in the intended sampling mode.
                modes = llm.llm_engine.collective_rpc(_is_sharded_sampling_active)
                assert all(mode == (not disable_sharding) for mode in modes)
                greedy = llm.chat(test_prompts, greedy_sampling())
                seeded = llm.chat(
                    test_prompts,
                    SamplingParams(temperature=1.0, seed=33, max_tokens=10),
                )
                return greedy, seeded
            finally:
                del llm
                torch.accelerator.empty_cache()
                cleanup_dist_env_and_memory()
                wait_for_rocm_memory_to_settle()

        ref_greedy, ref_seeded = run(disable_sharding=True)
        shard_greedy, shard_seeded = run(disable_sharding=False)

        # Engine boots are not bitwise deterministic (kernel/collective
        # selection shifts logits by ~1 ulp, flipping near-tie tokens), so
        # allow a small number of divergent prompts. A sharding bug produces
        # wholesale divergence, not isolated near-tie flips.
        for name, ref_outputs, shard_outputs in (
            ("greedy", ref_greedy, shard_greedy),
            ("seeded", ref_seeded, shard_seeded),
        ):
            num_divergent = 0
            for i, (ref, out) in enumerate(zip(ref_outputs, shard_outputs)):
                if list(ref.outputs[0].token_ids) != list(out.outputs[0].token_ids):
                    num_divergent += 1
                    print(
                        f"{name} prompt {i} diverged:\n"
                        f"  replicated: {ref.outputs[0].text!r}\n"
                        f"  sharded:    {out.outputs[0].text!r}"
                    )
            assert num_divergent <= 2, (
                f"{name}: {num_divergent}/{len(ref_outputs)} prompts diverged, "
                "beyond near-tie boot noise"
            )
