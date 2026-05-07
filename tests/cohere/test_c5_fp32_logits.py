#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import TypedDict, cast

import vllm.envs as envs
from vllm import SamplingParams

from .test_utils_c5 import (
    C5_SANITY_PROMPTS,
    build_c5_llm,
    shutdown_llm,
    validate_model_path,
)


class GenerationCaseSnapshot(TypedDict):
    text: str
    token_ids: list[int]
    sampled_logprobs: list[float | None]
    prompt_token_ids: list[int]
    prompt_logprobs: list[float | None]


class GenerationSnapshot(TypedDict):
    prompts: list[str]
    snapshots: list[GenerationCaseSnapshot]
    logits_output_dtypes: list[str | None]
    fp32_debug_info: object


def _required_model_path() -> str:
    model_path = os.environ.get("C5_MODEL_DIR")
    if not model_path:
        raise ValueError(
            "C5_MODEL_DIR is required for tests/cohere/test_c5_fp32_logits.py."
        )
    return validate_model_path(model_path)


def _tensor_parallel_size() -> int:
    raw_value = os.environ.get("C5_TENSOR_PARALLEL_SIZE")
    if raw_value is None:
        return 1
    return int(raw_value)


def _engine_args() -> str | None:
    return os.environ.get("C5_ENGINE_ARGS")


def _max_logprob_abs_diff() -> float:
    raw_value = os.environ.get("C5_FP32_LOGITS_MAX_ABS_DIFF")
    if raw_value is None:
        return 0.5
    return float(raw_value)


def _max_prompt_logprob_abs_diff() -> float:
    raw_value = os.environ.get("C5_FP32_LOGITS_MAX_PROMPT_ABS_DIFF")
    if raw_value is None:
        return 5.0
    return float(raw_value)


def _min_shared_prefix() -> int:
    raw_value = os.environ.get("C5_FP32_LOGITS_MIN_SHARED_PREFIX")
    if raw_value is None:
        return 8
    return int(raw_value)


def _extract_sampled_logprobs(
    token_ids: list[int], token_logprobs: list[dict] | None
) -> list[float | None]:
    if token_logprobs is None:
        return []

    sampled_logprobs: list[float | None] = []
    for step_idx, per_token_info in enumerate(token_logprobs):
        sampled_logprob: float | None = None
        sampled_token_id = token_ids[step_idx] if step_idx < len(token_ids) else None
        if sampled_token_id is not None and isinstance(per_token_info, dict):
            sampled_entry = per_token_info.get(sampled_token_id)
            if sampled_entry is not None:
                if hasattr(sampled_entry, "logprob"):
                    sampled_logprob = float(sampled_entry.logprob)
                elif isinstance(sampled_entry, dict) and "logprob" in sampled_entry:
                    sampled_logprob = float(sampled_entry["logprob"])
                elif isinstance(sampled_entry, (int, float)):
                    sampled_logprob = float(sampled_entry)

        sampled_logprobs.append(sampled_logprob)

    return sampled_logprobs


def _get_logits_processor(model):
    logits_processor = getattr(model, "logits_processor", None)
    if logits_processor is not None:
        return logits_processor

    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        return getattr(language_model, "logits_processor", None)

    return None


def _get_lm_head(model):
    model_core = getattr(model, "model", None)
    if model_core is not None and hasattr(model_core, "embed_tokens"):
        embed_tokens = model_core.embed_tokens
        return (
            embed_tokens
            if hasattr(embed_tokens, "weight")
            else getattr(embed_tokens, "base_layer", None)
        )

    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        return _get_lm_head(language_model)

    return None


def _install_logits_dtype_hook(model) -> bool:
    logits_processor = _get_logits_processor(model)
    if logits_processor is None:
        return False
    if getattr(logits_processor, "_dtype_hook_installed", False):
        return True

    def _capture_dtype(module, _inputs, output) -> None:
        if output is not None:
            module._last_logits_output_dtype = str(output.dtype)

    logits_processor.register_forward_hook(_capture_dtype)
    logits_processor._dtype_hook_installed = True
    logits_processor._last_logits_output_dtype = None
    return True


def _get_last_logits_output_dtype(model) -> str | None:
    logits_processor = _get_logits_processor(model)
    if logits_processor is None:
        return None
    return getattr(logits_processor, "_last_logits_output_dtype", None)


def _get_fp32_logits_debug_info(model) -> dict[str, object]:
    lm_head = _get_lm_head(model)
    lm_head_weight = getattr(lm_head, "weight", None) if lm_head else None
    return {
        "fp32_env_enabled": envs.VLLM_USE_LOGITS_FP32_COMPUTATION,
        "lm_head_has_weight": bool(lm_head is not None and hasattr(lm_head, "weight")),
        "lm_head_weight_dtype": (
            str(lm_head_weight.dtype) if lm_head_weight is not None else None
        ),
    }


def _capture_generation_snapshot(
    model_path: str,
    tensor_parallel_size: int,
    engine_args: str | None,
) -> GenerationSnapshot:
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=32, logprobs=1, prompt_logprobs=1
    )
    llm = build_c5_llm(model_path, tensor_parallel_size, engine_args)
    try:
        llm.apply_model(_install_logits_dtype_hook)
        outputs = llm.generate(C5_SANITY_PROMPTS, sampling_params=sampling_params)
        logits_output_dtypes = cast(
            list[str | None], llm.apply_model(_get_last_logits_output_dtype)
        )
        fp32_debug_info = llm.apply_model(_get_fp32_logits_debug_info)
    finally:
        shutdown_llm(llm)

    snapshots: list[GenerationCaseSnapshot] = []
    for output in outputs:
        completion = output.outputs[0]
        token_ids = [int(token_id) for token_id in completion.token_ids]
        sampled_logprobs = _extract_sampled_logprobs(token_ids, completion.logprobs)
        prompt_token_ids = (
            [int(t) for t in output.prompt_token_ids] if output.prompt_token_ids else []
        )
        prompt_logprobs = _extract_sampled_logprobs(
            prompt_token_ids, output.prompt_logprobs
        )
        snapshots.append(
            {
                "text": completion.text,
                "token_ids": token_ids,
                "sampled_logprobs": sampled_logprobs,
                "prompt_token_ids": prompt_token_ids,
                "prompt_logprobs": prompt_logprobs,
            }
        )

    return {
        "prompts": C5_SANITY_PROMPTS,
        "snapshots": snapshots,
        "logits_output_dtypes": logits_output_dtypes,
        "fp32_debug_info": fp32_debug_info,
    }


def run_fp32_logits_consistency_test(
    model_path: str,
    tensor_parallel_size: int = 1,
    engine_args: str | None = None,
    max_logprob_abs_diff: float = 0.5,
    max_prompt_logprob_abs_diff: float = 5.0,
    min_shared_prefix: int = 8,
) -> bool:
    """
    Compare generation outputs with and without fp32 logits projection.

    Runs sequentially in a single process, resetting envs cache before each run
    so VLLM_USE_LOGITS_FP32_COMPUTATION is re-evaluated.
    """
    env_key = "VLLM_USE_LOGITS_FP32_COMPUTATION"
    insecure_serialization_env_key = "VLLM_ALLOW_INSECURE_SERIALIZATION"
    original_env_values = {
        env_key: os.environ.get(env_key),
        insecure_serialization_env_key: os.environ.get(insecure_serialization_env_key),
    }
    baseline_data: GenerationSnapshot
    fp32_data: GenerationSnapshot
    try:
        # apply_model in this test sends Python callables over RPC.
        # Enable insecure serialization only for this test flow.
        os.environ[insecure_serialization_env_key] = "1"
        os.environ[env_key] = "0"
        envs.disable_envs_cache()
        baseline_data = _capture_generation_snapshot(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            engine_args=engine_args,
        )

        os.environ[env_key] = "1"
        envs.disable_envs_cache()
        fp32_data = _capture_generation_snapshot(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            engine_args=engine_args,
        )
    finally:
        for key, original_value in original_env_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        envs.disable_envs_cache()

    baseline_snapshots = baseline_data["snapshots"]
    fp32_snapshots = fp32_data["snapshots"]
    fp32_logits_output_dtypes = fp32_data["logits_output_dtypes"]
    fp32_debug_info = fp32_data["fp32_debug_info"]

    if len(baseline_snapshots) != len(fp32_snapshots):
        print("Mismatch in number of outputs between baseline and fp32 runs.")
        return True

    available_fp32_dtypes = [dtype for dtype in fp32_logits_output_dtypes if dtype]
    if not available_fp32_dtypes:
        print("No logits output dtype captured for fp32 run.")
        return True
    invalid_fp32_dtypes = [
        dtype for dtype in available_fp32_dtypes if dtype != "torch.float32"
    ]
    if invalid_fp32_dtypes:
        print(
            "Unexpected logits output dtype(s) for fp32 run: "
            f"{sorted(set(invalid_fp32_dtypes))}"
        )
        print(f"FP32 run debug info: {fp32_debug_info}")
        return True

    max_seen_sampled_diff = 0.0
    comparable_sampled_tokens = 0
    max_seen_prompt_diff = 0.0
    comparable_prompt_tokens = 0
    token_divergences = 0
    errors = 0

    for case_idx, (baseline_case, fp32_case) in enumerate(
        zip(baseline_snapshots, fp32_snapshots)
    ):
        baseline_tokens = baseline_case["token_ids"]
        fp32_tokens = fp32_case["token_ids"]

        shared_prefix_len = 0
        for bt, ft in zip(baseline_tokens, fp32_tokens):
            if bt != ft:
                break
            shared_prefix_len += 1

        if baseline_tokens != fp32_tokens:
            token_divergences += 1
            if shared_prefix_len < min_shared_prefix:
                errors += 1
                print("\n" + "=" * 80)
                print(f"✗ FP32 logits consistency FAILED (case {case_idx})")
                print("-" * 80)
                print(
                    f"Tokens diverged at position {shared_prefix_len}, "
                    f"minimum shared prefix is {min_shared_prefix}."
                )
                print(f"Baseline tokens: {baseline_tokens}")
                print(f"FP32 tokens:     {fp32_tokens}")
                print("=" * 80)
            else:
                print(
                    f"\n  Case {case_idx}: token divergence at position "
                    f"{shared_prefix_len}/{len(baseline_tokens)} "
                    "(expected with precision change)"
                )

        baseline_logprobs = baseline_case["sampled_logprobs"]
        fp32_logprobs = fp32_case["sampled_logprobs"]
        compare_len = (
            shared_prefix_len
            if baseline_tokens != fp32_tokens
            else len(baseline_tokens)
        )
        for token_idx, (baseline_lp, fp32_lp) in enumerate(
            zip(baseline_logprobs[:compare_len], fp32_logprobs[:compare_len])
        ):
            if baseline_lp is None or fp32_lp is None:
                continue
            comparable_sampled_tokens += 1
            abs_diff = abs(baseline_lp - fp32_lp)
            max_seen_sampled_diff = max(max_seen_sampled_diff, abs_diff)
            if abs_diff > max_logprob_abs_diff:
                errors += 1
                print("\n" + "=" * 80)
                print(f"✗ FP32 sampled logprob consistency FAILED (case {case_idx})")
                print("-" * 80)
                print(
                    f"Sampled token index {token_idx} exceeded tolerance "
                    f"({abs_diff:.6f} > {max_logprob_abs_diff:.6f})."
                )
                print(f"Baseline logprob: {baseline_lp:.6f}")
                print(f"FP32 logprob:     {fp32_lp:.6f}")
                print("=" * 80)
                break

        baseline_prompt_lps = baseline_case["prompt_logprobs"]
        fp32_prompt_lps = fp32_case["prompt_logprobs"]
        for token_idx, (baseline_lp, fp32_lp) in enumerate(
            zip(baseline_prompt_lps, fp32_prompt_lps)
        ):
            if baseline_lp is None or fp32_lp is None:
                continue
            comparable_prompt_tokens += 1
            abs_diff = abs(baseline_lp - fp32_lp)
            max_seen_prompt_diff = max(max_seen_prompt_diff, abs_diff)
            if abs_diff > max_prompt_logprob_abs_diff:
                errors += 1
                print("\n" + "=" * 80)
                print(f"✗ FP32 prompt logprob consistency FAILED (case {case_idx})")
                print("-" * 80)
                print(
                    f"Prompt token index {token_idx} exceeded tolerance "
                    f"({abs_diff:.6f} > {max_prompt_logprob_abs_diff:.6f})."
                )
                print(f"Baseline logprob: {baseline_lp:.6f}")
                print(f"FP32 logprob:     {fp32_lp:.6f}")
                print("=" * 80)
                break

    total_comparable = comparable_sampled_tokens + comparable_prompt_tokens
    if total_comparable == 0 and token_divergences == 0:
        print("No comparable logprobs found across runs.")
        return True

    max_seen_diff = max(max_seen_sampled_diff, max_seen_prompt_diff)
    if max_seen_diff <= 0.0 and token_divergences == 0:
        print(
            "Unexpected zero logprob delta between fp32 and baseline runs; "
            "expected max_seen_diff > 0."
        )
        return True

    if errors:
        print(f"\n✗ FP32 logits consistency test failed with {errors} issue(s).")
        return True

    print(
        "\n✓ FP32 logits consistency test PASSED "
        f"(logits dtypes: {sorted(set(available_fp32_dtypes))}, "
        f"max sampled abs diff: {max_seen_sampled_diff:.6f}, "
        f"max prompt abs diff: {max_seen_prompt_diff:.6f}, "
        f"comparable sampled tokens: {comparable_sampled_tokens}, "
        f"comparable prompt tokens: {comparable_prompt_tokens}, "
        f"token divergences: {token_divergences}/{len(baseline_snapshots)} cases)"
    )
    return False


def test_c5_fp32_logits_consistency() -> None:
    assert not run_fp32_logits_consistency_test(
        model_path=_required_model_path(),
        tensor_parallel_size=_tensor_parallel_size(),
        engine_args=_engine_args(),
        max_logprob_abs_diff=_max_logprob_abs_diff(),
        max_prompt_logprob_abs_diff=_max_prompt_logprob_abs_diff(),
        min_shared_prefix=_min_shared_prefix(),
    )
