# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for MoE routing replay on Blackwell GPUs.

Correctness strategy:
  - DeepSeek V3 series: Triton vs FI monolithic outputs should match.
"""

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "This test only runs on Blackwell GPUs (SM10x).",
        allow_module_level=True,
    )

HF_OVERRIDE_TEXT = {
    "num_layers": 4,
    "num_hidden_layers": 4,
}

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

DEEPSEEK_COMMON_ARGS: dict[str, Any] = {
    "quantization": "fp8",
}


@pytest.fixture(autouse=True)
def _set_test_environment() -> None:
    os.environ["VLLM_HAS_FLASHINFER_CUBIN"] = "1"
    os.environ["FLASHINFER_NVCC_THREADS"] = "16"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _collect_routed_experts(
    model: str,
    hf_overrides: dict[str, Any] | Callable,
    enforce_eager: bool,
    extra_args: dict[str, Any] | None = None,
) -> list[np.ndarray]:
    defaults: dict[str, Any] = {
        "max_model_len": 2048,
        "max_num_batched_tokens": 256,
        "load_format": "dummy",
        "trust_remote_code": True,
    }
    defaults.update(extra_args or {})
    llm = LLM(
        model=model,
        enable_return_routed_experts=True,
        enforce_eager=enforce_eager,
        hf_overrides=hf_overrides,
        **defaults,
    )

    hf_config = llm.llm_engine.model_config.hf_text_config
    num_experts: int = (
        getattr(hf_config, "num_experts", 0)
        or getattr(hf_config, "num_local_experts", 0)
        or getattr(hf_config, "n_routed_experts", 0)
    )

    outputs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=4))

    results: list[np.ndarray] = []
    assert len(outputs) == len(PROMPTS)
    for i, request_output in enumerate(outputs):
        completion = request_output.outputs[0]
        routed = completion.routed_experts

        assert routed is not None, f"Prompt {i}: routed_experts is None"
        assert routed.ndim >= 2, (
            f"Prompt {i}: expected at least 2D, got shape {routed.shape}"
        )
        assert np.all(routed >= 0), (
            f"Prompt {i}: negative expert IDs, min={routed.min()}"
        )
        assert np.all(routed < num_experts), (
            f"Prompt {i}: expert ID out of range [0, {num_experts}), "
            f"max={routed.max()}"
        )
        results.append(routed)

    del llm
    cleanup_dist_env_and_memory()
    return results


def test_deepseek_fp8_triton_vs_fi_monolithic_routing_replay() -> None:
    """DeepSeek V3: Triton and FI monolithic should produce identical expert IDs."""
    model = "deepseek-ai/DeepSeek-V3.1"

    triton_results = _collect_routed_experts(
        model,
        hf_overrides=HF_OVERRIDE_TEXT,
        enforce_eager=True,
        extra_args={**DEEPSEEK_COMMON_ARGS, "moe_backend": "triton"},
    )

    fi_results = _collect_routed_experts(
        model,
        hf_overrides=HF_OVERRIDE_TEXT,
        enforce_eager=True,
        extra_args={**DEEPSEEK_COMMON_ARGS, "moe_backend": "flashinfer_trtllm"},
    )

    assert len(triton_results) == len(fi_results)
    for i, (triton, fi) in enumerate(zip(triton_results, fi_results, strict=True)):
        assert triton.shape == fi.shape, (
            f"Prompt {i}: shape mismatch triton={triton.shape} vs fi={fi.shape}"
        )
        np.testing.assert_array_equal(
            triton,
            fi,
            err_msg=f"Prompt {i}: expert IDs differ between Triton and "
            f"FI monolithic backends",
        )


def test_deepseek_fp8_block_moe_routing_replay_with_mtp() -> None:
    """FI monolithic + MTP speculative decoding still captures routing."""
    _collect_routed_experts(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        enforce_eager=True,
        extra_args={
            **DEEPSEEK_COMMON_ARGS,
            "moe_backend": "flashinfer_trtllm",
            "speculative_config": {
                "method": "mtp",
                "num_speculative_tokens": 1,
            },
        },
    )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_deepseek_fp8_triton_moe_routing_replay_with_mtp(
    enforce_eager: bool,
) -> None:
    """Triton MoE backend + MTP speculative decoding + routing replay."""
    _collect_routed_experts(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        enforce_eager=enforce_eager,
        extra_args={
            **DEEPSEEK_COMMON_ARGS,
            "moe_backend": "triton",
            "speculative_config": {
                "method": "mtp",
                "num_speculative_tokens": 1,
            },
        },
    )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_deepseek_fp8_triton_moe_routing_replay_with_eagle3(
    enforce_eager: bool,
) -> None:
    """Triton MoE backend + eagle3 spec decode + routing replay.

    The eagle3 draft model is dense (MLP, no MoE) but shares the global
    compilation_config with the target MoE model. With dummy weights the
    draft model uses the same model path as the target.
    """
    _collect_routed_experts(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        enforce_eager=enforce_eager,
        extra_args={
            **DEEPSEEK_COMMON_ARGS,
            "moe_backend": "triton",
            "speculative_config": {
                "method": "eagle3",
                "model": "deepseek-ai/DeepSeek-V3.1",
                "num_speculative_tokens": 3,
            },
        },
    )
