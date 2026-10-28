# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm import SamplingParams
from vllm.model_executor.models.qwen3_longspark import _LongSparkRMSNorm


def _rms_norm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    normalized = hidden_states.float()
    if residual is not None:
        normalized = normalized + residual.float()
        residual = normalized.to(hidden_states.dtype)
    variance = normalized.pow(2).mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    output = (normalized.to(weight.dtype) * weight).to(hidden_states.dtype)
    return output, residual


@pytest.mark.parametrize(
    ("batch_size", "rows", "hidden_size"),
    [(1, 3, 64), (5, 7, 2560)],
)
@pytest.mark.parametrize("add_residual", [False, True])
def test_longspark_rms_norm_matches_native_bf16(
    batch_size: int,
    rows: int,
    hidden_size: int,
    add_residual: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(53)
    eps = 1e-6
    norm = _LongSparkRMSNorm(hidden_size, eps).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.normal_(generator=generator)
    hidden_states = torch.randn(
        batch_size,
        rows,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    residual = (
        torch.randn_like(hidden_states, generator=generator) if add_residual else None
    )
    expected, expected_residual = _rms_norm_reference(
        hidden_states, norm.weight, eps, residual
    )

    if residual is None:
        actual = norm(hidden_states)
        assert isinstance(actual, torch.Tensor)
    else:
        actual, actual_residual = norm(hidden_states.clone(), residual.clone())
        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)

    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-3)


@pytest.mark.slow_test
def test_longspark_draft_width_preserves_greedy_outputs(vllm_runner, monkeypatch):
    """Shorter drafts must preserve target outputs and propose exactly K tokens."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    target = os.getenv("LONGSPARK_TEST_TARGET_MODEL", "Qwen/Qwen3-4B")
    draft = os.getenv("LONGSPARK_TEST_DRAFT_MODEL", "Hehy/LongSpark-Qwen3-4B")
    prompts = [
        "Explain why the night sky is dark.",
        "Write a Python function that returns the first n Fibonacci numbers.",
    ]
    sampling = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    runner_kwargs = dict(
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        attention_config={"backend": "FLASH_ATTN"},
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
        disable_log_stats=False,
    )
    outputs_by_width = {}
    for width, enforce_eager in ((1, False), (3, True), (7, False)):
        with vllm_runner(
            target,
            **runner_kwargs,
            enforce_eager=enforce_eager,
            speculative_config={
                "method": "longspark",
                "model": draft,
                "num_speculative_tokens": width,
            },
        ) as runner:
            outputs_by_width[width] = runner.generate(prompts, sampling)
            metrics = {m.name: m for m in runner.llm.get_metrics()}
            steps = metrics["vllm:spec_decode_num_drafts"].value
            proposed = metrics["vllm:spec_decode_num_draft_tokens"].value
            accepted = metrics["vllm:spec_decode_num_accepted_tokens"].value
            assert steps > 0 and proposed == width * steps
            assert accepted > 0
            print(
                f"LongSpark width={width}: drafts={steps}, proposed={proposed}, "
                f"accepted={accepted}, enforce_eager={enforce_eager}"
            )
    with vllm_runner(target, **runner_kwargs) as runner:
        reference = runner.generate(prompts, sampling)
    for width, outputs in outputs_by_width.items():
        assert outputs == reference, f"Greedy outputs changed at draft width {width}"
