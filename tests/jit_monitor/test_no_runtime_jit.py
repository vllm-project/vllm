# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Catch runtime (post-warmup) JIT compilations for JIT-heavy backends running
e2e tests on popular models, for which we only load a few blocks for performance.

NOTE(NickLucche) With cuda graphs on, kernels fully covered by graphs captured during
warmup do not re-trigger the Python JIT hooks. The targeted paths (prefill
MoE/MLA/SSM and the sampler) run mixed, so they are unaffected.
"""

from dataclasses import dataclass
from functools import partial

import pytest

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from ..models.utils import dummy_hf_overrides
from ..utils import create_new_process_for_each_test


@dataclass(frozen=True)
class JitModel:
    model: str
    draft: str | None = None
    trust_remote_code: bool = False
    num_dummy_layers: int | None = None
    attention_backend: str | None = None


JIT_MONITOR_MODELS = [
    JitModel(
        "deepseek-ai/DeepSeek-V4-Flash",
        draft="deepseek-ai/DeepSeek-V4-Flash",
        trust_remote_code=True,
        num_dummy_layers=4,
        attention_backend="FLASH_ATTN"
    ),
    JitModel(
        "deepseek-ai/DeepSeek-V4-Flash",
        draft="deepseek-ai/DeepSeek-V4-Flash",
        trust_remote_code=True,
        num_dummy_layers=4,
    ),
    JitModel(
        "deepseek-ai/DeepSeek-V4-Pro",
        trust_remote_code=True,
        num_dummy_layers=4,
    ),
    JitModel(
        "nvidia/DeepSeek-V4-Flash-NVFP4",
        draft="deepseek-ai/DeepSeek-V4-Flash",
        trust_remote_code=True,
        num_dummy_layers=4,
    ),
    JitModel(
        "google/gemma-4-E2B-it", trust_remote_code=True
    ),
    JitModel(
        "google/gemma-4-31B-it",
        draft="google/gemma-4-31B-it",
        trust_remote_code=True),
    JitModel(
        "openai/gpt-oss-120b", trust_remote_code=True
    ),
]


def _run_shape_battery(llm: LLM, *, speculative: bool) -> None:
    """Exercise diverse compile keys so missing warmup keys surface.

    Token-id prompts keep shapes exact and avoid depending on a tokenizer.
    Outputs are meaningless under dummy weights; we assert only that no JIT
    fired.
    """
    short = TokensPrompt(prompt_token_ids=[1, 2, 3, 4])
    medium = TokensPrompt(prompt_token_ids=list(range(1, 33)))
    long = TokensPrompt(prompt_token_ids=list(range(1, 129)))

    # Greedy single-sequence multi-step decode: prefill + autoregressive decode
    # + greedy sampler.
    llm.generate(medium, SamplingParams(temperature=0.0, max_tokens=16))

    # Batched prefill with mixed lengths: varlen prefill + padded decode.
    llm.generate([short, medium, long], SamplingParams(temperature=0.0, max_tokens=8))

    # Long prefill with only one output token: exercise the large-query FA4
    # path without mixing in a multi-step decode workload.
    llm.generate(
        TokensPrompt(prompt_token_ids=list(range(1, 1537))),
        SamplingParams(temperature=0.0, max_tokens=1),
    )

    # Triton sampler kernels: top_k / top_p each specialize.
    for sampling_params in (
        SamplingParams(temperature=0.8, top_k=20, max_tokens=8, seed=0),
        SamplingParams(temperature=0.8, top_p=0.9, max_tokens=8, seed=0),
        SamplingParams(
            temperature=0.8, top_k=20, top_p=0.9, max_tokens=8, seed=0
        ),
    ):
        llm.generate(medium, sampling_params)

    # Heterogeneous SamplingParams in one step, where missing sampler warmup
    # keys most often hide.
    llm.generate(
        [medium] * 3,
        [
        SamplingParams(temperature=0.0, max_tokens=8),
        SamplingParams(temperature=0.8, top_k=20, max_tokens=8, seed=0),
        SamplingParams(temperature=0.8, top_p=0.9, max_tokens=8, seed=0),
        ],
    )


@create_new_process_for_each_test("spawn")
def can_run_without_jit(spec: JitModel):
    """Boot ``spec`` with the monitor armed and run the shape battery.

    A subprocess per model is required: the monitor's hooks are process-global
    and, once armed in ``error`` mode, stay armed. It must be spawned rather
    than forked, since forking a pytest process that already initialized CUDA
    poisons the child.
    """
    llm = LLM(
        spec.model,
        trust_remote_code=spec.trust_remote_code,
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.80,
        load_format="dummy",
        hf_overrides=(
            partial(dummy_hf_overrides, num_dummy_layers=spec.num_dummy_layers)
            if spec.num_dummy_layers is not None
            else dummy_hf_overrides
        ),
        attention_config=(
            {"backend": spec.attention_backend}
            if spec.attention_backend is not None
            else None
        ),
        # cuda graphs cover captured decode shapes, run eager.
        enforce_eager=False,
        jit_monitor_mode="error",
        jit_monitor_triton=False, #TODO(roberto): set to True
        jit_monitor_verbose=True,
        speculative_config={
            "model": spec.draft,
            "num_speculative_tokens": 2,
        }
        if spec.draft
        else None,
    )

    try:
        _run_shape_battery(llm, speculative=spec.draft is not None)
    except Exception as e:
        # The monitor's message contains "during inference"; distinguish a real
        # JIT miss from an unrelated crash.
        if "during inference" in str(e):
            pytest.fail(
                f"{spec.model}: post-warmup JIT compilation detected - a warmup "
                f"key is missing for a shape in the battery.\n{e}"
            )
        raise


@pytest.mark.parametrize(
    "spec",
    JIT_MONITOR_MODELS,
    ids=lambda s: f"{s.model}-{'mtp' if s.draft else 'base'}",
)
def test_no_runtime_jit(spec: JitModel, monkeypatch: pytest.MonkeyPatch):
    """Assert JIT-heavy backends do not JIT-compile during inference."""
    # Set here rather than in the child so the spawned process inherits it:
    # the engine core must not be forked once the test process has CUDA up.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    can_run_without_jit(spec)
