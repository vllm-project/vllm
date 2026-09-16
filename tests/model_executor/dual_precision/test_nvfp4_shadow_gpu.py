# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-model residency gate for an NVFP4 shadow: Qwen3.5-9B BF16 + ModelOpt NVFP4.

The GPTQ sibling of this test is ``test_shadow_validation_gpu.py``. Nothing in the
residency or binding path is format-aware -- a binding holds two ``LinearBase``
objects and the forward runs through whichever is active -- so what this test
actually gates is that an NVFP4 checkpoint gets past the format gate, that its
linears are recognised as shadow layers (they pack into the plain ``weight`` name,
so the marker is a scale), and that the attached layers still track their BF16
twins numerically.

One GPU, about 30 GiB. Run it through the decision-13 launcher.
"""

import os
from types import SimpleNamespace  # noqa: F401  (parity with the GPTQ sibling)

import pytest
import torch

pytestmark = pytest.mark.gpu_smoke

HF_HUB = os.environ.get("DUAL_PRECISION_HF_HUB", "/data/huggingface/hub")
QWEN35_9B_BF16 = (
    f"{HF_HUB}/models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
QWEN35_9B_NVFP4 = (
    f"{HF_HUB}/models--AxionML--Qwen3.5-9B-NVFP4/snapshots/"
    "97aef92393f126bf649f310cd40861be8dad3279"
)

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
for _path in (QWEN35_9B_BF16, QWEN35_9B_NVFP4):
    if not os.path.isdir(_path):
        pytest.skip(f"checkpoint not available: {_path}", allow_module_level=True)


def _inspect_shadow(model) -> dict:
    """Runs inside the worker via ``LLM.apply_model``."""
    from vllm.model_executor.dual_precision import (
        SHADOW_MODULE_NAME,
        get_dual_precision_state,
    )
    from vllm.model_executor.dual_precision.validation import compare_shadow_numerics

    state = get_dual_precision_state(model)
    assert state is not None
    store = model._modules[SHADOW_MODULE_NAME]
    generator = torch.Generator(device="cuda").manual_seed(0)
    results = []
    nvfp4_marked = 0
    for binding in state.bindings:
        if not binding.shadow_active:
            continue
        shadow = binding.int4_or_fallback
        if hasattr(shadow, "weight_global_scale") or hasattr(shadow, "weight_scale_2"):
            nvfp4_marked += 1
        item = compare_shadow_numerics(
            binding.module_name,
            binding.bf16,
            shadow,
            torch.bfloat16,
            generator=generator,
        )
        results.append((item.name, item.cosine, item.relative_rmse))
    results.sort(key=lambda item: item[1])
    return {
        "attached": state.attached,
        "fallback": state.fallback,
        "policy_bf16": state.policy_bf16,
        "unwrapped": state.unwrapped,
        "store_layers": len(store),
        "store_gib": state.shadow_bytes / (1 << 30),
        "nvfp4_marked": nvfp4_marked,
        "worst": results[:5],
    }


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def llm(monkeypatch_module):
    from vllm import LLM

    # apply_model ships a function to the worker; allow pickle for the test.
    monkeypatch_module.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_ROLLOUT", "1")
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_INT4_MODEL", QWEN35_9B_NVFP4)
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_BF16_LAYERS", "none")
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_INT4_MODULES", "all")
    llm = LLM(
        model=QWEN35_9B_BF16,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
    )
    yield llm
    del llm


def test_nvfp4_shadow_attaches_and_tracks_bf16(llm):
    (report,) = llm.apply_model(_inspect_shadow)

    assert report["attached"] > 0, report
    assert report["store_layers"] == report["attached"]
    assert report["policy_bf16"] == 0 and report["unwrapped"] == 0
    # Every attached shadow layer really is NVFP4, not a BF16 layer that slipped
    # through the predicate.
    assert report["nvfp4_marked"] == report["attached"], report
    _, worst_cosine, _ = report["worst"][0]
    assert worst_cosine >= 0.95, report["worst"]
    print(
        "nvfp4 shadow store GiB:",
        report["store_gib"],
        "attached:",
        report["attached"],
        "worst:",
        report["worst"][:3],
    )


def test_engine_generates_with_nvfp4_shadow_attached(llm):
    from vllm import SamplingParams

    outputs = llm.generate(
        ["The capital of France is"], SamplingParams(temperature=0, max_tokens=8)
    )

    assert outputs[0].outputs[0].text.strip()
