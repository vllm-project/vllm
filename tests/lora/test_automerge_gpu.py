# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU integration tests for the automerge (enable_lora_weight_merge) feature.

Uses Qwen3-0.6B with Meow/Woof LoRA adapters for fast validation.
Requires a single GPU.
"""

import pytest

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_RANK = 8
MAX_MODEL_LEN = 512


def _make_llm(enable_lora_weight_merge: bool = True, **kwargs) -> vllm.LLM:
    defaults = dict(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=LORA_RANK,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        enable_lora_weight_merge=enable_lora_weight_merge,
    )
    defaults.update(kwargs)
    return vllm.LLM(**defaults)


def _chat(llm: vllm.LLM, prompt: str, lora_request=None) -> str:
    messages = [
        {"role": "system", "content": "Follow the instructions to make animal noises"},
        {"role": "user", "content": prompt},
    ]
    params = vllm.SamplingParams(temperature=0, max_tokens=20)
    outputs = llm.chat(
        [messages],
        params,
        chat_template_kwargs={"enable_thinking": False},
        lora_request=lora_request,
    )
    return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# Test: basic automerge produces correct output
# ---------------------------------------------------------------------------


def test_automerge_single_adapter(qwen3_meowing_lora_files):
    """Automerge with a single adapter should produce the same output
    as the standard LoRA path."""
    llm = _make_llm(enable_lora_weight_merge=True)
    lora_req = LoRARequest("meow", 1, qwen3_meowing_lora_files)
    output = _chat(llm, "Make your favorite animal noise.", lora_request=lora_req)
    assert "Meow" in output, f"Expected meowing output, got: {output}"


# ---------------------------------------------------------------------------
# Test: automerge matches standard LoRA output
# ---------------------------------------------------------------------------


def test_automerge_matches_standard_lora(qwen3_meowing_lora_files):
    """Output with automerge should match output without it."""
    prompt = "Make your favorite animal noise."
    lora_req = LoRARequest("meow", 1, qwen3_meowing_lora_files)

    # Standard path
    llm_std = _make_llm(enable_lora_weight_merge=False)
    out_std = _chat(llm_std, prompt, lora_request=lora_req)
    del llm_std

    # Automerge path
    llm_am = _make_llm(enable_lora_weight_merge=True)
    out_am = _chat(llm_am, prompt, lora_request=lora_req)
    del llm_am

    assert out_std == out_am, (
        f"Standard LoRA output: {out_std!r}\nAutomerge output:     {out_am!r}"
    )


# ---------------------------------------------------------------------------
# Test: base model output is clean after adapter removal
# ---------------------------------------------------------------------------


def test_automerge_base_clean_after_lora(qwen3_meowing_lora_files):
    """After serving LoRA requests, base-only requests should produce
    clean base model output (no LoRA contamination)."""
    llm = _make_llm(enable_lora_weight_merge=True)
    lora_req = LoRARequest("meow", 1, qwen3_meowing_lora_files)

    # First: LoRA request (triggers merge)
    out_lora = _chat(llm, "Make your favorite animal noise.", lora_request=lora_req)
    assert "Meow" in out_lora

    # Second: base-only request (should trigger unmerge)
    out_base = _chat(llm, "What is 2 + 2?")
    # Base model should not produce meowing
    assert "Meow" not in out_base


# ---------------------------------------------------------------------------
# Test: repeated merge/unmerge cycles (no drift)
# ---------------------------------------------------------------------------


def test_automerge_no_drift(qwen3_meowing_lora_files):
    """Merge and unmerge 5 times — output should remain consistent."""
    llm = _make_llm(enable_lora_weight_merge=True)
    lora_req = LoRARequest("meow", 1, qwen3_meowing_lora_files)
    prompt = "Make your favorite animal noise."

    first_output = None
    for i in range(5):
        out = _chat(llm, prompt, lora_request=lora_req)
        if first_output is None:
            first_output = out
            assert "Meow" in out, f"Iteration {i}: expected Meow, got {out}"
        else:
            assert out == first_output, (
                f"Drift at iteration {i}: first={first_output!r}, now={out!r}"
            )
        # Trigger unmerge with a base request
        _chat(llm, "Hello")


# ---------------------------------------------------------------------------
# Test: adapter switching (meow -> woof)
# ---------------------------------------------------------------------------


def test_automerge_adapter_switch(qwen3_meowing_lora_files, qwen3_woofing_lora_files):
    """Switching adapters should unmerge the old one and merge the new one."""
    llm = _make_llm(enable_lora_weight_merge=True, max_loras=2)
    prompt = "Make your favorite animal noise."

    meow_req = LoRARequest("meow", 1, qwen3_meowing_lora_files)
    woof_req = LoRARequest("woof", 2, qwen3_woofing_lora_files)

    out_meow = _chat(llm, prompt, lora_request=meow_req)
    assert "Meow" in out_meow, f"Expected meowing, got: {out_meow}"

    out_woof = _chat(llm, prompt, lora_request=woof_req)
    assert "Woof" in out_woof, f"Expected woofing, got: {out_woof}"

    # Switch back to meow
    out_meow2 = _chat(llm, prompt, lora_request=meow_req)
    assert "Meow" in out_meow2, f"Expected meowing again, got: {out_meow2}"


# ---------------------------------------------------------------------------
# Test: fallback with max_loras > 1 and multi-adapter batch
# ---------------------------------------------------------------------------


def test_automerge_fallback_multi_adapter():
    """With max_loras > 1, automerge should fall back to standard path
    when multiple adapters are in the same batch. This test just verifies
    the engine starts and serves without errors."""
    llm = _make_llm(enable_lora_weight_merge=True, max_loras=4)
    # Base-only request should work fine
    out = _chat(llm, "What is the capital of France?")
    assert len(out) > 0


# ---------------------------------------------------------------------------
# Test: golden_device modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("golden_device", ["cpu", "gpu", "off"])
def test_automerge_golden_device_modes(qwen3_meowing_lora_files, golden_device):
    """All three golden_device modes should produce correct output."""
    llm = _make_llm(
        enable_lora_weight_merge=True,
        lora_weight_merge_golden_device=golden_device,
    )
    lora_req = LoRARequest("meow", 1, qwen3_meowing_lora_files)
    out = _chat(llm, "Make your favorite animal noise.", lora_request=lora_req)
    assert "Meow" in out, f"golden_device={golden_device}: expected Meow, got: {out}"


# ---------------------------------------------------------------------------
# Test: off mode adapter switching (the bug that was found)
# ---------------------------------------------------------------------------


def test_automerge_off_mode_adapter_switch(
    qwen3_meowing_lora_files, qwen3_woofing_lora_files
):
    """Off mode must correctly restore packed layers when switching adapters."""
    llm = _make_llm(
        enable_lora_weight_merge=True,
        lora_weight_merge_golden_device="off",
        max_loras=2,
    )
    prompt = "Make your favorite animal noise."
    meow_req = LoRARequest("meow", 1, qwen3_meowing_lora_files)
    woof_req = LoRARequest("woof", 2, qwen3_woofing_lora_files)

    out_meow = _chat(llm, prompt, lora_request=meow_req)
    assert "Meow" in out_meow, f"Expected meowing, got: {out_meow}"

    out_woof = _chat(llm, prompt, lora_request=woof_req)
    assert "Woof" in out_woof, f"Expected woofing, got: {out_woof}"

    out_meow2 = _chat(llm, prompt, lora_request=meow_req)
    assert "Meow" in out_meow2, f"Expected meowing again, got: {out_meow2}"
