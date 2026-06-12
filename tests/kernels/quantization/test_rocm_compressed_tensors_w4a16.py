#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for CT W4A16 models on ROCm.

This validates that a real compressed-tensors W4A16 model can run inference
end-to-end (which will exercise the Hybrid W4A16 kernel when selected).

Run `pytest tests/kernels/quantization/test_rocm_compressed_tensors_w4a16.py`.
"""

import pytest

from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "model_path",
    [
        # Listed in tests/weight_loading/models.txt
        "nm-testing/tinyllama-oneshot-w4a16-group128-v2",
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="Should only run on ROCm")
def test_rocm_compressed_tensors_w4a16_e2e(
    vllm_runner, example_prompts, model_path, max_tokens
):
    # Use fp16 activations for maximum compatibility.
    # gpu_memory_utilization=0.35: must satisfy two constraints simultaneously:
    #   1. High enough that vLLM has a non-negative KV-cache budget after
    #      torch + MIOpen reserve memory on top of the model weights.
    #   2. Low enough that (1 - gmu) > baseline_vram_ratio so that
    #      _wait_for_rocm_memory_release can clear after teardown.
    #      On Strix Halo CI runners amdsmi reports 512 MiB dedicated VRAM with
    #      ~314 MiB (61%) in use at ROCm baseline; threshold = 1 - 0.35 = 0.65
    #      gives a 332 MiB ceiling, safely above the 314 MiB floor.
    with vllm_runner(
        model_path, dtype="float16", gpu_memory_utilization=0.35
    ) as vllm_model:
        # Note: we cannot assert HybridW4A16LinearKernel is selected here
        # because V1 engine runs the model in a subprocess and apply_model
        # requires serializable callables (msgpack can't serialize functions).
        # If the W4A16 kernel is broken, generate_greedy will throw.
        vllm_model.generate_greedy(example_prompts, max_tokens=max_tokens)
