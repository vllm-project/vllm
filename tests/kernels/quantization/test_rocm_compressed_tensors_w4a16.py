#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for CT W4A16 models on ROCm.

This validates that a real compressed-tensors W4A16 model can run inference
end-to-end (which will exercise the Triton W4A16 kernel when selected).

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
    # gpu_memory_utilization lowered to work on shared nodes.
    with vllm_runner(
        model_path, dtype="float16", gpu_memory_utilization=0.3
    ) as vllm_model:
        # If the W4A16 kernel is broken, this will typically throw.
        vllm_model.generate_greedy(example_prompts, max_tokens=max_tokens)
