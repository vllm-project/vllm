# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E test for the ECCPUConnector.

Verifies:
- Accuracy: outputs from EC CPU cache match fresh encoder computation.
- Latency: loading from EC CPU cache is faster than a cold encoder run.

Requires a CUDA GPU and the Qwen2-VL-2B-Instruct model.
"""

import time

import pytest
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import ECTransferConfig
from vllm.platforms import current_platform

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
EC_CPU_BYTES = 500 << 20  # 500 MiB


def _build_image_prompt(image):
    """Build a Qwen2-VL prompt with a single image."""
    return {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "Describe this image briefly.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {"image": image},
    }


def _make_unique_image(base_image, iteration: int):
    """Create a unique image by modifying a pixel to change the mm_hash."""
    img = base_image.copy()
    img.putpixel((0, 0), (iteration % 256, iteration // 256 % 256, 0))
    return img


def _wait_for_ec_ready(llm: LLM) -> None:
    """Send a dummy request to force a scheduler step.

    After generate() returns, the EC entry's data is in the mmap but
    readiness may not have fired yet (first-finish fires in the NEXT
    scheduler step). This forces that step.
    """
    llm.generate("hi", SamplingParams(max_tokens=1), use_tqdm=False)


def _latency_test(llm: LLM) -> None:
    """Verify EC CPU cache hit is faster than cold encoder computation."""
    if not current_platform.is_cuda():
        pytest.skip("Latency test requires CUDA")

    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    base_image = ImageAsset("stop_sign").pil_image

    num_times_ec_better_than_cold = 0
    num_tests = 10
    total_cold_time = 0.0
    total_ec_hit_time = 0.0

    for i in tqdm(range(num_tests), desc="EC latency test"):
        # Use a unique image each iteration so the cold run is truly cold
        # (EC has never seen this mm_hash before).
        image = _make_unique_image(base_image, i)
        prompt = _build_image_prompt(image)

        # Cold run: encoder computes from scratch, saves to EC CPU.
        llm.llm_engine.reset_encoder_cache()
        start_time = time.time()
        llm.generate([prompt], sampling_params, use_tqdm=False)
        cold_time = time.time() - start_time
        total_cold_time += cold_time

        # Ensure EC entry is marked ready.
        _wait_for_ec_ready(llm)

        # Clear GPU encoder cache but keep EC CPU cache intact.
        llm.llm_engine.reset_encoder_cache()

        # EC hit: scheduler loads from CPU mmap instead of re-encoding.
        start_time = time.time()
        llm.generate([prompt], sampling_params, use_tqdm=False)
        ec_hit_time = time.time() - start_time
        total_ec_hit_time += ec_hit_time

        if ec_hit_time < cold_time:
            num_times_ec_better_than_cold += 1

    print("Average times:")
    print(f"    Cold: {total_cold_time * 1000 / num_tests:.2f}ms")
    print(f"    EC hit: {total_ec_hit_time * 1000 / num_tests:.2f}ms")

    assert num_times_ec_better_than_cold >= 0.8 * num_tests, (
        f"EC hit was faster only {num_times_ec_better_than_cold}/{num_tests} "
        f"times (expected >= 80%)"
    )


def _accuracy_test(llm: LLM) -> None:
    """Verify EC CPU cache produces correct outputs."""
    sampling_params = SamplingParams(max_tokens=8, temperature=0)
    image = ImageAsset("stop_sign").pil_image
    prompt = _build_image_prompt(image)

    # Seed: generate with fresh encoder to establish baseline output.
    llm.llm_engine.reset_encoder_cache()
    baseline = llm.generate([prompt], sampling_params, use_tqdm=False)
    baseline_text = baseline[0].outputs[0].text

    # Ensure EC entry is marked ready.
    _wait_for_ec_ready(llm)

    # Clear GPU encoder cache so subsequent runs go through EC CPU path.
    llm.llm_engine.reset_encoder_cache()

    # Generate multiple times from EC cache and check correctness.
    test_count = 20
    results = llm.generate([prompt] * test_count, sampling_params, use_tqdm=False)

    success_count = sum(1 for r in results if r.outputs[0].text == baseline_text)
    assert success_count >= 0.5 * test_count, (
        f"Only {success_count}/{test_count} outputs matched baseline "
        f"(expected >= 50%). Baseline: {baseline_text!r}"
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_ec_cpu_offloading() -> None:
    """Tests ECCPUConnector accuracy and latency with a VLM model."""
    ec_transfer_config = ECTransferConfig(
        ec_connector="ECCPUConnector",
        ec_role="ec_both",
        ec_connector_extra_config={"ec_cpu_bytes": EC_CPU_BYTES},
    )

    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        ec_transfer_config=ec_transfer_config,
    )

    try:
        _latency_test(llm)
        _accuracy_test(llm)
    finally:
        del llm
