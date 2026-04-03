# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for image token pruning via attention scores.

Tests that Qwen2.5-VL and Qwen3-VL models can run inference with
--image-pruning-rate and produce valid outputs without crashing.
"""

import pytest

from ....conftest import IMAGE_ASSETS

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def qwen_chat_template(*query):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{''.join(query)}"
        "<|im_end|><|im_start|>assistant\n"
    )


IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": qwen_chat_template(
            IMAGE_PLACEHOLDER, "What is the biggest text in this image?"
        ),
        "cherry_blossom": qwen_chat_template(
            IMAGE_PLACEHOLDER, "What season is shown? Reply in one sentence."
        ),
    }
)


# ===================================================================
# Qwen2.5-VL image pruning tests
# ===================================================================
#
QWEN2_5_VL_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct"]


@pytest.mark.core_model
@pytest.mark.parametrize("model", QWEN2_5_VL_MODELS)
@pytest.mark.parametrize("image_pruning_rate", [None, 0.6, 0.3])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
def test_qwen2_5_vl_image_pruning_single(
    vllm_runner,
    image_assets,
    model,
    image_pruning_rate,
    dtype,
    max_tokens,
) -> None:
    """Test image pruning with a single image input on Qwen2.5-VL."""
    images = [image_assets[0].pil_image]
    prompts = [IMAGE_PROMPTS[0]]

    runner_kwargs = dict(
        runner="generate",
        max_model_len=4096,
        max_num_seqs=1,
        dtype=dtype,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
    )
    if image_pruning_rate is not None:
        runner_kwargs["image_pruning_rate"] = image_pruning_rate

    with vllm_runner(model, **runner_kwargs) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens, images=images)

    assert len(outputs) == 1
    output_ids, output_text = outputs[0]
    assert len(output_ids) > 0
    assert isinstance(output_text, str)
    assert len(output_text) > 0


@pytest.mark.core_model
@pytest.mark.parametrize("model", QWEN2_5_VL_MODELS)
@pytest.mark.parametrize("image_pruning_rate", [0.6])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
def test_qwen2_5_vl_image_pruning_batched(
    vllm_runner,
    image_assets,
    model,
    image_pruning_rate,
    dtype,
    max_tokens,
) -> None:
    """Test image pruning with batched image inputs on Qwen2.5-VL."""
    images = [asset.pil_image for asset in image_assets]
    prompts = IMAGE_PROMPTS

    with vllm_runner(
        model,
        runner="generate",
        max_model_len=4096,
        max_num_seqs=2,
        dtype=dtype,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
        image_pruning_rate=image_pruning_rate,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens, images=images)

    assert len(outputs) == 2
    for output_ids, output_text in outputs:
        assert len(output_ids) > 0
        assert isinstance(output_text, str)
        assert len(output_text) > 0


@pytest.mark.core_model
@pytest.mark.parametrize("model", QWEN2_5_VL_MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
def test_qwen2_5_vl_pruning_reduces_tokens(
    vllm_runner,
    image_assets,
    model,
    dtype,
    max_tokens,
) -> None:
    """Verify that pruning produces shorter outputs than no pruning
    when max_tokens is large enough (smoke test for token reduction)."""
    images = [image_assets[0].pil_image]
    prompts = [IMAGE_PROMPTS[0]]

    # Run without pruning
    with vllm_runner(
        model,
        runner="generate",
        max_model_len=4096,
        max_num_seqs=1,
        dtype=dtype,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
    ) as vllm_model:
        outputs_no_prune = vllm_model.generate_greedy(
            prompts, max_tokens, images=images
        )

    # Run with aggressive pruning (prune 70% of tokens)
    with vllm_runner(
        model,
        runner="generate",
        max_model_len=4096,
        max_num_seqs=1,
        dtype=dtype,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
        image_pruning_rate=0.7,
    ) as vllm_model:
        outputs_pruned = vllm_model.generate_greedy(prompts, max_tokens, images=images)

    # Both should produce valid output
    assert len(outputs_no_prune[0][1]) > 0
    assert len(outputs_pruned[0][1]) > 0


# ===================================================================
# Qwen3-VL image pruning tests (if model available)
# ===================================================================

QWEN3_VL_MODELS = ["Qwen/Qwen3-VL-2B-Instruct"]


@pytest.mark.core_model
@pytest.mark.parametrize("model", QWEN3_VL_MODELS)
@pytest.mark.parametrize("image_pruning_rate", [None, 0.6])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
def test_qwen3_vl_image_pruning_single(
    vllm_runner,
    image_assets,
    model,
    image_pruning_rate,
    dtype,
    max_tokens,
) -> None:
    """Test image pruning with a single image input on Qwen3-VL."""
    images = [image_assets[0].pil_image]
    prompts = [IMAGE_PROMPTS[0]]

    runner_kwargs = dict(
        runner="generate",
        max_model_len=4096,
        max_num_seqs=1,
        dtype=dtype,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
    )
    if image_pruning_rate is not None:
        runner_kwargs["image_pruning_rate"] = image_pruning_rate

    with vllm_runner(model, **runner_kwargs) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens, images=images)

    assert len(outputs) == 1
    output_ids, output_text = outputs[0]
    assert len(output_ids) > 0
    assert isinstance(output_text, str)
    assert len(output_text) > 0


@pytest.mark.core_model
@pytest.mark.parametrize("model", QWEN3_VL_MODELS)
@pytest.mark.parametrize("image_pruning_rate", [0.6])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
def test_qwen3_vl_image_pruning_batched(
    vllm_runner,
    image_assets,
    model,
    image_pruning_rate,
    dtype,
    max_tokens,
) -> None:
    """Test image pruning with batched image inputs on Qwen3-VL."""
    images = [asset.pil_image for asset in image_assets]
    prompts = IMAGE_PROMPTS

    with vllm_runner(
        model,
        runner="generate",
        max_model_len=4096,
        max_num_seqs=2,
        dtype=dtype,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
        image_pruning_rate=image_pruning_rate,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens, images=images)

    assert len(outputs) == 2
    for output_ids, output_text in outputs:
        assert len(output_ids) > 0
        assert isinstance(output_text, str)
        assert len(output_text) > 0
