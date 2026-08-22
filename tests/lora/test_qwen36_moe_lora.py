# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm
import vllm.config
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest

from ..utils import multi_gpu_test

MODEL_PATH = "Qwen/Qwen3.6-35B-A3B"

LORA_2D_ID = 1
LORA_3D_ID = 2

PROMPT_TEMPLATE = """<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What is in the image?<|im_end|>
<|im_start|>assistant
<think>

</think>
"""

# Visual captioning prompts: each image will be paired with one LoRA in the
# mixed-batch case so we can check per-prompt routing.
VL_TEST_IMAGES = [
    ImageAsset("stop_sign"),
    ImageAsset("cherry_blossom"),
]


def _build_prompts() -> list[dict]:
    return [
        {
            "prompt": PROMPT_TEMPLATE,
            "multi_modal_data": {"image": asset.pil_image},
        }
        for asset in VL_TEST_IMAGES
    ]


def _generate(llm: vllm.LLM, lora_request) -> list[str]:
    outputs = llm.generate(
        _build_prompts(),
        vllm.SamplingParams(temperature=0, max_tokens=128),
        lora_request=lora_request,
    )
    return [out.outputs[0].text.strip() for out in outputs]


def _run_mixed_2d_3d_lora_test(
    lora_2d_files: str,
    lora_3d_files: str,
    tensor_parallel_size: int,
    fully_sharded_loras: bool,
) -> None:
    llm = vllm.LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        enable_lora=True,
        enable_mixed_moe_lora_format=True,
        max_loras=2,
        max_lora_rank=8,
        max_num_seqs=4,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=not fully_sharded_loras,
        fully_sharded_loras=fully_sharded_loras,
        trust_remote_code=True,
        enable_tower_connector_lora=True,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 1},
        compilation_config=vllm.config.CompilationConfig(
            cudagraph_specialize_lora=False,
        ),
    )

    lora_2d = LoRARequest(
        "lora_2d",
        LORA_2D_ID,
        lora_2d_files,
        is_3d_lora_weight=False,
    )
    lora_3d = LoRARequest(
        "lora_3d",
        LORA_3D_ID,
        lora_3d_files,
        is_3d_lora_weight=True,
    )

    # Reference: each adapter alone over both prompts.
    outputs_2d_alone = _generate(llm, lora_2d)
    outputs_3d_alone = _generate(llm, lora_3d)

    assert len(outputs_2d_alone) == len(VL_TEST_IMAGES)
    assert len(outputs_3d_alone) == len(VL_TEST_IMAGES)
    for text in outputs_2d_alone + outputs_3d_alone:
        assert text, "Empty output from single-adapter LoRA generation"

    # Mixed batch: prompt 0 uses the 2D adapter, prompt 1 uses the 3D
    # adapter. Per-prompt outputs must match the standalone runs.
    mixed_outputs = _generate(llm, [lora_2d, lora_3d])

    assert mixed_outputs[0] == outputs_2d_alone[0], (
        f"Mixed-batch 2D output {mixed_outputs[0]!r} does not match "
        f"standalone 2D output {outputs_2d_alone[0]!r}"
    )
    assert mixed_outputs[1] == outputs_3d_alone[1], (
        f"Mixed-batch 3D output {mixed_outputs[1]!r} does not match "
        f"standalone 3D output {outputs_3d_alone[1]!r}"
    )

    # Reverse assignment: neither adapter should be silently aliased.
    swapped_outputs = _generate(llm, [lora_3d, lora_2d])
    assert swapped_outputs[0] == outputs_3d_alone[0], (
        f"Swapped-batch 3D output {swapped_outputs[0]!r} does not match "
        f"standalone 3D output {outputs_3d_alone[0]!r}"
    )
    assert swapped_outputs[1] == outputs_2d_alone[1], (
        f"Swapped-batch 2D output {swapped_outputs[1]!r} does not match "
        f"standalone 2D output {outputs_2d_alone[1]!r}"
    )


@pytest.mark.skip(reason="This model is too big, so skip this test temporarily.")
@pytest.mark.parametrize("fully_sharded_loras", [False, True])
@multi_gpu_test(num_gpus=2)
def test_qwen36_moe_mixed_2d_3d_lora_tp2(
    qwen36_moe_2d_lora_files,
    qwen36_moe_3d_lora_files,
    fully_sharded_loras,
):
    _run_mixed_2d_3d_lora_test(
        lora_2d_files=qwen36_moe_2d_lora_files,
        lora_3d_files=qwen36_moe_3d_lora_files,
        tensor_parallel_size=2,
        fully_sharded_loras=fully_sharded_loras,
    )


@pytest.mark.skip(reason="This model is too big, so skip this test temporarily.")
@pytest.mark.parametrize("fully_sharded_loras", [False, True])
@multi_gpu_test(num_gpus=4)
def test_qwen36_moe_mixed_2d_3d_lora_tp4(
    qwen36_moe_2d_lora_files,
    qwen36_moe_3d_lora_files,
    fully_sharded_loras,
):
    _run_mixed_2d_3d_lora_test(
        lora_2d_files=qwen36_moe_2d_lora_files,
        lora_3d_files=qwen36_moe_3d_lora_files,
        tensor_parallel_size=4,
        fully_sharded_loras=fully_sharded_loras,
    )
