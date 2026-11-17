# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from vllm import SamplingParams

from ....conftest import VllmRunner


@pytest.mark.core_model
@pytest.mark.parametrize("image_counts", [(1, 3), (3, 1)])
def test_deepseek_vl2_encoder_cache_cropping(
    vllm_runner: type[VllmRunner], image_counts: tuple[int, int]
):
    images = [Image.new("RGB", (768, 768), color) for color in ("red", "green", "blue")]
    with vllm_runner(
        "deepseek-ai/deepseek-vl2-tiny",
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        max_model_len=4096,
        max_num_seqs=1,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 3},
    ) as runner:
        for count in image_counts:
            prompt = "<|User|>: " + "<image>\n" * count + "Describe.\n\n<|Assistant|>:"
            outputs = runner.llm.generate(
                {"prompt": prompt, "multi_modal_data": {"image": images[:count]}},
                SamplingParams(temperature=0, max_tokens=1),
            )
            output = outputs[0].outputs[0]
            # A cache collision makes the scheduler reject the second request
            # with finish_reason="error" due to mismatched embedding counts.
            assert output.finish_reason == "length"
            assert len(output.token_ids) == 1


@pytest.mark.core_model
def test_deepseek_vl2_request_processor_kwargs(vllm_runner):
    """Generate with a request override and verify the processed image span."""
    from PIL import Image

    from vllm import SamplingParams

    model_id = "deepseek-ai/deepseek-vl2-tiny"
    with vllm_runner(
        model_id,
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.2,
        kv_cache_memory_bytes=1 << 29,
        attention_config={"backend": "TRITON_ATTN"},
        mm_encoder_attn_backend="TORCH_SDPA",
        limit_mm_per_prompt={"image": 1},
    ) as runner:
        result = runner.llm.generate(
            {
                "prompt": "<image>\nDescribe.",
                "multi_modal_data": {"image": Image.new("RGB", (1600, 1200))},
                "mm_processor_kwargs": {"candidate_resolutions": ((384, 384),)},
            },
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        )[0]
        assert result.outputs[0].finish_reason == "length"
        assert len(result.outputs[0].token_ids) == 1
        expected_tokens = 421
        image_token_id = runner.llm.get_tokenizer().convert_tokens_to_ids("<image>")
        assert result.prompt_token_ids.count(image_token_id) == expected_tokens
