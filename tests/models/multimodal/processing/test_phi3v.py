# SPDX-License-Identifier: Apache-2.0
"""Tests for phi3v's multimodal preprocessing kwargs."""
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from ....conftest import _ImageAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["microsoft/Phi-3.5-vision-instruct"])
# yapf: disable
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img"),
    [
        ({"num_crops": 4}, 757),
        ({"num_crops": 16}, 1921),
        # the default num_crops of phi-3.5-vision is 4
        ({}, 757),
    ])
# yapf: enable
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    image_assets: _ImageAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, int],
    expected_toks_per_img: int,
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure input_processor_for_phi3v handles num_crops properly."""
    # Avoid initializing CUDA early
    from vllm.model_executor.models.phi3v import _IMAGE_TOKEN_ID

    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        trust_remote_code=True,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    tokenizer = cached_tokenizer_from_config(ctx.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=tokenizer,
    )
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Build the image str / prompt based on the number of images we pass
    img_str = "".join([f"<|image_{idx}|>\n" for idx in range(1, num_imgs + 1)])
    prompt = f"<|user|>\n{img_str}<|end|>\n<|assistant|>\n"
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    img_tok_count = processed_inputs["prompt_token_ids"].count(_IMAGE_TOKEN_ID)
    assert img_tok_count == expected_toks_per_img * num_imgs
