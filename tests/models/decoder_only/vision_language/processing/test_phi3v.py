"""Tests for phi3v's multimodal preprocessing kwargs."""
import pytest
from transformers import AutoTokenizer

from vllm.inputs import InputProcessingContext
from vllm.model_executor.models.phi3v import _IMAGE_TOKEN_ID

from .....conftest import _ImageAssets
from ....utils import build_model_context


# Wrap lazy imports to avoid initializing CUDA during test collection
@pytest.fixture()
def processor_for_phi3v():
    from vllm.model_executor.models.phi3v import Phi3VMultiModalProcessor
    return Phi3VMultiModalProcessor


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
def test_processor_override(
    processor_for_phi3v,
    image_assets: _ImageAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, int],
    expected_toks_per_img: int,
    num_imgs: int,
):
    """Ensure input_processor_for_phi3v handles num_crops properly."""
    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": num_imgs},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ctx = InputProcessingContext(ctx.model_config, tokenizer)

    # Build the image str / prompt based on the number of images we pass
    img_str = "".join([f"<|image_{idx}|>\n" for idx in range(1, num_imgs + 1)])
    prompt = f"<|user|>\n{img_str}<|end|>\n<|assistant|>\n"
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processor = processor_for_phi3v(ctx)
    processed_inputs = processor.apply(prompt, mm_data, mm_processor_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    img_tok_count = processed_inputs["prompt_token_ids"].count(_IMAGE_TOKEN_ID)
    assert img_tok_count == expected_toks_per_img * num_imgs
