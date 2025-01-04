import pytest
from transformers import AutoTokenizer

from vllm.inputs import InputProcessingContext

from .....conftest import _ImageAssets
from ....utils import build_model_context


# Fixtures lazy import to avoid initializing CUDA during test collection
@pytest.fixture()
def processor_for_qwen2_vl():
    from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalProcessor
    return Qwen2VLMultiModalProcessor


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])
# yapf: disable
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img", "expected_pixels_shape"), [
        ({}, 1426, (5704, 1176)),
        ({"min_pixels": 64**2, "max_pixels": 512**2}, 330, (1320, 1176)),
    ])
# yapf: enable
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_override(
    processor_for_qwen2_vl,
    image_assets: _ImageAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    expected_toks_per_img: int,
    expected_pixels_shape: tuple[int, int],
    num_imgs: int,
):
    """Ensure Qwen2VLMultiModalProcessor handles min/max pixels properly."""
    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ctx = InputProcessingContext(ctx.model_config, tokenizer)

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processor = processor_for_qwen2_vl(ctx)
    processed_inputs = processor.apply(prompt, mm_data, mm_processor_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    hf_processor = processor._get_hf_processor(**mm_processor_kwargs)
    image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"]["pixel_values"].shape

    assert img_tok_count == expected_toks_per_img * num_imgs
    assert pixel_shape[0] == expected_pixels_shape[0] * num_imgs
    assert pixel_shape[1] == expected_pixels_shape[1]
