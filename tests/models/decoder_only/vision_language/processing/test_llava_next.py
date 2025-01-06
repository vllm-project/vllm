import pytest
from PIL import Image
from transformers import AutoTokenizer

from vllm.inputs import InputProcessingContext

from ....utils import build_model_context


# Fixtures lazy import to avoid initializing CUDA during test collection
@pytest.fixture()
def processor_for_llava_next():
    from vllm.model_executor.models.llava_next import (
        LlavaNextMultiModalProcessor)
    return LlavaNextMultiModalProcessor


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize("image_size", [(1669, 2560), (2560, 1669), (183, 488),
                                        (488, 183), (198, 176), (176, 198)])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_prompt_replacements(
    processor_for_llava_next,
    model_id: str,
    image_size: tuple[int, int],
    num_imgs: int,
):
    """
    Ensure LlavaNextMultiModalProcessor handles prompt replacement properly.
    """
    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ctx = InputProcessingContext(ctx.model_config, tokenizer)

    # Build the image str / prompt based on the number of images we pass
    prompt = "<image>" * num_imgs
    mm_data = {"image": [Image.new("RGB", size=image_size)] * num_imgs}

    # The processor will throw an error if there is a mismatch
    # in the prompt replacements
    processor = processor_for_llava_next(ctx)
    processed_inputs = processor.apply(prompt, mm_data, {})

    image_placeholders = processed_inputs["mm_placeholders"]["image"]
    assert len(image_placeholders) == num_imgs

    first_placeholder = image_placeholders[0]

    # NOTE: There is a BOS token
    assert first_placeholder["offset"] == 1
    assert first_placeholder["length"] == (
        len(processed_inputs["prompt_token_ids"]) - 1) // num_imgs
