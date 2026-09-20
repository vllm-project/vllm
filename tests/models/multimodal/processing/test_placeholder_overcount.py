# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A placeholder token written by the caller must not take an item's position.

Prompt updates are matched in prompt order, so a bare `<|image_pad|>` sitting
in front of the placeholder the chat template rendered consumes the image and
leaves the rendered placeholder unexpanded (vllm-project/vllm#57740).
"""

import pytest
from PIL import Image

from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

pytestmark = pytest.mark.cpu_test

# placeholder, bare placeholder token, surrounding tokens
_QWEN_VL = (
    "<|vision_start|><|image_pad|><|vision_end|>",
    "<|image_pad|>",
    ("<|vision_start|>", "<|vision_end|>"),
)

_LEAD = "Describe this for me: "


@pytest.mark.parametrize(
    ("model_id", "placeholder", "pad", "wrapper"),
    [
        ("Qwen/Qwen2.5-VL-3B-Instruct", *_QWEN_VL),
        ("Qwen/Qwen3-VL-4B-Instruct", *_QWEN_VL),
        ("llava-hf/llava-1.5-7b-hf", "<image>", "<image>", None),
    ],
)
def test_extra_placeholder_token_in_prompt(
    model_id: str,
    placeholder: str,
    pad: str,
    wrapper: tuple[str, str] | None,
):
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()
    pad_id = tokenizer.get_vocab()[pad]
    mm_data = {"image": [Image.new("RGB", (224, 224), "white")]}

    def process(prompt: str):
        return processor(prompt, mm_items=processor.info.parse_mm_data(mm_data))

    with pytest.raises(VLLMValidationError, match="Found more"):
        process(f"{_LEAD}what does {pad} mean? {placeholder}")

    processed_inputs = process(f"{_LEAD}{placeholder}")
    prompt_ids = processed_inputs["prompt_token_ids"]
    (image_range,) = processed_inputs["mm_placeholders"]["image"]

    # The image lands on the placeholder, with the lead text before it.
    assert image_range.offset > 0
    assert pad_id not in prompt_ids[: image_range.offset]
    assert prompt_ids[image_range.offset] == pad_id

    if wrapper is not None:
        vocab = tokenizer.get_vocab()
        start_id, end_id = (vocab[token] for token in wrapper)
        assert prompt_ids[image_range.offset - 1] == start_id
        assert prompt_ids[image_range.offset + image_range.length] == end_id
