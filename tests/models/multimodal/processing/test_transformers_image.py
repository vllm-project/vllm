# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.assets.image import ImageAsset
from vllm.config import ModelConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config


@pytest.mark.parametrize("model_id", ["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"])
def test_multimodal_processor(model_id):
    model_config = ModelConfig(
        model=model_id,
        model_impl="transformers",
    )

    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    image_pil = ImageAsset("cherry_blossom").pil_image
    mm_data = {"image": image_pil}
    str_prompt = "<|im_start|>user <image>\nWhat is the content of this image?<|im_end|><|im_start|>assistant\n"  # noqa: E501
    str_processed_inputs = mm_processor(
        prompt=str_prompt,
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    ids_prompt = [
        151644,
        872,
        220,
        151646,
        198,
        3838,
        374,
        279,
        2213,
        315,
        419,
        2168,
        30,
        151645,
        151644,
        77091,
        198,
    ]
    ids_processed_inputs = mm_processor(
        prompt=ids_prompt,
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    assert (
        str_processed_inputs["prompt_token_ids"]
        == ids_processed_inputs["prompt_token_ids"]
    )


@pytest.mark.parametrize("model_id", ["llava-hf/llava-1.5-7b-hf"])
def test_ids_prompt_does_not_duplicate_special_tokens(model_id):
    """Token-ids prompts are decoded back to text and re-tokenized by the HF
    processor; special tokens already present in the ids (e.g. BOS added by
    the renderer) must not be added a second time."""
    model_config = ModelConfig(
        model=model_id,
        model_impl="transformers",
    )

    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    tokenizer = mm_processor.info.get_tokenizer()

    image_pil = ImageAsset("cherry_blossom").pil_image
    mm_data = {"image": image_pil}
    str_prompt = "USER: <image>\nWhat is the content of this image? ASSISTANT:"
    ids_prompt = tokenizer.encode(str_prompt, add_special_tokens=True)
    assert ids_prompt[0] == tokenizer.bos_token_id

    ids_processed_inputs = mm_processor(
        prompt=ids_prompt,
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    ids_token_ids = ids_processed_inputs["prompt_token_ids"]
    assert ids_token_ids.count(tokenizer.bos_token_id) == 1


@pytest.mark.parametrize(
    ("model_id", "prompt"),
    [
        (
            "llava-hf/llava-1.5-7b-hf",
            "USER: <image>\nWhat is the content of this image? ASSISTANT:",
        ),
        (
            "google/gemma-3-4b-it",
            "<start_of_image>What is the content of this image?",
        ),
    ],
)
def test_renderer_defers_tokenization_to_hf_processor(model_id, prompt):
    """Text prompts with mm data are tokenized exactly once, by the HF
    processor: the engine input ids must equal the HF processor's own output
    (no duplicated BOS, no metaspace drift from a decode/re-encode round
    trip)."""
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    renderer = HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )

    image_pil = ImageAsset("cherry_blossom").pil_image
    (engine_input,) = renderer.render_cmpl(
        [{"prompt": prompt, "multi_modal_data": {"image": image_pil}}]
    )

    hf_processor = renderer.get_mm_processor().info.get_hf_processor()
    direct = hf_processor(text=prompt, images=image_pil)["input_ids"][0]
    direct_ids = direct.tolist() if hasattr(direct, "tolist") else list(direct)

    bos_token_id = renderer.get_tokenizer().bos_token_id
    assert engine_input["prompt_token_ids"] == direct_ids
    assert direct_ids.count(bos_token_id) == 1


@pytest.mark.parametrize("model_id", ["google/gemma-3-4b-it"])
def test_renderer_chat_template_prompt_matches_hf(model_id):
    """A chat template renders special tokens itself (gemma3's starts with
    <bos>), so tokenization must not add them again, mirroring the fallback
    in `ProcessorMixin.apply_chat_template`. The reference ids are the
    processor's own chat-template tokenization."""
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    renderer = HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )
    hf_processor = renderer.get_mm_processor().info.get_hf_processor()

    image_pil = ImageAsset("cherry_blossom").pil_image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": "What is the content of this image?"},
            ],
        }
    ]
    rendered = hf_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    assert rendered.startswith(hf_processor.tokenizer.bos_token)

    (engine_input,) = renderer.render_cmpl(
        [{"prompt": rendered, "multi_modal_data": {"image": image_pil}}]
    )

    reference = hf_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True
    )
    reference_ids = reference["input_ids"]
    if hasattr(reference_ids, "tolist"):
        reference_ids = reference_ids.tolist()
    reference_ids = reference_ids[0]

    assert engine_input["prompt_token_ids"] == reference_ids
    assert reference_ids.count(hf_processor.tokenizer.bos_token_id) == 1


def test_image_multiple_inputs():
    """Multiple images per prompt are each detected as a separate placeholder
    and multi-modal item by the Transformers backend."""
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    image = ImageAsset("cherry_blossom").pil_image
    prompt = (
        "<|im_start|>user <image>\n and <image>\n"
        "What do these images show?<|im_end|><|im_start|>assistant\n"
    )

    result = mm_processor(
        prompt=prompt,
        mm_items=mm_processor.info.parse_mm_data({"image": [image, image]}),
        hf_processor_mm_kwargs={},
    )

    assert len(result["mm_placeholders"]["image"]) == 2
    assert len(result["mm_kwargs"]["image"]) == 2
