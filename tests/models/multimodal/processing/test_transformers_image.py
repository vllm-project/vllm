# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest

from vllm.assets.image import ImageAsset
from vllm.model_executor.models.transformers.multimodal import (
    LegacyMultiModalProcessor,
    OffsetsMultiModalProcessor,
)

from .transformers_backend import (
    PROCESSOR_CLASSES,
    create_cached_processor,
    create_processor,
    offsets_only,
)


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
@pytest.mark.parametrize("model_id", ["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"])
def test_multimodal_processor(model_id, processor_cls):
    mm_processor = create_processor(model_id, processor_cls)

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


def _process_two_images(processor_cls, separator: str):
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    mm_processor = create_processor(model_id, processor_cls)

    image = ImageAsset("cherry_blossom").pil_image
    prompt = (
        f"<|im_start|>user <image>{separator}<image>\n"
        "What do these images show?<|im_end|><|im_start|>assistant\n"
    )

    return mm_processor(
        prompt=prompt,
        mm_items=mm_processor.info.parse_mm_data({"image": [image, image]}),
        hf_processor_mm_kwargs={},
    )


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
def test_image_multiple_inputs(processor_cls):
    """Multiple images per prompt are each detected as a separate placeholder
    and multi-modal item by the Transformers modelling backend."""
    result = _process_two_images(processor_cls, separator="\n and ")

    assert len(result["mm_placeholders"]["image"]) == 2
    assert len(result["mm_kwargs"]["image"]) == 2


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
def test_image_adjacent_inputs(processor_cls):
    """Adjacent images stay separate placeholders rather than merging into one."""
    result = _process_two_images(processor_cls, separator="")

    assert len(result["mm_placeholders"]["image"]) == 2
    assert len(result["mm_kwargs"]["image"]) == 2


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
def test_batch_padding_removed_from_image_items(processor_cls):
    """Emu3 pads every image up to the largest in the batch, which would leave an
    item's data dependent on what it was processed with and so uncacheable."""
    mm_processor = create_processor("BAAI/Emu3-Chat-hf", processor_cls)
    image_token = mm_processor.info.get_hf_processor().image_token

    images = [
        ImageAsset("cherry_blossom").pil_image,
        ImageAsset("cherry_blossom").pil_image.resize((256, 1024)),
    ]
    result = mm_processor(
        prompt=f"{image_token} and {image_token}",
        mm_items=mm_processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={},
    )

    items = result["mm_kwargs"]["image"]
    shapes = set()
    for item in items:
        height, width = item["image_sizes"].data.flatten().tolist()
        pixel_values = item["pixel_values"].data
        assert tuple(pixel_values.shape[-2:]) == (height, width)
        shapes.add(tuple(pixel_values.shape))

    # Both images would have been padded to a common shape had they been kept
    assert len(shapes) == 2


def _process_one_gemma3_image(processor_cls):
    mm_processor = create_processor("google/gemma-3-4b-it", processor_cls)
    hf_processor = mm_processor.info.get_hf_processor()
    result = mm_processor(
        prompt=f"{hf_processor.boi_token} What is this?",
        mm_items=mm_processor.info.parse_mm_data(
            {"image": ImageAsset("cherry_blossom").pil_image}
        ),
        hf_processor_mm_kwargs={},
    )
    return hf_processor, result


@offsets_only
def test_non_embedding_tokens_excluded_from_placeholders():
    """Gemma3 wraps each image in text that carries no embeddings, which must be
    inside the placeholder range but masked out of it."""
    _, result = _process_one_gemma3_image(OffsetsMultiModalProcessor)

    (placeholder,) = result["mm_placeholders"]["image"]
    assert placeholder.is_embed is not None
    assert 0 < int(placeholder.is_embed.sum()) < placeholder.length


def test_legacy_placeholders_hold_only_image_tokens():
    """The legacy path spans whatever `mm_token_type_ids` attributes to the image,
    which for Gemma3 excludes the text wrapping it, unlike the replacement the offsets
    path spans. Gemma3 is also the sharp case for the mask: its `image_token_id` is the
    marker in the unexpanded prompt, not the token the expansion repeats."""
    hf_processor, result = _process_one_gemma3_image(LegacyMultiModalProcessor)

    (placeholder,) = result["mm_placeholders"]["image"]
    assert placeholder.length == hf_processor.image_seq_length
    prompt_ids = result["prompt_token_ids"]
    covered = prompt_ids[placeholder.offset : placeholder.offset + placeholder.length]
    assert set(covered) == {hf_processor.tokenizer.image_token_id}


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
def test_tokens_structuring_an_image_are_masked_not_dropped(processor_cls):
    """SmolVLM splits each image into tiles introduced by tokens carrying no
    embeddings. Those belong inside the placeholder and masked out, because the token
    count the processor reports is over the whole span. Idefics3 also refuses a prompt
    holding `<image>` when no images are passed, which is how the offsets path has to
    tokenize it before splicing in the expansion."""
    mm_processor = create_processor(
        "HuggingFaceTB/SmolVLM-256M-Instruct", processor_cls
    )
    result = mm_processor(
        prompt="<image>What is this?",
        mm_items=mm_processor.info.parse_mm_data(
            {"image": ImageAsset("cherry_blossom").pil_image}
        ),
        hf_processor_mm_kwargs={},
    )

    (placeholder,) = result["mm_placeholders"]["image"]
    assert placeholder.is_embed is not None
    assert 0 < int(placeholder.is_embed.sum()) < placeholder.length


@offsets_only
def test_missing_replacement_offsets_names_the_processor():
    """A processor that reports no replacement offsets cannot be served, which must
    be said plainly rather than surfacing later as a field config mismatch."""
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    mm_processor = create_processor(model_id, OffsetsMultiModalProcessor)
    hf_processor_cls = type(mm_processor.info.get_hf_processor())
    hf_call = hf_processor_cls.__call__

    def without_offsets(self, *args, **kwargs):
        hf_inputs = hf_call(self, *args, **kwargs)
        hf_inputs.pop("text_replacement_offsets", None)
        return hf_inputs

    with (
        patch.object(hf_processor_cls, "__call__", without_offsets),
        pytest.raises(ValueError, match="LlavaOnevisionProcessor returned no"),
    ):
        mm_processor(
            prompt="<image>\nWhat is the content of this image?",
            mm_items=mm_processor.info.parse_mm_data(
                {"image": ImageAsset("cherry_blossom").pil_image}
            ),
            hf_processor_mm_kwargs={},
        )


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
def test_text_only_prompt(processor_cls):
    """An image model still accepts a prompt with no images."""
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    mm_processor = create_processor(model_id, processor_cls)

    result = mm_processor(
        prompt="<|im_start|>user Hello!<|im_end|><|im_start|>assistant\n",
        mm_items=mm_processor.info.parse_mm_data({}),
        hf_processor_mm_kwargs={},
    )

    assert len(result["prompt_token_ids"]) > 0
    assert not result["mm_placeholders"]


@offsets_only
def test_repeated_image_hits_the_processor_cache():
    """Check that mm caching is actually working."""
    mm_processor, cache = create_cached_processor(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", OffsetsMultiModalProcessor
    )
    image = ImageAsset("cherry_blossom").pil_image

    def process():
        return mm_processor(
            prompt="<image>\nWhat is this?",
            mm_items=mm_processor.info.parse_mm_data({"image": image}),
            hf_processor_mm_kwargs={},
        )

    first, second = process(), process()

    assert cache.make_stats().hits > 0
    assert first["prompt_token_ids"] == second["prompt_token_ids"]
    assert first["mm_hashes"] == second["mm_hashes"]


@offsets_only
@pytest.mark.parametrize(
    ("model_id", "prompt"),
    [
        ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "<image>\nWhat is this?"),
        ("google/gemma-3-4b-it", "<start_of_image> What is this?"),
        ("HuggingFaceTB/SmolVLM-256M-Instruct", "<image>What is this?"),
        pytest.param(
            "BAAI/Emu3-Chat-hf",
            "<image> and more text",
            marks=pytest.mark.xfail(
                reason="Emu3Processor prepends its BOS token only when images are "
                "passed, so the unexpanded prompt the offsets path tokenizes never "
                "gets one. Fixed by huggingface/transformers#47924, unreleased.",
                strict=False,
            ),
        ),
    ],
)
def test_spliced_prompt_matches_hf_expansion(model_id, prompt):
    """The offsets path splices the expansion into a prompt tokenized without any
    multi-modal data, so its token ids have to come out the same as the ones the HF
    processor produces itself, which is what the legacy path returns."""
    prompt_ids = []
    for processor_cls in (LegacyMultiModalProcessor, OffsetsMultiModalProcessor):
        mm_processor = create_processor(model_id, processor_cls)
        prompt_ids.append(
            mm_processor(
                prompt=prompt,
                mm_items=mm_processor.info.parse_mm_data(
                    {"image": ImageAsset("cherry_blossom").pil_image}
                ),
                hf_processor_mm_kwargs={},
            )["prompt_token_ids"]
        )

    legacy_ids, offsets_ids = prompt_ids
    assert legacy_ids == offsets_ids


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
def test_nested_image_fields_split_per_image(processor_cls):
    """Idefics3 returns image fields with a leading batch dimension, putting the rows
    belonging to each image one dimension further in. Slicing the batch dimension
    instead handed the first image every row and the second an empty tensor."""
    mm_processor = create_processor(
        "HuggingFaceTB/SmolVLM-256M-Instruct", processor_cls
    )
    image = ImageAsset("cherry_blossom").pil_image
    result = mm_processor(
        prompt="<image> and <image>",
        mm_items=mm_processor.info.parse_mm_data({"image": [image, image]}),
        hf_processor_mm_kwargs={},
    )

    items = result["mm_kwargs"]["image"]
    assert len(items) == 2
    for item in items:
        pixel_values = item["pixel_values"].data
        assert pixel_values.shape[1] == int(item["num_image_patches"].data)
