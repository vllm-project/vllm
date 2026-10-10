# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from unittest.mock import patch

import pytest
from transformers.models.idefics3.processing_idefics3 import Idefics3ProcessorKwargs

from vllm.assets.image import ImageAsset
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache


@pytest.mark.parametrize("model_id", ["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"])
def test_multimodal_processor(model_id):
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )

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


def _process_two_images(separator: str):
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )

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


def test_image_multiple_inputs():
    """Multiple images per prompt are each detected as a separate placeholder
    and multi-modal item by the Transformers modelling backend."""
    result = _process_two_images(separator="\n and ")

    assert len(result["mm_placeholders"]["image"]) == 2
    assert len(result["mm_kwargs"]["image"]) == 2


def test_image_adjacent_inputs():
    """Adjacent images stay separate placeholders rather than merging into one."""
    result = _process_two_images(separator="")

    assert len(result["mm_placeholders"]["image"]) == 2
    assert len(result["mm_kwargs"]["image"]) == 2


def test_batch_padding_removed_from_image_items():
    """Emu3 pads every image up to the largest in the batch, which would leave an
    item's data dependent on what it was processed with and so uncacheable."""
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model="BAAI/Emu3-Chat-hf", model_impl="transformers")
    )
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


def _process_one_gemma3_image():
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model="google/gemma-3-4b-it", model_impl="transformers")
    )
    boi_token = mm_processor.info.get_hf_processor().boi_token
    return mm_processor(
        prompt=f"{boi_token} What is this?",
        mm_items=mm_processor.info.parse_mm_data(
            {"image": ImageAsset("cherry_blossom").pil_image}
        ),
        hf_processor_mm_kwargs={},
    )


def test_non_embedding_tokens_excluded_from_placeholders():
    """Gemma3 wraps each image in text that carries no embeddings, which must be
    inside the placeholder range but masked out of it."""
    result = _process_one_gemma3_image()

    (placeholder,) = result["mm_placeholders"]["image"]
    assert placeholder.is_embed is not None
    assert 0 < int(placeholder.is_embed.sum()) < placeholder.length


def test_tokens_structuring_an_image_are_masked_not_dropped():
    """SmolVLM splits each image into tiles introduced by tokens carrying no
    embeddings. Those belong inside the placeholder and masked out, because the token
    count the processor reports is over the whole span. Idefics3 also refuses a prompt
    holding `<image>` when no images are passed, which is how the prompt has to be
    tokenized before splicing in the expansion."""
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
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


def test_missing_replacement_offsets_names_the_processor():
    """A processor that reports no replacement offsets cannot be served, which must
    be said plainly rather than surfacing later as a field config mismatch."""
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
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


def test_text_only_prompt():
    """An image model still accepts a prompt with no images."""
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )

    result = mm_processor(
        prompt="<|im_start|>user Hello!<|im_end|><|im_start|>assistant\n",
        mm_items=mm_processor.info.parse_mm_data({}),
        hf_processor_mm_kwargs={},
    )

    assert len(result["prompt_token_ids"]) > 0
    assert not result["mm_placeholders"]


def test_repeated_image_hits_the_processor_cache():
    """Check that mm caching is actually working."""
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", model_impl="transformers"
    )
    model_config.multimodal_config.mm_processor_cache_gb = 4
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    cache = MultiModalProcessorOnlyCache(model_config)
    image = ImageAsset("cherry_blossom").pil_image

    def process():
        return mm_processor(
            prompt="<image>\nWhat is this?",
            mm_items=mm_processor.info.parse_mm_data({"image": image}),
            hf_processor_mm_kwargs={},
            cache=cache,
        )

    first, second = process(), process()

    assert cache.make_stats().hits > 0
    assert first["prompt_token_ids"] == second["prompt_token_ids"]
    assert first["mm_hashes"] == second["mm_hashes"]


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
                reason="Emu3Processor prepends the BOS token itself because the "
                "tokenizer doesn't, so the unexpanded prompt vLLM tokenizes "
                "never gets one.",
                strict=False,
            ),
        ),
    ],
)
def test_spliced_prompt_matches_hf_expansion(model_id, prompt):
    """The prompt is tokenized without any multi-modal data and the expansion spliced
    in, so its token ids have to come out the same as the ones the HF processor
    produces itself."""
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
    info = mm_processor.info
    image = ImageAsset("cherry_blossom").pil_image

    hf_processor = info.get_hf_processor()
    prompt_ids = info.get_tokenizer().encode(
        prompt, **info.default_tok_params.get_encode_kwargs()
    )
    hf_ids = info.ctx.call_hf_processor(
        hf_processor,
        dict(text=hf_processor.decode(prompt_ids), images=[image]),
        dict(truncation=False, add_special_tokens=False),
    )["input_ids"][0].tolist()

    result = mm_processor(
        prompt=prompt,
        mm_items=info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )
    assert result["prompt_token_ids"] == hf_ids


def test_nested_image_fields_split_per_image():
    """Idefics3 returns image fields with a leading batch dimension, putting the rows
    belonging to each image one dimension further in. Slicing the batch dimension
    instead handed the first image every row and the second an empty tensor."""
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
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


def _num_image_patches(mm_processor_kwargs) -> list[int]:
    """Process two images with SmolVLM and return the rows each one got."""
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    # Idefics3's token count updates these class-level defaults in place.
    defaults = copy.deepcopy(Idefics3ProcessorKwargs._defaults)
    with patch.object(Idefics3ProcessorKwargs, "_defaults", defaults):
        mm_processor = MULTIMODAL_REGISTRY.create_processor(
            ModelConfig(
                model=model_id,
                model_impl="transformers",
                mm_processor_kwargs=mm_processor_kwargs,
            )
        )
        image = ImageAsset("cherry_blossom").pil_image
        result = mm_processor(
            prompt="<image> and <image>",
            mm_items=mm_processor.info.parse_mm_data({"image": [image, image]}),
            hf_processor_mm_kwargs={},
        )
    return [
        int(item["num_image_patches"].data) for item in result["mm_kwargs"]["image"]
    ]


def test_scoped_images_kwargs_reach_the_patch_count():
    """A nested ``images_kwargs`` override must reach vLLM's per-image patch count.

    The HF processor honors a nested ``images_kwargs`` in its ``__call__``, so a
    count read from the flat kwargs alone expects the stock number of crops and
    cannot attribute the processor's rows to the images.
    """
    size = {"longest_edge": 1024}
    flat = _num_image_patches({"size": size})
    # Precondition: the override really changes the crop count.
    assert flat != _num_image_patches(None)

    assert _num_image_patches({"images_kwargs": {"size": size}}) == flat


def _max_image_tokens(mm_processor_kwargs) -> int:
    """The per-image token budget SmolVLM is profiled with."""
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    # Idefics3's token count updates these class-level defaults in place.
    defaults = copy.deepcopy(Idefics3ProcessorKwargs._defaults)
    with patch.object(Idefics3ProcessorKwargs, "_defaults", defaults):
        mm_processor = MULTIMODAL_REGISTRY.create_processor(
            ModelConfig(
                model=model_id,
                model_impl="transformers",
                mm_processor_kwargs=mm_processor_kwargs,
            )
        )
        return mm_processor.info.get_max_image_tokens()


def test_scoped_images_kwargs_reach_the_profiled_token_budget():
    """A nested ``images_kwargs`` override must also reach the profiling budget.

    ``get_max_image_tokens`` sizes the memory profile from the same HF token
    count as the patch count above, so reading the flat kwargs alone profiles
    the stock number of crops while requests are served with the override.
    """
    size = {"longest_edge": 1024}
    flat = _max_image_tokens({"size": size})
    # Precondition: the override really changes the budget.
    assert flat != _max_image_tokens(None)

    assert _max_image_tokens({"images_kwargs": {"size": size}}) == flat
