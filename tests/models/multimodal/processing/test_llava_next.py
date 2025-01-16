import itertools
from functools import partial

import pytest
from PIL import Image
from pqdm.threads import pqdm

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.utils import cached_get_tokenizer

from ...utils import build_model_context


def _validate_image_max_tokens_one(
    processor: BaseMultiModalProcessor,
    max_tokens: int,
    failed_size_excs: list[tuple[ImageSize, Exception]],
    image_size: ImageSize,
) -> None:
    info = processor.info
    feature_size = info.get_num_image_tokens(image_width=image_size.width,
                                             image_height=image_size.height)

    try:
        assert feature_size <= max_tokens, f"{feature_size} <= {max_tokens}"
    except Exception as exc:
        failed_size_excs.append((image_size, exc))


@pytest.mark.skip("This test takes around 5 minutes to run. "
                  "Comment this out to run it manually.")
@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
def test_processor_max_tokens(model_id):
    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=cached_get_tokenizer(ctx.model_config.tokenizer),
    )
    info = processor.info

    seen_aspect_ratios = set[float]()
    image_sizes = list[ImageSize]()

    # The aspect ratio of the grid layout is between 1 and 2
    # NOTE: Assumes that feature size calculation is the same if we
    # swap the width and height of the image
    for w, h in itertools.product(range(32, 4096), repeat=2):
        aspect_ratio = w / h
        if 1 <= aspect_ratio <= 2 and aspect_ratio not in seen_aspect_ratios:
            image_sizes.append(ImageSize(w, h))
            seen_aspect_ratios.add(aspect_ratio)

    failed_size_excs = list[tuple[ImageSize, Exception]]()

    validate_one = partial(
        _validate_image_max_tokens_one,
        processor,
        info.get_max_image_tokens(),  # type: ignore
        failed_size_excs,
    )
    pqdm(image_sizes, validate_one, n_jobs=8, desc="Validating image sizes")

    if failed_size_excs:
        msg = "Found failing image sizes:" \
            + "\n========\n".join(f"[{size}]\n{exc}"
                                  for size, exc in failed_size_excs)
        raise AssertionError(msg)


def _validate_image_prompt_replacements_one(
    processor: BaseMultiModalProcessor,
    num_imgs: int,
    failed_size_excs: list[tuple[ImageSize, Exception]],
    image_size: ImageSize,
) -> None:
    prompt = "<image>" * num_imgs
    image = Image.new("RGB", size=image_size)
    mm_data = {"image": [image] * num_imgs}

    try:
        # The processor will throw an error if there is a mismatch
        # in the prompt replacements
        processed_inputs = processor.apply(prompt, mm_data, {})

        image_placeholders = processed_inputs["mm_placeholders"]["image"]
        assert len(image_placeholders) == num_imgs

        first_placeholder = image_placeholders[0]

        # NOTE: There is a BOS token
        assert first_placeholder["offset"] == 1
        assert first_placeholder["length"] == (
            len(processed_inputs["prompt_token_ids"]) - 1) // num_imgs

    except Exception as exc:
        failed_size_excs.append((image_size, exc))


def _test_image_prompt_replacements(
    processor,
    *,
    num_imgs: int,
    image_sizes: list[ImageSize],
) -> None:
    """
    Ensure LlavaNextMultiModalProcessor
    handles prompt replacement properly for input images.
    """
    failed_size_excs = list[tuple[ImageSize, Exception]]()

    validate_one = partial(
        _validate_image_prompt_replacements_one,
        processor,
        num_imgs,
        failed_size_excs,
    )
    pqdm(image_sizes, validate_one, n_jobs=8, desc="Validating image sizes")

    if failed_size_excs:
        msg = "Found failing image sizes:" \
            + "\n========\n".join(f"[{size}]\n{exc}"
                                  for size, exc in failed_size_excs)
        raise AssertionError(msg)


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_prompt_replacements_regression(model_id, num_imgs):
    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=cached_get_tokenizer(ctx.model_config.tokenizer),
    )

    image_ratios = [(171, 152), (184, 161), (198, 176), (333, 296), (369, 328),
                    (488, 183), (2560, 1669)]
    image_sizes = [
        size for w, h in image_ratios
        for size in [ImageSize(w, h), ImageSize(h, w)]
    ]

    _test_image_prompt_replacements(
        processor,
        num_imgs=num_imgs,
        image_sizes=image_sizes,
    )


@pytest.mark.skip("This test takes around 2 hours to run. "
                  "Comment this out to run it manually.")
@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize("num_imgs", [1])
def test_processor_prompt_replacements_all(model_id, num_imgs):
    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=cached_get_tokenizer(ctx.model_config.tokenizer),
    )

    seen_aspect_ratios = set[float]()
    image_sizes = list[ImageSize]()

    # The aspect ratio of the grid layout is between 1 and 2
    # NOTE: Assumes that feature size calculation is the same if we
    # swap the width and height of the image
    for w, h in itertools.product(range(64, 1024), repeat=2):
        aspect_ratio = w / h
        if 1 <= aspect_ratio <= 2 and aspect_ratio not in seen_aspect_ratios:
            image_sizes.append(ImageSize(w, h))
            seen_aspect_ratios.add(aspect_ratio)

    _test_image_prompt_replacements(
        processor,
        num_imgs=num_imgs,
        image_sizes=image_sizes,
    )
