from contextlib import nullcontext

import numpy as np
import pytest
from transformers import LlavaNextImageProcessor

from vllm.config import ModelConfig
from vllm.multimodal import MultiModalRegistry
from vllm.multimodal.utils import rescale_image_size


@pytest.fixture
def mm_registry():
    return MultiModalRegistry()


@pytest.mark.parametrize("dtype", ["half", "float"])
@pytest.mark.parametrize("size_factor", [0.25, 0.5, 1.0])
def test_llava_next_image_processor(image_assets, mm_registry, dtype,
                                    size_factor):
    MODEL_NAME = "llava-hf/llava-v1.6-vicuna-7b-hf"

    hf_processor = LlavaNextImageProcessor.from_pretrained(MODEL_NAME)
    assert isinstance(hf_processor, LlavaNextImageProcessor)

    model_config = ModelConfig(
        model=MODEL_NAME,
        task="auto",
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype=dtype,
        revision=None,
        limit_mm_per_prompt={"image": 1},
    )

    mm_registry.init_mm_limits_per_prompt(model_config)

    for asset in image_assets:
        image = rescale_image_size(asset.pil_image, size_factor)

        hf_result = hf_processor.preprocess(
            image,
            return_tensors="pt",
        )
        vllm_result = mm_registry.map_input(
            model_config,
            {"image": image},
        )

        assert hf_result.keys() == vllm_result.keys()
        for key, hf_tensor in hf_result.items():
            hf_arr: np.ndarray = hf_tensor.numpy()
            vllm_arr: np.ndarray = vllm_result[key].numpy()

            assert hf_arr.shape == vllm_arr.shape, f"Failed for key={key}"
            assert np.allclose(hf_arr, vllm_arr), f"Failed for key={key}"


@pytest.mark.parametrize(
    ("num_images", "limit", "is_valid"),
    [(0, 0, True), (0, 1, True), (1, 0, False), (1, 1, True), (1, 2, True),
     (2, 1, False), (2, 2, True)],
)
def test_mm_limits(image_assets, mm_registry, num_images, limit, is_valid):
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"

    model_config = ModelConfig(
        model=MODEL_NAME,
        task="auto",
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="half",
        revision=None,
        limit_mm_per_prompt={"image": limit},
    )

    mm_registry.init_mm_limits_per_prompt(model_config)

    image = image_assets[0].pil_image
    if num_images == 0:
        mm_inputs = {}
    elif num_images == 1:
        mm_inputs = {"image": image}
    else:
        mm_inputs = {"image": [image] * num_images}

    with nullcontext() if is_valid else pytest.raises(ValueError):
        mm_registry.map_input(model_config, mm_inputs)


# NOTE: We don't test zero images since the HF processor doesn't support it
@pytest.mark.parametrize("num_images", [1, 2])
def test_image_mapper_multi(image_assets, mm_registry, num_images):
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"

    model_config = ModelConfig(
        model=MODEL_NAME,
        task="auto",
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="half",
        revision=None,
        limit_mm_per_prompt={"image": num_images},
    )

    mm_registry.init_mm_limits_per_prompt(model_config)

    image = image_assets[0].pil_image
    mm_inputs = {"image": [image] * num_images}

    mapped_inputs = mm_registry.map_input(model_config, mm_inputs)
    assert len(mapped_inputs["pixel_values"]) == num_images
