import numpy as np
import pytest
from transformers import CLIPImageProcessor, LlavaNextImageProcessor

from vllm.config import ModelConfig, MultiModalConfig
from vllm.multimodal import MultiModalRegistry
from vllm.multimodal.utils import rescale_image_size


@pytest.fixture
def mm_registry():
    return MultiModalRegistry()


@pytest.fixture
def mm_config():
    return MultiModalConfig(limit_per_prompt={"image": 1})


@pytest.mark.parametrize("dtype", ["half", "float"])
@pytest.mark.parametrize("size_factor", [0.25, 0.5, 1.0])
def test_clip_image_processor(image_assets, mm_registry, mm_config, dtype,
                              size_factor):
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

    hf_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    assert isinstance(hf_processor, CLIPImageProcessor)

    model_config = ModelConfig(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype=dtype,
        revision=None,
    )
    mm_registry.init_mm_limits_per_prompt(model_config, mm_config)

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


@pytest.mark.parametrize("dtype", ["half", "float"])
@pytest.mark.parametrize("size_factor", [0.25, 0.5, 1.0])
def test_llava_next_image_processor(image_assets, mm_registry, mm_config,
                                    dtype, size_factor):
    MODEL_NAME = "llava-hf/llava-v1.6-vicuna-7b-hf"

    hf_processor = LlavaNextImageProcessor.from_pretrained(MODEL_NAME)
    assert isinstance(hf_processor, LlavaNextImageProcessor)

    model_config = ModelConfig(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype=dtype,
        revision=None,
    )
    mm_registry.init_mm_limits_per_prompt(model_config, mm_config)

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
