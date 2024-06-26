import numpy as np
import pytest
from transformers import CLIPImageProcessor, LlavaNextImageProcessor

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from ..conftest import _STR_DTYPE_TO_TORCH_DTYPE


@pytest.mark.parametrize("dtype", ["half", "float"])
def test_clip_image_processor(image_assets, dtype):
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

    for asset in image_assets:
        hf_result = hf_processor.preprocess(
            asset.pil_image,
            return_tensors="pt",
        ).to(dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype])
        vllm_result = MULTIMODAL_REGISTRY.map_input(
            model_config,
            {"image": asset.pil_image},
        )

        assert hf_result.keys() == vllm_result.keys()
        for key, hf_tensor in hf_result.items():
            hf_arr: np.ndarray = hf_tensor.numpy()
            vllm_arr: np.ndarray = vllm_result[key].numpy()

            assert hf_arr.shape == vllm_arr.shape, f"Failed for key={key}"
            assert np.allclose(hf_arr, vllm_arr), f"Failed for key={key}"


@pytest.mark.xfail(
    reason="Inconsistent image processor being used due to lack "
    "of support for dynamic image token replacement")
@pytest.mark.parametrize("dtype", ["half", "float"])
def test_llava_next_image_processor(image_assets, dtype):
    MODEL_NAME = "llava-hf/llava-v1.6-34b-hf"

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

    for asset in image_assets:
        hf_result = hf_processor.preprocess(
            asset.pil_image,
            return_tensors="pt",
        ).to(dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype])
        vllm_result = MULTIMODAL_REGISTRY.map_input(
            model_config,
            {"image": asset.pil_image},
        )

        assert hf_result.keys() == vllm_result.keys()
        for key, hf_tensor in hf_result.items():
            hf_arr: np.ndarray = hf_tensor.numpy()
            vllm_arr: np.ndarray = vllm_result[key].numpy()

            assert hf_arr.shape == vllm_arr.shape, f"Failed for key={key}"
            assert np.allclose(hf_arr, vllm_arr), f"Failed for key={key}"


@pytest.mark.xfail(
    reason="Example image pixels were not processed using HuggingFace")
@pytest.mark.parametrize("dtype", ["float"])
def test_image_pixel_types(image_assets, dtype):
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

    model_config = ModelConfig(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype=dtype,
        revision=None,
    )
    for asset in image_assets:
        image_result = MULTIMODAL_REGISTRY.map_input(
            model_config,
            {"image": asset.pil_image},
        )
        tensor_result = MULTIMODAL_REGISTRY.map_input(
            model_config,
            {"image": asset.pil_image},
        )

        assert image_result.keys() == tensor_result.keys()
        for key, image_arr in image_result.items():
            tensor_arr: np.ndarray = tensor_result[key].numpy()

            assert image_arr.shape == tensor_arr.shape, f"Failed for key={key}"
            assert np.allclose(image_arr, tensor_arr), f"Failed for key={key}"
