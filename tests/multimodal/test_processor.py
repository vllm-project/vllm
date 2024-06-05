import numpy as np
import pytest
from transformers import CLIPImageProcessor, LlavaNextImageProcessor

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import ImagePixelData

from ..conftest import _STR_DTYPE_TO_TORCH_DTYPE


@pytest.mark.parametrize("dtype", ["half", "float"])
def test_clip_image_processor(hf_images, dtype):
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    IMAGE_HEIGHT = IMAGE_WIDTH = 560

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
    vlm_config = VisionLanguageConfig(
        image_input_type=VisionLanguageConfig.ImageInputType.PIXEL_VALUES,
        image_token_id=32000,
        image_input_shape=(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
        image_feature_size=576,
        image_processor=MODEL_NAME,
        image_processor_revision=None,
    )

    for image in hf_images:
        hf_result = hf_processor.preprocess(
            image,
            return_tensors="pt",
        ).to(dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype])
        vllm_result = MULTIMODAL_REGISTRY.process_input(
            ImagePixelData(image),
            model_config=model_config,
            vlm_config=vlm_config,
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
def test_llava_next_image_processor(hf_images, dtype):
    MODEL_NAME = "llava-hf/llava-v1.6-34b-hf"
    IMAGE_HEIGHT = IMAGE_WIDTH = 560

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
    vlm_config = VisionLanguageConfig(
        image_input_type=VisionLanguageConfig.ImageInputType.PIXEL_VALUES,
        image_token_id=64000,
        image_input_shape=(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
        image_feature_size=2928,
        image_processor=MODEL_NAME,
        image_processor_revision=None,
    )

    for image in hf_images:
        hf_result = hf_processor.preprocess(
            image,
            return_tensors="pt",
        ).to(dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype])
        vllm_result = MULTIMODAL_REGISTRY.process_input(
            ImagePixelData(image),
            model_config=model_config,
            vlm_config=vlm_config,
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
def test_image_pixel_types(hf_images, vllm_image_tensors, dtype):
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    IMAGE_HEIGHT = IMAGE_WIDTH = 560

    model_config = ModelConfig(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype=dtype,
        revision=None,
    )
    vlm_config = VisionLanguageConfig(
        image_input_type=VisionLanguageConfig.ImageInputType.PIXEL_VALUES,
        image_token_id=32000,
        image_input_shape=(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
        image_feature_size=576,
        image_processor=MODEL_NAME,
        image_processor_revision=None,
    )

    for image, tensor in zip(hf_images, vllm_image_tensors):
        image_result = MULTIMODAL_REGISTRY.process_input(
            ImagePixelData(image),
            model_config=model_config,
            vlm_config=vlm_config,
        )
        tensor_result = MULTIMODAL_REGISTRY.process_input(
            ImagePixelData(tensor),
            model_config=model_config,
            vlm_config=vlm_config,
        )

        assert image_result.keys() == tensor_result.keys()
        for key, image_arr in image_result.items():
            tensor_arr: np.ndarray = tensor_result[key].numpy()

            assert image_arr.shape == tensor_arr.shape, f"Failed for key={key}"
            assert np.allclose(image_arr, tensor_arr), f"Failed for key={key}"
