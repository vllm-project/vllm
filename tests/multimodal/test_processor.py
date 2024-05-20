import numpy as np
import pytest
from transformers import CLIPImageProcessor

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.multimodal import MM_REGISTRY
from vllm.multimodal.image import ImagePixelData


@pytest.mark.parametrize("dtype", ["half", "bfloat16", "float"])
def test_clip_image_processor(hf_images, dtype):
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    IMAGE_HEIGHT = IMAGE_WIDTH = 33

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
            # To be compatible with the patch for LLaVA-NeXT
            image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            return_tensors="np",
        )
        vllm_result = MM_REGISTRY.process_input(
            ImagePixelData(image),
            model_config=model_config,
            vlm_config=vlm_config,
        )

        assert hf_result.keys() == vllm_result.keys()
        for key, hf_arr in hf_result.items():
            vllm_arr: np.ndarray = vllm_result[key].numpy()

            assert hf_arr.shape == vllm_arr.shape, f"Failed for key={key}"
            assert np.allclose(hf_arr, vllm_arr), f"Failed for key={key}"
