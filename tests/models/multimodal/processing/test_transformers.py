from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from PIL import Image
import requests
import pytest
import torch

@pytest.mark.parametrize("model_id",[
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
])
def test_multimodal_processor(model_id):
    model_config = ModelConfig(
        model=model_id,
        model_impl="transformers",
    )

    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        model_config,
    )

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    mm_data = {"image": Image.open(requests.get(image_url, stream=True).raw)}
    str_prompt = "<|im_start|>user <image>\nWhat is the content of this image?<|im_end|><|im_start|>assistant\n"
    str_processed_inputs = mm_processor.apply(
        prompt=str_prompt,
        mm_data=mm_data,
        hf_processor_mm_kwargs = {},
    )

    ids_prompt = [151644, 872, 220, 151646, 198, 3838, 374, 279, 2213, 315, 419, 2168, 30, 151645, 151644, 77091, 198]
    ids_processed_inputs = mm_processor.apply(
        prompt=ids_prompt,
        mm_data=mm_data,
        hf_processor_mm_kwargs = {},
    )

    assert str_processed_inputs["prompt"] == ids_processed_inputs["prompt"]
