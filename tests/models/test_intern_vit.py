from typing import Optional, Type

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPImageProcessor

from vllm import ModelRegistry
from vllm.model_executor.models.intern_vit import InternVisionModel
from vllm.utils import is_cpu

from ..conftest import VllmRunner, _ImageAssets, cleanup

models = [
    "OpenGVLab/InternViT-300M-448px", "OpenGVLab/InternViT-6B-448px-V1-5"
]


def run_intern_vit_test(
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    dtype: str,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    ModelRegistry.register_model("InternVisionModel", InternVisionModel)

    img_processor = CLIPImageProcessor.from_pretrained(model)
    images = [asset.pil_image for asset in image_assets]
    pixel_values = [
        img_processor(images, return_tensors='pt').pixel_values
        for images in images
    ]

    with vllm_runner(model,
                     max_model_len=4096,
                     dtype=dtype,
                     skip_tokenizer_init=True,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vit_model = vllm_model.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        vllm_outputs_per_image = [
            vit_model(pixel_value) for pixel_value in pixel_values
        ]

    hf_model = AutoModel.from_pretrained(model, trust_remote_code=True)
    hf_outputs_per_image = [
        hf_model(pixel_value).last_hidden_state for pixel_value in pixel_values
    ]
    del hf_model

    cos_similar = nn.CosineSimilarity(dim=-1)
    for vllm_output, hf_output in zip(vllm_outputs_per_image,
                                      hf_outputs_per_image):
        assert cos_similar(vllm_output, hf_output).mean() > 0.99

    cleanup()


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", [target_dtype])
@torch.inference_mode()
def test_models(vllm_runner, image_assets, model, dtype: str) -> None:
    run_intern_vit_test(
        vllm_runner,
        image_assets,
        model,
        dtype=dtype,
        tensor_parallel_size=1,
    )
