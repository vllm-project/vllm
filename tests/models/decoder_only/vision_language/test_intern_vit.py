# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

from ....conftest import _ImageAssets

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]


def run_intern_vit_test(
    image_assets: _ImageAssets,
    model_id: str,
    *,
    dtype: str,
    distributed_executor_backend: Optional[str] = None,
):
    model = snapshot_download(model_id, allow_patterns=DOWNLOAD_PATTERN)

    img_processor = CLIPImageProcessor.from_pretrained(model)
    images = [asset.pil_image for asset in image_assets]
    pixel_values = [
        img_processor(images, return_tensors='pt').pixel_values.to(dtype)
        for images in images
    ]

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if not getattr(config, "norm_type", None):
        config.norm_type = "rms_norm"

    hf_model = AutoModel.from_pretrained(model,
                                         torch_dtype=dtype,
                                         trust_remote_code=True).to("cuda")
    hf_outputs_per_image = [
        hf_model(pixel_value.to("cuda")).last_hidden_state
        for pixel_value in pixel_values
    ]

    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.model_executor.models.intern_vit import InternVisionModel
    vllm_model = InternVisionModel(config)
    vllm_model.load_weights(hf_model.state_dict().items())

    del hf_model
    cleanup_dist_env_and_memory()

    vllm_model = vllm_model.to("cuda", dtype)
    vllm_outputs_per_image = [
        vllm_model(pixel_values=pixel_value.to("cuda"))
        for pixel_value in pixel_values
    ]
    del vllm_model
    cleanup_dist_env_and_memory()

    cos_similar = nn.CosineSimilarity(dim=-1)
    for vllm_output, hf_output in zip(vllm_outputs_per_image,
                                      hf_outputs_per_image):
        assert cos_similar(vllm_output, hf_output).mean() > 0.99


@pytest.mark.parametrize("model_id", [
    "OpenGVLab/InternViT-300M-448px",
    "OpenGVLab/InternViT-6B-448px-V1-5",
])
@pytest.mark.parametrize("dtype", [torch.half])
@torch.inference_mode()
def test_models(dist_init, image_assets, model_id, dtype: str) -> None:
    run_intern_vit_test(
        image_assets,
        model_id,
        dtype=dtype,
    )
