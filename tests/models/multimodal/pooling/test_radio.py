# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.models.radio import RadioModel
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import ImageTestAssets

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]

VIT_DIMS = {
    "vit_small_patch16_224": (384, 12, 6, 1536),
    "vit_base_patch16_224": (768, 12, 12, 3072),
    "vit_large_patch16_224": (1024, 24, 16, 4096),
    "vit_huge_patch16_224": (1280, 32, 16, 5120),
}


def get_args_from_model_type(model_type):
    return VIT_DIMS[model_type]


@torch.inference_mode()
def run_radio_test(
    image_assets: ImageTestAssets,
    model_id: str,
    *,
    dtype: str,
):
    model = snapshot_download(model_id, allow_patterns=DOWNLOAD_PATTERN)
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    img_processor = CLIPImageProcessor.from_pretrained(model)
    images = [asset.pil_image for asset in image_assets]
    # Input resolution must be a multiple of `self.min_resolution_step`.
    # Using `self.get_nearest_supported_resolution`, for assets 432x642 the
    # nearest supported resolution is 432x640.
    pixel_values = [
        img_processor(
            images,
            return_tensors='pt').pixel_values.to(torch_dtype)[:, :, :, :640]
        for images in images
    ]

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    hidden_size, num_layers, num_heads, intermediate_size = (
        get_args_from_model_type(config.args["model"]))

    config.num_hidden_layers = num_layers
    config.hidden_size = hidden_size
    config.num_attention_heads = num_heads
    config.intermediate_size = intermediate_size
    config.norm_type = "layer_norm"
    config.image_size = 224
    config.hidden_act = "gelu"
    config.layer_norm_eps = 1e-6
    config.initializer_factor = 1.0
    config.qkv_bias = True
    config.qk_normalization = False
    config.max_img_size = 2048
    config.reg_tokens = config.args["register_multiple"]

    hf_model = AutoModel.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to("cuda")
    hf_model.eval()

    hf_outputs_per_image = [
        hf_model(pixel_value.to("cuda")).features
        for pixel_value in pixel_values
    ]

    vllm_model = RadioModel(config)
    vllm_model.load_weights(hf_model.state_dict())

    del hf_model
    cleanup_dist_env_and_memory()

    vllm_model = vllm_model.to("cuda", torch_dtype)
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
    "nvidia/C-RADIOv2-H",
])
@pytest.mark.parametrize("dtype", ["half"])
def test_radio(dist_init, image_assets, model_id, dtype: str) -> None:
    run_radio_test(
        image_assets,
        model_id,
        dtype=dtype,
    )
