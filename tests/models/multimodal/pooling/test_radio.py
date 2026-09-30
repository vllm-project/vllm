# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.models.radio import RadioModel
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import ImageTestAssets

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]

DEVICE_TYPE = current_platform.device_type


def test_radio_loads_hf_embeddings_qkv_and_layer_scales(default_vllm_config, dist_init):
    config = RadioConfig(
        model_name="vit_small_patch16_224",
        image_size=32,
        cpe_max_size=32,
        teachers=[{"name": 0}],
        register_multiple=1,
        video_temporal_patch_size=2,
    )
    model = RadioModel(config, num_hidden_layers_override=1)
    params = dict(model.named_parameters())
    mapping = {
        "embeddings.patch_projection.weight": "model.patch_generator.embedder.weight",
        "embeddings.video_patch_projection.weight": (
            "model.patch_generator.video_embedder.weight"
        ),
        "embeddings.position_embedding": "model.patch_generator.pos_embed",
        "embeddings.cls_register_token": "model.patch_generator.cls_token.token",
        "encoder.layer.0.attention.output.dense.weight": (
            "model.encoder.layers.0.attn.proj.weight"
        ),
        "encoder.layer.0.layer_scale1.lambda1": "model.encoder.layers.0.ls1",
        "encoder.layer.0.layer_scale2.lambda1": "model.encoder.layers.0.ls2",
    }
    sources = {
        source: torch.full_like(params[target], index + 0.25)
        for index, (source, target) in enumerate(mapping.items())
    }
    for suffix in ("weight", "bias"):
        target = params[f"model.encoder.layers.0.attn.qkv.{suffix}"]
        for index, projection in enumerate(("query", "key", "value")):
            sources[f"encoder.layer.0.attention.attention.{projection}.{suffix}"] = (
                torch.full_like(target.chunk(3)[index], index + 1.0)
            )
    loaded = model.load_weights(sources)
    assert set(mapping.values()) <= loaded
    for source, target in mapping.items():
        torch.testing.assert_close(params[target], sources[source], rtol=0, atol=0)
    for suffix in ("weight", "bias"):
        target = params[f"model.encoder.layers.0.attn.qkv.{suffix}"]
        for index, actual in enumerate(target.chunk(3)):
            torch.testing.assert_close(actual, torch.full_like(actual, index + 1.0))


@torch.inference_mode()
def run_radio_test(
    image_assets: ImageTestAssets,
    model_id: str,
    *,
    dtype: str,
):
    model = hf_api().snapshot_download(model_id, allow_patterns=DOWNLOAD_PATTERN)
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    img_processor = CLIPImageProcessor.from_pretrained(model)
    images = [asset.pil_image for asset in image_assets]
    # Input resolution must be a multiple of `self.min_resolution_step`.
    # Using `self.get_nearest_supported_resolution`, for assets 432x642 the
    # nearest supported resolution is 432x640.
    pixel_values = [
        img_processor(image, return_tensors="pt").pixel_values.to(torch_dtype)[
            :, :, :, :640
        ]
        for image in images
    ]

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # RADIO model on HF does not properly handle torch_dtype argument
    # And relies on args["dtype"] which we have to patch manually:
    hf_config.args["dtype"] = torch_dtype

    hf_model = AutoModel.from_pretrained(
        model_id,
        config=hf_config,
        dtype=torch_dtype,
        trust_remote_code=True,
    ).to(DEVICE_TYPE)
    hf_model.eval()

    # A HF model has image normalization as a part of model's forward
    # However in vLLM we don't make normalization a part of the model
    # forward step since mean/std stored as model's parameters and
    # subject to precision loss (when using fp16/bf16) which negatively
    # affects evaluation benchmarks.
    hf_model.make_preprocessor_external()

    hf_outputs_per_image = [
        hf_model(pixel_value.to(DEVICE_TYPE)) for pixel_value in pixel_values
    ]

    vllm_config = RadioConfig(
        model_name=hf_config.args["model"],
        **hf_config.args,
    )
    vllm_model = RadioModel(vllm_config)
    vllm_model.load_weights(hf_model.state_dict())
    vllm_model = vllm_model.to(DEVICE_TYPE, torch_dtype)

    vllm_outputs_per_image = [
        vllm_model(pixel_values=pixel_value.to(DEVICE_TYPE))
        for pixel_value in pixel_values
    ]
    del vllm_model, hf_model
    cleanup_dist_env_and_memory()

    cos_similar = nn.CosineSimilarity(dim=-1)
    for vllm_output, hf_output in zip(vllm_outputs_per_image, hf_outputs_per_image):
        assert cos_similar(vllm_output[0], hf_output[0]).mean() > 0.99
        assert cos_similar(vllm_output[1], hf_output[1]).mean() > 0.99


@pytest.mark.parametrize(
    "model_id",
    [
        "nvidia/C-RADIOv2-H",
    ],
)
@pytest.mark.parametrize("dtype", ["half", "bfloat16"])
def test_radio(
    default_vllm_config, dist_init, image_assets, model_id, dtype: str
) -> None:
    run_radio_test(
        image_assets,
        model_id,
        dtype=dtype,
    )
