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
    loaded = vllm_model.load_weights(hf_model.state_dict())
    # Guard the remote-code -> vLLM key remap: every parameter must be loaded
    # from the checkpoint, except the LayerScale gains (layer_scale{1,2}.lambda1),
    # which are identity-init and may be absent from the remote-code checkpoint.
    expected = {
        name
        for name, _ in vllm_model.named_parameters()
        if not name.endswith((".layer_scale1.lambda1", ".layer_scale2.lambda1"))
    }
    missing = expected - loaded
    assert not missing, f"parameters not loaded from checkpoint: {sorted(missing)}"
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


def _radio_summary(*, teachers, cls_token_per_teacher):
    """Summary tensor from a tiny RadioModel with no encoder layers, so it
    builds no parallel layers and needs neither a GPU nor a checkpoint."""
    config = RadioConfig(
        model_name="vit_small_patch16_224",
        teachers=teachers,
        cls_token_per_teacher=cls_token_per_teacher,
    )
    model = RadioModel(config, num_hidden_layers_override=0)
    # Synthetic encoder output: [batch, num_skip + num_patches, hidden].
    y = torch.randn(2, model.embeddings.num_skip + 4, config.hidden_size)
    summary, _ = model._extract_final(y)
    return summary


def test_summary_idxs_no_teachers_keeps_class_tokens():
    # No teachers -> summary_idxs is None -> keep the class-token summary.
    summary = _radio_summary(teachers=[], cls_token_per_teacher=False)
    assert summary.shape[1] > 0


def test_summary_idxs_all_use_summary_false_is_empty():
    # Teachers present but none flagged use_summary -> empty selection.
    summary = _radio_summary(
        teachers=[{"name": "a", "use_summary": False}],
        cls_token_per_teacher=True,
    )
    assert summary.shape[1] == 0


def _native_state_dict_from_model(model: RadioModel) -> dict[str, torch.Tensor]:
    """Reverse-map a model's parameters onto native Transformers checkpoint keys
    (``encoder.layer.N.*`` with split ``attention.attention.{query,key,value}``
    and ``attention.output.dense``), so loading it must round-trip back to the
    original parameters."""
    state_dict: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        data = param.detach().clone()
        # The module tree already mirrors the native ``encoder.layer.N.*`` names
        # (including ``layer_scale{1,2}.lambda1``); only the fused attention
        # projections differ from the native split ones.
        if ".attention.qkv." in name:
            base, suffix = name.split(".attention.qkv.")
            query, key, value = data.chunk(3, dim=0)
            for proj, shard in (("query", query), ("key", key), ("value", value)):
                state_dict[f"{base}.attention.attention.{proj}.{suffix}"] = (
                    shard.clone()
                )
        elif ".attention.proj." in name:
            base, suffix = name.split(".attention.proj.")
            state_dict[f"{base}.attention.output.dense.{suffix}"] = data
        else:
            state_dict[name] = data
    return state_dict


def test_native_format_weight_loading(default_vllm_config, dist_init):
    # Native Transformers checkpoints nest the encoder under ``encoder.layer.N``
    # with split attention projections; loading must fuse q/k/v into the packed
    # ``qkv`` in the right order and land every other tensor in place, matching
    # the legacy layout the integration test covers.
    config = RadioConfig(model_name="vit_small_patch16_224")
    model = RadioModel(config, num_hidden_layers_override=2)

    # Every parameter (including layer_scale) must load from the native keys.
    expected = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    native_weights = _native_state_dict_from_model(model)
    # Keys both layouts intentionally drop must be skipped, not error.
    native_weights["input_conditioner.norm_mean"] = torch.zeros(3)
    native_weights["input_conditioner.norm_std"] = torch.ones(3)
    native_weights["summary_idxs"] = torch.zeros(1, dtype=torch.long)

    # Zero every parameter so a weight that fails to load is visible as zeros.
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    loaded = model.load_weights(native_weights)

    missing = set(expected) - loaded
    assert not missing, f"native weights not loaded: {sorted(missing)}"
    params = dict(model.named_parameters())
    for name, original in expected.items():
        assert torch.equal(params[name], original), (
            f"parameter mismatch after native load: {name}"
        )
