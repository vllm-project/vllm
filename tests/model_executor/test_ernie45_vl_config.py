# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig
from transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
    Ernie4_5_VLMoeConfig,
)

from vllm.model_executor.models.ernie45_vl_config import (
    get_ernie4_5_vl_config,
    get_ernie4_5_vl_vision_norm_eps,
)
from vllm.model_executor.models.ernie45_vl_moe import _is_moe_layer


def test_native_config_view_preserves_ernie_semantics():
    layer_types = ["dense", "sparse", "dense", "sparse"]
    hf_config = Ernie4_5_VLMoeConfig(
        image_token_id=101,
        video_token_id=102,
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": len(layer_types),
            "moe_num_experts": 8,
            "mlp_layer_types": layer_types,
            "rms_norm_eps": 1e-5,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 128,
            "spatial_merge_size": 2,
            "temporal_merge_size": 3,
            "rms_norm_eps": 1e-6,
        },
    )

    config = get_ernie4_5_vl_config(hf_config)

    assert config.hidden_size == 64
    assert config.im_patch_id == 101
    assert config.video_token_id == 102
    assert config.spatial_conv_size == 2
    assert config.temporal_conv_size == 3
    assert config.moe_num_experts == [8, 8]
    assert get_ernie4_5_vl_vision_norm_eps(config) == 1e-6
    assert config.vision_config is hf_config.vision_config
    assert [_is_moe_layer(config, index) for index in range(4)] == [
        False,
        True,
        False,
        True,
    ]
    assert not hasattr(hf_config, "hidden_size")
    assert hf_config.text_config.moe_num_experts == 8


def test_legacy_flat_config_is_unchanged():
    hf_config = PretrainedConfig()
    hf_config.hidden_size = 64
    hf_config.im_patch_id = 101
    hf_config.rms_norm_eps = 2e-6
    hf_config.vision_config = PretrainedConfig()

    assert get_ernie4_5_vl_config(hf_config) is hf_config
    assert get_ernie4_5_vl_vision_norm_eps(hf_config) == 2e-6
