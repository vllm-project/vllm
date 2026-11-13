# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.bailing_mrope import (
    BailingMRotaryEmbedding,
)
from vllm.model_executor.models.bailing_moe_v3 import _build_mla_rotary_embedding
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.bailing_moe_v3_vl import BailingMoeV3TextConfig


def test_bailing_mrope_selects_alternating_spatial_then_temporal():
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        rotary_emb = BailingMRotaryEmbedding(
            head_size=12,
            rotary_dim=12,
            max_position_embeddings=8,
            base=10_000,
            is_neox_style=False,
            dtype=torch.float32,
            mrope_section=[2, 2, 2],
        )

    positions = torch.tensor([[-1, 2], [-3, 4], [5, -6]])
    frequencies = torch.arange(6).expand(8, -1)
    position_values = torch.arange(8).unsqueeze(1) * 10
    cos = position_values + frequencies
    sin = cos + 100
    cos_sin_cache = torch.cat((cos, sin), dim=-1)

    actual_cos, actual_sin = rotary_emb.select_cos_sin(
        positions,
        cos_sin_cache,
    )

    expected_cos = torch.tensor(
        [
            [30, 51, 32, 53, 14, 15],
            [40, 61, 42, 63, 24, 25],
        ]
    )
    torch.testing.assert_close(actual_cos, expected_cos)
    signs = torch.tensor([[-1, 1, -1, 1, -1, -1], [1, -1, 1, -1, 1, 1]])
    torch.testing.assert_close(actual_sin, (expected_cos + 100) * signs)


def test_bailing_factory_preserves_mla_width_and_config():
    config = BailingMoeV3TextConfig(
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 6_000_000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [8, 12, 12],
        },
    )
    original = copy.deepcopy(vars(config))
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        rope = _build_mla_rotary_embedding(config, head_size=64)
        assert rope is _build_mla_rotary_embedding(config, head_size=64)
        generic = get_rope(
            head_size=64,
            max_position=128,
            is_neox_style=False,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 6_000_000,
                "mrope_section": [8, 12, 12],
            },
        )
    assert isinstance(rope, BailingMRotaryEmbedding)
    assert rope is not generic
    assert rope.rotary_dim == 64
    assert vars(config) == original


def test_bailing_factory_rejects_unsupported_scaling():
    config = BailingMoeV3TextConfig(
        rope_parameters={
            "rope_type": "linear",
            "factor": 2.0,
            "mrope_section": [8, 12, 12],
        }
    )
    with pytest.raises(ValueError, match="only supports rope_type='default'"):
        _build_mla_rotary_embedding(config, head_size=64)


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("mscale_all_dim", [0.0, 1.2])
def test_bailing_yarn_matches_hf_without_halving_mla_width(
    multimodal: bool, mscale_all_dim: float
):
    """YaRN uses the training length and explicit MLA width for both text and VL."""
    parameters = dict(
        rope_type="yarn",
        rope_theta=6_000_000.0,
        factor=2.0,
        original_max_position_embeddings=128,
        partial_rotary_factor=0.5,
        mscale=0.7,
        mscale_all_dim=mscale_all_dim,
    )
    if multimodal:
        parameters["mrope_section"] = [8, 12, 12]
    config = BailingMoeV3TextConfig(
        max_position_embeddings=256, rope_parameters=parameters
    )
    original = copy.deepcopy(vars(config))
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        rope = _build_mla_rotary_embedding(config, head_size=64)
    # HF derives 64 rotary dimensions from head_dim=128 and partial_factor=0.5.
    reference = PreTrainedConfig(
        head_dim=128,
        hidden_size=128,
        num_attention_heads=1,
        max_position_embeddings=256,
        rope_parameters=parameters,
    )
    inv_freq, amplitude = ROPE_INIT_FUNCTIONS["yarn"](reference, torch.device("cpu"))
    positions = torch.tensor([0, 127, 128, 255])
    angles = positions.float()[:, None] * inv_freq
    expected = torch.cat((angles.cos(), angles.sin()), dim=-1) * amplitude
    assert rope.cos_sin_cache.shape == (256, 64)
    torch.testing.assert_close(rope.cos_sin_cache[positions], expected)
    from vllm.model_executor.models.bailing_moe_v3 import _get_mla_attention_scale

    assert _get_mla_attention_scale(config, 192) == pytest.approx(
        192**-0.5 * (1 + 0.1 * mscale_all_dim * math.log(2)) ** 2
    )
    assert vars(config) == original


def _load_yarn_config(tmp_path: Path, rope_field: str, parameters: dict):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bailing_moe_v3_vl",
                "mrope_section": [8, 12, 12],
                "text_config": {
                    "max_position_embeddings": 256,
                    rope_field: parameters,
                },
            }
        )
    )
    return get_config(tmp_path, trust_remote_code=False).get_text_config()


@pytest.mark.parametrize(
    ("rope_field", "type_field"),
    [("rope_parameters", "rope_type"), ("rope_scaling", "type")],
)
def test_bailing_yarn_loads_standard_and_legacy_config(
    tmp_path: Path, rope_field: str, type_field: str
):
    """Use the shared config loader to normalize YaRN before model construction."""
    config = _load_yarn_config(
        tmp_path,
        rope_field,
        {type_field: "yarn", "factor": 2.0, "original_max_position_embeddings": 128},
    )
    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 2.0
    assert config.rope_parameters["original_max_position_embeddings"] == 128
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        rope = _build_mla_rotary_embedding(config, head_size=64)
    assert isinstance(rope, BailingMRotaryEmbedding)
    assert rope.cos_sin_cache.shape == (256, 64)


def test_bailing_yarn_config_requires_explicit_factor(tmp_path: Path):
    with pytest.raises(KeyError, match="factor"):
        _load_yarn_config(
            tmp_path,
            "rope_parameters",
            {"rope_type": "yarn", "original_max_position_embeddings": 128},
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Skipping CUDA/ROCm-only test.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("position_kind", ["text", "multimodal"])
@pytest.mark.parametrize("scaling_factor", [None, 2.0])
@torch.inference_mode()
def test_bailing_mrope_cuda_matches_native_for_strided_mla_views(
    dtype: torch.dtype,
    position_kind: str,
    scaling_factor: float | None,
):
    device = torch.device("cuda")
    num_tokens = 17
    num_query_heads = 8
    with set_current_vllm_config(VllmConfig()):
        rotary_emb = BailingMRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=64,
            base=6_000_000,
            is_neox_style=False,
            dtype=dtype,
            mrope_section=[8, 12, 12],
            scaling_factor=scaling_factor,
        ).to(device)

    torch.manual_seed(0)
    multimodal = position_kind == "multimodal"
    positions = torch.randint(
        -32 if multimodal else 0,
        64,
        (3, 2 * num_tokens) if multimodal else (2 * num_tokens,),
        dtype=torch.int32 if multimodal else torch.int64,
        device=device,
    )[..., ::2]
    reference_positions = positions if multimodal else positions.expand(3, -1)
    query_buffer = torch.randn(
        num_tokens,
        num_query_heads,
        192,
        dtype=dtype,
        device=device,
    )
    key_buffer = torch.randn(
        num_tokens,
        576,
        dtype=dtype,
        device=device,
    )
    query = query_buffer[..., 128:]
    key = key_buffer[:, 512:].unsqueeze(1)
    assert not query.is_contiguous()
    assert not key.is_contiguous()

    query_prefix = query_buffer[..., :128].clone()
    key_prefix = key_buffer[:, :512].clone()
    expected_query, expected_key = rotary_emb.forward_native(
        reference_positions,
        query.clone(),
        key.clone(),
    )
    actual_query, actual_key = rotary_emb.forward_cuda(positions, query, key)
    assert expected_key is not None
    assert actual_key is not None

    atol, rtol = (0, 0) if position_kind == "multimodal" else (1e-2, 1.6e-2)
    torch.testing.assert_close(actual_query, expected_query, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_key, expected_key, atol=atol, rtol=rtol)
    assert torch.equal(query_buffer[..., :128], query_prefix)
    assert torch.equal(key_buffer[:, :512], key_prefix)
    if position_kind == "multimodal":
        assert actual_query.data_ptr() == query.data_ptr()
        assert actual_key.data_ptr() == key.data_ptr()
