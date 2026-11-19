# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.lfm3 import get_rotary_embedding
from vllm.transformers_utils.configs.lfm3 import Lfm3Config
from vllm.utils.torch_utils import set_default_torch_dtype

# rope_parameters as written by the released hub checkpoints
HUB_ROPE_PARAMETERS = {
    "full_attention": {
        "rope_type": "default",
        "rope_theta": 1_000_000.0,
        "partial_rotary_factor": 0.5,
    },
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
}


@pytest.fixture
def cpu_vllm_config():
    # get_rope uses the default dtype, which earlier tests can leave at bf16
    with (
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
        set_default_torch_dtype(torch.float32),
    ):
        yield


def _reference_rope(x: torch.Tensor, positions: torch.Tensor, params: dict):
    # Hub remote code: the rotated frequencies follow the full head dimension
    # and the unrotated tail gets zero frequency.
    head_dim = x.shape[-1]
    num_freqs = int(params.get("partial_rotary_factor", 1.0) * head_dim) // 2
    exponent = torch.arange(0, 2 * num_freqs, 2, dtype=torch.float) / head_dim
    inv_freq = torch.cat(
        [params["rope_theta"] ** -exponent, torch.zeros(head_dim // 2 - num_freqs)]
    )
    freqs = positions[:, None].float() * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)[:, None]
    x1, x2 = x.chunk(2, dim=-1)
    return x * emb.cos() + torch.cat([-x2, x1], dim=-1) * emb.sin()


def test_hub_config():
    config = Lfm3Config(
        rope_parameters={k: dict(v) for k, v in HUB_ROPE_PARAMETERS.items()},
        tie_embedding=False,
        is_causal=True,
    )
    assert config.rope_parameters["full_attention"]["rope_type"] == "proportional"
    assert config.rope_parameters["sliding_attention"]["rope_type"] == "default"
    assert config.use_attention_gate and config.use_sandwich_norm
    assert not config.tie_word_embeddings
    assert config.layer_types[:5] == [
        "conv",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
    ]


@pytest.mark.parametrize("layer_type", ["full_attention", "sliding_attention"])
def test_rotary_embedding_matches_hub(cpu_vllm_config, layer_type: str):
    config = Lfm3Config(
        hidden_size=64,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rope_parameters={k: dict(v) for k, v in HUB_ROPE_PARAMETERS.items()},
    )
    rotary_emb = get_rotary_embedding(config, layer_type)

    torch.manual_seed(0)
    query = torch.randn(4, 2, 32)
    key = torch.randn(4, 1, 32)
    positions = torch.tensor([0, 1, 63, 127])
    actual_q, actual_k = rotary_emb.forward_native(
        positions, query.flatten(1).clone(), key.flatten(1).clone()
    )
    params = HUB_ROPE_PARAMETERS[layer_type]
    expected_q = _reference_rope(query, positions, params).flatten(1)
    expected_k = _reference_rope(key, positions, params).flatten(1)
    torch.testing.assert_close(actual_q, expected_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-5, rtol=1e-5)
