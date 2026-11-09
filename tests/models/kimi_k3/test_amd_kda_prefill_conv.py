# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How the ROCm KDA `_forward` slices the packed Q/K/V tensors.

The conv weight and conv state are packed `[q | k | v]` along the channel dim.
Without the AITER fused conv, the prefill branch slices them per Q/K/V and runs
three causal convs; a depthwise conv is per channel, so that must equal one conv
over the packed tensors. Every branch then splits the packed head dim.
"""

from types import SimpleNamespace

import pytest
import torch
from einops import rearrange

from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.platforms import current_platform


@pytest.mark.parametrize("num_tokens", [1, 7])
def test_split_heads_matches_rearrange_on_a_strided_band(num_tokens: int) -> None:
    # The layer splits row-strided bands out of the packed projection.
    head_dim, heads = 4, 3
    packed = torch.randn(num_tokens, 3 * heads * head_dim + 5)
    band = packed[:, heads * head_dim : 2 * heads * head_dim]
    layer = SimpleNamespace(head_dim=head_dim)

    out = KimiK3DeltaAttention._split_heads(layer, band)

    expected = rearrange(band, "n (h d) -> 1 n h d", d=head_dim)
    assert out.shape == (1, num_tokens, heads, head_dim)
    assert out.stride() == expected.stride()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert out.data_ptr() == band.data_ptr(), "must stay a view"


HEAD_DIM = 128
CONV_WIDTH = 4
BLOCK_SIZE = 64


@pytest.fixture(params=["DS", "SD"])
def conv_layout(monkeypatch, request):
    from vllm.model_executor.layers.mamba import mamba_utils

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", request.param)
    mamba_utils.get_conv_state_layout.cache_clear()
    yield request.param
    mamba_utils.get_conv_state_layout.cache_clear()


def _layer(monkeypatch):
    from tests.v1.attention.utils import create_vllm_config
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import set_current_vllm_config
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    # Keep every step on the generic path: no fused decode, no AITER conv.
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(rocm_aiter_ops, "is_kda_gluon_decode_available", lambda: False)
    monkeypatch.setattr(rocm_aiter_ops, "is_kda_gluon_prefill_available", lambda: False)
    config = KimiLinearConfig(
        hidden_size=256,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": HEAD_DIM,
            "num_heads": 4,
            "short_conv_kernel_size": CONV_WIDTH,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, block_size=BLOCK_SIZE
    )
    vllm_config.cache_config.mamba_cache_mode = "none"
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        layer = KimiK3DeltaAttention(
            config, vllm_config, prefix="model.layers.0.self_attn"
        )
    torch.manual_seed(0)
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(std=0.5)
    layer.decode_conv1d_weight = None
    layer.decode_norm_weight = None
    return layer, vllm_config


def _metadata(vllm_config, batch):
    from tests.v1.attention.utils import create_common_attn_metadata
    from vllm.models.kimi_k3.amd.kda_metadata import KimiK3ROCmKDAMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, torch.device("cuda"), arange_block_indices=True
    )
    builder = KimiK3ROCmKDAMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode="none",
        ),
        layer_names=["model.layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )
    return builder.build(0, common), int(common.block_table_tensor.max()) + 1


def _packed_conv(x_t, weight, bias, conv_state, query_start_loc, dim_q, dim_k, **kw):
    """One causal conv over all packed channels, split afterwards."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    out = causal_conv1d_fn(
        x_t,
        weight,
        bias,
        activation=kw["activation"],
        conv_states=conv_state,
        has_initial_state=kw["has_initial_state"],
        cache_indices=kw["cache_indices"],
        query_start_loc=query_start_loc,
        metadata=kw["metadata"],
    ).transpose(0, 1)
    return out.split([dim_q, dim_k, out.shape[-1] - dim_q - dim_k], dim=-1)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm KDA layer")
def test_per_qkv_prefill_conv_matches_packed_conv(
    dist_init, monkeypatch, conv_layout: str
) -> None:
    from tests.v1.attention.utils import BatchSpec
    from vllm.model_executor.layers.mamba import mamba_utils

    layer, vllm_config = _layer(monkeypatch)
    assert mamba_utils.is_conv_state_dim_first() == (conv_layout == "DS")
    # Decode-first: one decode, a resumed chunk and a fresh prompt.
    batch = BatchSpec(seq_lens=[70, 200, 90], query_lens=[1, 72, 90])
    md, num_slots = _metadata(vllm_config, batch)
    assert md.num_decodes == 1 and md.num_prefills == 2
    monkeypatch.setattr(
        "vllm.models.kimi_k3.amd.kda.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: md}),
    )

    n = sum(batch.query_lens)
    dim = layer.local_projection_size
    heads = layer.local_num_heads
    torch.manual_seed(1)
    # mixed_qkv is a band of the wider in_proj output, as in forward().
    mixed_qkv = torch.randn(n, 3 * dim + 64, device="cuda", dtype=torch.bfloat16)
    mixed_qkv = mixed_qkv[:, : 3 * dim]
    g1 = torch.randn(1, n, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    g2 = torch.randn(n, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(1, n, heads, device="cuda", dtype=torch.bfloat16)
    conv_shape, state_shape = layer.get_state_shape()
    conv_dtype, state_dtype = layer.get_state_dtype()
    conv_seed = torch.randn(num_slots, *conv_shape, device="cuda").to(conv_dtype)
    state_seed = 0.1 * torch.randn(num_slots, *state_shape, device="cuda")
    state_seed = state_seed.to(state_dtype)

    def run(fused_conv) -> tuple[torch.Tensor, ...]:
        conv_state, state = conv_seed.clone(), state_seed.clone()
        layer.kv_cache = (conv_state, state)
        layer._fused_qkv_conv_fn = fused_conv
        out = torch.zeros(1, n, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        layer._forward(mixed_qkv=mixed_qkv, g1=g1, g2=g2, beta=beta, core_attn_out=out)
        return out, conv_state, state

    out, conv_state, state = run(None)
    ref_out, ref_conv_state, ref_state = run(_packed_conv)

    assert torch.count_nonzero(out) > 0
    assert not torch.equal(conv_state, conv_seed), "prefill must move the window"
    # The conv state holds raw inputs, so it must match exactly.
    torch.testing.assert_close(conv_state, ref_conv_state, atol=0, rtol=0)
    torch.testing.assert_close(state, ref_state)
    torch.testing.assert_close(out, ref_out)
