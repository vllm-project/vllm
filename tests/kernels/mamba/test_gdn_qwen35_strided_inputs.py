# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 GDN glue: strided b/a gating views and strided-z gated RMS norm.

The Qwen3.5 (non-interleaved) branch of
``QwenGatedDeltaNetAttention.forward_cuda`` now passes ``b``/``a`` as chunk
views of the packed ``ba`` projection (row stride ``2*HV``) and ``z`` as a
strided view of the packed ``qkvz`` projection, removing two ``contiguous()``
copies and one reshape copy per GDN layer. These tests pin bitwise
equivalence with the old contiguous path for each consumer.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="Qwen3.5 GDN strided-input kernels require CUDA.",
        allow_module_level=True,
    )

from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import (  # noqa: E402
    fused_gdn_gated_rms_norm,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (  # noqa: E402
    rmsnorm_fn,
)

H = 2  # num K heads
HV = 4  # num V heads
K = 128  # head_k_dim
V = 128  # head_v_dim
QKV_DIM = 2 * H * K + HV * V
Z_SIZE = HV * V


def _packed_qkvz_ba(num_tokens: int, dtype: torch.dtype):
    """Build packed projections and return the strided views the layer uses."""
    qkvz = torch.randn(num_tokens, QKV_DIM + Z_SIZE, dtype=dtype, device="cuda")
    ba = torch.randn(num_tokens, 2 * HV, dtype=dtype, device="cuda")
    mixed_qkv, z = qkvz.split([QKV_DIM, Z_SIZE], dim=-1)
    b, a = ba.chunk(2, dim=-1)
    return mixed_qkv, z, b, a


@pytest.mark.parametrize("num_tokens", [1, 9])
@torch.inference_mode()
def test_packed_decode_strided_inputs_bitwise(num_tokens: int) -> None:
    """Packed non-spec decode accepts strided mixed_qkv/b/a views and matches
    the contiguous path bitwise."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    mixed_qkv, _, b_view, a_view = _packed_qkvz_ba(num_tokens, dtype)
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")
    state = torch.randn(num_tokens + 1, HV, V, K, dtype=torch.float32, device="cuda")
    indices = torch.arange(1, num_tokens + 1, dtype=torch.int32, device="cuda")

    out_strided = torch.empty(num_tokens, 1, HV, V, dtype=dtype, device="cuda")
    state_strided = state.clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a_view,
        b=b_view,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_strided,
        out=out_strided,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )

    out_contig = torch.empty(num_tokens, 1, HV, V, dtype=dtype, device="cuda")
    state_contig = state.clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv.contiguous(),
        a=a_view.contiguous(),
        b=b_view.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_contig,
        out=out_contig,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out_strided, out_contig, rtol=0, atol=0)
    torch.testing.assert_close(state_strided, state_contig, rtol=0, atol=0)


@torch.inference_mode()
def test_fused_post_conv_prep_strided_ba_bitwise() -> None:
    """Prefill post-conv prep consumes strided b/a chunk views and matches the
    contiguous path bitwise."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    num_tokens = 33
    conv_output = torch.randn(num_tokens, QKV_DIM, dtype=dtype, device="cuda")
    _, _, b_view, a_view = _packed_qkvz_ba(num_tokens, dtype)
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")

    ref = fused_post_conv_prep(
        conv_output=conv_output,
        a=a_view.contiguous(),
        b=b_view.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=True,
        output_g_exp=False,
    )
    got = fused_post_conv_prep(
        conv_output=conv_output,
        a=a_view,
        b=b_view,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=True,
        output_g_exp=False,
    )
    for r, g in zip(ref, got, strict=True):
        torch.testing.assert_close(g, r, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_fused_gdn_gated_rms_norm_strided_z_bitwise(
    num_tokens: int, norm_before_gate: bool, activation: str
) -> None:
    """The fused gated RMS norm reads z from its packed strided view and
    matches ``rmsnorm_fn`` on the copied/reshaped z bitwise."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    _, z_view, _, _ = _packed_qkvz_ba(num_tokens, dtype)
    x = torch.randn(num_tokens, HV, V, dtype=dtype, device="cuda")
    weight = torch.randn(V, dtype=torch.float32, device="cuda")
    eps = 1e-6

    ref = rmsnorm_fn(
        x,
        weight,
        None,
        z=z_view.reshape(num_tokens, HV, V),
        eps=eps,
        group_size=None,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )

    out = torch.empty(num_tokens, HV * V, dtype=dtype, device="cuda")
    fused_gdn_gated_rms_norm(
        x=x.view(num_tokens, -1),
        z=z_view,
        weight=weight,
        out=out,
        eps=eps,
        num_v_heads=HV,
        head_v_dim=V,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )

    torch.testing.assert_close(out.view(num_tokens, HV, V), ref, rtol=0, atol=0)
