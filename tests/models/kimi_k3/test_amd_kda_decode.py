# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm fused KDA decode kernel must match the Triton chain it replaces.

The fused kernel folds the packed causal conv1d update, the gated delta-rule
recurrence and the gated output RMSNorm into one launch, and updates both the
conv state and the recurrent state in place. Every one of those outputs is
compared against the three-kernel path the AMD layer falls back to.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(),
    reason="The fused KDA decode kernel is only built for gfx950",
)

# Kimi-K3 KDA: 96 heads x 128, conv width 4, gate_lower_bound -5.0.
HEAD_DIM = 128
CONV_WIDTH = 4
GATE_LOWER_BOUND = -5.0
NORM_EPS = 1e-5
DTYPE = torch.bfloat16


def _requires_kernel() -> None:
    if not hasattr(torch.ops._C, "fused_kda_decode"):
        pytest.skip("vLLM was built without the fused KDA decode kernel")


class KdaDecodeInputs:
    """One decode step of a Kimi-K3 KDA layer, in the layer's own layout."""

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        num_slots: int,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        device = "cuda"
        dim = num_heads * HEAD_DIM
        self.num_heads = num_heads
        self.dim = dim

        self.mixed_qkv = (
            torch.randn(num_tokens, 3 * dim, device=device, dtype=DTYPE) * 0.5
        )
        # conv1d weight as the layer holds it: [3 * dim, width] fp32.
        self.conv_weights = (
            torch.randn(3 * dim, CONV_WIDTH, device=device, dtype=torch.float32) * 0.3
        )
        # Width-major fp32 copy the fused kernel indexes: [3, width, dim].
        self.decode_conv1d_weight = torch.stack(
            [
                self.conv_weights[i * dim : (i + 1) * dim].transpose(0, 1).contiguous()
                for i in range(3)
            ]
        )
        # SD cache layout: [slots, width - 1, 3 * dim]; the layer transposes it.
        self.conv_state = (
            torch.randn(num_slots, CONV_WIDTH - 1, 3 * dim, device=device, dtype=DTYPE)
            * 0.5
        )
        self.recurrent_state = (
            torch.randn(
                num_slots,
                num_heads,
                HEAD_DIM,
                HEAD_DIM,
                device=device,
                dtype=torch.float32,
            )
            * 0.1
        )
        self.g1 = (
            torch.randn(1, num_tokens, num_heads, HEAD_DIM, device=device, dtype=DTYPE)
            * 0.5
        )
        self.g2 = (
            torch.randn(num_tokens, num_heads, HEAD_DIM, device=device, dtype=DTYPE)
            * 0.5
        )
        self.beta = torch.randn(1, num_tokens, num_heads, device=device, dtype=DTYPE)
        self.A_log = torch.randn(num_heads, device=device, dtype=torch.float32) * 0.5
        self.dt_bias = torch.randn(dim, device=device, dtype=torch.float32) * 0.1
        self.norm_weight_bf16 = 1 + 0.1 * torch.randn(
            HEAD_DIM, device=device, dtype=DTYPE
        )
        self.decode_norm_weight = self.norm_weight_bf16.float()
        # Distinct, shuffled slots: the kernel must honour the indirection.
        # Slot 0 is NULL_BLOCK_ID and never allocated for live state, and the
        # Triton reference short-circuits it, so draw from [1, num_slots).
        assert num_slots > num_tokens, "need a spare slot to detect stray writes"
        self.state_indices = (
            torch.randperm(num_slots - 1, device=device)[:num_tokens] + 1
        ).to(torch.int32)

    def conv_state_view(self, state: torch.Tensor) -> torch.Tensor:
        return state.transpose(-1, -2)


def _gated_rmsnorm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    normed = x_float * torch.rsqrt(variance + eps) * weight.float()
    return (normed * torch.sigmoid(gate.float())).to(x.dtype)


def _run_triton_chain(
    inp: KdaDecodeInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """conv1d update -> recurrent decode -> gated norm, as the layer runs it."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.ops.third_party.kda import (
        fused_recurrent_kda_packed_decode,
    )

    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    conv_out = torch.empty_like(inp.mixed_qkv)
    causal_conv1d_update(
        inp.mixed_qkv,
        inp.conv_state_view(conv_state),
        inp.conv_weights,
        None,
        activation="silu",
        conv_state_indices=inp.state_indices,
        validate_data=True,
        out=conv_out,
    )
    core_attn_out, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=conv_out,
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=GATE_LOWER_BOUND,
        initial_state=recurrent_state,
        state_indices=inp.state_indices,
    )
    out = _gated_rmsnorm(core_attn_out, inp.g2, inp.norm_weight_bf16, NORM_EPS)
    return out, conv_state, recurrent_state


def _run_fused(
    inp: KdaDecodeInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops

    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    num_tokens = inp.mixed_qkv.shape[0]
    out = torch.empty(
        1,
        num_tokens,
        inp.num_heads,
        HEAD_DIM,
        device=inp.mixed_qkv.device,
        dtype=DTYPE,
    )
    ops.fused_kda_decode(
        x=inp.mixed_qkv,
        weight=inp.decode_conv1d_weight,
        bias=None,
        conv_state=inp.conv_state_view(conv_state),
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        state_indices=inp.state_indices,
        state=recurrent_state,
        out=out,
        lower_bound=GATE_LOWER_BOUND,
        output_gate=inp.g2,
        norm_weight=inp.decode_norm_weight,
        norm_eps=NORM_EPS,
    )
    return out, conv_state, recurrent_state


@pytest.mark.parametrize("num_heads", [12, 24, 96])
@pytest.mark.parametrize("num_tokens", [1, 7, 128])
@torch.inference_mode()
def test_fused_kda_decode_matches_triton_chain(num_heads: int, num_tokens: int) -> None:
    _requires_kernel()
    inp = KdaDecodeInputs(num_tokens, num_heads, num_slots=max(num_tokens, 4) + 3)

    expected_out, expected_conv, expected_state = _run_triton_chain(inp)
    actual_out, actual_conv, actual_state = _run_fused(inp)

    # The fused kernel keeps the recurrent output in fp32 through the norm,
    # while the Triton chain rounds it to BF16 in between.
    torch.testing.assert_close(actual_out, expected_out, atol=3e-2, rtol=3e-2)
    # The conv state is a pure shift-and-append: it must be bit-exact.
    torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-3, rtol=2e-3)


@torch.inference_mode()
def test_fused_kda_decode_leaves_untouched_slots_alone() -> None:
    """Only the slots named by state_indices may be written."""
    _requires_kernel()
    num_tokens, num_heads, num_slots = 4, 12, 9
    inp = KdaDecodeInputs(num_tokens, num_heads, num_slots, seed=3)

    _, actual_conv, actual_state = _run_fused(inp)

    touched = set(inp.state_indices.tolist())
    untouched = [slot for slot in range(num_slots) if slot not in touched]
    assert untouched, "test needs at least one unused slot"
    torch.testing.assert_close(
        actual_conv[untouched], inp.conv_state[untouched], atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual_state[untouched], inp.recurrent_state[untouched], atol=0, rtol=0
    )


@torch.inference_mode()
def test_fused_kda_decode_without_output_norm() -> None:
    """Omitting the gate/norm pair returns the raw recurrent output."""
    _requires_kernel()
    from vllm import _custom_ops as ops

    inp = KdaDecodeInputs(5, 12, num_slots=8, seed=7)
    _, _, expected_state = _run_triton_chain(inp)

    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    out = torch.empty(1, 5, inp.num_heads, HEAD_DIM, device="cuda", dtype=DTYPE)
    ops.fused_kda_decode(
        x=inp.mixed_qkv,
        weight=inp.decode_conv1d_weight,
        bias=None,
        conv_state=inp.conv_state_view(conv_state),
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        state_indices=inp.state_indices,
        state=recurrent_state,
        out=out,
        lower_bound=GATE_LOWER_BOUND,
    )

    torch.testing.assert_close(recurrent_state, expected_state, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(out).all()


@torch.inference_mode()
def test_fused_kda_decode_skips_null_block_padding() -> None:
    """A CUDA-graph decode batch is padded with NULL_BLOCK_ID (0).

    gdn_attn.py fills the tail of ``non_spec_state_indices_tensor`` with 0, and
    the Triton chain zeroes those rows' output while leaving slot 0 alone. The
    fused kernel must do the same rather than read-modify-write slot 0's state
    once per padded row.
    """
    _requires_kernel()
    num_real, num_padded, num_heads = 3, 5, 12
    inp = KdaDecodeInputs(num_real + num_padded, num_heads, num_slots=12, seed=11)
    inp.state_indices[num_real:] = 0

    expected_out, expected_conv, expected_state = _run_triton_chain(inp)
    actual_out, actual_conv, actual_state = _run_fused(inp)

    torch.testing.assert_close(actual_out, expected_out, atol=3e-2, rtol=3e-2)
    assert not actual_out[0, num_real:].any(), "padded rows must produce zeros"
    # Slot 0 must be untouched by both paths.
    torch.testing.assert_close(actual_conv[0], inp.conv_state[0], atol=0, rtol=0)
    torch.testing.assert_close(actual_state[0], inp.recurrent_state[0], atol=0, rtol=0)
    torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-3, rtol=2e-3)
