# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

# Kimi-K3 KDA: 96 heads x 128, conv width 4, gate_lower_bound -5.0.
HEAD_DIM = 128
CONV_WIDTH = 4
GATE_LOWER_BOUND = -5.0
NORM_EPS = 1e-5
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _on_supported_arch() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950, on_gfx1250

    return on_gfx950() or on_gfx1250()


pytestmark = pytest.mark.skipif(
    not _on_supported_arch(),
    reason="The AITER Gluon KDA decode kernels require gfx950 / gfx1250",
)


def _aiter_kda():
    return pytest.importorskip(
        "aiter.ops.triton.attention.kda",
        reason="the installed AITER has no Gluon KDA decode kernels",
    )


def _gated_rmsnorm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    normed = x_float * torch.rsqrt(variance + eps) * weight.float()
    return (normed * torch.sigmoid(gate.float())).to(x.dtype)


class KdaDecodeInputs:
    """One decode step of a Kimi-K3 KDA layer, in the layer's own layout.

    ``num_spec`` > 0 sizes the conv state for speculative decoding
    (``conv_width - 1 + num_spec`` taps, as ``kda_state_shape`` does).
    """

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        num_slots: int,
        num_spec: int = 0,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        dim = num_heads * HEAD_DIM
        self.num_heads = num_heads
        self.dim = dim

        self.mixed_qkv = (
            torch.randn(num_tokens, 3 * dim, device=DEVICE, dtype=DTYPE) * 0.5
        )
        # conv1d weight as the layer holds it: [3 * dim, width] fp32.
        self.conv_weights = (
            torch.randn(3 * dim, CONV_WIDTH, device=DEVICE, dtype=torch.float32) * 0.3
        )
        # Width-major copy the fused kernels index: [3, width, dim].
        self.decode_conv1d_weight = (
            self.conv_weights.view(3, dim, CONV_WIDTH).transpose(1, 2).contiguous()
        )
        # SD cache layout: [slots, taps, 3 * dim]; the layer transposes it.
        self.conv_state = (
            torch.randn(
                num_slots,
                CONV_WIDTH - 1 + num_spec,
                3 * dim,
                device=DEVICE,
                dtype=DTYPE,
            )
            * 0.5
        )
        self.recurrent_state = (
            torch.randn(
                num_slots,
                num_heads,
                HEAD_DIM,
                HEAD_DIM,
                device=DEVICE,
                dtype=torch.float32,
            )
            * 0.1
        )
        self.g1 = (
            torch.randn(1, num_tokens, num_heads, HEAD_DIM, device=DEVICE, dtype=DTYPE)
            * 0.5
        )
        self.g2 = (
            torch.randn(num_tokens, num_heads, HEAD_DIM, device=DEVICE, dtype=DTYPE)
            * 0.5
        )
        self.beta = torch.randn(1, num_tokens, num_heads, device=DEVICE, dtype=DTYPE)
        self.A_log = torch.randn(num_heads, device=DEVICE, dtype=torch.float32) * 0.5
        self.dt_bias = torch.randn(dim, device=DEVICE, dtype=torch.float32) * 0.1
        # The layer passes o_norm.weight (BF16) straight to the AITER kernels.
        self.norm_weight = 1 + 0.1 * torch.randn(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        # Distinct, shuffled slots: the kernel must honour the indirection.
        # Slot 0 is NULL_BLOCK_ID and never allocated for live state, so draw
        # from [1, num_slots).
        assert num_slots > num_tokens, "need a spare slot to detect stray writes"
        self.state_indices = (
            torch.randperm(num_slots - 1, device=DEVICE)[:num_tokens] + 1
        ).to(torch.int32)

    @staticmethod
    def conv_state_view(state: torch.Tensor) -> torch.Tensor:
        return state.transpose(-1, -2)

    def empty_out(self) -> torch.Tensor:
        return torch.empty(
            1, self.mixed_qkv.shape[0], self.num_heads, HEAD_DIM,
            device=DEVICE, dtype=DTYPE,
        )  # fmt: skip


# --------------------------------------------------------------------------
# non-spec decode: fused_recurrent_kda_packed_decode
# --------------------------------------------------------------------------


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
    out = _gated_rmsnorm(core_attn_out, inp.g2, inp.norm_weight, NORM_EPS)
    return out, conv_state, recurrent_state


def _run_aiter_packed_decode(
    inp: KdaDecodeInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly the call KimiK3DeltaAttention._forward makes for decode batches."""
    aiter_kda = _aiter_kda()
    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    num_tokens = inp.mixed_qkv.shape[0]
    core_attn_out = inp.empty_out()
    aiter_kda.fused_recurrent_kda_packed_decode(
        mixed_qkv=inp.mixed_qkv,
        g=inp.g1[0],
        beta=inp.beta[0],
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=GATE_LOWER_BOUND,
        initial_state=recurrent_state,
        ssm_state_indices=inp.state_indices[:num_tokens],
        conv_state=inp.conv_state_view(conv_state),
        conv_weight=inp.decode_conv1d_weight,
        out_gate=inp.g2[:num_tokens],
        norm_weight=inp.norm_weight,
        norm_eps=NORM_EPS,
        out=core_attn_out[0, :num_tokens],
    )
    return core_attn_out, conv_state, recurrent_state


@pytest.mark.parametrize("num_heads", [12, 24])
@pytest.mark.parametrize("num_tokens", [1, 7, 128])
@torch.inference_mode()
def test_aiter_packed_decode_matches_triton_chain(
    num_heads: int, num_tokens: int
) -> None:
    inp = KdaDecodeInputs(num_tokens, num_heads, num_slots=max(num_tokens, 4) + 3)

    expected_out, expected_conv, expected_state = _run_triton_chain(inp)
    actual_out, actual_conv, actual_state = _run_aiter_packed_decode(inp)

    # The fused kernel keeps the recurrent output in fp32 through the norm,
    # while the Triton chain rounds it to BF16 in between.
    torch.testing.assert_close(actual_out, expected_out, atol=3e-2, rtol=3e-2)
    # The conv state is a pure shift-and-append: it must be bit-exact.
    torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-3, rtol=2e-3)


@torch.inference_mode()
def test_aiter_packed_decode_leaves_untouched_slots_alone() -> None:
    """Only the slots named by the state indices may be written."""
    num_tokens, num_heads, num_slots = 4, 12, 9
    inp = KdaDecodeInputs(num_tokens, num_heads, num_slots, seed=3)

    _, actual_conv, actual_state = _run_aiter_packed_decode(inp)

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
def test_aiter_packed_decode_skips_null_block_padding() -> None:
    """A CUDA-graph decode batch is padded with NULL_BLOCK_ID (0).

    gdn_attn.py fills the tail of ``non_spec_state_indices_tensor`` with 0 and
    the Triton chain leaves slot 0 alone; the fused kernel's pad_slot_guard
    must do the same rather than read-modify-write slot 0 once per padded row.
    """
    num_real, num_padded, num_heads = 3, 5, 12
    inp = KdaDecodeInputs(num_real + num_padded, num_heads, num_slots=12, seed=11)
    inp.state_indices[num_real:] = 0

    expected_out, expected_conv, expected_state = _run_triton_chain(inp)
    actual_out, actual_conv, actual_state = _run_aiter_packed_decode(inp)

    torch.testing.assert_close(
        actual_out[0, :num_real], expected_out[0, :num_real], atol=3e-2, rtol=3e-2
    )
    assert torch.isfinite(actual_out).all()
    # Slot 0 must be untouched by both paths.
    torch.testing.assert_close(actual_conv[0], inp.conv_state[0], atol=0, rtol=0)
    torch.testing.assert_close(actual_state[0], inp.recurrent_state[0], atol=0, rtol=0)
    torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-3, rtol=2e-3)


# --------------------------------------------------------------------------
# spec-only decode: fused_recurrent_kda with conv / gate / norm fused in
# --------------------------------------------------------------------------


class SpecBatch:
    """A spec-only KDA batch: ``num_spec + 1`` query tokens per request at most.

    ``seq_lens`` are the per-request query lengths, ``num_accepted`` the tokens
    verified in the previous step; ``state_indices`` is the [N, num_spec + 1]
    per-token slot table the GDN metadata hands the layer.
    """

    def __init__(
        self,
        seq_lens: list[int],
        num_accepted: list[int],
        num_heads: int,
        num_spec: int,
        seed: int = 0,
    ) -> None:
        assert len(seq_lens) == len(num_accepted)
        assert max(seq_lens) <= num_spec + 1
        n_seqs = len(seq_lens)
        slots_per_seq = num_spec + 1
        num_slots = n_seqs * slots_per_seq + 2
        self.inp = KdaDecodeInputs(
            sum(seq_lens), num_heads, num_slots, num_spec=num_spec, seed=seed
        )
        # Distinct slots per (request, token), shuffled, none at slot 0.
        perm = torch.randperm(num_slots - 1, device=DEVICE)[: n_seqs * slots_per_seq]
        self.state_indices = (perm + 1).to(torch.int32).view(n_seqs, slots_per_seq)
        self.query_start_loc = torch.tensor(
            [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=DEVICE,
        )
        self.num_accepted_tokens = torch.tensor(
            num_accepted, dtype=torch.int32, device=DEVICE
        )
        self.max_query_len = slots_per_seq


def _run_spec_triton_chain(
    sb: SpecBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The layer's unfused spec path: conv update -> recurrent KDA -> norm."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda

    inp = sb.inp
    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    conv_out = causal_conv1d_update(
        inp.mixed_qkv,
        inp.conv_state_view(conv_state),
        inp.conv_weights,
        None,
        activation="silu",
        conv_state_indices=sb.state_indices[:, 0],
        num_accepted_tokens=sb.num_accepted_tokens,
        query_start_loc=sb.query_start_loc,
        max_query_len=sb.max_query_len,
        validate_data=False,
        out=torch.empty_like(inp.mixed_qkv),
    )
    q, k, v = (
        x.view(1, -1, inp.num_heads, HEAD_DIM) for x in conv_out.split(inp.dim, dim=-1)
    )
    core_attn_out, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=GATE_LOWER_BOUND,
        initial_state=recurrent_state,
        cu_seqlens=sb.query_start_loc,
        ssm_state_indices=sb.state_indices,
        num_accepted_tokens=sb.num_accepted_tokens,
    )
    out = _gated_rmsnorm(core_attn_out, inp.g2, inp.norm_weight, NORM_EPS)
    return out, conv_state, recurrent_state


def _run_aiter_spec(
    sb: SpecBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly the call KimiK3DeltaAttention._forward makes for spec-only batches."""
    aiter_kda = _aiter_kda()
    inp = sb.inp
    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    q, k, v = (
        x.view(1, -1, inp.num_heads, HEAD_DIM)
        for x in inp.mixed_qkv.split(inp.dim, dim=-1)
    )
    n_spec = q.shape[1]
    core_attn_out = inp.empty_out()
    aiter_kda.fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=inp.g1,
        beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=GATE_LOWER_BOUND,
        initial_state=recurrent_state,
        cu_seqlens=sb.query_start_loc,
        ssm_state_indices=sb.state_indices,
        num_accepted_tokens=sb.num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        pad_slot_guard=True,
        conv_state=inp.conv_state_view(conv_state),
        conv_weight=inp.decode_conv1d_weight,
        out_gate=inp.g2[:n_spec].unsqueeze(0),
        norm_weight=inp.norm_weight,
        norm_eps=NORM_EPS,
        out=core_attn_out[:, :n_spec],
    )
    return core_attn_out, conv_state, recurrent_state


@pytest.mark.parametrize("num_heads", [12, 24])
@pytest.mark.parametrize(
    "num_spec, seq_lens, num_accepted",
    [
        (3, [4, 4, 4], [1, 2, 4]),  # Kimi-K3 recipe: 3 draft tokens, uniform
        (3, [4, 2, 1, 4], [4, 1, 1, 3]),  # ragged after partial acceptance
        (6, [7, 3], [1, 4]),  # longer window, two requests
    ],
)
@torch.inference_mode()
def test_aiter_spec_decode_matches_triton_chain(
    num_heads: int, num_spec: int, seq_lens: list[int], num_accepted: list[int]
) -> None:
    sb = SpecBatch(seq_lens, num_accepted, num_heads, num_spec, seed=5)

    expected_out, expected_conv, expected_state = _run_spec_triton_chain(sb)
    actual_out, actual_conv, actual_state = _run_aiter_spec(sb)

    torch.testing.assert_close(actual_out, expected_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-3, rtol=3e-2)
    # Nothing outside the slot table may change.
    touched = set(sb.state_indices.flatten().tolist())
    untouched = [s for s in range(sb.inp.conv_state.shape[0]) if s not in touched]
    assert untouched
    torch.testing.assert_close(
        actual_conv[untouched], sb.inp.conv_state[untouched], atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual_state[untouched], sb.inp.recurrent_state[untouched], atol=0, rtol=0
    )
