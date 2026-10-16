# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm fused KDA decode kernels must match the Triton chain they replace.

Two fused paths fold the packed causal conv1d update, the gated delta-rule
recurrence and the gated output RMSNorm into one launch, updating the conv
state and the recurrent state in place:

* the vLLM HIP kernel ``ops.fused_kda_decode`` (gfx942 / gfx950), and
* AITER's Gluon kernels behind ``rocm_aiter_ops`` (CDNA4+): the packed
  non-spec decode and the speculative-decode ``fused_recurrent_kda``.

Every output of each is compared against the unfused path the AMD layer falls
back to, with the argument layout the layer itself uses.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _hip_kernel_arch() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    return on_gfx950() or on_gfx942()


def _aiter_gluon_kda_available() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() >= 4 and rocm_aiter_ops._kda_gluon_decode_importable()


pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only KDA decode kernels"
)
requires_hip_kernel_arch = pytest.mark.skipif(
    not _hip_kernel_arch(),
    reason="The fused KDA decode kernel is only built for gfx942 / gfx950",
)
requires_aiter_gluon_kda = pytest.mark.skipif(
    not _aiter_gluon_kda_available(),
    reason="AITER Gluon KDA decode needs CDNA4+ and an AITER build that ships it",
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
    """One decode step of a Kimi-K3 KDA layer, in the layer's own layout.

    ``num_spec`` > 0 sizes the conv state for speculative decoding
    (``conv_width - 1 + num_spec`` taps, as ``kda_state_shape`` does).
    """

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        num_slots: int,
        seed: int = 0,
        num_spec: int = 0,
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
        # SD cache layout: [slots, taps, 3 * dim]; the layer transposes it.
        self.conv_state = (
            torch.randn(
                num_slots,
                CONV_WIDTH - 1 + num_spec,
                3 * dim,
                device=device,
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

    def empty_out(self) -> torch.Tensor:
        num_tokens = self.mixed_qkv.shape[0]
        return torch.empty(
            1, num_tokens, self.num_heads, HEAD_DIM, device="cuda", dtype=DTYPE
        )


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


@requires_hip_kernel_arch
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


@requires_hip_kernel_arch
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


@requires_hip_kernel_arch
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


@requires_hip_kernel_arch
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


@requires_hip_kernel_arch
def test_decode_weights_staged_after_loading(dist_init) -> None:
    """The fused-decode weight copies must not depend on the weight loaders.

    Loaders that bypass ``weight_loader`` (e.g. ``--load-format dummy``) left
    them uninitialized, so decode could read NaN. Loading runs with grad mode
    on, so the copies must also stay out of autograd.
    """
    _requires_kernel()
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    config = KimiLinearConfig(
        hidden_size=256,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": HEAD_DIM,
            "num_heads": 12,
            "short_conv_kernel_size": CONV_WIDTH,
            "use_full_rank_gate": True,
            "gate_lower_bound": GATE_LOWER_BOUND,
        },
    )
    vllm_config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B", dtype=DTYPE)
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        layer = KimiK3DeltaAttention(
            config, vllm_config, prefix="model.layers.0.self_attn"
        )
    assert layer.decode_conv1d_weight is not None
    # The AITER Gluon decode path takes o_norm.weight directly, so only the HIP
    # kernel gets an fp32 norm copy.
    assert (layer.decode_norm_weight is None) == layer.use_aiter_decode
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_()
        layer.decode_conv1d_weight.fill_(float("nan"))
        if layer.decode_norm_weight is not None:
            layer.decode_norm_weight.fill_(float("nan"))

    process_weights_after_loading(
        torch.nn.ModuleList([layer]), vllm_config.model_config, torch.device("cuda")
    )

    dim = layer.local_projection_size
    conv_weights = layer.conv1d.weight.squeeze(1)
    expected_conv = torch.stack(
        [conv_weights[i * dim : (i + 1) * dim].transpose(0, 1) for i in range(3)]
    )
    torch.testing.assert_close(
        layer.decode_conv1d_weight, expected_conv, atol=0, rtol=0
    )
    if layer.decode_norm_weight is not None:
        torch.testing.assert_close(
            layer.decode_norm_weight, layer.o_norm.weight.float(), atol=0, rtol=0
        )
        assert not layer.decode_norm_weight.requires_grad
    assert not layer.decode_conv1d_weight.requires_grad


# --------------------------------------------------------------------------
# non-spec decode: fused_recurrent_kda_packed_decode
# --------------------------------------------------------------------------


def _run_aiter_packed_decode(
    inp: KdaDecodeInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly the call KimiK3DeltaAttention._forward makes for decode batches."""
    from vllm._aiter_ops import rocm_aiter_ops

    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    num_tokens = inp.mixed_qkv.shape[0]
    core_attn_out = inp.empty_out()
    rocm_aiter_ops.kda_fused_recurrent_packed_decode(
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
        norm_weight=inp.norm_weight_bf16,
        norm_eps=NORM_EPS,
        out=core_attn_out[0, :num_tokens],
    )
    return core_attn_out, conv_state, recurrent_state


@requires_aiter_gluon_kda
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


@requires_aiter_gluon_kda
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


@requires_aiter_gluon_kda
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
            sum(seq_lens), num_heads, num_slots, seed=seed, num_spec=num_spec
        )
        # Distinct slots per (request, token), shuffled, none at slot 0.
        perm = torch.randperm(num_slots - 1, device="cuda")[: n_seqs * slots_per_seq]
        self.state_indices = (perm + 1).to(torch.int32).view(n_seqs, slots_per_seq)
        self.query_start_loc = torch.tensor(
            [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device="cuda",
        )
        self.num_accepted_tokens = torch.tensor(
            num_accepted, dtype=torch.int32, device="cuda"
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
    out = _gated_rmsnorm(core_attn_out, inp.g2, inp.norm_weight_bf16, NORM_EPS)
    return out, conv_state, recurrent_state


def _run_aiter_spec(
    sb: SpecBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly the call KimiK3DeltaAttention._forward makes for spec-only batches."""
    from vllm._aiter_ops import rocm_aiter_ops

    inp = sb.inp
    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    q, k, v = (
        x.view(1, -1, inp.num_heads, HEAD_DIM)
        for x in inp.mixed_qkv.split(inp.dim, dim=-1)
    )
    n_spec = q.shape[1]
    core_attn_out = inp.empty_out()
    rocm_aiter_ops.kda_fused_recurrent_spec_decode(
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
        conv_state=inp.conv_state_view(conv_state),
        conv_weight=inp.decode_conv1d_weight,
        out_gate=inp.g2[:n_spec].unsqueeze(0),
        norm_weight=inp.norm_weight_bf16,
        norm_eps=NORM_EPS,
        out=core_attn_out[:, :n_spec],
    )
    return core_attn_out, conv_state, recurrent_state


@requires_aiter_gluon_kda
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
