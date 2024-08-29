from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x,
                       weight.unsqueeze(1),
                       bias,
                       padding=width - 1,
                       groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(x: torch.Tensor,
                             conv_state: torch.Tensor,
                             weight: torch.Tensor,
                             bias: Optional[torch.Tensor] = None,
                             activation: Optional[str] = None):
    """
    x: (batch, dim)
    conv_state: (batch, dim, width)
    weight: (dim, width)
    bias: (dim,)

    out: (batch, dim)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    batch, dim = x.shape
    width = weight.shape[1]
    assert conv_state.shape == (batch, dim, width)
    assert weight.shape == (dim, width)
    conv_state.copy_(torch.roll(conv_state, shifts=-1,
                                dims=-1))  # Update state (B D W)
    conv_state[:, :, -1] = x
    out = torch.sum(conv_state * weight, dim=-1)  # (B D)
    if bias is not None:
        out += bias
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@pytest.mark.parametrize("return_final_states", [False, True])
@pytest.mark.parametrize("has_initial_states", [False, True])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("seqlen", [128, 512, 4096])
@pytest.mark.parametrize('dim', [64, 4096 + 32])
@pytest.mark.parametrize('batch', [1, 2])
def test_causal_conv1d(batch, dim, seqlen, width, has_bias, silu_activation,
                       itype, channel_last, has_initial_states,
                       return_final_states):
    if not channel_last and (has_initial_states or return_final_states):
        pytest.skip(
            "Only channel_last support initial_states or return_final_states")
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    if not channel_last:
        x = torch.randn(batch,
                        4096 + dim + 64,
                        seqlen,
                        device=device,
                        dtype=itype)[:, 4096:4096 + dim, :]
    else:
        x = rearrange(
            torch.randn(batch,
                        seqlen,
                        4096 + dim + 64,
                        device=device,
                        dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s")
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    if has_initial_states:
        initial_states = torch.randn(batch,
                                     width - 1,
                                     dim,
                                     device=device,
                                     dtype=itype).transpose(1, 2)
    else:
        initial_states = None
    x_ref = x.detach().clone()
    weight_ref = weight.detach().clone()
    bias_ref = bias.detach().clone() if bias is not None else None
    initial_states_ref = initial_states.detach().clone(
    ) if initial_states is not None else None
    activation = None if not silu_activation else "silu"
    out, final_states = causal_conv1d_fn(
        x,
        weight,
        bias,
        initial_states=initial_states,
        return_final_states=return_final_states,
        activation=activation)
    out_ref, final_states_ref = causal_conv1d_ref(
        x_ref,
        weight_ref,
        bias_ref,
        initial_states=initial_states_ref,
        return_final_states=return_final_states,
        activation=activation)
    if return_final_states:
        assert final_states is not None and final_states_ref is not None
        assert torch.allclose(final_states,
                              final_states_ref,
                              rtol=rtol,
                              atol=atol)

    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    if return_final_states:
        out += F.sigmoid(final_states).sum(dim=-1, keepdim=True)
        out_ref += F.sigmoid(final_states_ref).sum(dim=-1, keepdim=True)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
@pytest.mark.parametrize("batch", [1, 2])
def test_causal_conv1d_update(batch, dim, width, has_bias, silu_activation,
                              itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch = 2
    x = torch.randn(batch, dim, device=device, dtype=itype)
    conv_state = torch.randn(batch, dim, width, device=device, dtype=itype)
    weight = torch.randn(dim,
                         width,
                         device=device,
                         dtype=itype,
                         requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=itype, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_update(x,
                               conv_state,
                               weight,
                               bias,
                               activation=activation)
    out_ref = causal_conv1d_update_ref(x,
                                       conv_state_ref,
                                       weight,
                                       bias,
                                       activation=activation)

    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [True])
# @pytest.mark.parametrize('silu_activation', [False])
@pytest.mark.parametrize("has_bias", [True])
# @pytest.mark.parametrize('has_bias', [False])
@pytest.mark.parametrize("width", [4])
# @pytest.mark.parametrize('width', [2])
@pytest.mark.parametrize(
    "seqlen", [4096]
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [2048])
@pytest.mark.parametrize('dim', [64 ,4096])
# @pytest.mark.parametrize('dim', [64])
def test_causal_conv1d_varlen(dim, seqlen, width, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(seqlen + dim + width)
    batch = 1
    seqlens = []
    for b in range(batch):
        nsplits = torch.randint(1, 5, (1,)).item()
        eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
        seqlens.append(torch.diff(torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])).tolist())
        assert sum(seqlens[-1]) == seqlen
        assert all(s > 0 for s in seqlens[-1])
    # Only support channel_last
    print(seqlens)
    x = rearrange(
        torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
    ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    seq_idx = torch.stack([torch.cat([torch.full((s,), i, dtype=torch.int32, device=device) for i, s in enumerate(sl)], dim=0)
                           for sl in seqlens], dim=0)
    print(seq_idx)
    print(x.shape)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    activation = None if not silu_activation else "silu"
    out,final_states = causal_conv1d_fn(x, weight, bias, seq_idx=seq_idx, activation=activation,return_final_states=True)
    out_ref = []
    for b in range(batch):
        out_ref_b = []
        for x_s in torch.split(x_ref[[b]], seqlens[b], dim=2):
            print(x_s.shape)
            out_ref_b.append(causal_conv1d_ref(x_s, weight_ref, bias_ref, activation=activation,return_final_states=True))
        out_ref.append(torch.cat(out_ref_b[0], dim=2))
    out_ref = torch.cat(out_ref, dim=0)

    print("out",out.shape,out_ref.shape)
    print("fs",final_states.shape)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    # g = torch.randn_like(out)
    # out_ref.backward(g)
    # out.backward(g)

    # print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    # print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    # if has_bias:
        # print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")

    # assert torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    # assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    # if has_bias:
        # assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)
