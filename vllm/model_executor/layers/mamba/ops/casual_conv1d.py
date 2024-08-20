# Copyright (c) 2024, Tri Dao.

from typing import Optional
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out=None,
    activation: str = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(2) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None
    if seq_idx is not None:
        assert (initial_states is
                None), "initial_states must be None if seq_idx is not None"
        assert (not return_final_states
                ), "If seq_idx is not None, we don't return final_states_out"
    seq_idx = seq_idx.contiguous() if seq_idx is not None else None
    if initial_states is not None and (initial_states.stride(2) != 1
                                       and initial_states.stride(1) != 1):
        initial_states = initial_states.contiguous()
    if return_final_states:
        assert (
            x.stride(1) == 1
        ), "Only channel-last layout support returning final_states_out"
        if final_states_out is not None:
            assert (final_states_out.stride(2) == 1
                    or final_states_out.stride(1) == 1)
        else:
            batch, dim, seqlen = x.shape
            width = weight.shape[1]
            final_states_out = torch.empty(batch,
                                           width - 1,
                                           dim,
                                           device=x.device,
                                           dtype=x.dtype).transpose(1, 2)
    else:
        final_states_out = None

    out = ops.causal_conv1d_fwd(x, weight, bias, seq_idx, initial_states,
                                final_states_out, activation
                                in ["silu", "swish"])
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out=None,
    activation: str = "silu",
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


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
    """
    x: (batch, dim)
    conv_state: (batch, dim, width)
    weight: (dim, width)
    bias: (dim,)

    out: (batch, dim)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation = activation in ["silu", "swish"]
    return ops.causal_conv1d_update(x, conv_state, weight, bias, activation)


def causal_conv1d_update_ref(x,
                             conv_state,
                             weight,
                             bias=None,
                             activation=None):
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
