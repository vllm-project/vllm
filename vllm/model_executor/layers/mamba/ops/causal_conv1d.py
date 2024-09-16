# Copyright (c) 2024, Tri Dao.

from typing import Optional

import torch

from vllm import _custom_ops as ops


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cu_seq_len: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    cu_seq_len: (batch)
        tensor contains cumulative input ids sequence lengths
        for exmaple: cu_seq_len = torch.Tensor([10,16,17]), x.shape=(dim,17)
    cache_indices: (batch) 
        indicates the corresponding state index, 
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch)
        indicates whether should the kernel take the current state as initial 
        state for the calculations
    conv_states: (...,dim,width - 1)
        updated inplace if provided
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    if conv_states is None:
        conv_states = torch.empty(
            x.shape[0],
            x.shape[1],
            weight.shape[1] - 1,
            device=x.device,
            dtype=x.dtype
        )

    out = ops.causal_conv1d_fwd(x, weight, bias, conv_states, cu_seq_len,
                                cache_indices, has_initial_state, activation
                                in ["silu", "swish"])
    return (out, conv_states)


def causal_conv1d_update(x: torch.Tensor,
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
    activation_bool = activation in ["silu", "swish"]
    return ops.causal_conv1d_update(x, conv_state, weight, bias,
                                    activation_bool)
