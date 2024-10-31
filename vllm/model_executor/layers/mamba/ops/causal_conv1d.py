# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py

from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.utils import PAD_SLOT_ID


def causal_conv1d_fn(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: Optional[torch.Tensor] = None,
                     query_start_loc: Optional[torch.Tensor] = None,
                     cache_indices: Optional[torch.Tensor] = None,
                     has_initial_state: Optional[torch.Tensor] = None,
                     conv_states: Optional[torch.Tensor] = None,
                     activation: Optional[str] = "silu",
                     pad_slot_id: int = PAD_SLOT_ID):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]), 
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index, 
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial 
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded 
            entries that will not be processed, 
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id] 
            in this case, the kernel will not process entries at 
            indices 0 and 3


    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    ops.causal_conv1d_fwd(x, weight, bias, conv_states, query_start_loc,
                          cache_indices, has_initial_state, activation
                          in ["silu", "swish"], pad_slot_id)
    return x


def causal_conv1d_update(x: torch.Tensor,
                         conv_state: torch.Tensor,
                         weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None,
                         activation: Optional[str] = None,
                         cache_seqlens: Optional[torch.Tensor] = None,
                         conv_state_indices: Optional[torch.Tensor] = None,
                         pad_slot_id: int = PAD_SLOT_ID):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state 
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim, 
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded 
            entries that will not be processed, 
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id] 
            in this case, the kernel will not process entries at 
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    ops.causal_conv1d_update(x, conv_state, weight, bias, activation_val,
                             cache_seqlens, conv_state_indices, pad_slot_id)
    if unsqueeze:
        x = x.squeeze(-1)
    return x
