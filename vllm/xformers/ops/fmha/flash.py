# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union

import torch

from .attn_bias import (
    AttentionBias,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalMask,
    BlockDiagonalMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
)
from .common import (
    AttentionFwOpBase,
    Context,
    Inputs,
    check_lastdim_alignment_stride1,
)

FLASH_VERSION = "0.0.0"
try:
    import flash_attn
    from flash_attn.flash_attn_interface import (_flash_attn_forward, 
                                                 _flash_attn_varlen_forward,
                                                 _flash_attn_backward,
                                                 _flash_attn_varlen_backward)

    FLASH_VERSION = flash_attn.__version__

    print("========== flash attention version: {}".format(FLASH_VERSION))

    # create library so that flash-attn goes through the PyTorch Dispatcher
    _flash_lib = torch.library.Library("xformers_flash", "DEF")

    _flash_lib.define(
        "flash_fwd(Tensor query, Tensor key, Tensor value, "
        "Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, "
        "int max_seqlen_q, int max_seqlen_k, "
        "float p, float softmax_scale, "
        "bool is_causal, int window_size, bool return_softmax) -> (Tensor, Tensor, Tensor)"
    )

    _flash_lib.define(
        "flash_bwd(Tensor dout, Tensor query, Tensor key, Tensor value, "
        "Tensor out, Tensor softmax_lse_, Tensor dq, Tensor dk, Tensor dv, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
        "int max_seqlen_q, int max_seqlen_k, "
        "float p, float softmax_scale, bool is_causal, int window_size, Tensor rng_state) -> (Tensor, Tensor, Tensor)"
    )

    def _flash_fwd(
        query,
        key,
        value,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_seq_len_q,
        max_seq_len_k,
        p,
        softmax_scale,
        is_causal,
        window_size,
        return_softmax,
    ):
        # out = query.new_empty(query.shape[0], query.shape[1], value.shape[2])
        # out, softmax_lse, S_dmask = _flash_attn_forward(
        #     query,
        #     key,
        #     value,
        #     out,
        #     cu_seq_lens_q,
        #     cu_seq_lens_k,
        #     max_seq_len_q,
        #     max_seq_len_k,
        #     p,
        #     softmax_scale,
        #     is_causal,
        #     return_softmax,
        #     0,
        #     None
        # )
        # return out, softmax_lse, None
        
        if cu_seq_lens_q is None:
            assert cu_seq_lens_k is None
            (
                out,
                q_padded,
                k_padded,
                v_padded,
                out_padded,
                softmax_lse,
                S_dmask,
                rng_state,
            ) = _flash_attn_forward(
                query,
                key,
                value,
                p,
                softmax_scale,
                is_causal,
                return_softmax
            )
        else:
            out = query.new_empty(query.shape[0], query.shape[1], value.shape[2])
            (
                out,
                q_padded,
                k_padded,
                v_padded,
                out_padded,
                softmax_lse,
                S_dmask,
                rng_state,
            ) = _flash_attn_varlen_forward(
                query,
                key,
                value,
                cu_seq_lens_q,
                cu_seq_lens_k,
                max_seq_len_q,
                max_seq_len_k,
                p,
                softmax_scale,
                is_causal,
                return_softmax,
            )
        return out, softmax_lse, rng_state

    def _flash_bwd(
        grad,
        query,
        key,
        value,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_seq_len_q,
        max_seq_len_k,
        p,
        softmax_scale,
        is_causal,
        window_size,
        rng_state,
    ):
        # _flash_attn_backward(
        #     grad,
        #     query,
        #     key,
        #     value,
        #     out,
        #     lse,
        #     dq,
        #     dk,
        #     dv,
        #     cu_seq_lens_q,
        #     cu_seq_lens_k,
        #     max_seq_len_q,
        #     max_seq_len_k,
        #     p,
        #     softmax_scale,
        #     is_causal,
        #     0,
        #     None
        # )
        # return dq, dk, dv

        if cu_seq_lens_k is None:
            assert cu_seq_lens_q is None
            (
                dq, dk, dv, softmax_d
            ) = _flash_attn_backward(
                grad,
                query,
                key,
                value,
                out,
                lse,
                dq,
                dk,
                dv,
                p,
                softmax_scale,
                is_causal,
                rng_state,
            )
        else:
            (
                dq, dk, dv, softmax_d
            ) = _flash_attn_varlen_backward(
                grad,
                query,
                key,
                value,
                out,
                lse,
                dq,
                dk,
                dv,
                cu_seq_lens_q,
                cu_seq_lens_k,
                max_seq_len_q,
                max_seq_len_k,
                p,
                softmax_scale,
                is_causal,
                rng_state,
            )
        return dq, dk, dv

    _flash_lib.impl("flash_fwd", _flash_fwd, "CUDA")
    _flash_lib.impl("flash_bwd", _flash_bwd, "CUDA")
except ImportError:
    pass


def _convert_input_format(
    inp: Inputs,
) -> Tuple[Inputs, Optional[torch.Tensor], int, Optional[torch.Tensor], int]:
    assert inp.query.ndim in [4, 5]
    query, key, value = inp.query, inp.key, inp.value
    batch = query.shape[0]
    seqlen_q = query.shape[1]
    seqlen_kv = key.shape[1]
    head_dim_q = query.shape[-1]
    head_dim_v = value.shape[-1]

    attn_bias = inp.attn_bias
    if isinstance(attn_bias, BlockDiagonalMask):
        attn_bias.k_seqinfo.seqstart = attn_bias.k_seqinfo.seqstart.to(
            inp.query.device, non_blocking=True
        )
        attn_bias.q_seqinfo.seqstart = attn_bias.q_seqinfo.seqstart.to(
            inp.query.device, non_blocking=True
        )

        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
    else:
        cu_seqlen_k = None
        cu_seqlen_q = None
        max_seqlen_q = inp.query.shape[1]
        max_seqlen_k = inp.key.shape[1]

    if query.ndim == 5:  # QGA
        # Fold the group/head_in_group dimensions together
        def fold(x):
            # Either the head is replicated
            if x.stride(3) == 0:
                return x[:, :, :, 0]
            # Or we reshape
            return x.reshape(
                [
                    x.shape[0],
                    x.shape[1],
                    -1,
                    x.shape[4],
                ]
            )

        query = fold(query)
        key = fold(key)
        value = fold(value)
    # Optimize for MHA
    if key.ndim == 4 and key.stride(2) == 0 and value.stride(2) == 0:
        key = key[:, :, :1]
        value = value[:, :, :1]
    # Initially we have `query.shape = [batch, seqlen, head_dim_q]`
    # We want format `[batch * seqlen, num_heads, head_dim_q]`
    if cu_seqlen_k is not None:
        query = query.reshape([batch * seqlen_q, -1, head_dim_q])
        key = key.reshape([batch * seqlen_kv, -1, head_dim_q])
        value = value.reshape([batch * seqlen_kv, -1, head_dim_v])
    #if query.is_contiguous() or key.is_contiguous() or value.is_contiguous():
    #    query = query.contiguous()
    #    key = key.contiguous()
    #    value = value.contiguous()
    new_inp = replace(
        inp,
        query=query,
        key=key,
        value=value,
    )
    return new_inp, cu_seqlen_q, max_seqlen_q, cu_seqlen_k, max_seqlen_k


def _is_causal(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> bool:
    return isinstance(
        attn_bias,
        (
            LowerTriangularMask,
            BlockDiagonalCausalMask,
            BlockDiagonalCausalLocalAttentionMask,
            BlockDiagonalCausalFromBottomRightMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        ),
    )


def _window_size(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> int:
    if isinstance(
        attn_bias,
        (BlockDiagonalCausalLocalAttentionMask,),
    ):
        return attn_bias._window_size or 0
    if isinstance(attn_bias, BlockDiagonalCausalLocalAttentionFromBottomRightMask):
        return attn_bias._window_size
    return 0


def _check_needs_no_topleft(d: Inputs, reasons: List[str]) -> None:
    # Flash does not support TopLeft, so only allow causal masks with TopLeft
    # if each batch element has equal number of queries and keys.
    if isinstance(d.attn_bias, BlockDiagonalCausalMask):
        # Flash does not support TopLeft, so only allow BlockDiagonalCausalMask
        # if each batch element has equal number of queries and keys.
        for k_start, q_start in zip_longest(
            d.attn_bias.k_seqinfo.seqstart_py, d.attn_bias.q_seqinfo.seqstart_py
        ):
            if k_start != q_start:
                reasons.append(
                    "Only support BlockDiagonalCausalMask if equal"
                    " numbers of keys and queries"
                )
                break
    elif isinstance(d.attn_bias, LowerTriangularMask):
        if d.query.shape[1] != d.key.shape[1]:
            reasons.append(
                "Only support LowerTriangularMask if equal number of" "keys and queries"
            )


def _check_strides_for_bmghk(x: torch.Tensor, name: str, reasons: List[str]) -> None:
    """
    We want to be able to collapse the G/H dimensions together
    """
    if x.ndim == 5:
        stride_g, stride_h = x.stride(2), x.stride(3)
        if x.shape[2] == 1:
            return
        if x.shape[3] == 1 or stride_h == 0:
            return
        if stride_g != stride_h * x.shape[-2]:
            reasons.append(
                f"GQA is only supported when the G/H dimensions are contiguous\n"
                f"    {name}.stride:  {x.stride()}\n"
                f"    {name}.shape :  {list(x.shape)}"
            )


class FwOp(AttentionFwOpBase):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention <https://github.com/HazyResearch/flash-attention>`_ \
        implementation.
    """

    OPERATOR = _flash_fwd
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    #CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (7, 5)
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 256
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        LowerTriangularMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        BlockDiagonalCausalFromBottomRightMask,
        LowerTriangularMaskWithTensorBias,
    }
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    NAME = f"flshattF@{FLASH_VERSION}"
    VERSION = FLASH_VERSION

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        _check_needs_no_topleft(d, reasons)
        _check_strides_for_bmghk(d.query, "query", reasons)
        _check_strides_for_bmghk(d.key, "key", reasons)
        _check_strides_for_bmghk(d.value, "value", reasons)
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        return_softmax = False
        out_shape = [
            *inp.query.shape[:-1],
            inp.value.shape[-1],
        ]
        # no cumulative seqlen
        (
            inp,
            cu_seqlens_q,
            max_seqlen_q,
            cu_seqlens_k,
            max_seqlen_k,
        ) = _convert_input_format(inp)

        softmax_scale = inp.query.shape[-1] ** (-0.5) if inp.scale is None else inp.scale

        ret = cls.OPERATOR(
            inp.query,
            inp.key,
            inp.value,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            inp.p,
            softmax_scale,
            _is_causal(inp.attn_bias),
            _window_size(inp.attn_bias),
            return_softmax,
        )

        out = ret[0].reshape(out_shape)
        ctx = Context(out=out, lse=ret[1])
        return (out, ctx)

    @classmethod
    # type: ignore
    def operator_flop(
        cls,
        query,
        key,
        value,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_seq_len_q,
        max_seq_len_k,
        p,
        softmax_scale,
        causal,
        return_softmax,
    ) -> int:
        return cls.attn_operator_flop(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            causal=causal,
            seqstart_k=cu_seq_lens_k,
            seqstart_q=cu_seq_lens_q,
        )
