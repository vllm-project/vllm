# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Flash Attention wrapper with attention score extraction support.

This module provides a wrapper around flash_attn_varlen_func that supports
returning attention scores. It uses the softmax_lse (logsumexp) values
returned by Flash Attention to reconstruct the attention scores without
requiring kernel modifications.

Usage:
    from vllm.model_executor.layers.attention.flash_attn_with_score import flash_attn_varlen_func_with_score
    
    output, attention_score = flash_attn_varlen_func_with_score(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        return_attention_score=True,
    )
"""

from typing import List, Optional, Tuple, Union
import torch
from .compute_flash_attn_score_triton import compute_varlen_importance


def flash_attn_varlen_func_with_score(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[List[int]] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_softmax_lse: bool = False,
    return_attention_score: bool = False,
    out: Optional[torch.Tensor] = None,
    # FA3 Only parameters
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    fa_version: Optional[int] = None,
    s_aux=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Flash Attention with optional attention score extraction.
    
    This function wraps flash_attn_varlen_func and adds support for returning
    attention scores by using the logsumexp values to reconstruct them.
    
    Args:
        q: Query tensor, shape [total_q, nheads, head_dim]
        k: Key tensor, shape [total_k, nheads_k, head_dim]
        v: Value tensor, shape [total_k, nheads_k, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for keys, shape [batch_size + 1]
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        dropout_p: Dropout probability
        softmax_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
        causal: Whether to use causal masking
        window_size: Sliding window size [left, right]
        softcap: Softcap value for attention (0.0 means disabled)
        alibi_slopes: ALiBi slopes tensor
        deterministic: Whether to use deterministic backward pass
        return_softmax_lse: Whether to return logsumexp values
        return_attention_score: Whether to return attention scores (softmax probs)
        out: Optional pre-allocated output tensor
        scheduler_metadata: FA3 scheduler metadata
        q_descale, k_descale, v_descale: FA3 descale tensors
        num_splits: FA3 number of splits
        fa_version: Flash Attention version (2 or 3)
        s_aux: FA3 auxiliary tensor
        
    Returns:
        If return_attention_score=False and return_softmax_lse=False:
            output: [total_q, nheads, head_dim]
        If return_attention_score=True:
            (output, attention_score): 
                output: [total_q, nheads, head_dim]
                attention_score: [total_q, nheads, max_seqlen_k]
        If return_softmax_lse=True:
            (output, softmax_lse): 
                output: [total_q, nheads, head_dim]
                softmax_lse: [nheads, total_q]
        If both True:
            (output, attention_score, softmax_lse)
            
    Note:
        When return_attention_score=True, this function needs to:
        1. Store q and k tensors temporarily
        2. Request softmax_lse from Flash Attention
        3. Reconstruct attention scores from lse
        
        This may increase memory usage and computation time.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # If we need attention scores, we must get softmax_lse
    need_lse = return_softmax_lse or return_attention_score

    # Prefer FA3 on Hopper (sm90): FA2 wheels ship sm80 PTX (e.g. PTX ISA 8.8) that
    # older drivers cannot JIT for sm90, causing cudaErrorUnsupportedPtxVersion.
    # FA3 ships native sm90a ELF for Hopper.
    resolved_fa_version = fa_version
    if resolved_fa_version is None:
        try:
            from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

            resolved_fa_version = get_flash_attn_version(
                requires_alibi=alibi_slopes is not None
            )
        except Exception:
            resolved_fa_version = None
        if resolved_fa_version is None:
            resolved_fa_version = 2

    resolved_scheduler_metadata = scheduler_metadata
    if resolved_fa_version == 3 and resolved_scheduler_metadata is None:
        win = window_size if window_size is not None else (-1, -1)
        batch_size = int(cu_seqlens_q.shape[0] - 1)
        cache_seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(dtype=torch.int32)
        resolved_scheduler_metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=q.shape[1],
            num_heads_kv=k.shape[1],
            headdim=q.shape[-1],
            cache_seqlens=cache_seqlens,
            qkv_dtype=q.dtype,
            cu_seqlens_q=cu_seqlens_q,
            causal=causal,
            window_size=(win[0], win[1]),
            has_softcap=softcap > 0.0,
            num_splits=num_splits,
        )

    # Build kwargs for flash_attn_varlen_func
    flash_kwargs = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "window_size": window_size,
        "softcap": softcap,
        "alibi_slopes": alibi_slopes,
        "deterministic": deterministic,
        "return_softmax_lse": need_lse,
        "out": out,
        "fa_version": resolved_fa_version,
    }
    if resolved_scheduler_metadata is not None:
        flash_kwargs["scheduler_metadata"] = resolved_scheduler_metadata
    if q_descale is not None:
        flash_kwargs["q_descale"] = q_descale
    if k_descale is not None:
        flash_kwargs["k_descale"] = k_descale
    if v_descale is not None:
        flash_kwargs["v_descale"] = v_descale
    if num_splits != 0:
        flash_kwargs["num_splits"] = num_splits
    if s_aux is not None:
        flash_kwargs["s_aux"] = s_aux
    
    # Run flash attention
    result = flash_attn_varlen_func(q, k, v, **flash_kwargs)
    
    # Parse result
    if need_lse:
        output, softmax_lse = result
    else:
        output = result
        softmax_lse = None
    
    # Compute attention scores if requested
    attention_score = None
    if return_attention_score:
        attention_score = compute_varlen_importance(q, k, cu_seqlens_q, max_seqlen_q, softmax_lse, softmax_scale)
    
    # Build return value
    if return_attention_score and return_softmax_lse:
        return output, attention_score, softmax_lse
    elif return_attention_score:
        return output, attention_score
    elif return_softmax_lse:
        return output, softmax_lse
    else:
        return output

