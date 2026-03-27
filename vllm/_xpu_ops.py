# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

try:
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
    _HAS_XPU_KERNELS = True
except ImportError:
    _HAS_XPU_KERNELS = False

    def _flash_attn_varlen_sdpa_fallback(
        *,
        out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        block_table: torch.Tensor | None = None,
        s_aux: torch.Tensor | None = None,
        window_size: tuple[int, int] = (-1, -1),
        return_softmax_lse: bool | None = False,
        **kwargs,
    ):
        """Pure-PyTorch fallback for flash_attn_varlen_func.

        Uses torch.nn.functional.scaled_dot_product_attention per-sequence.
        Slow but works on any device (including XPU simulator).
        """
        import torch.nn.functional as F

        num_heads_q = q.shape[1]
        num_heads_k = k.shape[1]
        head_dim = q.shape[2]

        if softmax_scale is None:
            softmax_scale = head_dim ** -0.5

        batch_size = cu_seqlens_q.shape[0] - 1
        lse_list = [] if return_softmax_lse else None

        for i in range(batch_size):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            seq_len_q = q_end - q_start

            if cu_seqlens_k is not None:
                k_start = cu_seqlens_k[i].item()
                k_end = cu_seqlens_k[i + 1].item()
            else:
                k_start = 0
                k_end = max_seqlen_k

            seq_len_k = k_end - k_start

            # q: [total_q, num_heads, head_dim] → [1, num_heads, seq_len_q, head_dim]
            q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)

            if block_table is not None:
                # Paged KV cache: gather from pages
                # k, v are cache tensors: [num_blocks, block_size, num_heads, head_dim]
                page_table_i = block_table[i]
                block_size = k.shape[1]
                num_tokens = seq_len_k
                # Flatten the page table into token indices
                k_pages = []
                v_pages = []
                tokens_left = num_tokens
                for page_idx in page_table_i:
                    if tokens_left <= 0:
                        break
                    page = page_idx.item()
                    take = min(block_size, tokens_left)
                    k_pages.append(k[page, :take])  # [take, num_heads_k, head_dim]
                    v_pages.append(v[page, :take])
                    tokens_left -= take
                k_i = torch.cat(k_pages, dim=0).transpose(0, 1).unsqueeze(0)
                v_i = torch.cat(v_pages, dim=0).transpose(0, 1).unsqueeze(0)
            else:
                k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
                v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)

            # GQA: repeat KV heads to match Q heads
            if num_heads_q != num_heads_k:
                repeat = num_heads_q // num_heads_k
                k_i = k_i.repeat_interleave(repeat, dim=1)
                v_i = v_i.repeat_interleave(repeat, dim=1)

            # Use SDPA (works on CPU, CUDA, XPU)
            attn_out = F.scaled_dot_product_attention(
                q_i.to(torch.float32),
                k_i.to(torch.float32),
                v_i.to(torch.float32),
                is_causal=causal and seq_len_q > 1,
                scale=softmax_scale,
            )

            # attn_out: [1, num_heads, seq_len_q, head_dim] → [seq_len_q, num_heads, head_dim]
            attn_out = attn_out.squeeze(0).transpose(0, 1).to(out.dtype)
            out[q_start:q_end].copy_(attn_out)

            if return_softmax_lse and lse_list is not None:
                # Compute log-sum-exp for compatibility (approximate)
                lse_i = torch.zeros(num_heads_q, seq_len_q, device=q.device, dtype=torch.float32)
                lse_list.append(lse_i)

        if return_softmax_lse:
            return out, torch.cat(lse_list, dim=-1) if lse_list else None
        return out

    flash_attn_varlen_func = _flash_attn_varlen_sdpa_fallback

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake

if hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):

    @register_fake("_xpu_C::fp8_gemm_w8a16")
    def _fp8_gemm_w8a16_fake(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=input.dtype, device=input.device)


if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):

    @register_fake("_xpu_C::int4_gemm_w4a16")
    def _int4_gemm_w4a16_fake(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        bias: torch.Tensor | None,
        weight_scale: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        group_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=input.dtype, device=input.device)


class xpu_ops:
    @staticmethod
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        out: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
        window_size: list[int] | None = None,
        softcap: float | None = 0.0,
        seqused_k: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        # passed in qwen vl
        dropout_p: float = 0.0,
        # The following parameters are not used in xpu kernel currently,
        # we keep API compatible to CUDA's.
        scheduler_metadata=None,
        fa_version: int = 2,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits=0,
        return_softmax_lse: bool | None = False,
        s_aux: torch.Tensor | None = None,
    ):
        assert cu_seqlens_k is not None or seqused_k is not None, (
            "cu_seqlens_k or seqused_k must be provided"
        )
        assert cu_seqlens_k is None or seqused_k is None, (
            "cu_seqlens_k and seqused_k cannot be provided at the same time"
        )
        assert block_table is None or seqused_k is not None, (
            "when enable block_table, seqused_k is needed"
        )
        assert block_table is not None or cu_seqlens_k is not None, (
            "when block_table is disabled, cu_seqlens_k is needed"
        )
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        real_window_size: tuple[int, int]
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])  # noqa: F841

        # In encode attention, k and v maybe not contiguous and current
        # kernel can't handle it
        if block_table is None:
            k = k.contiguous()
            v = v.contiguous()
        return flash_attn_varlen_func(
            out=out,
            q=q.contiguous(),
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            s_aux=s_aux,
            window_size=real_window_size,
            # alibi_slopes = alibi_slopes,
            # softcap=softcap,
            return_softmax_lse=return_softmax_lse,
        )

    @staticmethod
    def get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens: torch.Tensor,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k_new: torch.Tensor | None = None,
        cache_leftpad: torch.Tensor | None = None,
        page_size: int | None = None,
        max_seqlen_k_new=0,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        has_softcap=False,
        num_splits=0,  # Can be tuned for speed
        pack_gqa=None,  # Can be tuned for speed
        sm_margin=0,  # Can be tuned if some SMs are used for communication
    ) -> None:
        logger.warning_once(
            "get_scheduler_metadata is not implemented for xpu_ops, returning None."
        )
        return None
