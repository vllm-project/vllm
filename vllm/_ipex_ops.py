# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.debug("Import error msg: %s", e.msg)


class ipex_ops:

    @staticmethod
    def _reshape_activation_tensor(
            x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num = x.size(0)
        d = x.size(1) // 2
        x = x.reshape(num, 2, d)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = x1.reshape(num, d)
        x2 = x2.reshape(num, d)
        return x1, x2

    @staticmethod
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.silu_and_mul(x, out)

    @staticmethod
    def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_and_mul(x, out)

    @staticmethod
    def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_and_mul(x, out)

    @staticmethod
    def gelu_fast(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    @staticmethod
    def gelu_new(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    @staticmethod
    def gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_quick(x, out)

    @staticmethod
    def paged_attention_v1(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        assert kv_cache_dtype == "auto"
        num_heads = out.size(1)
        num_queries_per_tokens = num_heads // num_kv_heads
        ipex.llm.modules.PagedAttention.single_query_kv_attention(
            out,
            query.contiguous(),
            key_cache.view_as(value_cache),
            value_cache,
            num_queries_per_tokens,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )

    @staticmethod
    def paged_attention_v2(
        out: torch.Tensor,
        exp_sum: torch.Tensor,
        max_logits: torch.Tensor,
        tmp_out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        assert kv_cache_dtype == "auto"
        num_heads = out.size(1)
        num_queries_per_tokens = num_heads // num_kv_heads
        ipex.llm.modules.PagedAttention.single_query_kv_attention(
            out,
            query.contiguous(),
            key_cache.view_as(value_cache),
            value_cache,
            num_queries_per_tokens,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )

    @staticmethod
    def rotary_embedding(
        positions: torch.Tensor,  # [batch_size, seq_len]
        query: torch.Tensor,  # [batch_size, seq_len, num_heads*head_size]
        key: torch.Tensor,  # [batch_size, seq_len, num_kv_heads*head_size]
        head_size: int,
        cos_sin_cache: torch.Tensor,  # [cos_sin_dim, rot_dim]
        is_neox: bool,
    ) -> None:
        rot_dim = cos_sin_cache.size(1)
        ipex.llm.functional.rotary_embedding_batched(positions, query, key,
                                                     head_size, cos_sin_cache,
                                                     is_neox, rot_dim)

    @staticmethod
    def rms_norm(input: torch.Tensor, weight: torch.Tensor,
                 epsilon: float) -> torch.Tensor:
        return ipex.llm.functional.rms_norm(input, weight, epsilon)

    @staticmethod
    def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                           weight: torch.Tensor, epsilon: float) -> None:
        tmp = ipex.llm.functional.add_rms_norm(residual, input, weight, None,
                                               epsilon, True)
        input.copy_(tmp)

    @staticmethod
    def varlen_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        seqlen_q: torch.Tensor,
        seqlen_k: torch.Tensor,
        alibi_slopes: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        pdropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        is_causal: bool,
        return_softmax: bool,
        gen_: torch.Generator,
        window_size_left: float,
        window_size_right: float,
        logits_soft_cap: float,
    ) -> None:
        if ipex.__version__.endswith("cpu"):
            if logits_soft_cap != 0.0:
                raise ValueError("IPEX CPU does not support logits_soft_cap")
            assert alibi_slopes is None
            assert window_size_left < 0 and window_size_right < 0
            ipex.llm.functional.varlen_attention(query.contiguous(),
                                                 key.contiguous(),
                                                 value.contiguous(), out,
                                                 seqlen_q.int(),
                                                 seqlen_k.int(), max_seqlen_q,
                                                 max_seqlen_k, pdropout,
                                                 softmax_scale, zero_tensors,
                                                 is_causal, return_softmax,
                                                 gen_)
        else:  # XPU build
            ipex.llm.functional.varlen_attention(
                query.contiguous(), key.contiguous(), value.contiguous(), out,
                seqlen_q.int(), seqlen_k.int(), alibi_slopes, max_seqlen_q,
                max_seqlen_k, pdropout, softmax_scale, zero_tensors, is_causal,
                return_softmax, gen_, window_size_left, window_size_right,
                logits_soft_cap)

    @staticmethod
    def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        assert kv_cache_dtype == "auto"
        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping)

    @staticmethod
    def reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        k_scale_float: float = 1.0,
        v_scale_float: float = 1.0,
    ) -> None:
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype,
            k_scale_float, v_scale_float)

    @staticmethod
    def flash_attn_varlen_func(
        out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,  # we don't support this in ipex kernel
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool,
        block_table: torch.Tensor,
        alibi_slopes: Optional[torch.Tensor],
        window_size: Optional[list[int]] = None,
        softcap: Optional[float] = 0.0,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        # The following parameters are not used in ipex kernel currently,
        # we keep API compatible to CUDA's.
        scheduler_metadata=None,
        fa_version: int = 2,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits=0,
        s_aux: Optional[torch.Tensor] = None,
    ):
        if cu_seqlens_k is None:
            # cu_seqlens_k is not used in ipex kernel.
            cu_seqlens_k = torch.cumsum(seqused_k, dim=0)
            cu_seqlens_k = torch.cat([
                torch.tensor([0], device=seqused_k.device, dtype=torch.int32),
                cu_seqlens_k
            ]).to(torch.int32)

        real_window_size: tuple[int, int]
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        return ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
            out,
            q.contiguous(),
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
            block_table,
            alibi_slopes,
            softcap=softcap,
            window_size_left=real_window_size[0],
            window_size_right=real_window_size[1],
            k_scale=1.0,
            v_scale=1.0,
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
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_k_new: Optional[torch.Tensor] = None,
            cache_leftpad: Optional[torch.Tensor] = None,
            page_size: Optional[int] = None,
            max_seqlen_k_new=0,
            causal=False,
            window_size=(-1, -1),  # -1 means infinite context window
            has_softcap=False,
            num_splits=0,  # Can be tuned for speed
            pack_gqa=None,  # Can be tuned for speed
            sm_margin=0,  # Can be tuned if some SMs are used for communication
    ) -> None:
        logger.warning_once(
            "get_scheduler_metadata is not implemented for ipex_ops, "
            "returning None.")
        return None

    @staticmethod
    def copy_blocks(key_caches: list[torch.Tensor],
                    value_caches: list[torch.Tensor],
                    block_mapping: torch.Tensor) -> None:
        torch.xpu.copy_blocks(  # type: ignore
            key_caches,
            value_caches,
            block_mapping,
        )

    @staticmethod
    def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                    block_mapping: torch.Tensor) -> None:
        torch.xpu.swap_blocks(src, dst, block_mapping)  # type: ignore

    @staticmethod
    def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        scale_ub: Optional[torch.Tensor] = None,
        use_per_token_if_dynamic: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor to FP8 and return quantized tensor and scale.
        
        This function is designed for both static and dynamic quantization:
        If you provide the scale, it will use static scaling and if you omit
        it, the scale will be determined dynamically. Currently, XPU platform
        only supports dynamic quantization. The function also allows optional
        padding of the output tensors for downstream kernels that will benefit
        from padding.

        Args:
            input: The input tensor to be quantized to FP8
            scale: Optional scaling factor for the FP8 quantization
            scale_ub: Optional upper bound for scaling factor in dynamic
                per token case
            num_token_padding: If specified, pad the first dimension
                of the output to at least this value.
            use_per_token_if_dynamic: Whether to do per_tensor or per_token
                in the dynamic quantization case.
    
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
                scaling factor.
        """
        # This code assumes batch_dim and num_tokens are flattened
        assert (input.ndim == 2)
        shape: Union[tuple[int, int], torch.Size] = input.shape
        out_dtype: torch.dtype = current_platform.fp8_dtype()
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        if output is None:
            output = torch.empty(shape, device=input.device, dtype=out_dtype)
        else:
            assert num_token_padding is None, \
                "padding not supported if output passed in"
            assert output.dtype == out_dtype
        assert scale is None, "only dynamic fp8 quantization supported on XPU"
        assert not use_per_token_if_dynamic, (
            "per token dynamic fp8 quantization not supported on XPU")
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        torch.ops.torch_ipex.dynamic_scaled_fp8_quant(output, input, scale)

        return output, scale
