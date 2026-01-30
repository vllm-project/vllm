# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


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
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        alibi_slopes: torch.Tensor | None,
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
        alibi_slopes: torch.Tensor | None,
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
        ipex.llm.functional.rotary_embedding_batched(
            positions, query, key, head_size, cos_sin_cache, is_neox, rot_dim
        )

    @staticmethod
    def rms_norm(
        input: torch.Tensor, weight: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        out = torch.empty_like(input)
        torch.ops.torch_ipex.rms_norm_vllm(out, input.contiguous(), weight, epsilon)
        return out

    @staticmethod
    def fused_add_rms_norm(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> None:
        torch.ops.torch_ipex.fused_add_rms_norm_vllm(input, residual, weight, epsilon)

    @staticmethod
    def varlen_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        seqlen_q: torch.Tensor,
        seqlen_k: torch.Tensor,
        alibi_slopes: torch.Tensor | None,
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
            ipex.llm.functional.varlen_attention(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                out,
                seqlen_q.int(),
                seqlen_k.int(),
                max_seqlen_q,
                max_seqlen_k,
                pdropout,
                softmax_scale,
                zero_tensors,
                is_causal,
                return_softmax,
                gen_,
            )
        else:  # XPU build
            ipex.llm.functional.varlen_attention(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                out,
                seqlen_q.int(),
                seqlen_k.int(),
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                pdropout,
                softmax_scale,
                zero_tensors,
                is_causal,
                return_softmax,
                gen_,
                window_size_left,
                window_size_right,
                logits_soft_cap,
            )

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
            key, value, key_cache, value_cache, slot_mapping
        )

    @staticmethod
    def reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        k_scale_float: float = 1.0,
        v_scale_float: float = 1.0,
    ) -> None:
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale_float,
            v_scale_float,
        )

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
        # The following parameters are not used in ipex kernel currently,
        # we keep API compatible to CUDA's.
        scheduler_metadata=None,
        fa_version: int = 2,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits=0,
        s_aux: torch.Tensor | None = None,
    ):
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        real_window_size: tuple[int, int]
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])

        if block_table is None:
            assert cu_seqlens_k is not None, (
                "cu_seqlens_k can't be None when calling varlen_attention."
            )
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            ipex_ops.varlen_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                None,
                max_seqlen_q,
                max_seqlen_k,
                0.0,
                softmax_scale,
                False,
                causal,
                False,
                None,
                real_window_size[0],
                real_window_size[1],
                -1,
            )
            return out
        else:
            return ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
                out,
                q.contiguous(),
                k,
                v,
                cu_seqlens_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                causal,
                block_table,
                alibi_slopes,
                sink=s_aux,
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
            "get_scheduler_metadata is not implemented for ipex_ops, returning None."
        )
        return None

    @staticmethod
    def swap_blocks(
        src: torch.Tensor, dst: torch.Tensor, block_mapping: torch.Tensor
    ) -> None:
        torch.xpu.swap_blocks(src, dst, block_mapping)  # type: ignore

    @staticmethod
    def scaled_fp8_quant(
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        num_token_padding: int | None = None,
        scale_ub: torch.Tensor | None = None,
        use_per_token_if_dynamic: bool = False,
        output: torch.Tensor | None = None,
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
        assert input.ndim == 2
        shape: tuple[int, int] | torch.Size = input.shape
        out_dtype: torch.dtype = current_platform.fp8_dtype()
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        if output is None:
            output = torch.empty(shape, device=input.device, dtype=out_dtype)
        else:
            assert num_token_padding is None, (
                "padding not supported if output passed in"
            )
            assert output.dtype == out_dtype
        assert scale is None, "only dynamic fp8 quantization supported on XPU"
        assert not use_per_token_if_dynamic, (
            "per token dynamic fp8 quantization not supported on XPU"
        )
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        torch.ops.torch_ipex.dynamic_scaled_fp8_quant(output, input, scale)

        return output, scale

    # Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
    @staticmethod
    def fp8_mqa_logits(
        q: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        """Compute FP8 MQA logits for a single sequence without KV paging.

        Args:
            q: Query tensor of shape [M, H, D]. Casted to
                `torch.float8_e4m3fn` by caller.
            kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
                dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
                [N, 1]) with dtype `torch.float32`.
            weights: weights of shape [M, H], dtype `torch.float32`.
            cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
                shape [M], dtype int32.
            cu_seqlen_ke: End indices (exclusive) for valid K per query position,
                shape [M], dtype int32.

        Returns:
            Logits tensor of shape [M, N], dtype `torch.float32`.
        """
        kv, scale = kv
        seq_len_kv = kv.shape[0]
        k = kv.to(torch.bfloat16)
        q = q.to(torch.bfloat16)

        mask_lo = (
            torch.arange(0, seq_len_kv, device=q.device)[None, :]
            >= cu_seqlen_ks[:, None]
        )
        mask_hi = (
            torch.arange(0, seq_len_kv, device=q.device)[None, :]
            < cu_seqlen_ke[:, None]
        )
        mask = mask_lo & mask_hi

        score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
        logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        logits = logits.masked_fill(~mask, float("-inf"))

        return logits

    # Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
    @staticmethod
    def fp8_paged_mqa_logits(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        max_model_len: int,
    ):
        from vllm.utils.math_utils import cdiv

        fp8_dtype = current_platform.fp8_dtype()
        batch_size, next_n, _, dim = q.size()
        kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
        scale = scale.contiguous().view(torch.float)
        q = q.float()
        kv_cache = kv_cache.view(fp8_dtype).float() * scale
        num_block, block_size, _, dim = kv_cache.size()
        logits = torch.full(
            [batch_size * next_n, max_model_len],
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        context_lens = context_lens.tolist()
        for i in range(batch_size):
            context_len = context_lens[i]
            q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
            weight_slice = (
                weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
            )
            for block_rk in range(cdiv(context_len, block_size)):
                block_idx = block_tables[i][block_rk]
                qx, kx = q[i], kv_cache[block_idx]
                k_offsets = torch.arange(
                    block_rk * block_size, (block_rk + 1) * block_size, device=q.device
                )
                mask = (k_offsets[None, :] < context_len) & (
                    k_offsets[None, :] <= q_offsets[:, None]
                )
                s = torch.where(
                    mask[None, :, :],
                    (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                        logits.dtype
                    ),
                    float("-inf"),
                )
                s = torch.relu(s) * weight_slice[..., None]
                s = s.sum(dim=0)
                logits[
                    i * next_n : (i + 1) * next_n,
                    block_rk * block_size : (block_rk + 1) * block_size,
                ] = torch.where(
                    k_offsets[None, :] <= q_offsets[:, None], s, float("-inf")
                )
        return logits

    @staticmethod
    def indexer_k_quant_and_cache(
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str | None,
    ) -> None:
        head_dim = k.shape[-1]
        k = k.view(-1, head_dim)  # [total_tokens, head_dim]

        def group_quant_torch(
            x: torch.Tensor,
            group_size: int,
            eps: float = 1e-10,
            dtype: torch.dtype | None = None,
            column_major_scales: bool = False,
            out_q: torch.Tensor | None = None,
            use_ue8m0: bool | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if use_ue8m0 is None:
                # Default fallback - could import is_deep_gemm_e8m0_used if needed
                use_ue8m0 = False

            if dtype is None:
                dtype = current_platform.fp8_dtype()

            # Validate inputs
            assert x.shape[-1] % group_size == 0, (
                f"Last dimension {x.shape[-1]} must be divisible by "
                f"group_size {group_size}"
            )
            assert x.stride(-1) == 1, "Input tensor groups must be contiguous"

            # Prepare output tensor
            if out_q is None:
                x_q = torch.empty_like(x, dtype=dtype)
            else:
                assert out_q.shape == x.shape
                x_q = out_q

            # Reshape input for group processing
            # Original shape: (..., last_dim)
            # Target shape: (..., num_groups, group_size)
            original_shape = x.shape
            num_groups = original_shape[-1] // group_size

            # Reshape to separate groups
            group_shape = original_shape[:-1] + (num_groups, group_size)
            x_grouped = x.view(group_shape)

            # Compute per-group absolute maximum values
            # Shape: (..., num_groups)
            abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)
            abs_max = torch.maximum(
                abs_max, torch.tensor(eps, device=x.device, dtype=x.dtype)
            )

            # Compute scales
            FP8_MAX = torch.finfo(dtype).max
            FP8_MIN = torch.finfo(dtype).min
            scale_raw = abs_max / FP8_MAX

            if use_ue8m0:
                # For UE8M0 format, scales must be powers of 2
                scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
            else:
                scales = scale_raw

            # Expand scales for broadcasting with grouped data
            # Shape: (..., num_groups, 1)
            scales_expanded = scales.unsqueeze(-1)

            # Quantize the grouped data
            x_scaled = x_grouped / scales_expanded
            x_clamped = torch.clamp(x_scaled, FP8_MIN, FP8_MAX)
            x_quantized = x_clamped.to(dtype)

            # Reshape back to original shape
            x_q.copy_(x_quantized.view(original_shape))

            # Prepare scales tensor in requested format
            if column_major_scales:
                # Column-major: (num_groups,) + batch_dims
                # Transpose the scales to put group dimension first
                scales_shape = (num_groups,) + original_shape[:-1]
                x_s = scales.permute(-1, *range(len(original_shape) - 1))
                x_s = x_s.contiguous().view(scales_shape)
            else:
                # Row-major: batch_dims + (num_groups,)
                x_s = scales.contiguous()

            # Ensure scales are float32
            return x_q, x_s.float()

        k_fp8, k_scale = group_quant_torch(
            k,
            group_size=quant_block_size,
            column_major_scales=False,
            use_ue8m0=(scale_fmt == "ue8m0"),
        )

        k_fp8_bytes = k_fp8.view(-1, head_dim).view(torch.uint8)
        scale_bytes = k_scale.view(torch.uint8).view(-1, 4)
        k = torch.cat(
            [k_fp8_bytes, scale_bytes], dim=-1
        )  # [total_tokens, head_dim + 4]

        slot_mapping = slot_mapping.flatten()
        # kv_cache: [num_block, block_size, head_dim + 4]
        kv_cache.view(-1, kv_cache.shape[-1]).index_copy_(0, slot_mapping, k)

    @staticmethod
    def cp_gather_indexer_k_quant_cache(
        kv_cache: torch.Tensor,
        dst_k: torch.Tensor,
        dst_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> None:
        """
        Args:
            kv_cache: [num_blocks, block_size, cache_stride] - quantized KV cache
                    Layout per block: [k_values, scale_values]
                    - k_values: [block_size * head_dim]
                    - scale_values: [block_size * head_dim * 4 / quant_block_size]
            dst_k: [num_tokens, head_dim] - output tensor for K values
            dst_scale: [num_tokens, head_dim / quant_block_size * 4]
                - output tensor for scale values
            block_table: [batch_size, num_blocks] - block table for indexing
            cu_seq_lens: [batch_size + 1] - cumulative sequence lengths
        """
        batch_size = block_table.size(0)
        num_tokens = dst_k.size(0)
        head_dim = dst_k.size(1)
        cache_block_size = kv_cache.size(1)
        num_blocks = block_table.size(1)
        quant_block_size = head_dim * 4 // dst_scale.size(1)

        # For each token, find which batch it belongs to using searchsorted
        token_indices = torch.arange(num_tokens, device=dst_k.device) + 1
        # cu_seq_lens is [batch_size + 1], we need to find which interval each
        # token belongs to
        batch_indices = torch.searchsorted(cu_seq_lens, token_indices) - 1
        batch_indices = torch.clamp(batch_indices, 0, batch_size - 1)

        # Calculate the in-batch sequence index for each token
        inbatch_seq_indices = token_indices - cu_seq_lens[batch_indices]

        # Find which block each token belongs to
        block_indices_in_table = inbatch_seq_indices // cache_block_size
        physical_block_indices = block_table[batch_indices, block_indices_in_table]

        # Calculate the offset within each block
        inblock_offsets = (inbatch_seq_indices - 1) % cache_block_size

        # Calculate strides
        block_stride = kv_cache.stride(0)  # stride for each block

        # Flatten kv_cache for easier indexing
        kv_cache_flat = kv_cache.view(-1)

        # Calculate source offset for K values for all tokens (vectorized)
        src_block_offsets = physical_block_indices * block_stride
        src_k_offsets = src_block_offsets + inblock_offsets * head_dim

        # Gather K values using advanced indexing
        # Create indices for all elements we need to gather
        k_indices = src_k_offsets.unsqueeze(1) + torch.arange(
            head_dim, device=dst_k.device
        )
        dst_k[:] = kv_cache_flat[k_indices]

        # Calculate source offset for scale values (vectorized)
        # Scales are stored after all K values for each block
        scale_size = head_dim * 4 // quant_block_size
        src_scale_offsets = src_block_offsets + head_dim + inblock_offsets * scale_size

        # Gather scale values
        scale_indices = src_scale_offsets.unsqueeze(1) + torch.arange(
            scale_size, device=dst_scale.device
        )
        dst_scale[:] = kv_cache_flat[scale_indices]

    @staticmethod
    def topk_with_bounds_torch(
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk_tokens: int,
    ) -> torch.Tensor:
        topk_indices = logits.topk(min(topk_tokens, logits.shape[-1]), dim=-1)[1].to(
            torch.int32
        )
        topk_indices -= cu_seqlen_ks[:, None]
        mask_lo = topk_indices >= 0
        mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks)[:, None] < 0
        mask = torch.full_like(
            topk_indices, False, dtype=torch.bool, device=topk_indices.device
        )
        mask = mask_lo & mask_hi
        topk_indices = topk_indices.masked_fill(~mask, -1)

    @staticmethod
    def decode_topk_with_masking_torch(
        logits: torch.Tensor,
        batch_size: int,
        next_n: int,
        topk_tokens: int,
        max_model_len: int,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        device = logits.device
        # padded query len
        padded_num_tokens = batch_size * next_n
        positions = (
            torch.arange(max_model_len, device=device)
            .unsqueeze(0)
            .expand(batch_size * next_n, -1)
        )
        row_indices = torch.arange(padded_num_tokens, device=device) // next_n
        next_n_offset = torch.arange(padded_num_tokens, device=device) % next_n
        index_end_pos = (seq_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)
        # index_end_pos: [B * N, 1]
        mask = positions <= index_end_pos
        # mask: [B * N, L]
        logits = logits.masked_fill(~mask, float("-inf"))
        topk_indices = logits.topk(topk_tokens, dim=-1)[1].to(torch.int32)  # [B * N, K]
        # ensure we don't set indices for the top k
        # that is out of range(masked already)
        # this will happen if context length is shorter than K
        topk_indices[topk_indices > index_end_pos] = -1

        return topk_indices
