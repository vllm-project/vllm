from typing import List, Optional, Tuple, Dict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)

import vllm._C.ops

class ipex_ops:

    @staticmethod
    def _reshape_activation_tensor(
            x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num = x.size(0)
        d = x.size(1) // 2
        x = x.reshape(num, 2, d)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = x1.reshape(num, d)
        x2 = x2.reshape(num, d)
        return x1, x2

    @staticmethod
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        # x1, x2 = ipex_ops._reshape_activation_tensor(x)
        # ipex.llm.functional.silu_mul(x1, x2, out)
        vllm._C.ops.silu_and_mul(out, x)

    @staticmethod
    def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        # x1, x2 = ipex_ops._reshape_activation_tensor(x)
        # ipex.llm.functional.gelu_mul(x1, x2, out, "none")
        vllm._C.ops.gelu_and_mul(out, x)

    @staticmethod
    def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        # x1, x2 = ipex_ops._reshape_activation_tensor(x)
        # ipex.llm.functional.gelu_mul(x1, x2, out, "tanh")
        vllm._C.ops.gelu_tanh_and_mul(out, x)

    @staticmethod
    def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
        # out.copy_(torch.nn.functional.gelu(x))
        vllm._C.ops.gelu_fast(out, x)

    @staticmethod
    def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
        # out.copy_(torch.nn.functional.gelu(x))
        vllm._C.ops.gelu_new(out, x)

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
        logits_soft_cap: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        # todo: ipex will refactor namespace
        import vllm._C.ops
        vllm._C.ops.paged_attention_v1(out, query,
                                     key_cache.view_as(value_cache),
                                     value_cache, num_kv_heads, scale,
                                     block_tables, context_lens, block_size,
                                     max_context_len, alibi_slopes, kv_cache_dtype, k_scale, logits_soft_cap)

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
        logits_soft_cap: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        # todo: ipex will refactor namespace
        import vllm._C.ops
        vllm._C.ops.paged_attention_v2(out, exp_sum, max_logits, tmp_out,
                                     query,
                                     key_cache.view_as(value_cache),
                                     value_cache, num_kv_heads, scale, block_tables,
                                     context_lens, block_size,
                                     max_context_len, alibi_slopes,kv_cache_dtype, k_scale, logits_soft_cap)

    @staticmethod
    def rotary_embedding(
        positions: torch.Tensor,  # [batch_size, seq_len]
        query: torch.Tensor,  # [batch_size, seq_len, num_heads*head_size]
        key: torch.Tensor,  # [batch_size, seq_len, num_kv_heads*head_size]
        head_size: int,
        cos_sin_cache: torch.Tensor,  # [cos_sin_dim, rot_dim]
        is_neox: bool,
    ) -> None:
        import vllm._C.ops
        vllm._C.ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
        # if positions.dim() == 1:
        #     positions = positions.unsqueeze(0)
        #     query = query.unsqueeze(0)
        #     key = key.unsqueeze(0)

        # rotary_dim = cos_sin_cache.size(1)
        # query = query.view(*query.shape[:-1], -1, head_size)
        # key = key.view(*key.shape[:-1], -1, head_size)

        # query_rot = query[..., :rotary_dim]
        # key_rot = key[..., :rotary_dim]

        # cos_sin = cos_sin_cache[positions.long()]
        # cos, sin = cos_sin.chunk(2, dim=-1)

        # if is_neox:
        #     cos = cos.repeat(1, 1, 2).unsqueeze(-2)
        #     sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        # else:
        #     cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        #     sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        
        # import vllm._C.ops
        # vllm._C.ops.rotary_embedding(query_rot, key_rot, sin, cos,
        #                                      rotary_dim, is_neox, positions)

    # def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
    #                              key: torch.Tensor, head_size: int,
    #                              cos_sin_cache: torch.Tensor, is_neox: bool,
    #                              rot_dim: int,
    #                              cos_sin_cache_offsets: torch.Tensor) -> None:
        
        # if positions.dim() == 1:
        #     positions = positions.unsqueeze(0)
        #     query = query.unsqueeze(0)
        #     key = key.unsqueeze(0)
        # cos_sin_cache_offsets = cos_sin_cache_offsets.view_as(positions)
        # rotary_dim = cos_sin_cache.size(1)
        # query = query.view(*query.shape[:-1], -1, head_size)
        # key = key.view(*key.shape[:-1], -1, head_size)

        # query_rot = query[..., :rotary_dim]
        # key_rot = key[..., :rotary_dim]

        # cos_sin = cos_sin_cache[torch.add(positions,
        #                                   cos_sin_cache_offsets).long()]
        # cos, sin = cos_sin.chunk(2, dim=-1)

        # if is_neox:
        #     cos = cos.repeat(1, 1, 2).unsqueeze(-2)
        #     sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        # else:
        #     cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        #     sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        # import vllm._C.ops
        # vllm._C.ops.rotary_embedding(query_rot, key_rot, sin, cos,
        #                                      rotary_dim, is_neox, positions)

    @staticmethod
    def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                 epsilon: float) -> None:
        # from intel_extension_for_pytorch.llm.modules.mha_fusion import RMSNorm
        # ipex_rms_norm = RMSNorm(weight, epsilon)
        # tmp = ipex_rms_norm.apply(input, weight, epsilon)
        # out.copy_(tmp)
        vllm._C.ops.rms_norm(out, input, weight, epsilon)

    @staticmethod
    def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                           weight: torch.Tensor, epsilon: float) -> None:
        # tmp = ipex.llm.functional.add_rms_norm(residual, input, weight, None,
        #                                        epsilon, True)
        # input.copy_(tmp)
        vllm._C.ops.fused_add_rms_norm(input, residual, weight, epsilon)

    @staticmethod
    def varlen_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        seqlen_q: torch.Tensor,
        seqlen_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        pdropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        is_causal: bool,
        return_softmax: bool,
        gen_: torch.Generator,
    ) -> None:
        pass

        # ipex.llm.functional.varlen_attention(query, key, value, out, seqlen_q,
        #                                      seqlen_k, max_seqlen_q,
        #                                      max_seqlen_k, pdropout,
        #                                      softmax_scale, zero_tensors,
        #                                      is_causal, return_softmax, gen_)

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
        # assert kv_cache_dtype == "auto"
        # ipex.llm.modules.PagedAttention.reshape_and_cache(
        #     key, value, key_cache, value_cache, slot_mapping)
        vllm._C.cache_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale)
    
    @staticmethod
    def reshape_and_cache_ipexllm(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        vllm._C.cache_ops.reshape_and_cache_ipexllm(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale)

    @staticmethod
    def copy_blocks(key_caches: List[torch.Tensor],
                    value_caches: List[torch.Tensor],
                    block_mapping) -> None:
        # torch.xpu.copy_blocks(  # type: ignore
        #     key_caches,
        #     value_caches,
        #     block_mapping,
        # )
        vllm._C.cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

    @staticmethod
    def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                    block_mapping) -> None:
        # torch.xpu.swap_blocks(src, dst, block_mapping)  # type: ignore
        vllm._C.cache_ops.swap_blocks(key_caches, value_caches, block_mapping)
