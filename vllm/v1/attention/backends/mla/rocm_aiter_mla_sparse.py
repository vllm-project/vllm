# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
)
from vllm.attention.backends.utils import get_mla_dims
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (
    MLACommonBaseImpl,
    is_rocm_aiter_fp8bmm_enabled,
)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)

if is_rocm_aiter_fp8bmm_enabled():
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as aiter_triton_fp8_bmm,  # noqa: E501
    )

    def dynamic_per_batched_tensor_quant(
        x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
    ):
        DTYPE_MAX = torch.finfo(dtype).max
        min_val, max_val = x.aminmax()
        amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
        scale = DTYPE_MAX / amax
        x_scl_sat = (x * scale).clamp(min=-DTYPE_MAX, max=DTYPE_MAX)
        return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


class ROCMAiterMLASparseBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ROCMAITERMLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return ROCMAiterMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["ROCMAiterMLASparseMetadataBuilder"]:
        return ROCMAiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["ROCMAiterMLASparseImpl"]:
        return ROCMAiterMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]


@dataclass
class ROCMAiterMLASparseMetadata:
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048


def ref_convert_to_global(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    # Ensure contiguous
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    max_num_blocks = block_table_c.size(-1)

    # Compute block index and intra-block offset
    idxs_in = token_indices_c // block_size
    idxs_out = token_indices_c % block_size

    block_table_indexed = block_table_c[req_id_c]

    invalid = (idxs_in < 0) | (idxs_in >= max_num_blocks)

    idxs_in_clamped = idxs_in.masked_fill(invalid, 0)

    num_tokens = idxs_in_clamped.size(0)
    rest = idxs_in_clamped.numel() // num_tokens
    gathered = torch.gather(
        block_table_indexed, 1, idxs_in_clamped.view(num_tokens, rest)
    ).view_as(idxs_in_clamped)

    # Compute global indices and apply invalid mask
    out = gathered * block_size + idxs_out
    out = out.masked_fill(invalid, -1)

    return out


@dataclass
class ROCMAiterMLASparseMetadataBuilder(
    AttentionMetadataBuilder[ROCMAiterMLASparseMetadata]
):
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], device=device, dtype=torch.int32
        )
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ROCMAiterMLASparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        metadata = ROCMAiterMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )
        return metadata


# Take from
# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_prefill.py#L72
def reference_mla_sparse_prefill(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import math

    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    skv = kv.shape[0]
    sq = q.shape[0]
    topk = indices.shape[-1]
    dqk = q.shape[-1]
    indices = indices[:, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= skv)
    qs = q.float()  # [s_q, h_q, d_qk]
    kvs = kv[:, 0, :].float()  # [s_kv, d_qk]

    kvs = torch.index_select(
        kvs, 0, indices.masked_fill(invalid_indices_mask, 0).flatten()
    ).view(sq, topk, dqk)  # [s_q, topk, d_qk]
    attn_score = qs @ kvs.transpose(1, 2)  # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float("-inf"))
    attn_score *= sm_scale * math.log2(math.e)
    max_logits = torch.max(attn_score, dim=-1)[0]  # [s_q, h_q]
    lse = log2sumexp2(attn_score, dim=-1)  # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    result = attn_score @ kvs[:, :, :d_v]
    return (result.to(q.dtype), max_logits, lse)


class ROCMAiterMLASparseImpl(MLACommonBaseImpl[ROCMAiterMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: Optional["Indexer"] = None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        self.softmax_scale = scale
        assert indexer is not None
        self.topk_indices_buffer = indexer.topk_indices_buffer

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = reference_mla_sparse_prefill(
            q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale, 512
        )[0]
        return output[:, : self.num_heads, :]

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for MLACommonImpl"
            )

        if attn_metadata is None:
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs

        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        if is_rocm_aiter_fp8bmm_enabled():
            # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
            ql_nope = aiter_triton_fp8_bmm(
                q_nope, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
            )
        else:
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            ql_nope = torch.bmm(q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            ql_nope = ql_nope.transpose(0, 1)

        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # Note: the above triton kernel may triggers some strange unexpected
        # crush on Mi300, although the code looks fine on memory access pattern,
        # this ref torch  impl can help to alleviate this issue.
        topk_indices_global = ref_convert_to_global(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            attn_metadata.block_size,
        )

        q = torch.cat([ql_nope, q_pe], dim=-1)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        attn_out = self._forward_bf16_kv(
            q, kv_cache, topk_indices_global, attn_metadata
        )

        self._v_up_proj(attn_out, out=output[:num_actual_toks])
        return output
