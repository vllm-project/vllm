# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class AiterMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor | None = None
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor | None = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor | None = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: torch.Tensor | None = None
    # The dtype of MLA out tensor
    attn_out_dtype: torch.dtype = torch.bfloat16
    # The max query output length: int
    max_qo_len: int | None = None


@dataclass
class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
        )

        self.compilation_config = vllm_config.compilation_config
        self.decode_attn_out_dtype = vllm_config.model_config.dtype
        # kernel block size is always 1.
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.

        # paged_kv_last_page_len is always 1s (kernel block size is always 1),
        # so we create it once and reuse slices in both eager and cudagraph modes.
        self.paged_kv_last_page_len = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )

        # Persistent buffer for paged_kv_indices to avoid blocking boolean mask
        # indexing (block_table_tensor[mask]) which has data-dependent output size.
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device
        )

        from aiter import dtypes, get_mla_metadata_info_v1

        # For num_attention_heads < 16 (e.g. kimi-k2.5 head=8 with TP8),
        # make sure get_mla_metadata_info_v1 / get_mla_metadata_v1 are consistent
        # with the actual tensor shape passed to mla_decode_fwd.
        self._num_attention_heads = max(16, self.num_heads)
        q_dtype = self.decode_attn_out_dtype
        kv_cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            kv_cache_dtype_str = "fp8"
        else:
            kv_cache_dtype_str = "bf16"
        kv_dtype = dtypes.d_dtypes.get(kv_cache_dtype_str, dtypes.bf16)
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_num_reqs,
            1,
            self._num_attention_heads,
            q_dtype,
            kv_dtype,
            is_sparse=False,
            fast_mode=True,
        )
        self._mla_work_meta_data = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        self._mla_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        self._mla_work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        )
        self._mla_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        self._mla_reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        self._mla_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_type,
            device=device,
        )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> AiterMLADecodeMetadata:
        # kernel block size is always 1, although the kv block size is not 1.
        device = self.device
        num_reqs = seq_lens_device.size(0)

        # kernel block size is always 1, so each page has exactly 1 token.
        # last_page_len is always 1 - just slice the pre-initialized buffer.
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens_device.dtype, device=device),
                seq_lens_device.cumsum(dim=0, dtype=torch.int32),
            ]
        )
        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indices.fill_(-1)
        _copy_page_indices_kernel[(num_reqs,)](
            self.paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )
        paged_kv_indices = self.paged_kv_indices

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr[: 1 + num_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

            # paged_kv_last_page_len already uses the pre-initialized buffer slice
            # (set above), so no copy needed - buffer is always 1s.

            self.qo_indptr[: 1 + num_reqs].copy_(
                query_start_loc_device, non_blocking=True
            )
            self.qo_indptr[1 + num_reqs :] = query_start_loc_device[-1]
            qo_indptr = self.qo_indptr[: 1 + num_reqs]

        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )

        from aiter import get_mla_metadata_v1

        get_mla_metadata_v1(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_last_page_len,
            self._num_attention_heads,
            1,
            True,
            self._mla_work_meta_data,
            self._mla_work_info_set,
            self._mla_work_indptr,
            self._mla_reduce_indptr,
            self._mla_reduce_final_map,
            self._mla_reduce_partial_map,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=max_qo_len,
            uni_seqlen_qo=max_qo_len,
            fast_mode=True,
        )

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            max_qo_len=max_qo_len,
            attn_out_dtype=self.decode_attn_out_dtype,
        )

        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AiterMLAMetadata:
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        attn_metadata.work_meta_data = self._mla_work_meta_data
        attn_metadata.work_indptr = self._mla_work_indptr
        attn_metadata.work_info_set = self._mla_work_info_set
        attn_metadata.reduce_indptr = self._mla_reduce_indptr
        attn_metadata.reduce_final_map = self._mla_reduce_final_map
        attn_metadata.reduce_partial_map = self._mla_reduce_partial_map
        return attn_metadata


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy block table rows into a flat page_indices buffer using indptr.
    Avoids blocking boolean mask indexing (tensor[mask]) which has
    data-dependent output size and forces sync.
    This is the same kernel as introduced in backends/flashinfer.py.
    """
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
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
        _valid_heads = num_heads in (4, 8) or (
            num_heads % 16 == 0 and 16 <= num_heads <= 128
        )
        assert _valid_heads, (
            f"Aiter MLA supports num_heads of 4, 8, or multiples of 16 "
            f"in [16, 128].\n"
            f"Provided {num_heads} number of heads.\n"
            "Try adjusting tensor_parallel_size value."
        )
        self._needs_head_repeat = num_heads < 16
        self._head_repeat_factor = 16 // num_heads if num_heads < 16 else 1
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self._absorption_weights_ready = False
        self._W_UK_T: torch.Tensor | None = None
        self._W_UV_ctx: torch.Tensor | None = None

    def _ensure_absorption_weights(self) -> None:
        """Lazily extract W_UK^T and W_UV from kv_b_proj for Q absorption.

        These are the same weights the decode path uses for the MQA approach.
        We duplicate them here so that the prefill context path can compute
        attention directly against the paged cache without expanding K/V
        through kv_b_proj.
        """
        if self._absorption_weights_ready:
            return
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_and_maybe_dequant_weights,
        )

        kv_b_w = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=torch.bfloat16
        ).T.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK = kv_b_w[..., : self.qk_nope_head_dim]
        W_UV = kv_b_w[..., self.qk_nope_head_dim :]
        self._W_UK_T = W_UK.permute(1, 2, 0).contiguous()  # [N, P, L]
        self._W_UV_ctx = W_UV.transpose(0, 1).contiguous()  # [N, L, V]
        self._absorption_weights_ready = True

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        k_scale: torch.Tensor,
    ):
        """Use AITER mla_prefill_fwd for context attention.

        Instead of the base class's chunk-by-chunk approach (gather cached KV,
        expand through kv_b_proj, flash_attn, merge), this computes
        absorbed-Q attention directly against the paged KV cache in a single
        assembly kernel call, eliminating the expensive KV expansion.
        """
        prefill = attn_metadata.prefill
        assert prefill is not None
        ctx = prefill.chunked_context
        assert ctx is not None

        if prefill.q_data_type == current_platform.fp8_dtype():
            return super()._compute_prefill_context(
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale
            )

        try:
            from aiter.mla import mla_prefill_fwd
        except ImportError:
            return super()._compute_prefill_context(
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale
            )

        self._ensure_absorption_weights()
        assert self._W_UK_T is not None and self._W_UV_ctx is not None

        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]
        T, N, _ = q_nope.shape
        L = self.kv_lora_rank

        # q_nope: [T, N, P]  W_UK_T: [N, P, L] -> q_absorbed: [T, N, L]
        q_absorbed = torch.bmm(
            q_nope.transpose(0, 1), self._W_UK_T
        ).transpose(0, 1)
        q_mqa = torch.cat([q_absorbed, q_pe], dim=-1)  # [T, N, L+R]

        iters = len(ctx.seq_tot)
        batch_size = ctx.cu_seq_lens[0].shape[0] - 1
        context_lens = torch.zeros(
            batch_size, dtype=torch.int32, device=q.device
        )
        for i in range(iters):
            cu = ctx.cu_seq_lens[i]
            context_lens += (cu[1:] - cu[:-1]).int()

        kv_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=q.device
        )
        torch.cumsum(context_lens, dim=0, out=kv_indptr[1:])

        total_pages = int(kv_indptr[-1].item())
        if total_pages == 0:
            output = torch.zeros(
                T, N, self.v_head_dim, dtype=q.dtype, device=q.device
            )
            lse = torch.full(
                (N, T), float("-inf"), dtype=torch.float32, device=q.device
            )
            return output, lse

        page_indices = torch.empty(
            total_pages, dtype=torch.int32, device=q.device
        )
        max_ctx = max(int(context_lens.max().item()), 1)
        _copy_page_indices_kernel[(batch_size,)](
            page_indices,
            prefill.block_table,
            prefill.block_table.stride(0),
            kv_indptr,
            BLOCK_SIZE=triton.next_power_of_2(max_ctx),
        )
        kv_last_page_lens = torch.where(
            context_lens > 0, 1, 0
        ).int()  # block_size=1

        o_compressed = torch.zeros(
            T, N, L, dtype=torch.float32, device=q.device
        )
        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        _, attn_lse = mla_prefill_fwd(
            q_mqa,
            kv_buffer,
            o_compressed,
            prefill.query_start_loc,
            kv_indptr,
            page_indices,
            kv_last_page_lens,
            prefill.max_query_len,
            sm_scale=self.scale,
        )

        output = torch.bmm(
            o_compressed.to(self._W_UV_ctx.dtype).transpose(0, 1),
            self._W_UV_ctx,
        ).transpose(0, 1)

        lse = attn_lse.squeeze(-1).squeeze(1).transpose(0, 1).contiguous()

        return output, lse

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        output = self.flash_attn_varlen_func(  # type: ignore[call-arg]
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert attn_metadata.decode.max_qo_len is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]

        if self._needs_head_repeat:
            q = q.repeat_interleave(self._head_repeat_factor, dim=1)
            kernel_num_heads = 16
        else:
            kernel_num_heads = self.num_heads

        o = torch.zeros(
            B,
            kernel_num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.decode.attn_out_dtype,
            device=q.device,
        )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        rocm_aiter_ops.mla_decode_fwd(
            q,
            kv_buffer,
            o,
            self.scale,
            attn_metadata.decode.qo_indptr,
            attn_metadata.decode.max_qo_len,
            attn_metadata.decode.paged_kv_indptr,
            attn_metadata.decode.paged_kv_indices,
            attn_metadata.decode.paged_kv_last_page_len,
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
            work_meta_data=attn_metadata.work_meta_data,
            work_indptr=attn_metadata.work_indptr,
            work_info_set=attn_metadata.work_info_set,
            reduce_indptr=attn_metadata.reduce_indptr,
            reduce_final_map=attn_metadata.reduce_final_map,
            reduce_partial_map=attn_metadata.reduce_partial_map,
        )

        if self._needs_head_repeat:
            o = o[:, :: self._head_repeat_factor, :]

        return o, None
