# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, Final

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec


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
        # The aiter MLA decode kernel always operates with page_size=1
        # internally (the wrapper flattens kv_buffer via .view(-1, 1, 1, H)).
        # We support any kernel_block_size by expanding block-level indices
        # into per-token flat indices in the metadata builder.
        return [MultipleOf(1)]

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
    # Whether persistent MLA metadata was computed (only for qseqlen=1)
    has_persistent_metadata: bool = False


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

        # Store the kernel block size from the spec. When kernel_block_size=1
        # (no spec-dec), behavior is identical to the original. When > 1
        # (e.g. 16 with Eagle3), we expand block-level indices into per-token
        # flat indices since the aiter kernel always uses page_size=1 internally.
        self.kernel_block_size = kv_cache_spec.block_size

        # In the flat view (.view(-1,1,1,H)), each token is its own page,
        # so max_num_pages_per_req = max_model_len regardless of
        # kernel_block_size.
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.

        # paged_kv_last_page_len is always 1s (the aiter kernel always sees
        # page_size=1 after .view(-1,1,1,H) flattening), so we create it
        # once and reuse slices in both eager and cudagraph modes.
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
        device = self.device
        num_reqs = seq_lens_device.size(0)

        # The aiter kernel always operates with page_size=1 (the wrapper
        # flattens kv_buffer). last_page_len is always 1.
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        # indptr: cumsum of seq_lens (one page per token in the flat view)
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

        # Expand block_table entries into per-token flat indices.
        # When kernel_block_size=1, this degrades to a direct copy (identical
        # to the original _copy_page_indices_kernel).
        # When kernel_block_size=K>1, block_table entry b covering K tokens
        # gets expanded to flat indices b*K, b*K+1, ..., b*K+(K-1).
        _expand_page_indices_kernel[(num_reqs,)](
            self.paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            seq_lens_device,
            KERNEL_BLOCK_SIZE=self.kernel_block_size,
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

        # The aiter MLA ASM kernel only supports qseqlen=1 (single-token
        # decode). With speculative decoding, the verification step has
        # qseqlen > 1 (e.g. 8 for spec7). get_mla_metadata_v1 calls
        # get_heuristic_kernel_mla which fails for qseqlen > 1.
        # We track whether persistent metadata was successfully computed
        # so forward_mqa can skip passing it (falling back to the kernel
        # computing its own metadata internally, like v0.18.0).
        has_persistent_metadata = False
        if max_qo_len == 1:
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
            has_persistent_metadata = True

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
            has_persistent_metadata=has_persistent_metadata,
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
        if (
            attn_metadata.decode is not None
            and attn_metadata.decode.has_persistent_metadata
        ):
            attn_metadata.work_meta_data = self._mla_work_meta_data
            attn_metadata.work_indptr = self._mla_work_indptr
            attn_metadata.work_info_set = self._mla_work_info_set
            attn_metadata.reduce_indptr = self._mla_reduce_indptr
            attn_metadata.reduce_final_map = self._mla_reduce_final_map
            attn_metadata.reduce_partial_map = self._mla_reduce_partial_map
        return attn_metadata


@triton.jit
def _expand_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_tokens,
    seq_lens,
    KERNEL_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Expand block table entries into per-token flat page indices.

    The aiter MLA kernel always operates with page_size=1 internally
    (kv_buffer is flattened via .view(-1, 1, 1, H)). This kernel converts
    block-level indices from the block table into individual token positions
    in the flattened KV buffer.

    When KERNEL_BLOCK_SIZE=1: block_idx=t, offset=0, flat=block_id
    (equivalent to a direct copy -- no regression from the original kernel).

    When KERNEL_BLOCK_SIZE=K: block table entry b (covering K tokens)
    is expanded to flat indices b*K, b*K+1, ..., b*K+(K-1).
    """
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_tokens + req_idx)
    num_tokens = tl.load(seq_lens + req_idx)

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_tokens, BLOCK_SIZE):
        token_offsets = i + offset
        mask = token_offsets < num_tokens

        # Which block in the block table does this token belong to?
        block_idx = token_offsets // KERNEL_BLOCK_SIZE
        # Offset within that block
        offset_in_block = token_offsets % KERNEL_BLOCK_SIZE

        # Load the block ID from the block table
        block_ids = tl.load(row_ptr + block_idx, mask=mask)

        # Compute flat index in the flattened kv_buffer
        flat_indices = block_ids * KERNEL_BLOCK_SIZE + offset_in_block

        tl.store(
            page_indices + start_idx + token_offsets,
            flat_indices,
            mask=mask,
        )


class AiterMLAHelper:
    """
    AITER MLA implementation requires num_heads >= 16. If num_heads < 16 and
    16 % num_heads == 0, we can pad q to 16 heads; otherwise AITER has to fail.
    """

    _AITER_MIN_MLA_HEADS: Final = 16

    @staticmethod
    def check_num_heads_validity(num_heads: int):
        assert AiterMLAHelper.is_valid_num_heads(num_heads), (
            f"Aiter MLA requires that num_heads be multiples or divisors of 16, "
            f"but provided {num_heads} number of heads.\n"
            f"Try adjusting tensor_parallel_size value."
        )

    @staticmethod
    def is_valid_num_heads(num_heads: int) -> bool:
        return (
            num_heads % AiterMLAHelper._AITER_MIN_MLA_HEADS == 0
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else AiterMLAHelper._AITER_MIN_MLA_HEADS % num_heads == 0
        )

    @staticmethod
    def get_actual_mla_num_heads(num_heads: int) -> int:
        return max(num_heads, AiterMLAHelper._AITER_MIN_MLA_HEADS)

    @staticmethod
    def get_mla_padded_q(num_heads: int, q: torch.Tensor) -> torch.Tensor:
        return (
            q
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else q.repeat_interleave(
                AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, dim=1
            )
        )

    @staticmethod
    def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
        return (
            o
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else o[:, :: AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, :]
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
        AiterMLAHelper.check_num_heads_validity(num_heads)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

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

        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)
        o = torch.empty(
            B,
            mla_num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.decode.attn_out_dtype,
            device=q.device,
        )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # Build kwargs for mla_decode_fwd. Pass persistent metadata only
        # when it was successfully computed (qseqlen=1 decode steps).
        # For multi-token verification steps (spec-dec), the kernel falls
        # back to computing metadata internally.
        mla_kwargs = dict(
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
        )
        if attn_metadata.work_meta_data is not None:
            mla_kwargs.update(
                work_meta_data=attn_metadata.work_meta_data,
                work_indptr=attn_metadata.work_indptr,
                work_info_set=attn_metadata.work_info_set,
                reduce_indptr=attn_metadata.reduce_indptr,
                reduce_final_map=attn_metadata.reduce_final_map,
                reduce_partial_map=attn_metadata.reduce_partial_map,
            )

        rocm_aiter_ops.mla_decode_fwd(
            mla_padded_q,
            kv_buffer,
            o,
            self.scale,
            attn_metadata.decode.qo_indptr,
            attn_metadata.decode.max_qo_len,
            attn_metadata.decode.paged_kv_indptr,
            attn_metadata.decode.paged_kv_indices,
            attn_metadata.decode.paged_kv_last_page_len,
            **mla_kwargs,
        )

        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, o), None
