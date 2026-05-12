# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass
from typing import ClassVar, Final

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
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@functools.lru_cache(maxsize=1)
def _fp8_mla_prefill_supported() -> bool:
    """Auto-detect FP8 MLA prefill via mla_prefill_ps_asm_fwd + mla_reduce_v1.

    The FP8 ASM prefill path requires (1) gfx950 (MI355X) and (2) a build of
    AITER that exports both ``mla_prefill_ps_asm_fwd`` and ``mla_reduce_v1``.
    When either is missing we fall back to the standard flash_attn_varlen_func
    prefill silently — no env knob, just "use it if you have it".

    NOTE: The AITER ``mla_prefill_ps_asm_fwd`` kernel currently only
    safely handles a *single* prefill segment per forward.  When the
    chunked-prefill scheduler packs multiple prefill segments together
    (e.g. 5×2-3k token segments to fill --max-num-batched-tokens=16384)
    the kernel faults.  We therefore auto-enable the path here, but
    ``forward_mha`` falls back to ``flash_attn_varlen_func`` whenever
    the prefill batch contains more than one segment — so single-
    request prefill (the common DSV3.2 1k/100/4 reference shape) gets
    the FP8 ASM speed-up while multi-segment batches stay safe.
    """
    try:
        from vllm.platforms.rocm import on_gfx950
    except Exception:  # noqa: BLE001
        return False
    if not on_gfx950():
        return False
    try:
        from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


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


# Tile size used by the mla_prefill_ps_asm_fwd assembly kernel.
_FP8_PREFILL_TILE_Q = 256


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

        # Pre-allocate FP8 MLA prefill PS metadata buffers.  This path is
        # auto-enabled on gfx950 when AITER provides the required kernels;
        # otherwise we silently fall back to flash_attn_varlen_func.
        self._fp8_prefill_enabled = _fp8_mla_prefill_supported()
        if self._fp8_prefill_enabled:
            # The PS metadata describes how to partition work for a single
            # prefill batch.  The max Q-length per request in any batch is
            # bounded by max_num_batched_tokens (chunked prefill), which is
            # typically much smaller than max_model_len.  Using the tighter
            # bound reduces the pre-allocated work_info buffer by ~20×.
            max_prefill_qlen = min(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            self._init_fp8_prefill_ps_buffers(max_num_reqs, max_prefill_qlen, device)

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

    def _init_fp8_prefill_ps_buffers(
        self,
        max_num_reqs: int,
        max_prefill_qlen: int,
        device: torch.device,
    ) -> None:
        """Pre-allocate persistent buffers for FP8 MLA prefill PS metadata.

        Uses ``get_ps_metadata_info_v1`` with max values so the buffers are
        large enough for any batch.  ``get_ps_metadata_v1`` fills them
        per-batch in ``build()``.

        Args:
            max_num_reqs: Maximum number of concurrent requests.
            max_prefill_qlen: Maximum Q-length for a single request in one
                prefill batch.  Should be ``min(max_model_len,
                max_num_batched_tokens)`` — the chunked-prefill scheduler
                never emits more than ``max_num_batched_tokens`` new tokens
                per batch.
            device: Target device for the buffers.
        """
        from aiter import get_ps_metadata_info_v1

        # After kv_b_proj decompression, K has num_heads heads (same as Q).
        # So gqa_ratio=1 and num_head_k=num_heads for the PS kernel.
        num_head_k = self.num_heads
        gqa_ratio = 1
        qlen_granularity = _FP8_PREFILL_TILE_Q // max(gqa_ratio, 1)

        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_size, work_info_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = get_ps_metadata_info_v1(
            batch_size=max_num_reqs,
            num_head_k=num_head_k,
            max_qlen=max_prefill_qlen,
            qlen_granularity=qlen_granularity,
        )

        self.fp8_ps_work_metadata = torch.empty(
            work_metadata_size, dtype=work_metadata_dtype, device=device
        )
        self.fp8_ps_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_dtype, device=device
        )
        self.fp8_ps_work_info = torch.empty(
            *work_info_size, dtype=work_info_dtype, device=device
        )
        self.fp8_ps_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
        )
        self.fp8_ps_reduce_final_map = torch.empty(
            *reduce_final_map_size, dtype=reduce_final_map_dtype, device=device
        )
        self.fp8_ps_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_dtype,
            device=device,
        )

        logger.info(
            "FP8 MLA prefill PS buffers allocated "
            "(max_batch=%d, max_qlen=%d, num_head_k=%d)",
            max_num_reqs,
            max_prefill_qlen,
            num_head_k,
        )

    def _build_fp8_prefill_ps_metadata(
        self,
        metadata: AiterMLAMetadata,
    ) -> None:
        """Build per-batch FP8 MLA prefill PS metadata and attach to *metadata*.

        Called from ``build()`` when prefill tokens are present and
        FP8 MLA prefill is enabled (auto-detected via
        ``_fp8_mla_prefill_supported()``).
        """
        from aiter import get_ps_metadata_v1

        prefill = metadata.prefill
        qo_indptr = prefill.query_start_loc
        kv_indptr = qo_indptr  # new tokens: KV length == Q length

        # get_ps_metadata_v1 reads CPU tensors.
        qo_indptr_cpu = qo_indptr.to("cpu", dtype=torch.int32)
        kv_indptr_cpu = qo_indptr_cpu.clone()
        seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)

        gqa_ratio = 1
        num_head_k = self.num_heads
        qhead_granularity = max(gqa_ratio, 1)
        qlen_granularity = _FP8_PREFILL_TILE_Q // qhead_granularity
        kvlen_granularity = 128
        block_size = 1  # non-paged: each "page" is one token

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_head_k,
            self.fp8_ps_work_metadata,
            self.fp8_ps_work_indptr,
            self.fp8_ps_work_info,
            self.fp8_ps_reduce_indptr,
            self.fp8_ps_reduce_final_map,
            self.fp8_ps_reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=True,
        )

        total_prefill_tokens = qo_indptr_cpu[-1].item()
        kv_indices = torch.arange(
            total_prefill_tokens, device=qo_indptr.device, dtype=torch.int32
        )

        # Attach PS metadata to the metadata object so forward_mha can read it.
        metadata.fp8_prefill_qo_indptr = qo_indptr
        metadata.fp8_prefill_kv_indptr = kv_indptr
        metadata.fp8_prefill_kv_indices = kv_indices
        metadata.fp8_prefill_work_indptr = self.fp8_ps_work_indptr
        metadata.fp8_prefill_work_info_set = self.fp8_ps_work_info
        metadata.fp8_prefill_reduce_indptr = self.fp8_ps_reduce_indptr
        metadata.fp8_prefill_reduce_final_map = self.fp8_ps_reduce_final_map
        metadata.fp8_prefill_reduce_partial_map = self.fp8_ps_reduce_partial_map
        metadata.fp8_prefill_max_q_len = prefill.max_query_len

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
        # Persistent (work-stealing) MLA decode metadata.
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
        # FP8 prefill persistent-scheduling (PS) metadata.  Populate
        # only when the FP8 ASM prefill kernels are auto-detected as
        # available AND there is actually prefill work in this batch;
        # forward_mha() checks hasattr(metadata, "fp8_prefill_qo_indptr")
        # and falls back to flash_attn_varlen_func otherwise.
        if self._fp8_prefill_enabled and attn_metadata.prefill is not None:
            self._build_fp8_prefill_ps_metadata(attn_metadata)
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
    _AITER_UNSUPPORTED_HEADS: ClassVar[tuple[int, ...]] = ()

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
        self._decode_out = None

        # FP8 MLA prefill kernel imports (lazy, only when enabled).
        # Auto-enabled on gfx950 when AITER ships the kernels.
        self._fp8_prefill_enabled = _fp8_mla_prefill_supported()
        if self._fp8_prefill_enabled:
            from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1

            self._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
            self._mla_reduce_v1 = mla_reduce_v1

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

    def _mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
    ) -> torch.Tensor:
        """Run FP8 MLA prefill via mla_prefill_ps_asm_fwd + mla_reduce_v1.

        Q, K, V are already decompressed (post-kv_b_proj).  After
        decompression K and V have ``num_heads`` heads (same as Q), so
        gqa_ratio = 1 which matches the Gqa=1 assembly kernel.

        Args:
            q: [total_tokens, num_heads, qk_head_dim]
            k: [total_tokens, num_heads, qk_head_dim]
            v: [total_tokens, num_heads, v_head_dim]
            attn_metadata: Must have fp8_prefill_* attributes set by the
                metadata builder.

        Returns:
            Output tensor [total_tokens, num_heads, v_head_dim] in the
            model's working dtype (e.g., bfloat16).
        """
        from vllm.platforms import current_platform

        fp8_dtype = current_platform.fp8_dtype()
        total_q = q.shape[0]
        nhead = self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q
        out_dtype = q.dtype if q.dtype != fp8_dtype else torch.bfloat16

        # Cast to FP8 if not already.
        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)

        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        # The actual number of active partial tiles for this batch is stored
        # in reduce_indptr[-1].  Using fp8_prefill_reduce_partial_map.size(0)
        # instead would allocate the *maximum* pre-allocated size, which can
        # be tens of GiB for large max_model_len values.
        num_partial_tiles = int(attn_metadata.fp8_prefill_reduce_indptr[-1].item())

        # Intermediate buffers for the two-phase PS kernel.
        logits = torch.empty(
            (num_partial_tiles * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (num_partial_tiles * tile_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = torch.empty(
            (total_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        output = torch.empty(
            (total_q, nhead, v_head_dim),
            dtype=out_dtype,
            device=q.device,
        )

        # Phase 1: persistent-scheduling assembly prefill kernel.
        self._mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            attn_metadata.fp8_prefill_qo_indptr,
            attn_metadata.fp8_prefill_kv_indptr,
            attn_metadata.fp8_prefill_kv_indices,
            attn_metadata.fp8_prefill_work_indptr,
            attn_metadata.fp8_prefill_work_info_set,
            attn_metadata.fp8_prefill_max_q_len,
            self.scale,
            True,  # is_causal
            logits,
            attn_lse,
            output,
            one_scale,
            one_scale,
            one_scale,
        )

        # Phase 2: reduction across KV splits.
        self._mla_reduce_v1(
            logits,
            attn_lse,
            attn_metadata.fp8_prefill_reduce_indptr,
            attn_metadata.fp8_prefill_reduce_final_map,
            attn_metadata.fp8_prefill_reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )

        return output

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Override to use FP8 MLA prefill when auto-detected as available.

        Auto-detection is in ``_fp8_mla_prefill_supported()`` and
        requires gfx950 plus AITER's ``mla_prefill_ps_asm_fwd`` /
        ``mla_reduce_v1`` symbols.  Falls back to the parent
        (``flash_attn_varlen_func``) when:
        - FP8 MLA prefill is not enabled
        - There is chunked context (prior KV cache to merge)
        - PS metadata was not built (e.g., pure decode batch)
        """
        if not self._fp8_prefill_enabled or not hasattr(
            attn_metadata, "fp8_prefill_qo_indptr"
        ):
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
            )

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        has_context = prefill_metadata.chunked_context is not None

        if has_context:
            # Chunked context requires merge_attn_states; fall back to parent
            # which handles the two-pass context + suffix merge.
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
            )

        # AITER mla_prefill_ps_asm_fwd faults on multi-segment varlen
        # input (e.g. 5 prefill segments packed into a single 16k-token
        # forward by the chunked-prefill scheduler).  Restrict the FP8 ASM
        # path to single-segment batches; multi-segment batches use the
        # safe flash_attn_varlen_func fallback below.
        prefill_qo_indptr = prefill_metadata.query_start_loc
        num_prefill_segments = prefill_qo_indptr.shape[0] - 1
        if num_prefill_segments > 1:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
            )

        # Decompress KV: kv_c_normed → k_nope, v via kv_b_proj
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        # FP8 MLA prefill: Q, K, V → mla_prefill_ps_asm_fwd + mla_reduce_v1
        output_prefill = self._mla_fp8_prefill_attn(q, k, v, attn_metadata)

        # Handle v_head_dim padding if present.
        if self._pad_v:
            output_prefill = output_prefill[..., : v.shape[-1]]

        output.copy_(output_prefill.flatten(start_dim=-2))

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

        # Pad heads up to AITER's MLA tile size (16) when num_heads < 16
        # (e.g. heavily TP'd small models).  For DSV3 / DSV3.2 / Kimi /
        # Linear with num_heads >= 16 this returns q unchanged.
        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)

        dtype = attn_metadata.decode.attn_out_dtype
        if (
            self._decode_out is None
            or self._decode_out.shape[0] < B
            or self._decode_out.shape[1] != mla_num_heads
            or self._decode_out.dtype != dtype
        ):
            self._decode_out = torch.zeros(
                B,
                mla_num_heads,
                self.kv_lora_rank,
                dtype=dtype,
                device=q.device,
            )
        o = self._decode_out[:B]

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
