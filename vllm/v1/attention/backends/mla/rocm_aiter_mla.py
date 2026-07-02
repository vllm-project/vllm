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

    Falls back to ``flash_attn_varlen_func`` when the kernels are unavailable.
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
    # Whether persistent MLA metadata was computed and can be passed to AITER.
    has_persistent_metadata: bool = False


@dataclass
class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None

    # FP8 ASM prefill persistent-scheduling (PS) metadata.  Populated by
    # AiterMLAMetadataBuilder._build_fp8_prefill_ps_metadata when prefill
    # tokens are present and FP8 MLA prefill is supported on the device.
    # Left as None on hosts/configs that fall back to flash_attn_varlen_func.
    fp8_prefill_qo_indptr: torch.Tensor | None = None
    fp8_prefill_kv_indptr: torch.Tensor | None = None
    fp8_prefill_kv_indices: torch.Tensor | None = None
    fp8_prefill_work_indptr: torch.Tensor | None = None
    fp8_prefill_work_info_set: torch.Tensor | None = None
    fp8_prefill_reduce_indptr: torch.Tensor | None = None
    fp8_prefill_reduce_final_map: torch.Tensor | None = None
    fp8_prefill_reduce_partial_map: torch.Tensor | None = None
    fp8_prefill_max_q_len: int | None = None
    fp8_prefill_num_partial_tiles: int | None = None


# Tile size used by the mla_prefill_ps_asm_fwd assembly kernel.
_FP8_PREFILL_TILE_Q = 256
_MAX_NATIVE_MTP_DECODE_QUERY_LEN = 4
_DEFAULT_MLA_NUM_KV_SPLITS = 32


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    # MTP verification presents uniform qlen>1 decode batches. AITER's dense
    # MLA decode path supports those batches when vLLM supplies causal
    # persistent metadata, so keep the native row layout by default.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.SINGLE_ONLY

    @classmethod
    def _mtp_decode_query_len(cls, vllm_config: VllmConfig) -> int | None:
        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            return None

        num_spec_tokens = speculative_config.num_speculative_tokens
        method = speculative_config.method
        if method not in ("mtp", "deepseek_mtp") or num_spec_tokens is None:
            return None

        return int(num_spec_tokens) + 1

    @classmethod
    def _split_uniform_mtp_decode(cls, vllm_config: VllmConfig) -> bool:
        mtp_query_len = cls._mtp_decode_query_len(vllm_config)
        return (
            mtp_query_len is not None
            and mtp_query_len > 1
            and mtp_query_len > _MAX_NATIVE_MTP_DECODE_QUERY_LEN
        )

    @classmethod
    def _allow_uniform_mtp_decode(cls, vllm_config: VllmConfig) -> bool:
        # MTP (num_speculative_tokens>=1) always yields qlen>=2, so native
        # dense AITER MLA consumes uniform MTP qlen>1 batches up to
        # _MAX_NATIVE_MTP_DECODE_QUERY_LEN; larger qlen uses the split fallback,
        # which still presents qlen=1 rows to AITER.
        return cls._mtp_decode_query_len(vllm_config) is not None

    @staticmethod
    def _uniform_padded_mtp_qo_len(
        qo_len: torch.Tensor,
        max_qo_len: int,
        num_decode_tokens: int,
    ) -> int:
        num_reqs = qo_len.numel()
        if num_reqs == 0 or num_decode_tokens <= 0:
            return 0

        # Full-CG pads q to a captured token count while leaving
        # query_start_loc flat for dummy requests. Only synthesize dummy rows
        # when every padded request maps to the same qlen and the q buffer has
        # exactly that many rows.
        if num_decode_tokens <= int(qo_len.sum().item()):
            return 0
        if num_decode_tokens % num_reqs != 0:
            return 0

        uniform_qo_len = num_decode_tokens // num_reqs
        if uniform_qo_len <= 1:
            return 0

        positive_qo_len = qo_len[qo_len > 0]
        if positive_qo_len.numel() == qo_len.numel():
            return 0
        if positive_qo_len.numel() > 0:
            if max_qo_len != uniform_qo_len:
                return 0
            if not torch.all(positive_qo_len == uniform_qo_len):
                return 0

        zero_positions = torch.nonzero(qo_len == 0, as_tuple=False).flatten()
        if zero_positions.numel() > 0:
            first_zero = int(zero_positions[0].item())
            if torch.any(qo_len[first_zero:] > 0):
                return 0

        return uniform_qo_len

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if cls._allow_uniform_mtp_decode(vllm_config):
            return AttentionCGSupport.UNIFORM_BATCH
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        if self._allow_uniform_mtp_decode(vllm_config):
            # Config-dependent upgrade: AITER MLA handles uniform MTP
            # verification decode (qlen<=4) natively. The ClassVar is
            # intentionally shadowed per-instance before super().__init__
            # consumes it to derive reorder_batch_threshold.
            self.query_len_support = QueryLenSupport.UNIFORM  # type: ignore[misc]

        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
        )

        self.compilation_config = vllm_config.compilation_config
        self.decode_attn_out_dtype = vllm_config.model_config.dtype
        self._split_mtp_decode = self._split_uniform_mtp_decode(vllm_config)
        self._mtp_decode_qlen = self._mtp_decode_query_len(vllm_config) or 1

        # Store the kernel block size from the spec. When kernel_block_size=1,
        # block-level indices already map to flat token indices. When > 1
        # (e.g. 16 with Eagle3), expand block-level indices because the aiter
        # kernel always uses page_size=1 internally.
        self.kernel_block_size = kv_cache_spec.block_size

        # In the flat view (.view(-1,1,1,H)), each token is its own page,
        # so max_num_pages_per_req = max_model_len regardless of
        # kernel_block_size.
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_decode_rows = max_num_reqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        if self._split_mtp_decode:
            max_decode_rows *= self._mtp_decode_qlen
            max_num_pages *= self._mtp_decode_qlen

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.

        # paged_kv_last_page_len is always 1s (the aiter kernel always sees
        # page_size=1 after .view(-1,1,1,H) flattening), so we create it
        # once and reuse slices in both eager and cudagraph modes.
        self.paged_kv_last_page_len = torch.ones(
            max_decode_rows, dtype=torch.int32, device=device
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
        kv_cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            kv_cache_dtype_str = "fp8"
            kv_dtype = dtypes.fp8
        else:
            kv_dtype = {
                torch.float16: dtypes.fp16,
                torch.bfloat16: dtypes.bf16,
            }.get(kv_cache_spec.dtype, kv_cache_spec.dtype)
        # MLAAttention quantizes decode Q to FP8 before calling this backend
        # whenever the KV cache is FP8 and supports_quant_query_input is true.
        q_dtype = (
            dtypes.fp8 if kv_cache_dtype_str == "fp8" else self.decode_attn_out_dtype
        )
        self._mla_metadata_q_dtype = q_dtype
        self._mla_metadata_kv_dtype = kv_dtype
        max_metadata_batch_size = (
            max_decode_rows if self._split_mtp_decode else max_num_reqs
        )
        max_metadata_qo_len = 1 if self._split_mtp_decode else self._mtp_decode_qlen
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_metadata_batch_size,
            max_metadata_qo_len,
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

        self._fp8_prefill_enabled = _fp8_mla_prefill_supported()
        if self._fp8_prefill_enabled:
            max_prefill_qlen = min(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            self._init_fp8_prefill_ps_buffers(
                max_num_reqs,
                max_prefill_qlen,
                vllm_config.scheduler_config.max_num_batched_tokens,
                device,
            )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_decode_rows + 1, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.zeros(
                max_decode_rows + 1, dtype=torch.int32, device=device
            )

    def _max_split_per_batch(self, batch_size: int) -> int:
        if self._num_attention_heads < 128 or batch_size <= 0:
            return -1
        device_properties = torch.cuda.get_device_properties(self.device)
        cu_num = device_properties.multi_processor_count
        return max(
            1,
            min(
                (cu_num + batch_size - 1) // batch_size,
                _DEFAULT_MLA_NUM_KV_SPLITS,
            ),
        )

    def _init_fp8_prefill_ps_buffers(
        self,
        max_num_reqs: int,
        max_prefill_qlen: int,
        max_num_batched_tokens: int,
        device: torch.device,
    ) -> None:
        """Pre-allocate persistent buffers for FP8 MLA prefill PS metadata.

        Uses ``get_ps_metadata_info_v1`` with max values so the buffers are
        large enough for any batch.  ``get_ps_metadata_v1`` fills them
        per-batch in ``build()``.  The FP8 prefill forward path also uses the
        global workspace manager for per-call scratch, so reserve its maximum
        shape here before the workspace manager is locked after warmup.

        Args:
            max_num_reqs: Maximum number of concurrent requests.
            max_prefill_qlen: Maximum Q-length for a single request in one
                prefill batch.  Should be ``min(max_model_len,
                max_num_batched_tokens)`` — a single request never exceeds
                ``max_model_len`` tokens, nor the per-batch token budget.
            max_num_batched_tokens: Maximum number of tokens scheduled in one
                batch.  The ``final_lse`` scratch is sized by ``total_q`` (the
                summed Q-length over all prefill requests in the batch), which
                is bounded by this budget rather than by a single request's
                ``max_prefill_qlen`` — concurrent requests can sum to more than
                ``max_model_len`` when ``max_model_len < max_num_batched_tokens``.
            device: Target device for the buffers.
        """
        from aiter import get_ps_metadata_info_v1

        # After kv_b_proj decompression, K has num_heads heads (same as Q).
        # So gqa_ratio=1 and num_head_k=num_heads for the PS kernel.
        num_head_k = self.num_heads
        v_head_dim = self.mla_dims.v_head_dim
        # gqa_ratio = 1
        # qlen_granularity = _FP8_PREFILL_TILE_Q // max(gqa_ratio, 1)
        qlen_granularity = _FP8_PREFILL_TILE_Q

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

        from vllm.v1.worker.workspace import current_workspace_manager

        max_num_partial_tiles = reduce_partial_map_size
        current_workspace_manager().get_simultaneous(
            (
                (max_num_partial_tiles * _FP8_PREFILL_TILE_Q, num_head_k, v_head_dim),
                torch.float32,
            ),
            (
                (max_num_partial_tiles * _FP8_PREFILL_TILE_Q, num_head_k),
                torch.float32,
            ),
            ((max_num_batched_tokens, num_head_k), torch.float32),
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
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        """Build per-batch FP8 MLA prefill PS metadata and attach to *metadata*.

        Called from ``build()`` when prefill tokens are present and
        FP8 MLA prefill is enabled (auto-detected via
        ``_fp8_mla_prefill_supported()``).
        """
        from aiter import get_ps_metadata_v1

        prefill = metadata.prefill
        # Caller (build()) only invokes this when prefill tokens exist, so
        # metadata.prefill is guaranteed non-None.  Assert to narrow for mypy.
        assert prefill is not None
        qo_indptr = prefill.query_start_loc
        kv_indptr = qo_indptr  # new tokens: KV length == Q length

        # Reuse the existing CPU view of query_start_loc instead of forcing a
        # device->host copy.  Prefill batches sit at the tail of the request
        # list, so we slice from num_decodes onwards and rebase to zero, the
        # same transform the parent build applies on device tensors.
        num_decodes = metadata.num_decodes
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        qo_indptr_cpu = (qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]).to(torch.int32)
        kv_indptr_cpu = qo_indptr_cpu.clone()
        seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)

        num_head_k = self.num_heads
        # gqa_ratio = 1
        # qhead_granularity = max(gqa_ratio, 1)
        # qlen_granularity = _FP8_PREFILL_TILE_Q // qhead_granularity
        gqa_ratio = 1
        qhead_granularity = 1
        qlen_granularity = _FP8_PREFILL_TILE_Q
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

        total_prefill_tokens = int(qo_indptr_cpu[-1].item())
        kv_indices = torch.arange(
            total_prefill_tokens, device=qo_indptr.device, dtype=torch.int32
        )

        # The actual number of active partial tiles for this batch is the
        # final value of reduce_indptr.  Resolving it here (during metadata
        # build) keeps it off the per-layer forward path where a sync would
        # break CUDA Graph capture.  Using the device-side reduce_indptr is
        # acceptable since build is allowed to incur an occasional sync.
        num_partial_tiles = int(self.fp8_ps_reduce_indptr[-1].item())

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
        metadata.fp8_prefill_num_partial_tiles = num_partial_tiles

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
        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()
        padded_mtp_qo_len = self._uniform_padded_mtp_qo_len(
            qo_len, max_qo_len, num_decode_tokens
        )
        if padded_mtp_qo_len > 0:
            max_qo_len = padded_mtp_qo_len
        pad_uniform_mtp = padded_mtp_qo_len > 0

        split_mtp_decode = self._split_mtp_decode and max_qo_len > 1
        if split_mtp_decode:
            qo_lens_device = (
                query_start_loc_device[1 : num_reqs + 1]
                - query_start_loc_device[:num_reqs]
            ).to(torch.int32)
            seq_lens_device_i32 = seq_lens_device.to(torch.int32)
            if pad_uniform_mtp:
                padded_rows = qo_lens_device == 0
                qo_lens_device = torch.where(
                    padded_rows,
                    qo_lens_device.new_full((), max_qo_len),
                    qo_lens_device,
                )
                seq_lens_device_i32 = torch.where(
                    padded_rows,
                    seq_lens_device_i32.new_full((), max_qo_len),
                    seq_lens_device_i32,
                )
            token_offsets = torch.arange(max_qo_len, dtype=torch.int32, device=device)
            flat_seq_lens = seq_lens_device_i32.unsqueeze(1) - (
                qo_lens_device.unsqueeze(1) - 1 - token_offsets.unsqueeze(0)
            )
            valid_tokens = token_offsets.unsqueeze(0) < qo_lens_device.unsqueeze(1)
            flat_seq_lens = torch.where(
                valid_tokens, flat_seq_lens, flat_seq_lens.new_zeros(())
            ).flatten()
            flat_seq_lens = flat_seq_lens[:num_decode_tokens]
            seq_lens_for_kernel = flat_seq_lens
            num_kernel_reqs = num_decode_tokens
        else:
            seq_lens_for_kernel = seq_lens_device
            num_kernel_reqs = num_reqs
            if pad_uniform_mtp:
                qo_lens_device = (
                    query_start_loc_device[1 : num_reqs + 1]
                    - query_start_loc_device[:num_reqs]
                ).to(torch.int32)
                seq_lens_for_kernel = torch.where(
                    qo_lens_device > 0,
                    seq_lens_for_kernel,
                    seq_lens_for_kernel.new_full((), max_qo_len),
                )

        # The aiter kernel always operates with page_size=1 (the wrapper
        # flattens kv_buffer). last_page_len is always 1.
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_kernel_reqs]

        # indptr: cumsum of seq_lens (one page per token in the flat view)
        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                seq_lens_for_kernel.cumsum(dim=0, dtype=torch.int32),
            ]
        )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indices.fill_(-1)

        if split_mtp_decode:
            # Present MTP qlen=2/4 as independent qlen=1 decode rows to AITER.
            # Each row sees only the causal prefix up to that token, so token i
            # never attends to later verification tokens in the same request.
            _expand_mtp_decode_page_indices_kernel[(num_kernel_reqs,)](
                self.paged_kv_indices,
                block_table_tensor,
                block_table_tensor.stride(0),
                paged_kv_indptr,
                seq_lens_device_i32,
                qo_lens_device,
                MAX_QO_LEN=max_qo_len,
                KERNEL_BLOCK_SIZE=self.kernel_block_size,
                BLOCK_SIZE=1024,
            )
            max_qo_len = 1
        else:
            # Expand block_table entries into per-token flat indices.
            # When kernel_block_size=1, block IDs already equal flat token
            # indices.
            # When kernel_block_size=K>1, block_table entry b covering K tokens
            # gets expanded to flat indices b*K, b*K+1, ..., b*K+(K-1).
            _expand_page_indices_kernel[(num_reqs,)](
                self.paged_kv_indices,
                block_table_tensor,
                block_table_tensor.stride(0),
                paged_kv_indptr,
                seq_lens_for_kernel,
                KERNEL_BLOCK_SIZE=self.kernel_block_size,
                BLOCK_SIZE=1024,
            )
        paged_kv_indices = self.paged_kv_indices

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr[: 1 + num_kernel_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_kernel_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_kernel_reqs]

            # paged_kv_last_page_len already uses the pre-initialized buffer slice
            # (set above), so no copy needed - buffer is always 1s.

            if split_mtp_decode:
                qo_indptr_src = torch.arange(
                    0,
                    num_kernel_reqs + 1,
                    step=1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                if pad_uniform_mtp:
                    qo_indptr_src = torch.arange(
                        0,
                        (num_kernel_reqs + 1) * max_qo_len,
                        step=max_qo_len,
                        dtype=torch.int32,
                        device=device,
                    )
                else:
                    qo_indptr_src = query_start_loc_device[: 1 + num_kernel_reqs]
            self.qo_indptr[: 1 + num_kernel_reqs].copy_(
                qo_indptr_src, non_blocking=True
            )
            self.qo_indptr[1 + num_kernel_reqs :] = qo_indptr_src[-1]
            qo_indptr = self.qo_indptr[: 1 + num_kernel_reqs]

        else:
            if max_qo_len == 1:
                qo_indptr = torch.arange(
                    0,
                    num_kernel_reqs + 1,
                    step=1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                if pad_uniform_mtp:
                    qo_indptr = torch.arange(
                        0,
                        (num_kernel_reqs + 1) * max_qo_len,
                        step=max_qo_len,
                        dtype=torch.int32,
                        device=device,
                    )
                else:
                    qo_indptr = query_start_loc_device[: 1 + num_kernel_reqs]

        # For native MTP verification (qlen>1), pass persistent metadata so
        # AITER gets explicit causal boundaries for each request.
        has_persistent_metadata = False
        use_persistent_metadata = (
            max_qo_len > 1
            and max_qo_len <= self._mtp_decode_qlen
            and not split_mtp_decode
        )
        if use_persistent_metadata:
            from aiter import get_mla_metadata_v1

            uni_qo_len = (
                max_qo_len if pad_uniform_mtp or torch.all(qo_len == max_qo_len) else -1
            )
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
                uni_seqlen_qo=uni_qo_len,
                fast_mode=True,
                max_split_per_batch=self._max_split_per_batch(num_kernel_reqs),
                dtype_q=self._mla_metadata_q_dtype,
                dtype_kv=self._mla_metadata_kv_dtype,
            )
            has_persistent_metadata = True

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_for_kernel,
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
        if self._fp8_prefill_enabled and attn_metadata.prefill is not None:
            self._build_fp8_prefill_ps_metadata(attn_metadata, common_attn_metadata)
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

    When KERNEL_BLOCK_SIZE=1: block_idx=t, offset=0, flat=block_id.

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


@triton.jit
def _expand_mtp_decode_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_tokens,
    seq_lens,
    qo_lens,
    MAX_QO_LEN: tl.constexpr,
    KERNEL_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Expand each MTP verification token as one qlen=1 decode row."""
    flat_row = tl.program_id(0)
    req_idx = flat_row // MAX_QO_LEN
    token_idx = flat_row - req_idx * MAX_QO_LEN

    qo_len = tl.load(qo_lens + req_idx)
    seq_len_full = tl.load(seq_lens + req_idx)
    seq_len = seq_len_full - (qo_len - 1 - token_idx)
    seq_len = tl.where(token_idx < qo_len, seq_len, 0)

    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_tokens + flat_row)

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, seq_len, BLOCK_SIZE):
        token_offsets = i + offset
        mask = token_offsets < seq_len

        block_idx = token_offsets // KERNEL_BLOCK_SIZE
        offset_in_block = token_offsets % KERNEL_BLOCK_SIZE
        block_ids = tl.load(row_ptr + block_idx, mask=mask)
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

        # FP8 MLA prefill kernel imports are lazy and only used when supported.
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
        out: torch.Tensor,
    ) -> None:
        """Run FP8 MLA prefill via mla_prefill_ps_asm_fwd + mla_reduce_v1.

        Q, K, V are already decompressed (post-kv_b_proj), so K and V have
        ``num_heads`` heads (same as Q) and gqa_ratio=1.  Writes the
        result in-place to ``out``, which is the [total_q, nhead * v_head_dim]
        output buffer supplied by ``forward_mha``; no extra allocation or
        copy is required.
        """
        from vllm.platforms import current_platform
        from vllm.v1.worker.workspace import current_workspace_manager

        fp8_dtype = current_platform.fp8_dtype()
        total_q = q.shape[0]
        nhead = self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q

        # The FP8 ASM kernel expects FP8 inputs; the q_scale/k_scale/v_scale
        # parameters select per-tensor dequant scales.  Q/K/V arrive as
        # bf16 from kv_b_proj, so cast here (one_scale=1.0 disables scaling).
        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)

        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        # num_partial_tiles is resolved during metadata build to avoid an
        # in-forward .item() sync that would prevent CUDA Graph capture.
        # forward_mha gates the FP8 path on fp8_prefill_qo_indptr being set,
        # and the builder always sets every fp8_prefill_* field together, so
        # num_partial_tiles is non-None here.
        num_partial_tiles = attn_metadata.fp8_prefill_num_partial_tiles
        assert num_partial_tiles is not None

        # Reuse the caller's output buffer to skip the per-call alloc + copy.
        # The ASM and reduce kernels both write to a [total_q, nhead, v_head_dim]
        # view, which aliases the [total_q, nhead * v_head_dim] storage of out.
        out_3d = out.view(total_q, nhead, v_head_dim)

        # Per-call scratch (logits, attn_lse, final_lse) is served from the
        # workspace manager so allocator churn in the prefill hot path is
        # bounded after warmup.
        logits, attn_lse, final_lse = current_workspace_manager().get_simultaneous(
            ((num_partial_tiles * tile_q, nhead, v_head_dim), torch.float32),
            ((num_partial_tiles * tile_q, nhead), torch.float32),
            ((total_q, nhead), torch.float32),
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
            out_3d,
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
            # num_kv_splits added by ROCm/aiter#3391; 0 selects the kernel
            # default max(cu_num, 0) == cu_num, matching pre-#3391 behavior.
            0,
            out_3d,
            final_lse,
        )

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        """Dispatch prefill to the FP8 ASM kernel when available.

        Falls back to the parent (``flash_attn_varlen_func``) when FP8
        MLA prefill is disabled, PS metadata is missing, or chunked
        context requires two-pass merge.

        The annotation uses the base ``MLACommonMetadata`` to honour LSP
        with ``MLACommonImpl.forward_mha``; the AITER builder always
        produces ``AiterMLAMetadata`` instances at runtime, so we narrow
        with ``isinstance`` before reading the AITER-specific FP8 fields.
        """
        if (
            not self._fp8_prefill_enabled
            or not isinstance(attn_metadata, AiterMLAMetadata)
            or attn_metadata.fp8_prefill_qo_indptr is None
        ):
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        has_context = prefill_metadata.chunked_context is not None

        if has_context:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )

        assert output_scale is None, (
            "fused FP8 output not supported by the AITER FP8 MLA prefill path"
        )

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        self._mla_fp8_prefill_attn(q, k, v, attn_metadata, output)

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
        if (
            attn_metadata.decode.max_qo_len > 1
            and not attn_metadata.decode.has_persistent_metadata
        ):
            # MTP verification can call the AITER MLA decode kernel with
            # qlen > 1. If that path is running without persistent metadata,
            # zero-fill so unwritten lanes cannot leak into logits.
            o.zero_()

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # Build kwargs for mla_decode_fwd. Native MTP qlen>1 uses this metadata
        # to supply causal verification boundaries; other paths only pass it
        # when the builder marked the metadata as valid.
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
