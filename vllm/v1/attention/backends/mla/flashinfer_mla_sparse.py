# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA attention backend."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
    max_decode_query_len,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.nvfp4_ds_mla_fp8_gather import (
    FP8_STAGING_PAGE_SIZE,
    FP8_STAGING_ROW_DIM,
    NVFP4_DS_MLA_KV_LORA_RANK,
    NVFP4_DS_MLA_ROPE_DIM,
    NVFP4GatherPrefillMetadata,
    build_nvfp4_gather_prefill_metadata,
    gather_nvfp4_ds_mla_context_to_fp8,
    gather_nvfp4_ds_mla_topk_to_fp8,
    nvfp4_fp8_gather_unsupported_reason,
    nvfp4_fp8_max_decode_tokens,
    nvfp4_fp8_prefill_workspace_rows,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    align_blocks_to_rows,
    flat_kv_row_view,
    prepare_sparse_mla_safe_lengths,
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.worker.workspace import current_workspace_manager

# FlashInfer's native NVFP4 sparse MLA decode serves 16 query heads per rank on
# SM100 and SM103 and was tested at these top-k widths.
_NVFP4_NATIVE_DECODE_NUM_HEADS = 16
_NVFP4_NATIVE_DECODE_TOPK_WIDTHS = (512, 1024, 2048)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)


class _FlashInferMLASparseBackendBase(AttentionBackend):
    """Common metadata for concrete FlashInfer sparse MLA backends."""

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseMetadataBuilder"]:
        return FlashInferMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 576 = 512 NoPE + 64 RoPE (with-rope layout); 512 = 512 NoPE only
        # (no-rope layout, qk_rope_head_dim == 0). Both share D_V = 512 and are
        # served by the TRTLLM-GEN sparse MLA kernel.
        return [512, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # The kernels read the rows as pages of their kernel block size.
        (page,) = cls.get_supported_kernel_block_sizes(spec)
        assert isinstance(page, MultipleOf)
        return align_blocks_to_rows(spec, page.base)


class FlashInferMLASparseTRTLLMBackend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend using the TRTLLM-gen launcher."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        # Native NVFP4 decode where supported; FP8 staging for prefill/fallback.
        "nvfp4_ds_mla",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(32)]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return FlashInferMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseTRTLLMMetadataBuilder"]:
        return FlashInferMLASparseTRTLLMMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            and parallel_config.decode_context_parallel_size > 1
        ):
            return (
                "FLASHINFER_MLA_SPARSE does not support combined PCP+DCP; "
                "use FLASHMLA_SPARSE, which gathers each DCP KV shard before "
                "running the rank-local PCP prefill queries"
            )
        if kv_cache_dtype == "fp8_ds_mla":
            return (
                "FLASHINFER_MLA_SPARSE SM10 does not support fp8_ds_mla kv-cache dtype"
            )
        if kv_cache_dtype == "nvfp4_ds_mla":
            reason = nvfp4_fp8_gather_unsupported_reason(vllm_config)
            if reason is not None:
                return reason

        # FlashInfer MLA sparse SM10 kernel requires qk_nope_head_dim in [128, 192].
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = hf_text_config.qk_nope_head_dim
            qk_rope_head_dim = hf_text_config.qk_rope_head_dim
            kv_lora_rank = hf_text_config.kv_lora_rank
            if qk_rope_head_dim == 0:
                # Native no-rope MLA: FlashInfer only ships the one shape
                # (nope_mla_dimensions in flashinfer.mla._core).
                if qk_nope_head_dim != 256 or kv_lora_rank != 512:
                    return (
                        "FlashInfer native no-rope MLA requires "
                        "qk_nope_head_dim=256 and kv_lora_rank=512, but got "
                        f"qk_nope_head_dim={qk_nope_head_dim}, "
                        f"kv_lora_rank={kv_lora_rank}"
                    )
            elif qk_nope_head_dim not in [128, 192]:
                return (
                    "FlashInfer MLA Sparse kernel requires qk_nope_head_dim "
                    f"in [128, 192], but got {qk_nope_head_dim}"
                )
            # Check for index_topk which indicates sparse model
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashInfer MLA Sparse requires model with index_topk config"
        return None


class FlashInferMLASparseSM120Backend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend for SM120."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_SM120"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
            FlashInferMLASparseSM120Impl,
        )

        return FlashInferMLASparseSM120Impl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            return (
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API"
            )
        if dtype != torch.bfloat16:
            return "dtype not supported"
        if kv_cache_dtype not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
        ):
            return "kv_cache_dtype not supported"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            index_topk = getattr(hf_text_config, "index_topk", None)
            if index_topk is None:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires a model with "
                    "index_topk config"
                )
            if int(index_topk) != 2048:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires index_topk=2048; "
                    f"got {index_topk}"
                )
        return None


@dataclass
class FlashInferMLASparseMetadata(SparseMLACommonMetadata):
    """Attention metadata for FlashInfer MLA Sparse backend."""

    # nvfp4_ds_mla only: context-gather plan for the batch's prefill tokens.
    # None without prefills.
    nvfp4_prefill: NVFP4GatherPrefillMetadata | None = None


class FlashInferMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashInferMLASparseMetadata]
):
    """Builder for FlashInfer MLA Sparse attention metadata."""

    metadata_cls = FlashInferMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )


class FlashInferMLASparseTRTLLMMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    """Metadata builder for the SM100 TRT-LLM sparse MLA kernel."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    use_nvfp4_gather: bool = False

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # Decode uses device request boundaries; prefill metadata is not graph-safe.
        return max_decode_query_len(vllm_config)

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Under DCP the workspace must hold the trtllm-gen softmax-stats slab.
        # The buffer address is baked into CUDA graphs, so it has to reach its
        # final size here, before the first forward/capture (no lazy regrow).
        if self.dcp_world_size > 1:
            num_q_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            _get_workspace_buffer(
                device,
                _required_workspace_bytes(
                    self.dcp_world_size,
                    num_q_heads,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                ),
            )

        self.use_nvfp4_gather = vllm_config.cache_config.cache_dtype == "nvfp4_ds_mla"
        if self.use_nvfp4_gather:
            # Only single-token (and spec-decode) rows take the per-token top-k
            # gather. Longer requests share one context gather per request,
            # which keeps short prefills off the query_len * top-k gather path.
            self._init_reorder_batch_threshold(
                1,
                supports_spec_as_decode=True,
                supports_dcp_with_varlen=True,
            )
            scheduler_config = vllm_config.scheduler_config
            self._nvfp4_request_ids = torch.empty(
                scheduler_config.max_num_batched_tokens,
                dtype=torch.int32,
                device=device,
            )
            self._nvfp4_workspace_starts = torch.empty(
                scheduler_config.max_num_seqs, dtype=torch.int32, device=device
            )
            self._nvfp4_workspace_rows = nvfp4_fp8_prefill_workspace_rows(
                vllm_config.model_config.max_model_len
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> FlashInferMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # Planned for every batch with prefills (as FlashMLA does): whether the
        # layer routes them to dense MHA is decided at forward time.
        if self.use_nvfp4_gather and metadata.num_prefills > 0:
            metadata.nvfp4_prefill = build_nvfp4_gather_prefill_metadata(
                common_attn_metadata,
                metadata.num_decodes,
                metadata.num_prefills,
                self._nvfp4_workspace_rows,
                self._nvfp4_request_ids,
                self._nvfp4_workspace_starts,
            )
        return metadata


# Global workspace buffer (lazily initialized)
_fi_sparse_workspace: torch.Tensor | None = None

# trtllm-gen carves a softmax-stats slab from the workspace whenever LSE is
# requested (FlashInfer csrc/trtllm_fmha_kernel_launcher.cu, unchanged from
# v0.6.14 through current main):
#   sizeof(float2) * num_qo_heads * batch_size * round_up(max_q_len, 256)
#   + 1 MiB guard
# forward_mqa always passes q_len == 1, so each (head, token) pair costs
# round_up(1, 256) == 256 slots.
_TRTLLM_GEN_SOFTMAX_STAT_BYTES = 8  # sizeof(float2)
_TRTLLM_GEN_SOFTMAX_SLOTS_PER_TOKEN = 256  # round_up(q_len=1, 256)
_TRTLLM_GEN_SOFTMAX_GUARD_BYTES = 1024 * 1024

# Keep in sync with the VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE default in
# vllm/envs.py.
_DEFAULT_WORKSPACE_BUFFER_SIZE = 394 * 1024 * 1024


def compute_trtllm_sparse_mla_workspace_bytes(
    base_workspace_bytes: int,
    dcp_world_size: int,
    num_heads_per_rank: int,
    max_num_batched_tokens: int,
) -> int:
    """Workspace bytes needed by the trtllm-gen sparse MLA decode kernel.

    Under DCP the query is all-gathered across the DCP group in the head dim
    and the kernel is asked for LSE, so trtllm-gen carves the softmax-stats
    slab described above from the workspace before its (batch-independent)
    counter and scratch regions. ``base_workspace_bytes`` must stay available
    for those regions, so the slab is added on top of it (see #50781).

    Without DCP no LSE is requested, no slab is carved, and the base size is
    returned unchanged.
    """
    if dcp_world_size <= 1:
        return base_workspace_bytes
    softmax_bytes = (
        _TRTLLM_GEN_SOFTMAX_STAT_BYTES
        * (num_heads_per_rank * dcp_world_size)
        * max_num_batched_tokens
        * _TRTLLM_GEN_SOFTMAX_SLOTS_PER_TOKEN
        + _TRTLLM_GEN_SOFTMAX_GUARD_BYTES
    )
    return base_workspace_bytes + softmax_bytes


def _required_workspace_bytes(
    dcp_world_size: int,
    num_heads_per_rank: int,
    max_num_batched_tokens: int,
) -> int:
    """Resolve the workspace size, honoring an explicit env override."""
    computed = compute_trtllm_sparse_mla_workspace_bytes(
        _DEFAULT_WORKSPACE_BUFFER_SIZE,
        dcp_world_size,
        num_heads_per_rank,
        max_num_batched_tokens,
    )
    if not envs.is_set("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"):
        return computed
    env_bytes = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
    if env_bytes < computed:
        logger.warning_once(
            "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=%d is below the %d bytes "
            "computed for the sparse MLA workspace with dcp_world_size=%d, "
            "%d heads/rank and max_num_batched_tokens=%d. Respecting the "
            "override, but the trtllm-gen kernel may crash with a workspace "
            "overflow; set VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=%d or unset "
            "it to use the computed size.",
            env_bytes,
            computed,
            dcp_world_size,
            num_heads_per_rank,
            max_num_batched_tokens,
            computed,
        )
    return env_bytes


def _get_workspace_buffer(
    device: torch.device, min_bytes: int | None = None
) -> torch.Tensor:
    global _fi_sparse_workspace
    required = (
        min_bytes
        if min_bytes is not None
        else envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
    )
    if _fi_sparse_workspace is None or _fi_sparse_workspace.numel() < required:
        # FlashInfer's CuteDSL MLA-decode tactic requires an int8 workspace;
        # the trtllm-gen path views it as uint8, so int8 is safe for all backends.
        _fi_sparse_workspace = torch.zeros(
            required,
            dtype=torch.int8,
            device=device,
        )
    return _fi_sparse_workspace


class FlashInferMLASparseImpl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """FlashInfer MLA Sparse implementation.

    Uses the TRT-LLM MLA kernel with sparse_mla_top_k parameter for
    sparse attention computation.
    """

    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    lse_base_on_e: bool = False
    # nvfp4_ds_mla staging, enabled in __init__ for that dtype.
    use_nvfp4_gather: bool = False
    _nvfp4_inv_k_scale: float | None = None

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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLASparseImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLASparseImpl"
            )

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
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )

        self._workspace_buffer: torch.Tensor | None = None
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

        # Native no-rope MLA additionally requires a per-query-token active
        # top-k length tensor.
        self.is_nope_mla = self.qk_rope_head_dim == 0

        # fp8 query quantization is required when using fp8 kv_cache,
        # as the TRTLLM-GEN sparse MLA kernel requires matching dtypes
        # for query and kv_cache (mixed bf16+fp8 is not supported).
        self.supports_quant_query_input = True

        # nvfp4_ds_mla: prefill is staged to FP8 rows for the fp8 kernel; decode
        # uses FlashInfer's native NVFP4 kernel where it applies and is staged
        # otherwise.
        self.use_nvfp4_gather = kv_cache_dtype == "nvfp4_ds_mla"
        if self.use_nvfp4_gather:
            self._init_nvfp4_staging()

    def _init_nvfp4_staging(self) -> None:
        if (self.kv_lora_rank, self.qk_rope_head_dim) != (
            NVFP4_DS_MLA_KV_LORA_RANK,
            NVFP4_DS_MLA_ROPE_DIM,
        ):
            raise ValueError(
                "The nvfp4_ds_mla kv-cache dtype requires kv_lora_rank="
                f"{NVFP4_DS_MLA_KV_LORA_RANK} and qk_rope_head_dim="
                f"{NVFP4_DS_MLA_ROPE_DIM}, got {self.kv_lora_rank} and "
                f"{self.qk_rope_head_dim}"
            )
        vllm_config = get_current_vllm_config()
        topk_indices_buffer = self.topk_indices_buffer
        topk_width = (
            topk_indices_buffer.shape[1]
            if topk_indices_buffer is not None
            else vllm_config.model_config.hf_text_config.index_topk
        )
        if topk_width % FP8_STAGING_PAGE_SIZE:
            raise ValueError(
                f"nvfp4_ds_mla staging needs a top-k width divisible by "
                f"{FP8_STAGING_PAGE_SIZE}, got {topk_width}"
            )
        self._nvfp4_max_decode_tokens = nvfp4_fp8_max_decode_tokens(vllm_config)
        self._nvfp4_native_decode = self._supports_native_nvfp4_decode(topk_width)
        if self._nvfp4_native_decode:
            logger.info_once(
                "FLASHINFER_MLA_SPARSE decodes nvfp4_ds_mla with FlashInfer's native "
                "NVFP4 kernel; prefill is staged to FP8"
            )
        fp8_dtype = current_platform.fp8_dtype()
        # Reserved up front (like FlashMLA's prefill workspace) so the memory
        # profile sees it; every layer shares the same workspace views. The
        # native decode kernel needs no decode staging.
        workspace_shapes: list[tuple[tuple[int, ...], torch.dtype]] = [
            (
                (
                    nvfp4_fp8_prefill_workspace_rows(
                        vllm_config.model_config.max_model_len
                    ),
                    FP8_STAGING_ROW_DIM,
                ),
                fp8_dtype,
            )
        ]
        if not self._nvfp4_native_decode:
            workspace_shapes += [
                (
                    (self._nvfp4_max_decode_tokens * topk_width, FP8_STAGING_ROW_DIM),
                    fp8_dtype,
                ),
                ((self._nvfp4_max_decode_tokens, topk_width), torch.int32),
            ]
        workspaces = current_workspace_manager().get_simultaneous(*workspace_shapes)
        self._nvfp4_prefill_rows = workspaces[0]
        if not self._nvfp4_native_decode:
            self._nvfp4_decode_rows, self._nvfp4_decode_indices = workspaces[1:]

    def _supports_native_nvfp4_decode(self, topk_width: int) -> bool:
        """Whether FlashInfer's native NVFP4 kernel can serve this layer's decode."""
        from vllm.utils.flashinfer import has_flashinfer_nvfp4_sparse_mla_decode

        return (
            self.num_heads == _NVFP4_NATIVE_DECODE_NUM_HEADS
            and topk_width in _NVFP4_NATIVE_DECODE_TOPK_WIDTHS
            and (
                current_platform.is_device_capability(100)
                or current_platform.is_device_capability(103)
            )
            and has_flashinfer_nvfp4_sparse_mla_decode()
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            if q_pe.shape[-1] == 0 and ql_nope.is_contiguous():
                q = ql_nope
            else:
                q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        self._prepare_mqa_kernel(layer, q.device)

        if self.use_nvfp4_gather:
            return (
                self._forward_mqa_nvfp4(
                    q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
                ),
                None,
            )

        index_group = self.index_group
        if isinstance(index_group, HiSparseMLAIndexGroup):
            num_decode_tokens = attn_metadata.num_decode_tokens
            decode_out: torch.Tensor | None = None
            decode_lse: torch.Tensor | None = None
            if num_decode_tokens > 0:
                physical_topk, valid_counts = (
                    index_group.convert_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices[:num_decode_tokens],
                        attn_metadata,
                        block_stride_rows=None,
                        return_valid_counts=True,
                    )
                )
                decode_out, decode_lse = self._run_mqa_kernel(
                    q[:num_decode_tokens],
                    index_group.physical_kv_cache(self.index_group_index).view(
                        kv_c_and_k_pe_cache.dtype
                    ),
                    physical_topk,
                    valid_counts,
                )
                if num_decode_tokens == num_actual_toks:
                    return decode_out, decode_lse

            cache = index_group.cache(self.index_group_index)
            if num_decode_tokens == 0 and cache.all_context_pages_resident:
                physical_topk, valid_counts = (
                    index_group.convert_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices,
                        attn_metadata,
                        block_stride_rows=None,
                        return_valid_counts=True,
                    )
                )
                return self._run_mqa_kernel(
                    q,
                    index_group.physical_kv_cache(self.index_group_index).view(
                        kv_c_and_k_pe_cache.dtype
                    ),
                    physical_topk,
                    valid_counts,
                )

            prefill_cache, block_table, req_ids = index_group.stage_prefill_rows(
                self.index_group_index, kv_c_and_k_pe_cache, attn_metadata
            )
            prefill_indices, prefill_lens = triton_convert_req_index_to_global_index(
                req_ids,
                block_table,
                topk_indices[num_decode_tokens:],
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
            prefill_out, prefill_lse = self._run_mqa_kernel(
                q[num_decode_tokens:],
                prefill_cache,
                prefill_indices,
                prefill_lens,
            )
            if decode_out is None:
                return prefill_out, prefill_lse
            output = torch.cat((decode_out, prefill_out))
            if decode_lse is None:
                return output, None
            assert prefill_lse is not None
            return output, torch.cat((decode_lse, prefill_lse))

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )

        if self.dcp_world_size > 1:
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices_physical, seq_lens = self._convert_logical_to_physical_topk(
                topk_indices,
                attn_metadata,
                block_stride_rows=block_stride_rows,
                return_valid_counts=True,
            )

        return self._run_mqa_kernel(
            q,
            # Block sizes and strides have been aligned to 32 rows.
            kv_rows.view(
                -1,
                math.gcd(block_stride_rows, attn_metadata.block_size, 64),
                kv_rows.shape[-1],
            ),
            topk_indices_physical,
            seq_lens,
        )

    def _forward_mqa_nvfp4(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
    ) -> torch.Tensor:
        """Attend over an nvfp4_ds_mla cache with native decode when supported.

        Decode tokens are at the front of ``q``; any tokens after them are
        prefill tokens (present unless the layer routed them to dense MHA).
        """
        if q.dtype != current_platform.fp8_dtype():
            raise ValueError(
                "FLASHINFER_MLA_SPARSE runs the nvfp4_ds_mla kv-cache dtype with "
                f"an fp8 query (quantized by the MLA layer), got {q.dtype}"
            )
        num_tokens = q.shape[0]
        num_decode_tokens = min(attn_metadata.num_decode_tokens, num_tokens)
        # The FlashInfer kernel writes bf16 output.
        out = torch.empty(
            (num_tokens, q.shape[1], self.kv_lora_rank),
            dtype=torch.bfloat16,
            device=q.device,
        )
        if num_decode_tokens > 0:
            self._nvfp4_decode(
                q[:num_decode_tokens],
                kv_cache,
                topk_indices[:num_decode_tokens],
                attn_metadata,
                out[:num_decode_tokens],
            )
        if num_tokens > num_decode_tokens:
            self._nvfp4_prefill(
                q[num_decode_tokens:],
                kv_cache,
                topk_indices[num_decode_tokens:],
                attn_metadata,
                out[num_decode_tokens:],
                token_offset=num_decode_tokens,
            )
        return out

    def _nvfp4_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        out: torch.Tensor,
    ) -> None:
        assert self._nvfp4_inv_k_scale is not None
        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_cache, attn_metadata.block_size
        )
        physical_topk, valid_counts = self._convert_logical_to_physical_topk(
            topk_indices,
            attn_metadata,
            block_stride_rows=block_stride_rows,
            return_valid_counts=True,
        )
        if self._nvfp4_native_decode:
            from flashinfer.mla import nvfp4_sparse_mla_decode

            assert self.bmm1_scale is not None and self.bmm2_scale is not None
            # The fp8 bmm scales carry k_scale because staged rows hold
            # dequant / k_scale; the native kernel reads the exact values.
            nvfp4_sparse_mla_decode(
                q.contiguous(),
                kv_rows,
                physical_topk,
                bmm1_scale=float(self.bmm1_scale) * self._nvfp4_inv_k_scale,
                bmm2_scale=float(self.bmm2_scale) * self._nvfp4_inv_k_scale,
                out=out,
                backend="cuda",
            )
            return
        topk = physical_topk.shape[1]
        step = self._nvfp4_max_decode_tokens
        for start in range(0, q.shape[0], step):
            end = min(q.shape[0], start + step)
            rows = self._nvfp4_decode_rows[: (end - start) * topk]
            staging_indices = self._nvfp4_decode_indices[: end - start]
            gather_nvfp4_ds_mla_topk_to_fp8(
                kv_cache,
                physical_topk[start:end],
                rows,
                staging_indices,
                self._nvfp4_inv_k_scale,
            )
            self._run_mqa_kernel(
                q[start:end],
                rows.view(-1, FP8_STAGING_PAGE_SIZE, FP8_STAGING_ROW_DIM),
                staging_indices,
                valid_counts[start:end],
                output=out[start:end],
            )

    def _nvfp4_prefill(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        out: torch.Tensor,
        token_offset: int,
    ) -> None:
        assert self._nvfp4_inv_k_scale is not None
        plan = attn_metadata.nvfp4_prefill
        if plan is None:
            raise RuntimeError(
                "Prefill tokens reached the nvfp4_ds_mla sparse MQA path without "
                "a context-gather plan"
            )
        num_tokens = q.shape[0]
        assert plan.request_ids.shape[0] == num_tokens, (
            plan.request_ids.shape,
            num_tokens,
        )
        # Top-k positions of prefill tokens map to their request's (chunk-local)
        # workspace rows.
        staging_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[token_offset : token_offset + num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            HAS_PREFILL_WORKSPACE=True,
            prefill_workspace_request_ids=plan.request_ids,
            prefill_workspace_starts=plan.workspace_starts,
            return_valid_counts=True,
        )
        workspace = self._nvfp4_prefill_rows
        workspace_pages = workspace.view(-1, FP8_STAGING_PAGE_SIZE, FP8_STAGING_ROW_DIM)
        for chunk in plan.chunks:
            gather_nvfp4_ds_mla_context_to_fp8(
                kv_cache,
                chunk.block_table,
                chunk.workspace_starts,
                chunk.num_rows,
                chunk.search_steps,
                workspace,
                self._nvfp4_inv_k_scale,
            )
            tokens = chunk.tokens_slice
            self._run_mqa_kernel(
                q[tokens],
                workspace_pages,
                staging_indices[tokens],
                valid_counts[tokens],
                output=out[tokens],
            )

    def _prepare_mqa_kernel(
        self,
        layer: AttentionLayer,
        device: torch.device,
    ) -> None:
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(device)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float
        if self.use_nvfp4_gather and self._nvfp4_inv_k_scale is None:
            # Staged rows hold dequant(x) / k_scale, so the fp8 bmm scales above
            # apply unchanged.
            self._nvfp4_inv_k_scale = 1.0 / layer._k_scale_float

    def autotune_hisparse_decode(self, layer: AttentionLayer) -> None:
        """Autotune the largest legal HiSparse decode batch."""
        assert isinstance(self.index_group, HiSparseMLAIndexGroup)
        cache = self.index_group.cache(self.index_group_index)
        assert self.topk_indices_buffer is not None

        runtime = cache.runtime
        kv_cache = runtime.hot.attention_cache
        num_tokens = runtime.max_num_reqs
        topk_tokens = self.topk_indices_buffer.shape[1]
        self._prepare_mqa_kernel(layer, kv_cache.device)

        q_dtype = (
            current_platform.fp8_dtype()
            if is_quantized_kv_cache(self.kv_cache_dtype)
            else kv_cache.dtype
        )
        q = torch.zeros(
            (
                num_tokens,
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=q_dtype,
            device=kv_cache.device,
        )
        topk_indices = (
            torch.arange(
                topk_tokens,
                dtype=torch.int32,
                device=kv_cache.device,
            )
            .expand(num_tokens, -1)
            .contiguous()
        )
        seq_lens = torch.full(
            (num_tokens,),
            topk_tokens,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._run_mqa_kernel(q, kv_cache, topk_indices, seq_lens)

    def _run_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self._workspace_buffer is not None
        assert self.bmm1_scale is not None
        assert self.bmm2_scale is not None

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        kv_cache = kv_cache.view(q.dtype)

        # Single-token sparse decode. trtllm-gen requires the q_len_per_request
        # dim, but the sparse attention mask is fully per-token (each query token
        # carries its own top-k index row), so unsqueeze is sufficient and
        # correct. The MTP/multi-token q_len grouping is a perf-only layout and is
        # deferred until MTP is validated end-to-end for this backend.
        query = q.unsqueeze(1)
        block_tables = topk_indices.unsqueeze(1)
        seq_lens_arg = seq_lens

        # page_table width = topk buffer width, which kpool widens past
        # index_topk (topk_tokens) and rounds up to a multiple of 128. The
        # kernel treats sparse_mla_top_k as the page-table *capacity* and bounds
        # the active per-query length by ``seq_lens`` (the compacted valid
        # count), so the -1 padding slots past seq_lens are never attended to.
        # Use the actual buffer width instead of the fixed topk_tokens, which
        # mismatches the page_table when index_kpool > 1.
        sparse_topk_capacity = topk_indices.shape[1]

        extra_kwargs: dict[str, torch.Tensor] = {}
        needs_empty_query_guard = self.need_to_return_lse_for_decode or isinstance(
            self.index_group, HiSparseMLAIndexGroup
        )
        if self.is_nope_mla:
            # Resident TP queries have nonempty selections. Preserve empty-query
            # handling for DCP's local selections and host-backed HiSparse.
            topk_lens = seq_lens
            if needs_empty_query_guard:
                topk_lens = prepare_sparse_mla_safe_lengths(topk_indices, seq_lens)
            extra_kwargs["sparse_mla_top_k_lens"] = topk_lens

        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_arg,
            max_seq_len=sparse_topk_capacity,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sparse_mla_top_k=sparse_topk_capacity,
            out=None if output is None else output.unsqueeze(1),
            return_lse=self.need_to_return_lse_for_decode,
            **extra_kwargs,
        )
        if self.need_to_return_lse_for_decode:
            assert isinstance(kernel_out, tuple)
            o, lse = kernel_out
        else:
            assert isinstance(kernel_out, torch.Tensor)
            o = kernel_out
            lse = None

        out = o.view(-1, o.shape[-2], o.shape[-1])
        if lse is not None:
            lse = self._normalize_lse(lse, out.shape[0], out.shape[1])
        if self.is_nope_mla and needs_empty_query_guard:
            empty_queries = seq_lens == 0
            if lse is not None:
                # DCP combine already suppresses outputs with zero LSE weight.
                lse.masked_fill_(empty_queries[:, None], float("-inf"))
            else:
                out.masked_fill_(empty_queries[:, None, None], 0.0)
        elif lse is not None:
            empty_rows = (topk_indices == -1).all(dim=-1)
            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse

    @staticmethod
    def _normalize_lse(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        # FlashInfer returns the decode LSE either as 2D (num_tokens, num_heads)
        # or 3D ((num_tokens, num_heads, 1) / (num_tokens, 1, num_heads)).
        # Collapse all of these to the (num_tokens, num_heads) the shared DCP
        # reducer expects.
        if lse.dim() == 3:
            if lse.shape[-1] == 1:
                lse = lse.squeeze(-1)
            elif lse.shape[1] == 1:
                lse = lse.squeeze(1)
            elif lse.shape[0] * lse.shape[1] == num_tokens:
                lse = lse.reshape(num_tokens, lse.shape[-1])
        if lse.shape != (num_tokens, num_heads):
            raise RuntimeError(
                "Unexpected FlashInfer sparse MLA LSE shape: "
                f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
            )
        return lse
