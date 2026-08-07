# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X causal paged attention backend for SM12x."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_B12X_SUPPORTED_PAGE_SIZES = (64, 128)
_B12X_PREFERRED_PAGE_SIZE = 128
_MIN_PAGED_TILE_Q = 16
_B12X_FP8_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")
_B12X_SUPPORTED_KV_CACHE_DTYPES = (
    "auto",
    "bfloat16",
    *_B12X_FP8_KV_CACHE_DTYPES,
)


def _max_page_table_width(
    max_model_len: int,
    block_size: int,
    max_num_batched_tokens: int,
    is_hybrid: bool,
) -> int:
    width = max(cdiv(max(max_model_len, 1), block_size), 1)
    if is_hybrid:
        # Hybrid cache setup can enlarge the storage block after attention
        # layers are initialized. Its expansion into kernel-sized blocks adds
        # at most one storage block of trailing page-table capacity.
        width += cdiv(max_num_batched_tokens, block_size)
    return width


def _kv_page_size(key_cache: torch.Tensor, value_cache: torch.Tensor) -> int:
    """Return the static kernel page geometry negotiated by vLLM.

    The KV manager can split the configured storage block into a smaller
    kernel page when another backend shares its cache group.  This notably
    happens when a B12X target shares a DFlash cache group with FlashInfer.
    Cache shapes are fixed before graph capture, so this is not a live-length
    policy decision.
    """
    if key_cache.ndim < 2 or value_cache.ndim < 2:
        raise ValueError(
            "B12X_ATTN expects paged K/V caches with a page dimension, got "
            f"{tuple(key_cache.shape)} and {tuple(value_cache.shape)}."
        )
    key_page_size = int(key_cache.shape[1])
    value_page_size = int(value_cache.shape[1])
    if key_page_size != value_page_size:
        raise ValueError(
            "B12X_ATTN requires matching K/V page sizes, got "
            f"{key_page_size} and {value_page_size}."
        )
    if key_page_size not in _B12X_SUPPORTED_PAGE_SIZES:
        raise ValueError(
            "B12X_ATTN requires runtime page size in "
            f"{_B12X_SUPPORTED_PAGE_SIZES}, got {key_page_size}."
        )
    return key_page_size


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed


def _env_optional_storage_limit(name: str, *, allow_zero: bool) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using B12X's planned capacity",
            name,
            value,
        )
        return None
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        logger.warning(
            "Ignoring %s=%r below the minimum %d; using B12X's planned capacity",
            name,
            value,
            minimum,
        )
        return None
    return parsed


def _capture_alloc_forbidden() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _ensure_i32_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        if _capture_alloc_forbidden():
            raise RuntimeError(
                f"B12X_ATTN would convert {name} to int32 during CUDA graph "
                "capture. Prepare int32 metadata before capture."
            )
        tensor = tensor.to(torch.int32)
    if not tensor.is_contiguous():
        if _capture_alloc_forbidden():
            raise RuntimeError(
                f"B12X_ATTN would make {name} contiguous during CUDA graph "
                "capture. Prepare contiguous metadata before capture."
            )
        tensor = tensor.contiguous()
    return tensor


def _dtype_from_cache_config(
    kv_cache_dtype: str,
    vllm_config: VllmConfig,
) -> torch.dtype:
    if kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if kv_cache_dtype in _B12X_FP8_KV_CACHE_DTYPES:
        return current_platform.fp8_dtype()
    if kv_cache_dtype != "auto":
        raise NotImplementedError(
            "B12X_ATTN currently supports only auto, bfloat16, "
            "fp8, and fp8_e4m3 "
            f"KV cache dtypes; got {kv_cache_dtype!r}."
        )
    return vllm_config.model_config.dtype


def _is_b12x_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in _B12X_FP8_KV_CACHE_DTYPES


class B12XPagedAttentionBackend(AttentionBackend):
    """Opt-in b12x paged attention backend for regular/GQA decoder layers."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "B12X_ATTN"

    @classmethod
    def get_impl_cls(cls) -> type[B12XPagedAttentionImpl]:
        return B12XPagedAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[B12XPagedMetadataBuilder]:
        return B12XPagedMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return list(_B12X_SUPPORTED_PAGE_SIZES)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or int(block_size) in _B12X_SUPPORTED_PAGE_SIZES

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if int(default_block_size) in _B12X_SUPPORTED_PAGE_SIZES:
            return int(default_block_size)
        return _B12X_PREFERRED_PAGE_SIZE

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 192, 256]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell SM120 / SM121. The b12x paged kernels also gate
        # internally, but keep vLLM selection fail-fast and explicit.
        return (capability.major, capability.minor) in ((12, 0), (12, 1))

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
        if dtype != torch.bfloat16:
            return "B12X_ATTN currently requires bfloat16 queries"
        if kv_cache_dtype == "float16":
            return "B12X_ATTN does not support float16 KV cache"
        if (
            kv_cache_dtype is not None
            and is_quantized_kv_cache(kv_cache_dtype)
            and not _is_b12x_fp8_kv_cache(kv_cache_dtype)
        ):
            return (
                "B12X_ATTN currently supports only fp8/fp8_e4m3 quantized "
                "KV cache dtypes"
            )
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        if parallel_config.decode_context_parallel_size > 1:
            return "B12X_ATTN does not yet support decode context parallelism"
        if parallel_config.prefill_context_parallel_size > 1:
            return "B12X_ATTN does not yet support prefill context parallelism"
        return None

    @classmethod
    def get_kv_cache_shape(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size not in _B12X_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                "B12X_ATTN requires block_size in "
                f"{_B12X_SUPPORTED_PAGE_SIZES}, got {block_size}."
            )
        if cache_dtype_str not in _B12X_SUPPORTED_KV_CACHE_DTYPES:
            raise ValueError(
                "B12X_ATTN currently supports only auto, bfloat16, "
                "fp8, and fp8_e4m3 "
                f"KV cache dtypes; got {cache_dtype_str!r}."
            )
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def get_kv_cache_stride_order(
        cls,
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout != "NHD":
            raise ValueError(
                f"B12X_ATTN requires NHD KV cache layout; got {cache_layout!r}."
            )
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        return "NHD"


@dataclass
class B12XPagedMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class B12XPagedMetadataBuilder(AttentionMetadataBuilder[B12XPagedMetadata]):
    """Metadata builder for B12X_ATTN.

    Decode and uniform speculative-verifier batches use preplanned graph
    buckets. Extend/prefill remains eager and does not affect uniform decode
    graph eligibility.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        del vllm_config, kv_cache_spec
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12XPagedMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        return B12XPagedMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            max_seq_len=cm.max_seq_len,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            causal=cm.causal,
        )

    def update_block_table(
        self,
        metadata: B12XPagedMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> B12XPagedMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata


class B12XPagedAttentionImpl(AttentionImpl[B12XPagedMetadata]):
    """b12x paged GQA attention implementation."""

    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("B12X_ATTN does not support ALiBi.")
        if logits_soft_cap not in (None, 0):
            raise NotImplementedError("B12X_ATTN does not support logits soft cap.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_ATTN currently supports decoder self-attention only."
            )
        if is_quantized_kv_cache(kv_cache_dtype) and not _is_b12x_fp8_kv_cache(
            kv_cache_dtype
        ):
            raise NotImplementedError(
                "B12X_ATTN currently supports only fp8/fp8_e4m3 quantized "
                "KV cache dtypes."
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError("B12X_ATTN requires q heads divisible by kv heads.")

        expected_scale = head_size**-0.5
        if not math.isclose(float(scale), expected_scale, rel_tol=1e-5, abs_tol=1e-7):
            raise NotImplementedError(
                "B12X_ATTN currently requires canonical softmax scale "
                f"head_dim**-0.5={expected_scale}, got {scale}."
            )
        if self.total_cp_world_size > 1:
            raise NotImplementedError(
                "B12X_ATTN does not yet support decode/prefill context parallelism."
            )

        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.output_head_size = self.head_size
        self.scale = float(scale)
        self.num_kv_heads = int(num_kv_heads)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.window_left = -1 if sliding_window is None else int(sliding_window) - 1

        self.sinks = sinks
        if self.sinks is not None and (
            self.sinks.ndim != 1 or int(self.sinks.shape[0]) != self.num_heads
        ):
            raise ValueError(
                "B12X_ATTN sinks must have shape "
                f"[{self.num_heads}], got {tuple(self.sinks.shape)}."
            )
        self._sinks_cache: dict[tuple[Any, ...], torch.Tensor] = {}

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        spec_config = vllm_config.speculative_config
        default_block_size = int(cache_config.block_size)
        if default_block_size not in _B12X_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                "B12X_ATTN requires --block-size in "
                f"{_B12X_SUPPORTED_PAGE_SIZES}, got "
                f"{cache_config.block_size}."
            )

        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.dtype = model_config.dtype
        self.kv_torch_dtype = _dtype_from_cache_config(kv_cache_dtype, vllm_config)
        if self.dtype != torch.bfloat16:
            raise NotImplementedError("B12X_ATTN currently requires bfloat16 queries.")
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        max_model_len = int(model_config.max_model_len)
        self._max_num_seqs = max_num_seqs
        max_page_table_widths = {
            page_size: _max_page_table_width(
                max_model_len,
                page_size,
                max_batched,
                model_config.is_hybrid,
            )
            for page_size in _B12X_SUPPORTED_PAGE_SIZES
        }

        # Extend dispatch may depend on the static Q tensor capacity, but never
        # on live per-request lengths. Keep a small set of capacity buckets so
        # short/tail prefills do not replay the maximum 8K CTA grid.
        self._extend_q_capacities = tuple(
            sorted(
                {
                    min(max_batched, q_capacity)
                    for q_capacity in (128, 512, 1024, 2048, 4096, max_batched)
                    if q_capacity > 0
                }
            )
        )

        def _extend_work_items(
            page_size: int,
            q_capacity: int,
            batch_size: int,
        ) -> int:
            capacity = plan_extend_graph_capacity(
                device=self.device,
                q_dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.output_head_size,
                page_size=page_size,
                batch=batch_size,
                total_q_capacity=q_capacity,
                max_cache_page_count=max_page_table_widths[page_size],
                window_left=self.window_left,
            )
            return _env_int(
                "VLLM_B12X_PAGED_EXTEND_MAX_WORK_ITEMS",
                capacity.max_work_items,
            )

        from b12x.attention.paged import (
            Caps as B12XPagedAttentionScratchCaps,
        )
        from b12x.attention.paged import (
            compile as compile_paged_attention,
        )
        from b12x.attention.paged import (
            decode_graph_capacity as plan_decode_graph_capacity,
        )
        from b12x.attention.paged import (
            decode_graph_scratch_envelope as plan_decode_graph_scratch_envelope,
        )
        from b12x.attention.paged import (
            extend_graph_capacity as plan_extend_graph_capacity,
        )
        from b12x.attention.paged import (
            plan as plan_paged_attention_scratch,
        )
        from b12x.attention.paged import (
            run as paged_attention_forward,
        )
        from b12x.attention.paged import (
            verify_graph_capacity as plan_verify_graph_capacity,
        )

        self._compile_paged_attention = compile_paged_attention
        self._paged_attention_forward = paged_attention_forward

        def _make_plan(
            page_size: int,
            mode: str,
            max_total_q: int,
            max_batch: int,
            max_work_items: int,
            max_partial_rows: int,
            use_cuda_graph: bool,
            num_cache_pages: int,
            copy_runtime_metadata: bool,
        ) -> Any:
            return plan_paged_attention_scratch(
                B12XPagedAttentionScratchCaps(
                    device=self.device,
                    mode=mode,
                    dtype=self.dtype,
                    kv_dtype=self.kv_torch_dtype,
                    num_q_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim_qk=self.head_size,
                    head_dim_vo=self.output_head_size,
                    page_size=page_size,
                    max_total_q=max_total_q,
                    max_batch=max_batch,
                    max_page_table_width=max_page_table_widths[page_size],
                    max_work_items=max_work_items,
                    max_partial_rows=max_partial_rows,
                    # Shape-only planning tensor; runtime cache shape is
                    # validated by head/page geometry, not page count.
                    num_cache_pages=num_cache_pages,
                    use_cuda_graph=use_cuda_graph,
                    copy_runtime_metadata=copy_runtime_metadata,
                )
            )

        capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
        decode_plan_sizes = {
            int(size) for size in capture_sizes if 0 < int(size) <= max_num_seqs
        }
        decode_plan_sizes.add(max_num_seqs)
        if os.getenv("VLLM_B12X_PAGED_DECODE_MAX_CHUNKS_PER_REQ"):
            logger.warning_once(
                "VLLM_B12X_PAGED_DECODE_MAX_CHUNKS_PER_REQ is ignored; "
                "B12X owns decode graph chunk policy. Use the fixed "
                "work/partial capacity controls only to constrain storage."
            )
        decode_work_items_limit = _env_optional_storage_limit(
            "VLLM_B12X_PAGED_DECODE_MAX_WORK_ITEMS",
            allow_zero=False,
        )
        decode_partial_rows_limit = _env_optional_storage_limit(
            "VLLM_B12X_PAGED_DECODE_MAX_PARTIAL_ROWS",
            allow_zero=True,
        )

        def _create_decode_plan(page_size: int, batch_size: int) -> Any:
            max_page_table_width = max_page_table_widths[page_size]
            capacity = plan_decode_graph_capacity(
                device=self.device,
                q_dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.output_head_size,
                page_size=page_size,
                batch=batch_size,
                max_cache_page_count=max_page_table_width,
                window_left=self.window_left,
                max_work_items=decode_work_items_limit,
                max_partial_rows=decode_partial_rows_limit,
            )
            plan = _make_plan(
                page_size,
                "decode",
                batch_size,
                batch_size,
                capacity.max_work_items,
                capacity.max_partial_rows,
                True,
                max_page_table_width,
                True,
            )
            plan.prepare_decode_graph_replay_state(
                batch=batch_size,
                total_q_capacity=batch_size,
                max_page_table_width=max_page_table_width,
                max_cache_page_count=max_page_table_width,
                window_left=self.window_left,
            )
            return plan

        self._create_decode_plan = _create_decode_plan
        self._verify_q_per_req = 0
        if spec_config is not None:
            self._verify_q_per_req = 1 + int(
                getattr(spec_config, "num_speculative_tokens", None) or 0
            )
        if self._verify_q_per_req <= 1:
            self._verify_q_per_req = 0

        def _create_verify_plan(page_size: int, batch_size: int) -> Any:
            if self._verify_q_per_req <= 1:
                raise RuntimeError(
                    "B12X_ATTN verifier plan requested without speculation"
                )
            max_page_table_width = max_page_table_widths[page_size]
            total_q = batch_size * self._verify_q_per_req
            capacity = plan_verify_graph_capacity(
                device=self.device,
                q_dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.output_head_size,
                page_size=page_size,
                batch=batch_size,
                query_len=self._verify_q_per_req,
                max_cache_page_count=max_page_table_width,
                window_left=self.window_left,
            )
            plan = _make_plan(
                page_size,
                "verify",
                total_q,
                batch_size,
                capacity.max_work_items,
                capacity.max_partial_rows,
                True,
                max_page_table_width,
                True,
            )
            page_ids = torch.arange(
                max_page_table_width,
                dtype=torch.int32,
                device=self.device,
            )
            max_page_table = page_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
            max_cache_seqlens = torch.full(
                (batch_size,),
                capacity.representative_cache_seqlen,
                dtype=torch.int32,
                device=self.device,
            )
            max_cu_seqlens_q = torch.arange(
                0,
                total_q + 1,
                self._verify_q_per_req,
                dtype=torch.int32,
                device=self.device,
            )
            plan.prepare_graph_replay_state(
                page_table=max_page_table,
                cache_seqlens=max_cache_seqlens,
                cu_seqlens_q=max_cu_seqlens_q,
                active_total_q=total_q,
                window_left=self.window_left,
            )
            return plan

        self._create_verify_plan = _create_verify_plan

        def _create_extend_plan(
            page_size: int,
            batch_size: int,
            q_capacity: int,
        ) -> Any:
            """Prepare a fixed-capacity extend plan without reading live lengths."""
            max_page_table_width = max_page_table_widths[page_size]
            plan = _make_plan(
                page_size,
                "extend",
                q_capacity,
                batch_size,
                _extend_work_items(page_size, q_capacity, batch_size),
                0,
                True,
                max_page_table_width,
                False,
            )
            page_ids = torch.arange(
                max_page_table_width,
                dtype=torch.int32,
                device=self.device,
            )
            max_page_table = page_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
            max_cache_seqlens = torch.full(
                (batch_size,),
                min(max_model_len, max_page_table_width * page_size),
                dtype=torch.int32,
                device=self.device,
            )
            # Put one row in every request except the last, which owns the
            # remainder. This represents the full total-Q capacity while the
            # replay kernel remains responsible for packing arbitrary live
            # per-request lengths from device cu_seqlens_q.
            max_cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=torch.int32,
                device=self.device,
            )
            max_cu_seqlens_q[-1] = q_capacity
            plan.prepare_graph_replay_state(
                page_table=max_page_table,
                cache_seqlens=max_cache_seqlens,
                cu_seqlens_q=max_cu_seqlens_q,
                active_total_q=q_capacity,
                window_left=self.window_left,
            )
            return plan

        self._create_extend_plan = _create_extend_plan
        decode_scratch_envelopes = {
            page_size: plan_decode_graph_scratch_envelope(
                device=self.device,
                q_dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.output_head_size,
                page_size=page_size,
                max_batch=max_num_seqs,
                max_page_table_width=max_page_table_widths[page_size],
                max_cache_page_count=max_page_table_widths[page_size],
                window_left=self.window_left,
                max_work_items=decode_work_items_limit,
                max_partial_rows=decode_partial_rows_limit,
                copy_runtime_metadata=True,
            )
            for page_size in _B12X_SUPPORTED_PAGE_SIZES
        }
        self._decode_plans: dict[tuple[int, int], Any] = {}
        self._verify_plans: dict[tuple[int, int], Any] = {}
        self._extend_plans: dict[tuple[int, int, int], Any] = {}
        for page_size in _B12X_SUPPORTED_PAGE_SIZES:
            for batch_size in sorted(decode_plan_sizes):
                self._decode_plans[page_size, batch_size] = self._create_decode_plan(
                    page_size, batch_size
                )
            if self._verify_q_per_req > 1:
                for batch_size in range(1, max_num_seqs + 1):
                    self._verify_plans[page_size, batch_size] = (
                        self._create_verify_plan(page_size, batch_size)
                    )
            for batch_size in range(1, max_num_seqs + 1):
                for q_capacity in self._extend_q_capacities:
                    # Equal capacity is necessarily one query token per
                    # request, which is handled by the decode plan.
                    if batch_size >= q_capacity:
                        continue
                    self._extend_plans[page_size, batch_size, q_capacity] = (
                        self._create_extend_plan(
                            page_size,
                            batch_size,
                            q_capacity,
                        )
                    )
        self._scratch_nbytes = max(
            *(int(envelope.nbytes) for envelope in decode_scratch_envelopes.values()),
            *(int(plan.layout.nbytes) for plan in self._verify_plans.values()),
            *(int(plan.layout.nbytes) for plan in self._extend_plans.values()),
        )

        current_workspace_manager().get_simultaneous(
            ((self._scratch_nbytes,), torch.uint8),
        )

        self.supports_quant_query_input = False
        register_cutedsl_warmup_provider(self)

        logger.info_once(
            "Using B12X_ATTN with q_heads=%d kv_heads=%d head_dim_qk=%d "
            "head_dim_vo=%d window_left=%d planned_page_sizes=%s "
            "verify_q_per_req=%d extend_q_capacities=%s scratch=%d bytes.",
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.output_head_size,
            self.window_left,
            _B12X_SUPPORTED_PAGE_SIZES,
            self._verify_q_per_req,
            self._extend_q_capacities,
            self._scratch_nbytes,
        )

    def _compile_paged_extend_entry(self, page_size: int) -> None:
        """Compile fixed-capacity paged-prefill entries without a live plan."""
        warmup_plans: list[tuple[int, int, Any, bool]] = []
        for batch_size in range(1, self._max_num_seqs + 1):
            candidates = sorted(
                (q_capacity, plan)
                for (plan_page_size, plan_batch, q_capacity), plan in (
                    self._extend_plans.items()
                )
                if plan_page_size == page_size and plan_batch == batch_size
            )
            for index, (q_capacity, plan) in enumerate(candidates):
                warmup_plans.append((batch_size, q_capacity, plan, index == 0))
        if not warmup_plans:
            return

        max_q_rows = max(
            min(q_capacity, max(64, batch_size + 1))
            for batch_size, q_capacity, _, _ in warmup_plans
        )
        q = torch.zeros(
            (max_q_rows, self.num_heads, self.head_size),
            dtype=self.dtype,
            device=self.device,
        )
        output = torch.zeros(
            (max_q_rows, self.num_heads, self.output_head_size),
            dtype=self.dtype,
            device=self.device,
        )
        kv_cache = torch.zeros(
            (1, 2, page_size, self.num_kv_heads, self.head_size),
            dtype=self.kv_torch_dtype,
            device=self.device,
        )
        key_cache, value_cache = self._kv_cache_views(kv_cache)
        sinks = self._prepare_sinks(self.sinks, self.device)
        (scratch_storage,) = current_workspace_manager().get_simultaneous(
            ((self._scratch_nbytes,), torch.uint8),
        )
        for batch_size, q_capacity, plan, execute in warmup_plans:
            q_rows = min(q_capacity, max(64, batch_size + 1))
            page_table = torch.zeros(
                (batch_size, plan.caps.max_page_table_width),
                dtype=torch.int32,
                device=self.device,
            )
            cache_seqlens = torch.full(
                (batch_size,), page_size, dtype=torch.int32, device=self.device
            )
            cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=torch.int32,
                device=self.device,
            )
            cu_seqlens_q[-1] = q_rows
            k_descale = None
            v_descale = None
            if _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
                k_descale = torch.ones(
                    (), dtype=torch.float32, device=self.device
                ).expand(batch_size)
                v_descale = torch.ones(
                    (), dtype=torch.float32, device=self.device
                ).expand(batch_size)
            binding = plan.bind(
                scratch=scratch_storage,
                q=q[:q_rows],
                k_cache=key_cache,
                v_cache=value_cache,
                output=output[:q_rows],
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                window_left=self.window_left,
                attention_sink_bias=sinks,
                k_descale=k_descale,
                v_descale=v_descale,
            )
            self._compile_paged_attention(binding=binding)
            if execute:
                # Compile-only warmup does not launch the device-side compact
                # scheduler. One execution per batch covers the capture-static
                # metadata variant shared by its Q-capacity plans.
                self._paged_attention_forward(binding=binding)

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        common_key = (
            "b12x_paged_extend",
            str(self.device),
            str(self.dtype),
            str(self.kv_torch_dtype),
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.output_head_size,
            self.window_left,
            self.sinks is not None,
        )
        return tuple(
            CuTeDSLCompileUnit(
                name="b12x_paged_extend",
                key=(*common_key, page_size),
                compile=partial(self._compile_paged_extend_entry, page_size),
            )
            for page_size in _B12X_SUPPORTED_PAGE_SIZES
        )

    def _prepare_sinks(
        self,
        sinks: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if sinks is None:
            return None
        if sinks.device != device:
            raise RuntimeError(
                "B12X_ATTN sinks must be on the same CUDA device as query."
            )
        if sinks.dtype == torch.float32 and sinks.is_contiguous():
            return sinks
        key = (
            int(sinks.data_ptr()),
            tuple(sinks.shape),
            tuple(sinks.stride()),
            str(sinks.dtype),
            str(sinks.device),
        )
        cached = self._sinks_cache.get(key)
        if cached is not None:
            cached.copy_(sinks)
            return cached
        if _capture_alloc_forbidden():
            raise RuntimeError(
                "B12X_ATTN would convert attention sinks during CUDA graph "
                "capture. Warm the layer eagerly or store sinks as contiguous "
                "float32."
            )
        cached = sinks.to(dtype=torch.float32, device=device).contiguous()
        self._sinks_cache[key] = cached
        return cached

    def _prepare_fp8_descales(
        self,
        layer: AttentionLayer,
        num_reqs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            return None, None
        if num_reqs <= 0:
            raise ValueError("B12X_ATTN fp8 KV descale request count must be positive.")

        def _prepare(scale: torch.Tensor, name: str) -> torch.Tensor:
            if scale.device != device:
                raise RuntimeError(f"B12X_ATTN {name} must be on the query device.")
            if scale.dtype != torch.float32:
                raise RuntimeError(f"B12X_ATTN {name} must be float32.")
            if scale.ndim == 0:
                return scale.expand(num_reqs)
            if scale.ndim == 1:
                if int(scale.shape[0]) == 1:
                    return scale.expand(num_reqs)
                if int(scale.shape[0]) >= num_reqs:
                    return scale[:num_reqs]
            raise ValueError(
                f"B12X_ATTN {name} must be scalar or rank-1 with at least "
                f"{num_reqs} values; got shape {tuple(scale.shape)}."
            )

        return _prepare(layer._k_scale, "k_scale"), _prepare(layer._v_scale, "v_scale")

    def _kv_cache_views(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            fp8_dtype = current_platform.fp8_dtype()
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(fp8_dtype)
            if value_cache.dtype == torch.uint8:
                value_cache = value_cache.view(fp8_dtype)
        if (
            key_cache.dtype != self.kv_torch_dtype
            or value_cache.dtype != self.kv_torch_dtype
        ):
            raise TypeError(
                f"B12X_ATTN plan expects KV dtype {self.kv_torch_dtype}, got "
                f"{key_cache.dtype}/{value_cache.dtype}."
            )
        return key_cache, value_cache

    def _select_plan(
        self,
        attn_metadata: B12XPagedMetadata,
        total_q: int,
        q_capacity: int,
        num_reqs: int,
        page_size: int,
    ) -> Any:
        if attn_metadata.max_query_len <= 1 and int(total_q) == int(num_reqs):
            batch_size = int(total_q)
            plan_key = (page_size, batch_size)
            plan = self._decode_plans.get(plan_key)
            if plan is None:
                if _capture_alloc_forbidden():
                    raise RuntimeError(
                        "B12X_ATTN decode plan was not prepared before CUDA graph "
                        f"capture for page size {page_size}, batch size "
                        f"{batch_size}."
                    )
                plan = self._create_decode_plan(page_size, batch_size)
                if int(plan.layout.nbytes) > self._scratch_nbytes:
                    raise RuntimeError(
                        "B12X_ATTN lazily created decode plan exceeds reserved "
                        f"scratch: {int(plan.layout.nbytes)} > "
                        f"{self._scratch_nbytes} bytes."
                    )
                self._decode_plans[plan_key] = plan
            return plan
        elif (
            self._verify_q_per_req > 1
            and attn_metadata.max_query_len == self._verify_q_per_req
            and int(total_q) == int(num_reqs) * self._verify_q_per_req
        ):
            plan_key = (page_size, int(num_reqs))
            plan = self._verify_plans.get(plan_key)
            if plan is None:
                if _capture_alloc_forbidden():
                    raise RuntimeError(
                        "B12X_ATTN verifier plan was not prepared before CUDA "
                        f"graph capture for page size {page_size}, batch size "
                        f"{num_reqs}."
                    )
                plan = self._create_verify_plan(page_size, int(num_reqs))
                if int(plan.layout.nbytes) > self._scratch_nbytes:
                    raise RuntimeError(
                        "B12X_ATTN lazily created verifier plan exceeds reserved "
                        f"scratch: {int(plan.layout.nbytes)} > "
                        f"{self._scratch_nbytes} bytes."
                    )
                self._verify_plans[plan_key] = plan
            return plan
        extend_q_capacity = next(
            (
                capacity
                for capacity in self._extend_q_capacities
                if q_capacity <= capacity
            ),
            None,
        )
        if extend_q_capacity is None:
            raise ValueError(
                f"B12X_ATTN extend Q capacity {q_capacity} exceeds prepared "
                f"maximum {self._extend_q_capacities[-1]}."
            )
        return self._extend_plans[
            page_size,
            int(num_reqs),
            extend_q_capacity,
        ]

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: B12XPagedMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "B12X_ATTN does not support fused output quantization."
            )
        if attn_metadata is None:
            return output.fill_(0)
        if output.shape[-1] != self.output_head_size:
            raise ValueError(
                f"B12X_ATTN expected output head dim {self.output_head_size}, got "
                f"{output.shape[-1]}."
            )
        if kv_cache.numel() == 0:
            return output.fill_(0)

        # In FULL cudagraph mode vLLM may pad attention metadata to the graph
        # bucket while still passing per-layer Q/output tensors with only the
        # real rows. Use tensor capacity as the launch contract and avoid
        # selecting decode graph replay for padded virtual requests.
        q_capacity = min(
            int(query.shape[0]),
            int(output.shape[0]),
        )
        num_actual_tokens = min(
            int(attn_metadata.num_actual_tokens),
            q_capacity,
        )
        if num_actual_tokens <= 0:
            return output
        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]
        if q.dtype != self.dtype or out.dtype != self.dtype:
            raise TypeError(
                f"B12X_ATTN plan expects dtype {self.dtype}, got "
                f"q={q.dtype}, output={out.dtype}."
            )

        key_cache, value_cache = self._kv_cache_views(kv_cache)
        page_size = _kv_page_size(key_cache, value_cache)
        if not attn_metadata.causal:
            raise NotImplementedError("B12X_ATTN supports causal attention only.")

        page_table = _ensure_i32_contiguous(attn_metadata.block_table, "block_table")
        cache_seqlens = _ensure_i32_contiguous(attn_metadata.seq_lens, "seq_lens")
        cu_seqlens_q = _ensure_i32_contiguous(
            attn_metadata.query_start_loc,
            "query_start_loc",
        )
        num_reqs = int(cache_seqlens.shape[0])
        if attn_metadata.max_query_len <= 1 and num_actual_tokens < num_reqs:
            num_reqs = num_actual_tokens
            page_table = page_table[:num_reqs]
            cache_seqlens = cache_seqlens[:num_reqs]
            cu_seqlens_q = cu_seqlens_q[: num_reqs + 1]
        sinks = self._prepare_sinks(self.sinks, q.device)
        k_descale, v_descale = self._prepare_fp8_descales(
            layer,
            num_reqs,
            q.device,
        )
        plan = self._select_plan(
            attn_metadata,
            num_actual_tokens,
            q_capacity,
            num_reqs,
            page_size,
        )
        (scratch_storage,) = current_workspace_manager().get_simultaneous(
            ((self._scratch_nbytes,), torch.uint8),
        )
        binding = plan.bind(
            scratch=scratch_storage,
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            output=out,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            window_left=self.window_left,
            active_total_q=(None if plan.caps.mode == "extend" else num_actual_tokens),
            attention_sink_bias=sinks,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        self._paged_attention_forward(binding=binding)
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

        key_cache, value_cache = kv_cache.unbind(1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
