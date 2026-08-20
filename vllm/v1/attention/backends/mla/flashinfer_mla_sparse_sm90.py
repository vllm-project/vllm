# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA backend for SM90 (Hopper) NoPE models.

Wraps FlashInfer's ``BatchMLAPagedAttentionWrapper`` (FA2/FA3 paths), which
as of FlashInfer 0.6.18 supports ``head_dim_kpe=0`` (GLM-5-Next's NoPE MLA)
and FP8 E4M3 KV caches on SM90 with in-kernel dequantization: the FP8 cache
is read directly (half the bf16 HBM traffic) and converted to BF16 in shared
memory, while queries stay BF16 (no query quantization).

Sparsity rides the same trick the FA-based sparse backend uses: with
``page_size=1`` the per-token top-k slot indices ARE the page table, so each
query token becomes one varlen batch row whose ``kv_indices`` slice is its
top-k row and whose ``kv_len`` is its valid count. Causality is already
encoded by the indexer's selection, so ``causal=False``.

CUDA-graph handling: ``plan()`` copies its inputs to host unconditionally,
so it must stay outside graph capture. A process-wide state object owns the
wrapper (created eagerly at impl construction with the model's head count
and KV dtype), reserved capture-stable device buffers, and the planned
shape. The metadata builder re-plans outside the graph when the batch shape
changes; per-step varying content (top-k slots, valid counts) is written
into the reserved buffers by kernels inside the captured forward, and the
kernel bounds each row by the device-side ``kv_len``. Planning uses the
constant worst-case length (the full top-k width), which is valid because
the schedule only sizes the work partitioning.

KV cache format: plain contiguous E4M3 ``[num_blocks, block_size, 512]``
(uint8 storage) with a per-tensor ``k_scale``; BF16 caches also work. The
per-token x 128-channel-group ``ckv_scale_arr`` layout is supported by the
kernel but not wired yet (it needs a group-quantizing cache-write op).
"""

from typing import Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType
from vllm.v1.worker.workspace import current_workspace_manager

_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")
_WORKSPACE_BYTES = 128 * 1024 * 1024


class FlashInferMLASparseSM90Backend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_SM90"

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseSM90Builder"]:
        return FlashInferMLASparseSM90Builder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return FlashInferMLASparseSM90Impl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512 = ckv 512 + kpe 0 (NoPE); 576 = ckv 512 + kpe 64.
        return [512, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

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
        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf = vllm_config.model_config.hf_text_config
            # The SM90 FA2/FA3 kernel covers ckv=512 with kpe in {0, 64}
            # (NoPE models and DeepSeek-style rope MLA alike).
            if getattr(hf, "kv_lora_rank", 512) != 512:
                return "FLASHINFER_MLA_SPARSE_SM90 requires kv_lora_rank=512"
            if getattr(hf, "qk_rope_head_dim", 0) not in (0, 64):
                return "FLASHINFER_MLA_SPARSE_SM90 requires qk_rope_head_dim in (0, 64)"
            if not hasattr(hf, "index_topk"):
                return "FLASHINFER_MLA_SPARSE_SM90 requires a sparse model"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return "HND"


class _SM90State:
    """Process-wide wrapper, capture-stable buffers, and plan parameters.

    One instance serves every MLA layer: the plan depends only on the batch
    shape, not the layer. Created eagerly at the first impl construction
    (always before any graph capture), so the head count and KV dtype are
    known when planning.
    """

    def __init__(
        self,
        device: torch.device,
        num_heads: int,
        kv_dtype: torch.dtype,
        max_tokens: int,
        topk_width: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        sm_scale: float,
    ) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        (float_workspace,) = current_workspace_manager().get_simultaneous(
            ((_WORKSPACE_BYTES,), torch.uint8),
        )
        self.device = device
        self.num_heads = num_heads
        self.kv_dtype = kv_dtype
        self.max_tokens = max_tokens
        self.topk_width = topk_width
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.sm_scale = sm_scale
        # User-reserved buffers: with use_cuda_graph=True plan() refreshes
        # these in place, so run()'s captured kernels always read them.
        self.kv_indices = torch.zeros(
            max_tokens * topk_width, dtype=torch.int32, device=device
        )
        self.kv_len_arr = torch.full(
            (max_tokens,), topk_width, dtype=torch.int32, device=device
        )
        self.wrapper = BatchMLAPagedAttentionWrapper(
            float_workspace,
            qo_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device),
            kv_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device),
            kv_indices=self.kv_indices,
            kv_len_arr=self.kv_len_arr,
            use_cuda_graph=True,
            backend="fa3",
        )
        self.planned_shape: tuple[int, int] | None = None

    def plan(self, num_tokens: int, topk_width: int) -> None:
        """Plan outside graph capture; CPU indptrs keep plan() sync-free."""
        # Callers disagree on topk_width: the builder passes
        # metadata.topk_tokens (index_topk) while forward_mqa passes the
        # topk buffer's actual row width (kpool-widened). kv_indices rows
        # are always spaced by the buffer width, so normalize to it —
        # otherwise the two call sites never share the planned_shape cache
        # (forcing a replan inside CUDA graph capture) and the schedule is
        # built with the wrong row stride.
        topk_width = self.topk_width
        if self.planned_shape == (num_tokens, topk_width):
            return
        # use_cuda_graph=True makes the wrapper copy qo/kv indptr into its
        # fixed (max_tokens+1)-sized buffers with exact-size copy_, so the
        # indptr must always be full-size. Rows past num_tokens are padded
        # empty (qo_indptr flat at num_tokens) — zero-query rows read no q
        # and schedule no work.
        qo = torch.arange(self.max_tokens + 1, dtype=torch.int32).clamp_(max=num_tokens)
        kv = qo * topk_width
        self.wrapper.plan(
            qo.to(self.device),
            kv.to(self.device),
            self.kv_indices,
            # Worst case: the schedule only sizes work partitioning; actual
            # per-row lengths are refreshed inside the captured graph.
            self.kv_len_arr,
            self.num_heads,
            self.kv_lora_rank,  # head_dim_ckv
            self.qk_rope_head_dim,  # 0 (NoPE) or 64 (rope MLA)
            1,  # page_size: top-k slots are the page table
            False,  # causal: encoded by the indexer's selection
            self.sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=self.kv_dtype,
        )
        self.planned_shape = (num_tokens, topk_width)


_SM90_STATE: _SM90State | None = None


def _get_sm90_state() -> "_SM90State | None":
    return _SM90_STATE


class FlashInferMLASparseSM90Builder(FlashInferMLASparseMetadataBuilder):
    """Reuse the common sparse metadata (req ids, topk buffer access)."""

    metadata_cls = FlashInferMLASparseMetadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # Re-plan outside any CUDA graph capture when the batch shape
        # changes; capture and replay hit the planned-shape cache.
        if _SM90_STATE is not None:
            _SM90_STATE.plan(metadata.num_actual_tokens, metadata.topk_tokens)
        return metadata


class FlashInferMLASparseSM90Impl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        global _SM90_STATE
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FlashInferMLASparseSM90Impl does not support alibi, sliding "
                "window, or logits soft cap."
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
        assert self.topk_indices_buffer is not None
        self.supports_quant_query_input = False
        self.use_fp8_kv_cache = self.kv_cache_dtype in _FP8_KV_DTYPES
        if _SM90_STATE is None:
            from vllm.config import get_current_vllm_config

            assert topk_indices_buffer is not None
            max_tokens = (
                get_current_vllm_config().scheduler_config.max_num_batched_tokens
            )
            _SM90_STATE = _SM90State(
                topk_indices_buffer.device,
                num_heads,
                (torch.float8_e4m3fn if self.use_fp8_kv_cache else torch.bfloat16),
                max_tokens,
                topk_indices_buffer.shape[1],
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                sm_scale=self.scale,
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "FlashInferMLASparseSM90Impl expects split (q_nope, q_rope)."
            )
        q_nope, q_rope = q
        num_tokens = q_rope.shape[0]
        # NoPE models hand a zero-width rope tensor through; rope MLA hands
        # the real 64-dim part. The kernel takes both as-is.
        q_pe = q_rope.reshape(num_tokens, self.num_heads, self.qk_rope_head_dim)

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        topk_slots, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        state = _SM90_STATE
        assert state is not None
        # Refresh the reserved buffers' contents with in-graph device copies
        # so captured runs observe this step's top-k rows and valid counts.
        width = topk_slots.shape[1]
        state.kv_indices[: num_tokens * width].copy_(
            topk_slots.reshape(-1).to(torch.int32)
        )
        state.kv_len_arr[:num_tokens].copy_(valid_counts.to(torch.int32))
        state.plan(num_tokens, width)

        flat = (
            kv_c_and_k_pe_cache.view(torch.float8_e4m3fn)
            if self.use_fp8_kv_cache
            else kv_c_and_k_pe_cache
        ).reshape(-1, 1, self.head_size)
        ckv = flat[..., : self.kv_lora_rank]
        kpe = flat[..., self.kv_lora_rank :]

        scale_kwargs = (
            {"ckv_scale": float(layer._k_scale_float or 1.0), "kpe_scale": 1.0}
            if self.use_fp8_kv_cache
            else {}
        )
        out = state.wrapper.run(q_nope, q_pe, ckv, kpe, **scale_kwargs)
        return out, None
