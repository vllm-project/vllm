# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA backend for SM90 (Hopper) NoPE models.

Wraps FlashInfer's ``BatchMLAPagedAttentionWrapper`` (FA2/FA3 paths), which
as of FlashInfer 0.6.18 supports ``head_dim_kpe=0`` (GLM-5.3-Flash NoPE MLA)
and FP8 E4M3 KV caches on SM90 with in-kernel dequantization: the FP8 cache
is read directly (half the bf16 HBM traffic) and converted to BF16 in shared
memory, while queries stay BF16 (no query quantization).

Sparsity rides the same trick the FA-based sparse backend uses: with
``page_size=1`` the per-token top-k slot indices ARE the page table, so each
query token becomes one varlen batch row whose ``kv_indices`` slice is its
top-k row and whose ``kv_len`` is its valid count. Causality is already
encoded by the indexer's selection, so ``causal=False``.

CUDA-graph handling: ``plan()`` copies its inputs to host unconditionally,
so it must stay outside graph capture. Each metadata builder owns a wrapper,
reserved capture-stable device buffers, and the plan parameters. The wrapper
bakes the per-row ``kv_len`` into its int schedule at plan() time — ``run()``
never reads the device-side buffer. The builder plans (outside capture) from
sync-free host upper bounds and clamps each scheduled work item's ``kv_end``
on device to the exact valid count, so the kernel never reads past a row's
valid prefix (the -1 tail of the converted index buffer) and async scheduling
needs no D2H sync. Per-step content (top-k slots) is written into the
reserved buffers by kernels inside the captured forward, and captured runs
read the refreshed plan buffers on replay.

KV cache format: plain contiguous E4M3 ``[num_blocks, block_size, 512]``
(uint8 storage) with a per-tensor ``k_scale``; BF16 caches also work. The
per-token x 128-channel-group ``ckv_scale_arr`` layout is supported by the
kernel but not wired yet (it needs a group-quantizing cache-write op).
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
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
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")
_WORKSPACE_BYTES = 128 * 1024 * 1024
# FlashInfer MLAPlanInfo layout (scheduler.cuh) field indices.
_PLAN_INFO_LEN = 18
_PI_NUM_BLKS_Y = 1
_PI_Q_INDPTR = 2
_PI_KV_INDPTR = 3
_PI_KV_START = 13
_PI_KV_END = 14
_PI_WORK_INDPTR = 15
# Planned lengths exceed the host upper bound by up to this many tokens so a
# plan stays reusable across draft steps and consecutive decode steps; the
# device-side clamp restores the exact lengths.
_PLAN_SLACK = 32


@triton.jit
def _pack_topk_indices(
    slots,
    indptr,
    indices,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    start = tl.load(indptr + row)
    end = tl.load(indptr + row + 1)
    mask = (cols < end - start) & (cols < WIDTH)
    values = tl.load(slots + row * WIDTH + cols, mask=mask, other=0)
    tl.store(indices + start + cols, tl.maximum(values, 0), mask=mask)


@triton.jit
def _clamp_work_kv_end(
    int_ws,
    saved_kv_end,
    seq_lens,
    query_start_loc,
    req_id_per_token,
    num_works_idx,
    q_indptr_off,
    kv_start_off,
    kv_end_off,
    TOPK: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Clamp each planned work item to its row's exact valid count, derived
    # from the device batch layout exactly as the convert kernel counts it.
    works = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = works < tl.load(int_ws + num_works_idx)
    row = tl.load(int_ws + q_indptr_off + works, mask=mask, other=0)
    req = tl.load(req_id_per_token + row, mask=mask, other=0)
    seq_len = tl.load(seq_lens + req, mask=mask, other=0)
    query_end = tl.load(query_start_loc + req + 1, mask=mask, other=0)
    ctx = seq_len - query_end + row + 1
    valid = tl.where(ctx <= TOPK, ctx, TOPK + ctx % KPOOL)
    start = tl.load(int_ws + kv_start_off + works, mask=mask, other=0)
    end = tl.load(saved_kv_end + works, mask=mask, other=0)
    tl.store(
        int_ws + kv_end_off + works,
        tl.maximum(tl.minimum(end, valid), start),
        mask=mask,
    )


class FlashInferMLASparseSM90Backend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
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
        if not use_sparse:
            return "FLASHINFER_MLA_SPARSE_SM90 requires sparse MLA"
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf = vllm_config.model_config.hf_text_config
            # The SM90 FA2/FA3 kernel covers ckv=512 with kpe in {0, 64}
            # (NoPE models and DeepSeek-style rope MLA alike).
            if hf.kv_lora_rank != 512:
                return "FLASHINFER_MLA_SPARSE_SM90 requires kv_lora_rank=512"
            if hf.qk_rope_head_dim not in (0, 64):
                return "FLASHINFER_MLA_SPARSE_SM90 requires qk_rope_head_dim in (0, 64)"
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
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)


class _SM90State:
    """Builder-owned wrapper, capture-stable buffers, and plan parameters.

    One instance serves every MLA layer in an attention group because the plan
    depends only on the batch shape, not the layer.
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
        index_topk: int,
        index_kpool: int,
    ) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        self.workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.device = device
        self.num_heads = num_heads
        self.kv_dtype = kv_dtype
        self.max_tokens = max_tokens
        self.topk_width = topk_width
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.sm_scale = sm_scale
        self.index_topk = index_topk
        self.index_kpool = max(index_kpool, 1)
        # Largest valid count the convert kernel can produce for any context.
        self.max_valid = min(self.index_topk + self.index_kpool - 1, topk_width)
        # User-reserved buffers: with use_cuda_graph=True plan() refreshes
        # these in place, so run()'s captured kernels always read them.
        self.kv_indices = torch.zeros(
            max_tokens * topk_width, dtype=torch.int32, device=device
        )
        self.kv_len_arr = torch.full(
            (max_tokens,), topk_width, dtype=torch.int32, device=device
        )
        self.kv_indptr = torch.zeros(max_tokens + 1, dtype=torch.int32, device=device)
        self.wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace,
            qo_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device),
            kv_indptr=self.kv_indptr,
            kv_indices=self.kv_indices,
            kv_len_arr=self.kv_len_arr,
            use_cuda_graph=True,
            backend="fa3",
        )
        self._arange_cpu = torch.arange(self.max_tokens + 1, dtype=torch.int32)
        self._qo_cpu = torch.empty(self.max_tokens + 1, dtype=torch.int32)
        self._kv_cpu = torch.empty(self.max_tokens + 1, dtype=torch.int32)
        self._lens_cpu = torch.full(
            (self.max_tokens,), self.topk_width, dtype=torch.int32
        )
        # Planned per-work kv_end of the current plan, restored by each clamp;
        # sized on first plan to the planner's work capacity.
        self._saved_kv_end = torch.empty(0, dtype=torch.int32, device=device)
        self._planned_num_tokens = -1
        self._int_ws: torch.Tensor | None = None
        self._plan_offsets: tuple[int, int, int, int] | None = None

    def plan(
        self,
        num_tokens: int,
        kv_lens: torch.Tensor,
        cam: CommonAttentionMetadata | None,
        req_id_per_token: torch.Tensor | None,
    ) -> None:
        """Plan per-row KV lengths (CPU int32, ``[num_tokens]``).

        The wrapper bakes kv_len into its int schedule from host values, so
        rows past their valid count would send the kernel into the -1 tail of
        the converted index buffer. With ``cam`` and ``req_id_per_token``
        (device batch layout), ``kv_lens`` only needs to upper-bound the
        valid counts: the plan is padded by ``_PLAN_SLACK`` and reused while
        the bound still fits, and every call clamps each work item's kv_end
        on device to the exact count, so no D2H sync is needed. Without them
        the lengths must be exact. Must run outside CUDA graph capture: the
        in-place refreshed plan_info/indptr buffers are what captured runs
        read.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInferMLASparseSM90 plan() called inside CUDA graph "
                "capture; lengths must be planned host-side before capture."
            )
        kv_lens = kv_lens.to(torch.int32)
        if cam is None or req_id_per_token is None:
            self._plan(num_tokens, kv_lens)
            self._planned_num_tokens = -1
            return
        planned = self._lens_cpu[:num_tokens]
        if num_tokens != self._planned_num_tokens or not bool(
            ((planned >= kv_lens) & (planned <= kv_lens + _PLAN_SLACK)).all()
        ):
            self._plan(
                num_tokens, torch.clamp(kv_lens + _PLAN_SLACK, max=self.max_valid)
            )
            self._planned_num_tokens = num_tokens
        self._clamp_kv_end(cam, req_id_per_token)

    def _plan(self, num_tokens: int, kv_lens: torch.Tensor) -> None:
        # CPU staging buffers are filled in place: plan() runs per step
        # (once per draft/verify metadata build), so per-call allocations and
        # device round trips are on the hot path. Passing CPU tensors lets
        # the wrapper's internal .to("cpu") no-op; its reserved-buffer
        # copy_ then performs the single H2D transfer per tensor.
        # use_cuda_graph=True makes the wrapper copy qo/kv indptr into its
        # fixed (max_tokens+1)-sized buffers with exact-size copy_, so the
        # indptr must always be full-size. Rows past num_tokens are padded
        # empty (qo_indptr flat at num_tokens) — zero-query rows read no q
        # and schedule no work. FlashInfer 0.7 requires packed CSR offsets
        # matching the exact lengths, including zero-length padded rows.
        torch.clamp(self._arange_cpu, max=num_tokens, out=self._qo_cpu)
        self._lens_cpu.zero_()
        self._lens_cpu[:num_tokens] = kv_lens
        self._kv_cpu[0] = 0
        torch.cumsum(self._lens_cpu, dim=0, out=self._kv_cpu[1:])
        # Only the live prefix of kv_indices: the wrapper snapshots and
        # copies whatever is passed.
        num_indices = max(int(self._kv_cpu[-1]), 1)
        self.wrapper.plan(
            self._qo_cpu,
            self._kv_cpu,
            self.kv_indices[:num_indices],
            self._lens_cpu,
            self.num_heads,
            self.kv_lora_rank,  # head_dim_ckv
            self.qk_rope_head_dim,  # 0 (NoPE) or 64 (rope MLA)
            1,  # page_size: top-k slots are the page table
            False,  # causal: encoded by the indexer's selection
            self.sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=self.kv_dtype,
        )
        plan_info = [int(x) for x in self.wrapper._plan_info]
        assert len(plan_info) == _PLAN_INFO_LEN, plan_info
        # Per-work arrays are laid out back to back at the planner's capacity.
        max_works = (plan_info[_PI_KV_INDPTR] - plan_info[_PI_Q_INDPTR]) // 4
        if self._saved_kv_end.numel() != max_works:
            self._saved_kv_end = torch.empty(
                max_works, dtype=torch.int32, device=self.device
            )
        self._int_ws = self.wrapper._planned_backend._int_workspace_buffer.view(
            torch.int32
        )
        self._plan_offsets = (
            plan_info[_PI_WORK_INDPTR] // 4 + plan_info[_PI_NUM_BLKS_Y],
            plan_info[_PI_Q_INDPTR] // 4,
            plan_info[_PI_KV_START] // 4,
            plan_info[_PI_KV_END] // 4,
        )
        kv_end_off = self._plan_offsets[3]
        self._saved_kv_end.copy_(self._int_ws[kv_end_off : kv_end_off + max_works])

    def _clamp_kv_end(
        self, cam: CommonAttentionMetadata, req_id_per_token: torch.Tensor
    ) -> None:
        assert self._int_ws is not None and self._plan_offsets is not None
        block = 1024
        _clamp_work_kv_end[(triton.cdiv(self._saved_kv_end.numel(), block),)](
            self._int_ws,
            self._saved_kv_end,
            cam.seq_lens,
            cam.query_start_loc,
            req_id_per_token,
            *self._plan_offsets,
            TOPK=self.index_topk,
            KPOOL=self.index_kpool,
            BLOCK_SIZE=block,
        )

    def pack_indices(self, topk_slots: torch.Tensor) -> None:
        num_tokens, width = topk_slots.shape
        _pack_topk_indices[(num_tokens, triton.cdiv(width, 256))](
            topk_slots, self.kv_indptr, self.kv_indices, width, 256
        )


@dataclass
class FlashInferMLASparseSM90Metadata(FlashInferMLASparseMetadata):
    state: _SM90State | None = None


class FlashInferMLASparseSM90Builder(FlashInferMLASparseMetadataBuilder):
    """Reuse the common sparse metadata (req ids, topk buffer access)."""

    metadata_cls = FlashInferMLASparseSM90Metadata

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        attention_layer = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ]
        self._attention_layer = attention_layer
        impl = attention_layer.impl
        if not isinstance(impl, FlashInferMLASparseSM90Impl):
            raise TypeError(
                "FlashInferMLASparseSM90Builder requires an SM90 FlashInfer "
                f"implementation, got {type(impl).__name__}."
            )
        topk_indices_buffer = impl.topk_indices_buffer
        assert topk_indices_buffer is not None
        hf_config = vllm_config.model_config.hf_text_config
        assert hf_config.index_topk is not None
        self.state = _SM90State(
            device,
            impl.num_heads,
            kv_cache_spec.dtype,
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_indices_buffer.shape[1],
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            sm_scale=impl.scale,
            index_topk=int(hf_config.index_topk),
            index_kpool=int(getattr(hf_config, "index_kpool", 1) or 1),
        )
        spec_config = vllm_config.speculative_config
        self._adaptive_verification = bool(
            spec_config is not None and spec_config.enable_adaptive_verification
        )

    def _kv_lens_host(
        self, cam: CommonAttentionMetadata
    ) -> tuple[int, torch.Tensor, bool]:
        """Host per-row KV lengths for the plan and whether they are exact.

        A row for the j-th query token of request i attends
        ``seq_lens[i] - q_len[i] + j + 1`` tokens. The indexer's selection
        then bounds the valid count: contexts up to ``index_topk`` select
        everything (valid == context); longer contexts keep the top
        ``index_topk`` pool-expanded tokens plus the trailing incomplete
        pool (valid == ``index_topk + context % index_kpool``). Both match
        the count of non -1 entries the convert kernel produces.

        With ``seq_lens_cpu_upper_bound`` (optimistic under async spec
        decode) returns sync-free upper bounds, which the plan clamps on
        device. Otherwise returns exact lengths via a D2H copy.
        """
        num_reqs = cam.num_reqs
        qsl = cam.query_start_loc_cpu[: num_reqs + 1]
        num_rows = int(qsl[-1])
        if num_rows == 0:
            return 0, torch.zeros(0, dtype=torch.int32), True
        state = self.state
        sl_host = cam.seq_lens_cpu_upper_bound
        if sl_host is None:
            ctx = self._row_contexts(qsl, cam.seq_lens[:num_reqs].cpu())
            topk, kpool = state.index_topk, state.index_kpool
            lens = torch.where(ctx <= topk, ctx, topk + ctx % kpool)
            return num_rows, lens.to(torch.int32), True
        if self._adaptive_verification:
            # The host query split may not match the device one, so bound
            # every row by the longest request.
            ctx = torch.full((num_rows,), int(sl_host[:num_reqs].max()))
        else:
            ctx = self._row_contexts(qsl, sl_host[:num_reqs])
        return num_rows, torch.clamp(ctx, max=state.max_valid), False

    @staticmethod
    def _row_contexts(qsl: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        num_reqs = qsl.shape[0] - 1
        q_lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
        first_ctx = seq_lens.to(torch.int64) - q_lens + 1
        req_of_row = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int64), q_lens
        )
        rows = torch.arange(int(qsl[-1]), dtype=torch.int64)
        return first_ctx[req_of_row] + rows - qsl.to(torch.int64)[req_of_row]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMLASparseSM90Metadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        assert isinstance(metadata, FlashInferMLASparseSM90Metadata)
        # Plan outside any CUDA graph capture; captured runs read the
        # refreshed buffers.
        num_rows, kv_lens, exact = self._kv_lens_host(common_attn_metadata)
        # MHA prefills are removed from q before forward_mqa. Plan only the
        # decode prefix in that case, using the same routing decision as MLA.
        if metadata.num_prefills > 0 and self._attention_layer._use_sparse_mha(
            metadata
        ):
            num_rows = metadata.num_decode_tokens
            kv_lens = kv_lens[:num_rows]
        if exact:
            self.state.plan(num_rows, kv_lens, None, None)
        else:
            self.state.plan(
                num_rows, kv_lens, common_attn_metadata, metadata.req_id_per_token
            )
        metadata.state = self.state
        return metadata


class FlashInferMLASparseSM90Impl(SparseMLACommonImpl[FlashInferMLASparseSM90Metadata]):
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

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseSM90Metadata,
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
        # return_valid_counts=True keeps the compacted-prefix layout: valid
        # entries at [0, valid_count), -1 past it — exactly the prefix the
        # planned per-row lengths address.
        topk_slots, _ = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        state = attn_metadata.state
        assert state is not None
        # Pack valid prefixes at the offsets refreshed by plan() before replay.
        state.pack_indices(topk_slots)

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
