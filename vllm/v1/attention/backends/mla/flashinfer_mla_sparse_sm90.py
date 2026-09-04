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
bakes the per-row ``kv_len`` into its int schedule
at plan() time — ``run()`` never reads the device-side buffer — so the
metadata builder replans every step (outside capture) with exact host-side
lengths derived from the batch's sequence lengths; a full-width schedule
would send the kernel past each row's valid count into the -1 tail of the
converted index buffer (illegal address). Per-step content (top-k slots)
is written into the reserved buffers by kernels inside the captured
forward, and captured runs read the refreshed plan buffers on replay.

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
        # User-reserved buffers: with use_cuda_graph=True plan() refreshes
        # these in place, so run()'s captured kernels always read them.
        self.kv_indices = torch.zeros(
            max_tokens * topk_width, dtype=torch.int32, device=device
        )
        self.kv_len_arr = torch.full(
            (max_tokens,), topk_width, dtype=torch.int32, device=device
        )
        self.wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace,
            qo_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device),
            kv_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device),
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

    def plan(self, num_tokens: int, kv_lens: torch.Tensor) -> None:
        """Replan with exact per-row KV lengths (CPU int32, ``[num_tokens]``).

        The wrapper bakes kv_len into its int schedule from host values;
        run() never reads the device kv_len_arr buffer. Scheduling with the
        full buffer width while kv_indices rows carry a -1 tail past each
        row's valid count makes the kernel compute ``-1 * ckv_stride_page``
        (illegal address), so the lengths must be exact at plan time and
        replanned every step as contexts grow. Must run outside CUDA graph
        capture: the in-place refreshed plan_info/indptr buffers are what
        captured runs read.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInferMLASparseSM90 plan() called inside CUDA graph "
                "capture; lengths must be planned host-side before capture."
            )
        # CPU staging buffers are filled in place: plan() runs per step
        # (once per draft/verify metadata build), so per-call allocations and
        # device round trips are on the hot path. Passing CPU tensors lets
        # the wrapper's internal .to("cpu") no-op; its reserved-buffer
        # copy_ then performs the single H2D transfer per tensor.
        # use_cuda_graph=True makes the wrapper copy qo/kv indptr into its
        # fixed (max_tokens+1)-sized buffers with exact-size copy_, so the
        # indptr must always be full-size. Rows past num_tokens are padded
        # empty (qo_indptr flat at num_tokens) — zero-query rows read no q
        # and schedule no work. Padded rows keep the full width lens; the
        # value is never dereferenced.
        torch.clamp(self._arange_cpu, max=num_tokens, out=self._qo_cpu)
        torch.mul(self._qo_cpu, self.topk_width, out=self._kv_cpu)
        self._lens_cpu.fill_(self.topk_width)
        self._lens_cpu[:num_tokens] = kv_lens.to(torch.int32)
        self.wrapper.plan(
            self._qo_cpu,
            self._kv_cpu,
            self.kv_indices,
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
        impl = attention_layer.impl
        if not isinstance(impl, FlashInferMLASparseSM90Impl):
            raise TypeError(
                "FlashInferMLASparseSM90Builder requires an SM90 FlashInfer "
                f"implementation, got {type(impl).__name__}."
            )
        topk_indices_buffer = impl.topk_indices_buffer
        assert topk_indices_buffer is not None
        self.state = _SM90State(
            device,
            impl.num_heads,
            kv_cache_spec.dtype,
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_indices_buffer.shape[1],
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            sm_scale=impl.scale,
        )
        # seq_lens_cpu_upper_bound is optimistic on decode rows under async
        # spec decode, so the fast sync-free path is only safe without it;
        # under async scheduling the exact (device) positions are used at
        # the cost of one D2H sync per metadata build.
        self._async_scheduling = bool(vllm_config.scheduler_config.async_scheduling)
        hf_config = vllm_config.model_config.hf_text_config
        assert hf_config.index_topk is not None
        self._index_topk = int(hf_config.index_topk)
        self._index_kpool = int(kv_cache_spec.tokens_per_state)

    def _kv_lens_host(self, cam: CommonAttentionMetadata) -> tuple[int, torch.Tensor]:
        """Exact per-row KV lengths, host-side (the flashinfer wrapper bakes
        them into its schedule at plan time; there is no device-side path).

        A row for the j-th query token of request i attends
        ``seq_lens[i] - q_len[i] + j + 1`` tokens. The indexer's selection
        then bounds the valid count: contexts up to ``index_topk`` select
        everything (valid == context); longer contexts keep the top
        ``index_topk`` pool-expanded tokens plus the trailing incomplete
        pool (valid == ``index_topk + context % index_kpool``). Both match
        the count of non -1 entries the convert kernel produces.
        """
        num_reqs = cam.num_reqs
        qsl = cam.query_start_loc_cpu[: num_reqs + 1]
        num_rows = int(qsl[-1])
        if num_rows == 0:
            return 0, torch.zeros(0, dtype=torch.int32)
        # Row context == position + 1. Without async scheduling the
        # maintained host upper bound is exact, giving a sync-free path;
        # under async scheduling it is optimistic on decode rows, so fall
        # back to the exact device positions (one D2H sync per build). The
        # seq_lens derivation equals position + 1 because positions are
        # contiguous per request.
        sl_host = cam.seq_lens_cpu_upper_bound
        positions = cam.positions
        if not self._async_scheduling and sl_host is not None:
            seq_lens = sl_host[:num_reqs].to(torch.int32)
            q_lens = qsl[1:] - qsl[:-1]
            first_pos = seq_lens - q_lens
            req_of_row = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int64), q_lens.to(torch.int64)
            )
            rows = torch.arange(num_rows, dtype=torch.int32)
            ctx = (
                first_pos.to(torch.int64)[req_of_row]
                + rows.to(torch.int64)
                - qsl.to(torch.int64)[req_of_row]
                + 1
            )
        elif positions is not None and num_rows <= positions.shape[0]:
            ctx = positions[:num_rows].cpu().to(torch.int64) + 1
        else:
            seq_lens = cam.seq_lens[:num_reqs].cpu().to(torch.int32)
            q_lens = qsl[1:] - qsl[:-1]
            first_pos = seq_lens - q_lens
            req_of_row = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int64), q_lens.to(torch.int64)
            )
            rows = torch.arange(num_rows, dtype=torch.int32)
            ctx = (
                first_pos.to(torch.int64)[req_of_row]
                + rows.to(torch.int64)
                - qsl.to(torch.int64)[req_of_row]
                + 1
            )
        topk = self._index_topk
        kpool = max(self._index_kpool, 1)
        lens = torch.where(ctx <= topk, ctx, topk + ctx % kpool)
        return num_rows, lens.to(torch.int32)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMLASparseSM90Metadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        assert isinstance(metadata, FlashInferMLASparseSM90Metadata)
        # Replan every step outside any CUDA graph capture with this step's
        # exact per-row lengths; captured runs read the refreshed buffers.
        num_rows, kv_lens = self._kv_lens_host(common_attn_metadata)
        self.state.plan(num_rows, kv_lens)
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
        # Refresh top-k rows in graph and clamp masked tails to a valid slot;
        # per-row lengths are already baked into the host-side plan.
        width = topk_slots.shape[1]
        state.kv_indices[: num_tokens * width].copy_(
            topk_slots.reshape(-1).clamp_(min=0).to(torch.int32)
        )

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
