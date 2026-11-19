# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink-capable FlashMLA sparse backend for HY V4 (NVIDIA).

HY V4 adds a per-head learnable attention sink on top of sparse MLA. The
vendored FlashMLA kernels already accept an `attn_sink` argument, but vLLM's
shared `FLASHMLA_SPARSE` backend neither advertises sink support nor forwards
the tensor, so the bias would be silently dropped.

This module supplies the missing wiring inside the model package, mirroring how
`vllm.models.deepseek_v4.nvidia.flashinfer_sparse` hands `sinks` to the
FlashInfer sparse MLA kernels: subclass the platform backend, declare
`supports_sink`, and thread the sink into every kernel call.

The subclass intentionally keeps the inherited `get_name()`
(`"FLASHMLA_SPARSE"`). Several shared code paths key off that exact string —
`_canonicalize_sparse_mla_kv_cache_dtype` promotes a quantized KV cache to
`fp8_ds_mla` for it, and `FlashMLASparseImpl` asserts that layout — so a new
name would silently change KV cache behaviour. This subclass also supplies
full-head sinks for ShardQ prefill.
"""

import math
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


class HYV4FlashMLASparseImpl(FlashMLASparseImpl):
    """FlashMLA sparse impl that applies HY V4's per-head learnable sink.

    The sink enters as the `sinks` impl kwarg of
    `vllm.model_executor.layers.attention.MLAAttention` and is consumed by the
    FlashMLA kernels, which fold it into the softmax denominator:
    `out *= exp(lse) / (exp(lse) + exp(sink))`.
    """

    # ShardQ computes every head for each rank's token slice.
    supports_shardq: ClassVar[bool] = True
    # The sink-aware kernel has not been validated with HiSparse's
    # host-backed prefill path; keep the combination closed.
    supports_shardq_with_hisparse: ClassVar[bool] = False

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
        # ``SparseMLACommonImpl`` takes explicit keyword arguments only, so the
        # sink has to be removed before the base classes see``mla_args``.
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
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
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        self._validate_sinks(sinks, num_heads)
        self.sinks = sinks
        self._sinks_full: torch.Tensor | None = None
        # FlashMLA's BF16 sparse kernel is a prefill kernel (one CTA per query
        # row), so on SM100 decode rows go to FlashInfer's trtllm-gen sparse
        # MLA decode, which splits each row's top-k across CTAs.
        self._flashinfer_decode = (
            current_platform.is_device_capability_family(100)
            and has_flashinfer()
            and self.dcp_world_size == 1
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and self.qk_nope_head_dim in (128, 192)
        )
        # Leading decode rows of the batch in flight, and the KV page height
        # (set per step by `_forward_bf16_kv`).
        self._decode_rows = 0
        self._decode_page_rows = 64

    @staticmethod
    def _validate_sinks(sinks: torch.Tensor | None, num_heads: int) -> None:
        """Reject sink tensors the FlashMLA kernels cannot consume.

        Args:
            sinks: Candidate sink tensor, or None when the layer has no sink.
            num_heads: Local (TP-sharded) query head count of this layer.

        Raises:
            ValueError: If the dtype is not float32 or the shape is not
                `(num_heads,)`.

        """
        if sinks is None:
            return
        if sinks.dtype != torch.float32:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have dtype "
                f"torch.float32, but got {sinks.dtype}."
            )
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have shape "
                f"({num_heads},), but got {tuple(sinks.shape)}."
            )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        super().process_weights_after_loading(act_dtype)
        if self.sinks is None or not self.use_shardq:
            return
        from vllm.distributed import get_tp_group

        tp_group = get_tp_group()
        full = tp_group.all_gather(self.sinks.to(tp_group.device), dim=0)
        # Reload in place to preserve pointers held by captured CUDA graphs.
        if self._sinks_full is None:
            self._sinks_full = torch.empty_like(full)
        self._sinks_full.copy_(full)

    def reserve_workspace(self) -> None:
        super().reserve_workspace()
        if not self._flashinfer_decode:
            return
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
            _get_workspace_buffer,
            _reserve_kv_counters,
            trtllm_gen_query_heads,
        )

        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        _get_workspace_buffer(device)
        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        _reserve_kv_counters(device, max_tokens, trtllm_gen_query_heads(self.num_heads))

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        actual_num_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The parent hands decode rows, which lead the batch, to
        # `_bf16_flash_mla_kernel` together with any prefill rows.
        if (
            self._flashinfer_decode
            and not self.need_to_return_lse_for_decode
            and not isinstance(self.index_group, HiSparseMLAIndexGroup)
        ):
            self._decode_rows = attn_metadata.num_decode_tokens
            # Whole pages never cut off a block's rows.
            self._decode_page_rows = math.gcd(attn_metadata.block_size, 64)
        try:
            return super()._forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata, actual_num_heads
            )
        finally:
            self._decode_rows = 0

    def _sinks_for_query(
        self,
        q: torch.Tensor,
        head_dim: int,
        kernel_heads: int,
    ) -> torch.Tensor | None:
        """Return the sink laid out for the kernel's query head count.

        Args:
            q: Query tensor, before any head padding.
            head_dim: Axis of `q` holding the query heads.
            kernel_heads: Head count the kernel is invoked with, which may
                exceed the query head count because of padding.

        Returns:
            The sink tensor padded to `kernel_heads` with `-inf` (a no-op
            sink) for the padded lanes, or None when the layer has no sink.

        Raises:
            ValueError: If the sink and query head layouts disagree, or if they
                live on different devices.

        """
        sinks = self.sinks
        if sinks is None:
            return None

        query_heads = q.shape[head_dim]
        if sinks.shape[0] != query_heads:
            full = self._sinks_full
            if full is not None and full.shape[0] == query_heads:
                sinks = full
            else:
                full_heads = None if full is None else full.shape[0]
                raise ValueError(
                    "HYV4 FlashMLA sparse attention sink head count must match "
                    f"the runtime query layout: sinks={sinks.shape[0]}, "
                    f"full_sinks={full_heads}, query_heads={query_heads}. "
                    "The sink must use the same unpadded head layout as the "
                    "query."
                )
        if sinks.device != q.device:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks and query must be on the "
                f"same device, but got sinks={sinks.device}, query={q.device}."
            )
        if kernel_heads < query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse kernel head count cannot be smaller than "
                f"the runtime query layout: query_heads={query_heads}, "
                f"kernel_heads={kernel_heads}."
            )
        if kernel_heads == query_heads:
            return sinks

        # Mirror the query padding the kernels require. Reading ``sinks`` here
        # (rather than caching a padded copy at construction time) keeps the
        # values correct for weights loaded after the module is built, and the
        # allocation plus copy are captured in the CUDA graph.
        padded_sinks = sinks.new_full((kernel_heads,), float("-inf"))
        padded_sinks[:query_heads] = sinks
        return padded_sinks

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
        topk_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self._compute_fp8_decode_padded_heads(actual_num_heads)
        attn_sink = self._sinks_for_query(q, head_dim=2, kernel_heads=padded_num_heads)

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            # The shared schedule is only valid without `topk_length`.
            tile_scheduler_metadata=(
                kernel_metadata.scheduler_metadata
                if topk_length is None
                else get_mla_metadata()[0]
            ),
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )

        # Slice output and lse back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]
            lse = lse[:, :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        decode_rows = min(self._decode_rows, q.shape[0]) if out is None else 0
        if decode_rows == 0:
            return self._bf16_prefill_kernel(
                q, kv_c_and_k_pe_cache, topk_indices, topk_length, actual_num_heads, out
            )
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        if topk_length is None:
            topk_length = topk_indices.new_full((q.shape[0],), topk_indices.shape[-1])
        # No caller reads the LSE without DCP, which this path excludes.
        decode_out = self._flashinfer_decode_kernel(
            q[:decode_rows, :actual_num_heads],
            kv_c_and_k_pe_cache,
            topk_indices[:decode_rows],
            topk_length[:decode_rows],
        )
        if decode_rows == q.shape[0]:
            return decode_out, None
        prefill_out, _ = self._bf16_prefill_kernel(
            q[decode_rows:],
            kv_c_and_k_pe_cache,
            topk_indices[decode_rows:],
            topk_length[decode_rows:],
            actual_num_heads,
        )
        return torch.cat([decode_out, prefill_out]), None

    def _flashinfer_decode_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse MLA decode through FlashInfer's trtllm-gen kernel.

        Args:
            q: `[T, heads, 576]` decode queries.
            kv_c_and_k_pe_cache: `[rows, 576]` flat BF16 KV rows.
            topk_indices: `[T, topk]` flat KV row per selected token (-1 unused).
            topk_length: `[T]` valid entries per row.

        Returns:
            `[T, heads, 512]` outputs with the sink applied.

        """
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
            _get_workspace_buffer,
            _kv_counters,
            _pad_query_heads,
            trtllm_gen_query_heads,
        )

        num_tokens, num_heads, _ = q.shape
        heads = trtllm_gen_query_heads(num_heads)
        attn_sink = self._sinks_for_query(q, head_dim=1, kernel_heads=heads)
        q = _pad_query_heads(q, heads) if heads != num_heads else q.contiguous()
        kv_rows = kv_c_and_k_pe_cache.view(-1, kv_c_and_k_pe_cache.shape[-1])
        page = self._decode_page_rows
        kv_pages = kv_rows[: kv_rows.shape[0] // page * page].view(
            -1, 1, page, kv_rows.shape[-1]
        )
        out = q.new_empty((num_tokens, 1, heads, self.kv_lora_rank))
        topk = topk_indices.shape[-1]
        trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_pages,
            workspace_buffer=_get_workspace_buffer(q.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices.unsqueeze(1),
            seq_lens=topk_length,
            max_seq_len=topk,
            sparse_mla_top_k=topk,
            out=out,
            bmm1_scale=self.softmax_scale,
            bmm2_scale=1.0,
            sinks=attn_sink,
            **_kv_counters(num_tokens, heads),
        )
        return out[:, 0, :num_heads]

    def _bf16_prefill_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # DCP can all-gather heads before this call.
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        padded_num_heads = (
            (actual_num_heads + self.prefill_padding - 1)
            // self.prefill_padding
            * self.prefill_padding
        )
        use_small_heads = self._should_use_bf16_small_head_kernel(
            num_tokens, actual_num_heads
        )
        kernel_heads = actual_num_heads if use_small_heads else padded_num_heads
        # q_concat_buffer may already include the kernel-required padding.
        # The sink remains defined only for the real query heads.
        attn_sink = self._sinks_for_query(
            q[:, :actual_num_heads], head_dim=1, kernel_heads=kernel_heads
        )

        if use_small_heads:
            q = q[:, :actual_num_heads]
            if out is not None:
                out = out[:, :actual_num_heads]
        elif q.shape[1] < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for BF16 sparse prefill kernel"
            )
            # Zero (not new_empty) the padded lanes: topk_indices is shared by
            # all heads, so the kernel reduces across the head group and NaNs
            # from uninitialized memory would leak into the real heads.
            if out is None:
                q_padded = q.new_zeros((num_tokens, padded_num_heads, q.shape[2]))
            else:
                *_, q_workspace, _, _ = current_workspace_manager().get_simultaneous(
                    *self.workspace_specs
                )
                q_padded = q_workspace[:num_tokens, :padded_num_heads]
            q_padded[:, actual_num_heads:padded_num_heads].zero_()
            q_padded[:, :actual_num_heads].copy_(q[:, :actual_num_heads])
            q = q_padded

        q = q[:, :padded_num_heads]
        if out is not None:
            out = out[:, :padded_num_heads]

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output, _, lse = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
        )

        output = output[:, :actual_num_heads, :]
        lse = lse[:, :actual_num_heads]
        return output, lse


class HYV4FlashMLASparseBackend(FlashMLASparseBackend):
    """`FLASHMLA_SPARSE` with attention-sink support for HY V4.

    Keeps the parent's name, metadata and builder; only the impl class and the
    sink capability differ. See the module docstring for why the name is reused.
    """

    @staticmethod
    def get_impl_cls() -> type[HYV4FlashMLASparseImpl]:
        return HYV4FlashMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
