# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling short-conv state managed as a sliding-window KV cache.

Each decoder layer owns one `InklingConvState` (an `AttentionLayerBase`) that
emits a single sliding-window KV cache for the layer's 4 sconv streams (K, V,
attn-output, mlp-output), packed head-major into one block:

    H = num_kv_heads (per-rank), N = block_size = sconv_kernel_size,
    D = head_dim(K) + head_dim(V) + hidden/H(attn) + hidden/H(mlp)

`D` is TP-invariant; per rank we store `H/TP` heads of width `D`. The conv
reads/writes this cache out-of-band via a custom backend; the (smaller) conv page
is padded up to the uniform attention page by `unify_kv_cache_spec_page_size`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    DFlashSlidingWindowManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig

from .ops.sconv import sconv_seq_metadata

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheGroup, ManagerConfig
    from vllm.v1.request import Request


class InklingSconvTailManager(DFlashSlidingWindowManager):
    """Conv-state cache backing each step's trailing window and the blocks a
    prefix hit can resume from.

    The sconv kernels read earlier steps' taps only for the `W - 1` positions
    before a step's first token; taps inside the step come from `x`. Like the
    DFlash drafter KV, allocations back blocks from `num_tokens - W` on and
    leave lower ones null (the kernels skip null-block stores), so a prefill
    chunk holds one tail instead of `W - 1 + max_in_flight_tokens` positions.

    Prefix hits stay plain sliding window: a hit at `h` reads `[h - W + 1, h)`.
    Under sparse retention the other sliding-window groups only keep the tails
    ending on the replay boundaries and the shared-prefix junction, floored to
    the hit alignment (`SlidingWindowManager.reachable_block_mask`); those
    blocks are backed here too. The prompt length arrives via
    `remove_skipped_blocks` before each allocation and the junction via
    `cache_blocks`, so a junction inside a request's first chunk is not backed.
    """

    # Resend and full-prompt replay boundaries plus one shared-prefix junction.
    NUM_BOUNDARIES: ClassVar[int] = 3

    # Plain sliding-window hits and retention, not DFlash's end-of-request ones.
    @classmethod
    def find_longest_cache_hit(cls, *args, **kwargs):
        return super(DFlashSlidingWindowManager, cls).find_longest_cache_hit(
            *args, **kwargs
        )

    @classmethod
    def reachable_block_mask(cls, *args, **kwargs):
        return super(DFlashSlidingWindowManager, cls).reachable_block_mask(
            *args, **kwargs
        )

    @classmethod
    def max_admission_blocks(
        cls,
        config: ManagerConfig,
        block_size: int,
        max_in_flight_tokens: int,
        max_model_len: int,
    ) -> int | None:
        params = config.params
        assert isinstance(params, DFlashSlidingWindowManager.Params)
        tails = super().max_admission_blocks(
            config, block_size, max_in_flight_tokens, max_model_len
        )
        window_blocks = SlidingWindowManager.max_admission_blocks(
            config,
            block_size,
            max_in_flight_tokens + params.num_lookahead_tokens,
            max_model_len,
        )
        assert tails is not None and window_blocks is not None
        need = cdiv(params.sliding_window - 1, block_size)
        return min(window_blocks, tails + cls.NUM_BOUNDARIES * need)

    def __init__(self, kv_cache_group: KVCacheGroup, **kwargs) -> None:
        super().__init__(kv_cache_group, **kwargs)
        assert not self.enable_caching or self.retention_interval == 0
        params = self.manager_config.params
        assert isinstance(params, SlidingWindowManager.Params)
        self._boundary_need = cdiv(params.sliding_window - 1, self.block_size)
        # req_id -> indices of blocks below the tail that a prefix hit reads.
        self._boundary_blocks: dict[str, set[int]] = {}

    def _note_boundaries(self, request_id: str, positions: Iterable[int]) -> None:
        align = self.cache_hit_alignment_tokens
        blocks = self._boundary_blocks.setdefault(request_id, set())
        for position in positions:
            end = position // align * align // self.block_size
            blocks.update(range(max(end - self._boundary_need, 0), end))

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        if num_prompt_tokens is not None and self.enable_caching:
            from vllm.v1.core.kv_cache_coordinator import replay_boundaries

            self._note_boundaries(
                request_id,
                replay_boundaries(
                    num_prompt_tokens, self.block_pool.num_hash_lookahead_tokens
                ),
            )
        super().remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    def cache_blocks(self, request: Request, *args, **kwargs) -> None:
        if request.shared_prefix_boundary:
            self._note_boundaries(request.request_id, (request.shared_prefix_boundary,))
        super().cache_blocks(request, *args, **kwargs)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        num_blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
            prefill_end=prefill_end,
        )
        boundary = self._boundary_blocks.get(request_id)
        if not boundary:
            return num_blocks
        # The base counts the tail only; add boundary blocks below it.
        start = len(self.req_to_blocks.get(request_id, ()))
        if request_id not in self.num_cached_block:
            start += len(new_computed_blocks)
        tail_start = self._num_unbacked_leading_blocks(num_tokens_main_model)
        end = min(tail_start, cdiv(num_tokens, self.block_size))
        num_blocks += sum(start <= j < end for j in boundary)
        if apply_admission_cap and self.max_admission_blocks_per_request is not None:
            num_blocks = min(num_blocks, self.max_admission_blocks_per_request)
        return num_blocks

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        assert request_id not in self._partial_hit_reqs
        req_blocks = self.req_to_blocks[request_id]
        start, end = len(req_blocks), cdiv(num_tokens, self.block_size)
        if end <= start:
            return []
        tail_start = self._num_unbacked_leading_blocks(num_tokens_main_model)
        boundary = self._boundary_blocks.get(request_id, ())
        backed = [j >= tail_start or j in boundary for j in range(start, end)]
        fresh = iter(self.block_pool.get_new_blocks(sum(backed)))
        new_blocks = [next(fresh) if b else self._null_block for b in backed]
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in new_blocks if not b.is_null)
        return new_blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._boundary_blocks.pop(request_id, None)
        return super().pop_blocks_for_free(request_id)


# Stream order within the per-head packed D (== contiguous sub-ranges).
_K, _V, _ATTN, _MLP = 0, 1, 2, 3


@dataclass
class InklingSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks] physical blocks per req
    slot_mapping: torch.Tensor  # [T] int64 flat slot of each token (-1 => skip)
    seq_idx: torch.Tensor  # [T] int32 token -> batch request
    query_start: torch.Tensor  # [T] int32 first x-row of each token's request


class InklingSconvMetadataBuilder(AttentionMetadataBuilder[InklingSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, AttentionSpec)
        # Persistent per-token buffers for CUDA graph capture.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.query_start_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingSconvMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        assert num_padded_tokens >= num_actual_tokens

        # Per-token seq_idx (owning request) and query_start (first x-row of
        # that request; the fused kernel uses it to tell same-forward taps,
        # read from x, from pre-forward taps, read from cache) in one launch.
        sconv_seq_metadata(
            common_attn_metadata.query_start_loc,
            num_reqs,
            num_actual_tokens,
            self.seq_idx_buffer,
            self.query_start_buffer,
            num_padded_tokens,
        )

        return InklingSconvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
        )


class InklingSconvBackend(AttentionBackend):
    """Custom dummy backend for the sconv sliding-window cache management."""

    @staticmethod
    def get_name() -> str:
        return "INKLING_SCONV_SWA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "InklingSconvBackend has no attention impl; the conv runs out-of-band."
        )

    @staticmethod
    def get_builder_cls() -> type[InklingSconvMetadataBuilder]:
        return InklingSconvMetadataBuilder


class InklingConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer owner emitting one sliding-window conv-state spec."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
        replicate_hidden: bool = False,
        tail_only: bool = False,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Back only each step's trailing window (InklingSconvTailManager).
        # Needs kernels that skip null-block stores (Inkling2 ops/sconv.py).
        self.tail_only = tail_only
        # Bound to the manager-allocated paged cache by bind_kv_cache; a
        # placeholder until then. Read out-of-band by InklingShortConv.
        self.kv_cache = torch.tensor([])
        tp_size = get_tensor_model_parallel_world_size()
        # Guardrails for the conv-state layout below; only these are exercised.
        # tp_size <= num_kv_heads keeps >=1 whole KV head per rank (no
        # replication/clamping).
        assert tp_size <= num_kv_heads, (
            f"sconv SWA cache supports tp_size <= num_kv_heads ({num_kv_heads}), "
            f"got {tp_size}"
        )
        # By default the attn-/mlp-output sconv streams are hidden-sharded:
        # each rank owns its H/tp chunk (the sublayer outputs are
        # reduce-scattered / all-gathered around the conv), so D is
        # TP-invariant. With ``replicate_hidden``, each rank spreads the full
        # hidden width over its local heads, so D grows with TP.
        self.num_kv_heads = num_kv_heads // tp_size
        hidden_per_head = hidden_size // (
            self.num_kv_heads if replicate_hidden else num_kv_heads
        )
        # Packed per-head width: K + V + attn-output chunk + mlp-output chunk,
        # padded to a power of two so every layer's conv page is the same size.
        # The page unifier may still scale the logical conv block size up to
        # match a larger attention page; index the bound cache with its runtime
        # token dim (``cache_block_size``), not this kernel window size.
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        # Per-head D-sub-range (offset, width) for each stream. Streams share
        # the cache; each writes/reads its own width across all H heads.
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),  # _K
            (head_dim, head_dim),  # _V
            (2 * head_dim, hidden_per_head),  # _ATTN
            (2 * head_dim + hidden_per_head, hidden_per_head),  # _MLP
        )
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        assert self._dtype == torch.bfloat16, (
            f"sconv SWA cache supports bfloat16 only, got {self._dtype}"
        )
        # Register in the forward context so the runner enumerates this owner as
        # an attention-like layer (get_kv_cache_config / get_attn_backend).
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self): ...

    @property
    def cache_block_size(self) -> int:
        """Return the block size used by cache metadata and physical indexing."""
        if self.kv_cache.numel() > 0:
            return self.kv_cache.shape[2]
        return self.block_size

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingSconvBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        spec = AttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,  # all 4 streams packed into head_size
            dtype=self._dtype,
        )
        cache_config = vllm_config.cache_config
        if self.tail_only and (
            not cache_config.enable_prefix_caching
            or cache_config.prefix_cache_retention_interval == 0
        ):
            return InklingSconvTailManager.layer_config(
                spec,
                sliding_window=self.sliding_window,
                num_speculative_tokens=vllm_config.num_speculative_tokens,
                max_in_flight_steps=vllm_config.max_concurrent_batches,
                num_lookahead_tokens=vllm_config.num_lookahead_tokens,
            )
        return SlidingWindowManager.layer_config(
            spec, sliding_window=self.sliding_window
        )
