# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

logger = logging.getLogger(__name__)



def _get_num_experts_per_tok(hf_config) -> int:
    """Resolve the per-token expert count from the HF config.

    Different model families store this under different attribute names
    (e.g. ``num_experts_per_tok`` for DeepSeek, ``top_k_experts`` for Gemma 4).
    """
    val = getattr(hf_config, "num_experts_per_tok", None)
    if val is None:
        val = getattr(hf_config, "top_k_experts", None)
    if val is None:
        raise ValueError(
            "Cannot determine num_experts_per_tok: HF config has neither "
            "'num_experts_per_tok' nor 'top_k_experts'"
        )
    return val


def get_num_experts(hf_config) -> int:
    """Resolve ``num_experts`` across HuggingFace config naming conventions.

    Different MoE model families expose this under different keys:
      - ``num_experts``: Mixtral, Qwen2-MoE, Qwen3-MoE
      - ``n_routed_experts``: DeepSeek-V2/V3
      - ``num_local_experts``: Mixtral (older exports)
    """
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(hf_config, key, None)
        if val is not None:
            return val
    raise ValueError(
        "Could not resolve num_experts from model config. "
        "Expected one of 'num_experts', 'n_routed_experts', "
        "or 'num_local_experts'."
    )


def _expert_id_dtype(num_experts: int, *, numpy: bool = False):
    """Pick the smallest unsigned int type that fits all expert IDs.

    Expert IDs are 0..num_experts-1; uint8 fits 256 distinct values
    (0..255), so the boundary is ``<= 256`` (NOT ``< 256``).
    """
    if numpy:
        return np.uint8 if num_experts <= 256 else np.uint16
    return torch.uint8 if num_experts <= 256 else torch.uint16


class RoutedExpertsCapturer:
    """Worker-side capturer for routed experts, lives on GPU.

    Layer-level hooks call :meth:`capture` from inside the forward pass
    with the per-layer ``topk_ids`` tensor. The tensor is sliced to the
    tokens owned by this DP rank and written into a preallocated device
    buffer. At the end of the step, :class:`GPUModelRunner` reads the
    device buffer, issues a D2H copy into a pinned CPU buffer, and hands
    the result to the scheduler via :class:`RoutedExpertsLists`.

    Invariants:
        - One instance per worker; shape is fixed at init and covers the
          worst-case step (``max_num_batched_tokens`` tokens).
        - :meth:`clear_buffer` is called at the start of every step, so
          unused slots stay zero.
        - ``device_buffer.dtype`` is picked by ``num_experts``; callers
          should not assume int32.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                hf_config.num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=_expert_id_dtype(num_experts),
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Capture expert routing decisions for a specific layer.

        Under data parallelism, ``topk_ids`` may have two different batch
        layouts depending on where the DP combine happens:
          - ``n == total`` (naive dispatch): all DP ranks' tokens are
            concatenated before routing; we slice out this rank's span
            using the cumulative per-rank counts.
          - ``n == token_num_per_dp`` (modular-kernel path): DP combine
            happens inside ``quant_method.apply``; ``select_experts`` only
            ever sees this rank's tokens, so we take the whole tensor.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]

            if n == total:
                # Naive dispatch: all DP ranks' tokens concatenated
                # before routing. This rank owns tokens
                # [end_loc - token_num_per_dp, end_loc).
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif n == token_num_per_dp:
                # Modular-kernel path: DP combine happens inside
                # quant_method.apply; select_experts only sees this
                # rank's tokens, take the whole tensor.
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch dim "
                    f"{n} (expected {total} or {token_num_per_dp} "
                    f"for dp_rank={self.dp_rank})"
                )

        # Defensive: model may expose more layers than the capture buffer
        # was sized for (unusual, but guards against miss-config).
        if layer_id >= self.device_buffer.shape[1]:
            return

        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def clear_buffer(self) -> None:
        """Zero the device buffer. Called at the start of every step so
        slots belonging to finished / preempted tokens don't leak into
        the next step.
        """
        self.device_buffer.zero_()

    def get_device_buffer(self) -> torch.Tensor:
        """Return the underlying device buffer so the model runner can
        issue the D2H copy. The tensor is shared; callers must either
        clone or fully drain it before the next forward pass runs
        :meth:`clear_buffer`.
        """
        return self.device_buffer


class RoutedExpertsManager:
    """Scheduler-side slot-indexed buffer for routed experts.

    Lives on CPU in the scheduler process. Each slot corresponds to
    ``block_id * block_size + offset_in_block`` where ``block_id`` is
    drawn from the physical KV-cache block pool, so routing data is
    tied to physical blocks and naturally survives preemption for
    prefix-cached blocks (prefix hits re-expose the same slots).

    Data flow per step:
      1. Worker D2Hs its device capture buffer into
         :class:`RoutedExpertsLists` and returns it via
         :class:`ModelRunnerOutput`.
      2. Scheduler calls :meth:`store_batch` with that step's
         ``(routing_data, slot_mapping)`` — a single CPU->CPU
         fancy-index assign, ~few MB per step.
      3. On request completion / abort / preemption, the scheduler
         calls :meth:`get` with the request's block IDs to recover
         the full per-token routing.

    Memory: ``routed_experts_by_slot`` is sized for the whole block
    pool (``num_blocks * block_size`` slots). For large block pools
    this can reach multiple GB; see the init log for the exact size.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Pick the attention group for block/slot mapping. We require
        # a FullAttentionSpec group rather than any AttentionSpec to
        # stay consistent with the worker-side lookup in
        # ``GPUModelRunner._get_attention_kv_cache_gid``; hybrid models
        # (Mamba / linear attention) also have other AttentionSpec
        # groups whose slot layout differs.
        self.attn_gid = next(
            gid
            for gid, g in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(g.kv_cache_spec, FullAttentionSpec)
        )
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

        # All kv_cache_groups share the same physical block pool, so
        # block IDs span [0, num_blocks) regardless of how many groups
        # exist. Sizing to the full pool avoids index-out-of-range
        # when different groups happen to land on the same block.
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        self.routed_experts_by_slot = np.zeros(
            (
                max_num_slots,
                hf_config.num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=_expert_id_dtype(num_experts, numpy=True),
        )
        logger.info(
            "RoutedExpertsManager CPU buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s)",
            self.routed_experts_by_slot.nbytes / 1e9,
            max_num_slots,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
        )

    def store_batch(self, data: np.ndarray, slot_mapping: np.ndarray) -> None:
        """Persist one step's routed experts into the slot buffer.

        Equivalent to ``slot_buffer[slot_mapping] = data``; numpy fancy
        indexing handles repeated / out-of-order indices. Called once
        per scheduler step in ``update_from_output``.
        """
        self.routed_experts_by_slot[slot_mapping] = data

    def get(self, block_ids: list[int], num_tokens: int) -> np.ndarray:
        """Read routed experts data for a completed / preempted request.

        Reconstructs a per-token slot_mapping from the request's block
        IDs and returns the routing slice. Because numpy fancy indexing
        returns a **copy** (not a view), the returned ndarray is safe
        to hold across subsequent :meth:`store_batch` calls — do not
        replace the fancy index with a slice without re-verifying.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of tokens that have gone through a forward
                pass and therefore have routing data written to their
                slots (typically ``request.num_tokens - 1``; the last
                sampled token has not been forwarded yet). Slots beyond
                ``request.num_computed_tokens`` are zero-initialized.

        Returns:
            Array of shape (num_tokens, num_layers, num_experts_per_tok).
        """
        bs = self.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(bs)
        # slot = block_id * block_size + offset_in_block; flatten the
        # (num_blocks, block_size) grid and trim to num_tokens.
        slot_mapping = (
            block_ids_array.reshape(-1, 1) * bs + block_offsets.reshape(1, -1)
        ).flatten()[:num_tokens]
        return self.routed_experts_by_slot[slot_mapping]


@dataclass
class RoutedExpertsEntry:
    """Cached routed experts for a single preempted / aborted request.

    ``data``: routing snapshot taken before blocks were freed.
    ``pending_block_ids``: when not None, the slot buffer was incomplete
        at capture time (store_batch for the in-flight step hadn't run
        yet). :meth:`RoutedExpertsCache.refresh_pending` re-reads using
        these block_ids after store_batch to get the complete snapshot.
    """

    data: np.ndarray
    pending_block_ids: list[int] | None = None
    pending_num_tokens: int = 0

    @property
    def length(self) -> int:
        return self.data.shape[0]


@dataclass
class RoutedExpertsCache:
    """Per-request cache of routed experts, used by the scheduler.

    Wraps ``aborted_routed_experts`` and ``_preempted_re_pending`` into
    a single structure with clear semantics.
    """

    mgr: RoutedExpertsManager
    _entries: dict[str, RoutedExpertsEntry] = field(default_factory=dict)

    def capture(
        self,
        req_id: str,
        data: np.ndarray,
        block_ids: list[int],
        num_tokens: int,
    ) -> None:
        """Cache routing at preemption time, with deferred re-read info."""
        existing = self._entries.get(req_id)
        if existing is None or data.shape[0] > existing.length:
            self._entries[req_id] = RoutedExpertsEntry(
                data=data,
                pending_block_ids=block_ids,
                pending_num_tokens=num_tokens,
            )
        else:
            existing.pending_block_ids = block_ids
            existing.pending_num_tokens = num_tokens

    def refresh_pending(self) -> None:
        """Re-read routing from slot buffer for entries that had
        incomplete data at capture time. Call after store_batch."""
        for entry in self._entries.values():
            if entry.pending_block_ids is not None:
                refreshed = self.mgr.get(
                    entry.pending_block_ids, entry.pending_num_tokens
                )
                if refreshed.shape[0] > entry.length:
                    entry.data = refreshed
                entry.pending_block_ids = None
                entry.pending_num_tokens = 0

    def pop(self, req_id: str) -> np.ndarray | None:
        """Pop and return the cached routing data, or None."""
        entry = self._entries.pop(req_id, None)
        return entry.data if entry is not None else None

    def remove(self, req_id: str) -> None:
        """Remove entry without returning data."""
        self._entries.pop(req_id, None)

    def get_best(self, req_id: str, current: np.ndarray) -> np.ndarray:
        """Pop cached entry and return whichever is more complete."""
        cached = self.pop(req_id)
        if cached is not None and cached.shape[0] > current.shape[0]:
            return cached
        return current
