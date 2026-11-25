# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped PD storage retention, independent of attention and hit rules."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.exceptions import VLLMValidationError
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_coordinator import replay_boundaries
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheLayerConfig
    from vllm.v1.request import Request


def supports_pd_replay(manager_cls: type[SingleTypeKVCacheManager]) -> bool:
    return manager_cls in (FullAttentionManager, SlidingWindowManager)


def pd_replay_retention_enabled(config: "KVTransferConfig | None") -> bool:
    """Validate and resolve the MoRI-IO opt-in storage policy."""
    if config is None:
        return False
    options = config.kv_connector_extra_config
    enabled = options.get("pd_replay_retention", False)
    if not isinstance(enabled, bool):
        raise ValueError("pd_replay_retention must be a JSON boolean")
    if enabled:
        if config.kv_connector != "MoRIIOConnector" or str(
            options.get("read_mode", False)
        ).strip().lower() not in ("true", "1"):
            raise ValueError("pd_replay_retention requires MoRIIOConnector READ mode")
        if config.kv_role not in ("kv_producer", "kv_consumer"):
            raise ValueError("PD replay retention requires a producer or consumer role")
    return enabled


def get_pd_replay_storage_allowance(
    vllm_config: "VllmConfig", layers: dict[str, "KVCacheLayerConfig"]
) -> dict[str, int] | None:
    """Return storage-only token allowances before grouping and pool sizing."""
    if not pd_replay_retention_enabled(vllm_config.kv_transfer_config):
        return None
    margin = (
        max(1, vllm_config.num_hash_lookahead_tokens)
        + math.lcm(*(layer.spec.block_size for layer in layers.values()))
        - 1
        if layers
        else 0
    )
    allowance = {}
    for name, layer in layers.items():
        if not supports_pd_replay(layer.manager_cls):
            raise ValueError(
                "PD replay retention requires full attention or "
                "plain SlidingWindowManager groups"
            )
        config = layer.manager_config
        if isinstance(config.params, SlidingWindowManager.Params):
            allowance[name] = 1 if config.params.bounded_replay else margin
    return allowance


@dataclass(frozen=True)
class ReplayAllocation:
    generation: int
    ranges: tuple[tuple[int, int], ...]
    local_tokens: int


class PDReplayRetention:
    def __init__(self, manager: "KVCacheManager", lookahead: int, producer: bool):
        self.manager, self.lookahead, self.producer = manager, lookahead, producer
        self.managers = manager.coordinator.single_type_managers
        self.alignment = (
            manager.kv_cache_config.cache_hit_alignment_tokens
            or manager.coordinator.scheduler_block_size
        )
        self.allocations: dict[str, ReplayAllocation] = {}
        self._completed: set[str] = set()
        self._generation = 0
        for group in self.managers:
            if not supports_pd_replay(type(group)):
                raise ValueError(
                    "PD replay retention supports only full attention and "
                    "plain SlidingWindowManager groups"
                )
            if isinstance(group, SlidingWindowManager) and group.enable_caching:
                params = group.manager_config.params
                assert isinstance(params, SlidingWindowManager.Params)
                required = (
                    1
                    if params.bounded_replay
                    else max(1, lookahead) + self.alignment - 1
                )
                if params.storage_retention_tokens < required:
                    raise ValueError(
                        "PD replay retention requires a matching "
                        "storage planning margin"
                    )

        if manager.storage_retention_provider is not None:
            raise ValueError("Storage retention is already bound to this pool")
        manager.storage_retention_provider = self

    def signature(self) -> list[Any]:
        """Return wire geometry in tokens and per-group block sizes."""
        policies = []
        for group in self.managers:
            params = group.manager_config.params
            if isinstance(group, SlidingWindowManager):
                assert isinstance(params, SlidingWindowManager.Params)
                policies.append(
                    [
                        group.block_size,
                        group.num_retained_tokens,
                        params.sliding_window,
                        params.extra_retained_tokens,
                        params.bounded_replay,
                    ]
                )
            else:
                policies.append([group.block_size])
        return [self.lookahead, self.alignment, policies]

    def ranges(
        self, prompt: int, computed: int, *, local_tokens: int = 0
    ) -> tuple[tuple[int, int], ...]:
        """Select contiguous replay tails as absolute group-block [start, end).

        Args:
            prompt: Full prompt length in tokens.
            computed: Computed prompt prefix length in tokens.
            local_tokens: Locally reused prefix length in tokens; only its
                attention tail has backing storage.

        Returns:
            One absolute block range per group, covering both aligned replay
            boundaries and the computed attention tail.

        """
        result = []
        boundaries = replay_boundaries(prompt, self.lookahead)
        for group in self.managers:
            size = group.block_size
            end = cdiv(computed, size)
            start = group.get_num_skipped_tokens(max(0, computed - 1)) // size
            if isinstance(group, SlidingWindowManager) and group.prefix_cacheable(
                group.manager_config
            ):
                params = group.manager_config.params
                assert isinstance(params, SlidingWindowManager.Params)
                first_boundary = min(boundaries) // self.alignment * self.alignment
                first = max(
                    0, first_boundary // size - cdiv(params.sliding_window - 1, size)
                )
                mask = group.reachable_block_mask(
                    first,
                    end,
                    self.alignment,
                    size,
                    group.manager_config,
                    retention_interval=0,
                    reachable_boundaries=boundaries,
                )
                if mask is None:
                    start = min(start, first)
                else:
                    start = min(
                        start,
                        next((first + i for i, keep in enumerate(mask) if keep), start),
                    )
                # A local hit only backs its attention tail, not re-prefill storage.
                start = max(
                    start,
                    local_tokens // size - cdiv(params.sliding_window - 1, size),
                )
            result.append((min(start, end), end))
        return tuple(result)

    def source(self, request: "Request") -> dict[str, Any]:
        params = request.kv_transfer_params or {}
        source = params.get("pd_replay")
        if not isinstance(source, dict) or source.get("version") != 1:
            raise VLLMValidationError("Enable pd_replay_retention on both PD instances")
        if source.get("signature") != self.signature():
            raise VLLMValidationError(
                "PD replay geometry or speculative settings differ between instances"
            )
        prompt, computed = source.get("prompt_tokens"), source.get("computed_tokens")
        if (
            not isinstance(prompt, int)
            or not isinstance(computed, int)
            or not 0 <= computed <= prompt == request.num_prompt_tokens
        ):
            raise VLLMValidationError(
                "Invalid PD replay prompt or computed-token range"
            )
        local = source.get("local_tokens", 0)
        max_hit = max(0, prompt - max(1, self.lookahead))
        max_hit = max_hit // self.alignment * self.alignment
        if not isinstance(local, int) or not 0 <= local <= min(computed, max_hit):
            raise VLLMValidationError("Invalid PD replay producer local-hit length")
        ranges = source.get("ranges")
        if ranges != [
            list(r) for r in self.ranges(prompt, computed, local_tokens=local)
        ]:
            raise VLLMValidationError("Invalid PD replay absolute block ranges")
        blocks = params.get("remote_block_ids")
        if (
            not isinstance(blocks, (list, tuple))
            or len(blocks) != len(self.managers)
            or any(
                not isinstance(ids, (list, tuple))
                or len(ids) != end - start
                or any(not isinstance(b, int) or b <= 0 for b in ids)
                for ids, (start, end) in zip(blocks, ranges, strict=True)
            )
        ):
            raise VLLMValidationError(
                "PD replay source contains null or missing blocks"
            )
        return source

    def validate_request(self, request: "Request") -> None:
        params = request.kv_transfer_params or {}
        if self.producer and params.get("do_remote_decode"):
            if request.max_tokens != 1:
                raise VLLMValidationError(
                    "PD replay producer requests require max_tokens=1"
                )
        elif not self.producer and params.get("do_remote_prefill"):
            self.source(request)

    def prepare(
        self, request: "Request", local_tokens: int, external_tokens: int
    ) -> bool:
        params = request.kv_transfer_params or {}
        if request.request_id in self._completed:
            return False
        active = (
            params.get("do_remote_decode")
            if self.producer
            else (params.get("do_remote_prefill") and external_tokens > 0)
        )
        if not active:
            return False
        self.validate_request(request)
        if request.request_id in self.allocations:
            return False
        if self.producer:
            ranges = self.ranges(
                request.num_prompt_tokens,
                request.num_prompt_tokens,
                local_tokens=local_tokens,
            )
        else:
            source = self.source(request)
            ranges = tuple((r[0], r[1]) for r in source["ranges"])
            backed = self.ranges(
                request.num_prompt_tokens,
                source["computed_tokens"],
                local_tokens=local_tokens,
            )
            ranges = tuple(
                (max(start, local_start), end)
                for (start, end), (local_start, _) in zip(ranges, backed, strict=True)
            )
        self._generation += 1
        allocation = ReplayAllocation(self._generation, ranges, local_tokens)
        self.allocations[request.request_id] = allocation
        self.manager.set_storage_retention(
            request.request_id, [start for start, _ in ranges]
        )
        return True

    def release(self, request_id: str, generation: int | None = None) -> None:
        current = self.allocations.get(request_id)
        if current is None or (
            generation is not None and generation != current.generation
        ):
            return
        self.manager.clear_storage_retention(request_id)
        self._completed.add(request_id)

    def forget(self, request_id: str) -> None:
        """Discard an allocation lifetime on free or preemption."""
        self.allocations.pop(request_id, None)
        self._completed.discard(request_id)

    def step_generations(self, request_ids: Sequence[str]) -> dict[str, int]:
        """Snapshot allocation generations for reads issued in this step."""
        return {
            rid: self.allocations[rid].generation
            for rid in request_ids
            if rid in self.allocations
        }

    def complete_step(self, generations: dict[str, int]) -> None:
        """Lift retention after the scheduler reconciles this step's load errors."""
        for rid, generation in generations.items():
            self.release(rid, generation)

    def export(
        self, request: "Request", blocks: Sequence[Sequence[int]]
    ) -> tuple[list[list[int]], dict[str, Any]]:
        """Export real block IDs for absolute group-block replay ranges.

        Args:
            request: Producer request, with computed and prompt lengths in tokens.
            blocks: Per-group block IDs indexed by absolute group-block position.

        Returns:
            Range-relative block ID lists and wire metadata describing their
            absolute half-open ranges, excluding uncomputed speculative slots.

        """
        computed = min(request.num_computed_tokens, request.num_prompt_tokens)
        local = self.allocations[request.request_id].local_tokens
        ranges = self.ranges(request.num_prompt_tokens, computed, local_tokens=local)
        selected = []
        for group, (start, end) in zip(blocks, ranges, strict=True):
            chosen = list(group[start:end])
            if len(chosen) != end - start or any(b <= 0 for b in chosen):
                raise VLLMValidationError(
                    "PD replay export requires real blocks throughout each range"
                )
            selected.append(chosen)
        return selected, dict(
            version=1,
            prompt_tokens=request.num_prompt_tokens,
            computed_tokens=computed,
            local_tokens=local,
            signature=self.signature(),
            ranges=[list(r) for r in ranges],
        )

    def pair(
        self, request: "Request", local: Sequence[Sequence[int]], external: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Pair destination and source IDs at identical absolute block positions.

        Args:
            request: Consumer request carrying the producer's replay ranges.
            local: Per-group IDs indexed by absolute group-block position.
            external: Tokens to load after the locally reused prefix.

        Returns:
            Per-group destination and source lists for the external prefix,
            excluding locally reused blocks and the locally recomputed tail.

        """
        source = self.source(request)
        remote = (request.kv_transfer_params or {})["remote_block_ids"]
        local_tokens = self.allocations[request.request_id].local_tokens
        load_end = local_tokens + external
        if load_end > source["computed_tokens"]:
            raise ValueError("PD replay read exceeds the producer's computed tokens")
        # Local cache hits are measured before num_computed_tokens is updated.
        dst, src = [], []
        for group, local_ids, remote_ids, (start, end), (backed_start, _) in zip(
            self.managers,
            local,
            remote,
            source["ranges"],
            self.allocations[request.request_id].ranges,
            strict=True,
        ):
            read_end = min(end, cdiv(load_end, group.block_size))
            read_start = max(backed_start, local_tokens // group.block_size)
            if read_end > len(local_ids) or any(
                b <= 0 for b in local_ids[backed_start:read_end]
            ):
                raise RuntimeError(
                    "PD replay destination contains null or missing blocks"
                )
            dst.append(list(local_ids[read_start:read_end]))
            src.append(list(remote_ids[read_start - start : read_end - start]))
        return dst, src
