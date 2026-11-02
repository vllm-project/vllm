# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup workspace profiles for FlashInfer MegaMoE."""

from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from weakref import WeakSet

import torch

from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = init_logger(__name__)


@dataclass
class _ProfileContext:
    decode_tokens: int
    prefill_tokens: int


_active_context: _ProfileContext | None = None
_profile_owners: WeakSet[FlashInferMoeEpProfiles] = WeakSet()


def has_flashinfer_moe_ep_profiles() -> bool:
    """Whether this worker owns live-token workspaces with dynamic sizing."""
    return bool(_profile_owners)


def _runner_config(runner: Any, name: str) -> Any:
    config = getattr(runner, name, None)
    if config is not None:
        return config
    return getattr(runner.vllm_config, name)


def _runner_limit(runner: Any, name: str, config_name: str) -> int:
    value = getattr(runner, name, None)
    if value is None:
        value = getattr(_runner_config(runner, "scheduler_config"), config_name)
    return int(value)


def _decode_query_len(runner: Any) -> int:
    for name in ("uniform_decode_query_len", "decode_query_len"):
        value = getattr(runner, name, None)
        if value is not None:
            return max(1, int(value))
    speculator = getattr(runner, "speculator", None)
    return max(1, int(getattr(speculator, "num_query_per_req", 1)))


def _decode_token_limit(runner: Any) -> int:
    kv_config = runner.vllm_config.kv_transfer_config
    if (
        kv_config is not None
        and kv_config.is_kv_producer
        and not kv_config.is_kv_consumer
    ):
        return 1
    # MRV1 materializes its final capture sizes on the runner. MRV2 keeps them
    # on CompilationConfig until its CUDA Graph manager is initialized.
    capture_sizes = getattr(runner, "cudagraph_batch_sizes", None)
    if capture_sizes is None:
        compilation_config = _runner_config(runner, "compilation_config")
        capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if capture_sizes:
        return max(int(size) for size in capture_sizes)
    max_num_tokens = _runner_limit(runner, "max_num_tokens", "max_num_batched_tokens")
    max_num_reqs = _runner_limit(runner, "max_num_reqs", "max_num_seqs")
    return min(max_num_tokens, max_num_reqs * _decode_query_len(runner))


@contextmanager
def flashinfer_moe_ep_profile_token_context(
    *, decode_tokens: int, prefill_tokens: int
) -> Iterator[None]:
    """Expose serving token limits while preparing persistent workspaces."""
    global _active_context
    previous = _active_context
    _active_context = _ProfileContext(
        decode_tokens=decode_tokens,
        prefill_tokens=prefill_tokens,
    )
    try:
        yield
    finally:
        _active_context = previous


@contextmanager
def flashinfer_moe_ep_profile_context(runner: Any) -> Iterator[None]:
    """Expose serving token limits during the startup memory-profile run."""
    with flashinfer_moe_ep_profile_token_context(
        decode_tokens=_decode_token_limit(runner),
        prefill_tokens=_runner_limit(
            runner, "max_num_tokens", "max_num_batched_tokens"
        ),
    ):
        yield


def _profile_capacities(decode_tokens: int, prefill_tokens: int) -> tuple[int, ...]:
    return tuple(sorted({max(1, decode_tokens), max(1, prefill_tokens)}))


class FlashInferMoeEpProfiles:
    """Prepare offline-tuned MegaMoE workspaces before CUDA Graph capture."""

    def __init__(
        self, layer: Any, bootstrap: Any, fleet: Any, *, sp_size: int = 1
    ) -> None:
        if sp_size < 1:
            raise ValueError(f"sp_size must be positive, got {sp_size}")
        self.layer = layer
        self.bootstrap = bootstrap
        self.fleet = fleet
        self.sp_size = sp_size
        self.workspaces: dict[int, list[tuple[int | None, Any]]] = {}
        self.capacities: tuple[int, ...] = ()
        _profile_owners.add(self)

    def destroy(self) -> None:
        """Release workspace handles and remove this owner from memory planning."""
        _profile_owners.discard(self)
        self.workspaces.clear()

    @staticmethod
    def is_supported(layer: Any) -> bool:
        return callable(getattr(layer, "create_workspace", None)) and callable(
            getattr(layer, "live_token_buckets", None)
        )

    def _agree(self, value: Any) -> None:
        if self.bootstrap.world_size == 1:
            return
        values = [None] * self.bootstrap.world_size
        torch.distributed.all_gather_object(
            values, value, group=self.bootstrap.process_group
        )
        if any(item != values[0] for item in values):
            raise RuntimeError("MegaMoE ranks selected different workspace profiles")

    def _profile_inputs(self, tensors: Any, tokens: int) -> Any:
        count = tensors.num_tokens

        def resize(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if count == 0:
                return tensor.new_zeros((tokens, *tensor.shape[1:]))
            if tokens <= count:
                return tensor[:tokens]
            repeats = ((tokens + count - 1) // count,) + (1,) * (tensor.ndim - 1)
            return tensor.repeat(repeats)[:tokens]

        if count == 0:
            hidden = tensors.hidden_states.new_zeros(
                tokens, self.fleet.token_hidden_size
            )
            top_k = tensors.topk_ids.shape[1]
            ids = (
                torch.arange(
                    tokens * top_k,
                    dtype=tensors.topk_ids.dtype,
                    device=hidden.device,
                ).view(tokens, top_k)
                % self.fleet.num_experts
            )
            weights = tensors.topk_weights.new_full(ids.shape, 1.0 / top_k)
        else:
            hidden = resize(tensors.hidden_states)
            ids = resize(tensors.topk_ids)
            weights = resize(tensors.topk_weights)
        assert hidden is not None and ids is not None and weights is not None
        return replace(
            tensors,
            hidden_states=hidden,
            topk_ids=ids,
            topk_weights=weights,
            scales=resize(tensors.scales),
            output=None,
        )

    def _prepare(self, tensors: Any) -> None:
        context = _active_context
        if context is None:
            raise RuntimeError("MegaMoE profiles require the startup profile context")

        self.capacities = _profile_capacities(
            cdiv(context.decode_tokens, self.sp_size),
            cdiv(context.prefill_tokens, self.sp_size),
        )
        self._agree(self.capacities)
        for capacity in self.capacities:
            live_buckets = tuple(self.layer.live_token_buckets(capacity))
            if any(tokens < 0 or tokens > capacity for tokens in live_buckets):
                raise ValueError(
                    f"invalid MegaMoE live-token buckets for capacity {capacity}: "
                    f"{live_buckets}"
                )
            self._agree((capacity, live_buckets))
            profiles: tuple[int | None, ...] = live_buckets or (None,)
            handles = []
            for live_tokens in profiles:
                workspace = self.layer.create_workspace(
                    capacity, live_tokens=live_tokens
                )
                warmup_tokens = live_tokens if live_tokens is not None else capacity
                self.layer.warmup(
                    self._profile_inputs(tensors, warmup_tokens),
                    workspace=workspace,
                )
                handles.append((live_tokens, workspace))
            self.workspaces[capacity] = handles
            logger.info(
                "Prepared FlashInfer MegaMoE capacity=%d live_tokens=%s",
                capacity,
                live_buckets or "default",
            )

    def _token_count(self, local_rows: int) -> int:
        if self.bootstrap.world_size == 1:
            return local_rows
        if not is_forward_context_available():
            return local_rows
        metadata = get_forward_context().dp_metadata
        if metadata is None:
            return local_rows
        if metadata.local_sizes is not None:
            return max(metadata.local_sizes)
        return cdiv(int(metadata.num_tokens_across_dp_cpu.max()), self.sp_size)

    def _workspace(self, capacity: int, tokens: int) -> Any:
        handles = self.workspaces[capacity]
        if handles[0][0] is None:
            return handles[0][1]
        buckets = [bucket for bucket, _ in handles if bucket is not None]
        assert len(buckets) == len(handles)
        index = bisect_left(buckets, tokens)
        if index == len(handles):
            index -= 1
        return handles[index][1]

    def forward(self, tensors: Any, *, return_workspace_view: bool) -> torch.Tensor:
        if not self.workspaces:
            if _active_context is None:
                return self.layer.forward(
                    tensors, return_workspace_view=return_workspace_view
                )
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE workspace setup must finish before CUDA Graph capture"
                )
            self._prepare(tensors)

        count = self._token_count(tensors.num_tokens)
        index = bisect_left(self.capacities, max(count, 1))
        if index == len(self.capacities):
            raise ValueError(
                f"MegaMoE received {count} rows beyond prepared {self.capacities}"
            )
        capacity = self.capacities[index]
        return self.layer.forward(
            tensors,
            workspace=self._workspace(capacity, count),
            return_workspace_view=return_workspace_view,
        )
