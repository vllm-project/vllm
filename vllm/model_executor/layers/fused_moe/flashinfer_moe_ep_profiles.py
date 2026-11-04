# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup tuning and fixed workspace profiles for FlashInfer MegaMoE."""

from __future__ import annotations

import json
import os
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from pathlib import Path
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
    records: dict[str, Any] = field(default_factory=dict)
    dirty: bool = False


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
    *,
    decode_tokens: int,
    prefill_tokens: int,
    cache_file: Path | None = None,
    cache_identity: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Share tuning results across layers and persist them between startups."""
    global _active_context
    previous = _active_context
    context = _ProfileContext(decode_tokens, prefill_tokens)
    leader = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    identity = {"version": 1, "environment": cache_identity}
    if cache_file is not None and leader:
        try:
            payload = json.loads(cache_file.read_text())
            if (
                isinstance(payload, dict)
                and payload.get("identity") == identity
                and isinstance(payload.get("records"), dict)
            ):
                context.records = payload["records"]
        except (OSError, ValueError):
            pass
    _active_context = context
    try:
        yield
        if cache_file is not None and leader and context.dirty:
            from vllm.model_executor.warmup.flashinfer_autotune_cache import (
                write_flashinfer_autotune_cache,
            )

            try:
                write_flashinfer_autotune_cache(
                    cache_file,
                    json.dumps(
                        {"identity": identity, "records": context.records},
                        sort_keys=True,
                    ).encode(),
                )
            except OSError:
                logger.warning("Could not save MegaMoE tuning cache %s", cache_file)
    finally:
        _active_context = previous


@contextmanager
def flashinfer_moe_ep_profile_context(runner: Any) -> Iterator[None]:
    """Prepare fixed workspaces before measuring serving memory requirements."""
    import flashinfer

    from vllm.model_executor.warmup.flashinfer_autotune_cache import (
        resolve_flashinfer_autotune_file,
    )

    packages: dict[str, str | None] = {}
    for package in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "nvshmem4py-cu13"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    device = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    identity = {
        "flashinfer": flashinfer.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": [
            device.name,
            device.major,
            device.minor,
            device.multi_processor_count,
        ],
        "packages": packages,
        "mega_use_ncu": os.environ.get("MEGA_USE_NCU", "0"),
    }
    with flashinfer_moe_ep_profile_token_context(
        decode_tokens=_decode_token_limit(runner),
        prefill_tokens=_runner_limit(
            runner, "max_num_tokens", "max_num_batched_tokens"
        ),
        cache_file=resolve_flashinfer_autotune_file(runner).with_name(
            "megamoe_knobs_v1.json"
        ),
        cache_identity=identity,
    ):
        yield


class FlashInferMoeEpProfiles:
    """Tune once per geometry and retain fixed workspaces for serving."""

    def __init__(
        self,
        layer: Any,
        bootstrap: Any,
        fleet: Any,
        *,
        backend: Any,
        sp_size: int = 1,
    ) -> None:
        if sp_size < 1:
            raise ValueError(f"sp_size must be positive, got {sp_size}")
        self.layer = layer
        self.bootstrap = bootstrap
        self.fleet = fleet
        self.backend = backend
        self.sp_size = sp_size
        self.workspaces: dict[int, list[tuple[int, Any]]] = {}
        self.capacities: tuple[int, ...] = ()
        _profile_owners.add(self)

    def destroy(self) -> None:
        """Release workspace handles and remove this owner from memory planning."""
        _profile_owners.discard(self)
        self.workspaces.clear()

    @staticmethod
    def is_supported(layer: Any) -> bool:
        create = getattr(layer, "create_workspace", None)
        if not callable(create) or not callable(
            getattr(layer, "get_tuned_knobs", None)
        ):
            return False
        return {"live_tokens", "knobs"} <= signature(create).parameters.keys()

    def _broadcast(self, value: Any) -> Any:
        if self.bootstrap.world_size == 1:
            return value
        values = [value]
        torch.distributed.broadcast_object_list(
            values, group=self.bootstrap.process_group, group_src=0
        )
        return values[0]

    def _profile_key(self, capacity: int, tokens: int) -> str:
        config = self.backend.megakernel
        settings = {
            item.name: (
                getattr(config, item.name).value
                if isinstance(getattr(config, item.name), Enum)
                else getattr(config, item.name)
            )
            for item in fields(config)
            if item.name
            not in {
                "knobs",
                "input_norm_const",
                "fc1_alpha",
                "fc2_alpha",
                "fc1_norm_const",
            }
        }
        return json.dumps(
            {
                "backend": type(config).__name__,
                "settings": settings,
                "quantize_input": self.backend.quantize_input,
                "world_size": self.bootstrap.world_size,
                "hidden": self.fleet.token_hidden_size,
                "experts": self.fleet.num_experts,
                "capacity": capacity,
                "live_tokens": tokens,
            },
            sort_keys=True,
        )

    def _tune(self, tensors: Any, capacity: int, tokens: int) -> dict[str, Any]:
        temporary = type(self.layer)(
            self.bootstrap,
            replace(
                self.fleet, max_tokens_per_rank=capacity, tuning_live_tokens=tokens
            ),
            None,
            replace(
                self.backend,
                megakernel=replace(self.backend.megakernel, knobs="auto"),
            ),
        )
        try:
            temporary.warmup(tensors)
            return temporary.get_tuned_knobs()
        finally:
            temporary.destroy()

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

        if (
            torch.distributed.is_initialized()
            and self.bootstrap.world_size != torch.distributed.get_world_size()
        ):
            raise RuntimeError("MegaMoE startup tuning requires EP to span WORLD")
        decode = max(1, cdiv(context.decode_tokens, self.sp_size))
        prefill = max(1, cdiv(context.prefill_tokens, self.sp_size))
        self.capacities = tuple(sorted({decode, prefill}))
        self._agree(self.capacities)
        for capacity in self.capacities:
            buckets = {capacity}
            if capacity == decode:
                buckets.update(1 << i for i in range(decode.bit_length()))
            live_buckets = tuple(sorted(buckets))
            self._agree((capacity, live_buckets))
            handles = []
            for live_tokens in live_buckets:
                inputs = self._profile_inputs(tensors, live_tokens)
                key = self._profile_key(capacity, live_tokens)
                knobs = self._broadcast(
                    context.records.get(key) if self.bootstrap.rank == 0 else None
                )
                if not isinstance(knobs, dict) or not knobs:
                    knobs = self._broadcast(self._tune(inputs, capacity, live_tokens))
                    if not isinstance(knobs, dict) or not knobs:
                        raise RuntimeError(
                            "MegaMoE autotune returned no selected knobs"
                        )
                    context.records[key] = knobs
                    context.dirty = True
                # FlashInfer tactics contain tuple-valued tile/cluster shapes.
                knobs = {
                    name: tuple(value) if isinstance(value, list) else value
                    for name, value in knobs.items()
                }
                workspace = self.layer.create_workspace(
                    capacity, live_tokens=live_tokens, knobs=knobs
                )
                self.layer.warmup(inputs, workspace=workspace)
                handles.append((live_tokens, workspace))
            self.workspaces[capacity] = handles

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
        buckets = [bucket for bucket, _ in handles]
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
