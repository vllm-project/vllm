# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape keyed Punica metadata for shared model Uno drafting.

The regular LoRA path rebuilds Punica metadata from a per token mapping on
every forward.  ``LoRAKernelMeta.prepare_tensors`` is correct, but its data
dependent device operations update CPU scalar flags and therefore introduce a
host wait.  Uno has two mappings whose contents are known from the batch shape:
an all base target mapping and one base row followed by adapter rows for each
draft request.  This module snapshots the native result for each shape and
restores it with in place copies on later uses.

The first use of a shape deliberately calls the native wrapper builder.  That
keeps this helper tied to the backend's actual metadata contract.  A resident
cache hit only copies prepared tensors; it does not inspect device
values or replace buffers, so CUDA graph captured addresses remain valid.
Evicted shapes and adapter-slot changes require another native preparation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# These names mirror PunicaWrapperBase and LoRAKernelMeta.  Keeping the field
# list here avoids changing the native LoRA implementation for this focused
# optimization; moving or resizing these known buffers invalidates the cache.
_BASE_FIELDS = (
    "_token_lora_indices",
    "_sampler_indices",
    "_sampler_indices_padded",
    "_embeddings_indices",
)
_KERNEL_PREFIX = ("token_lora_mapping", "token_indices_sorted_by_lora_ids")
_KERNEL_RESET = (
    "active_lora_ids",
    "num_tokens_per_lora",
    "lora_token_start_loc",
    "no_lora_flag_cpu",
    "num_active_loras_cpu",
)
_METAS = ("token_mapping_meta", "prompt_mapping_meta")


def draft_lora_mapping(
    num_input_tokens: int,
    batch_size: int,
    k: int,
    uno_lora_id: int,
) -> tuple[int, ...]:
    """Return the draft token mapping, including graph padding.

    Each request has one base weighted seed row and ``k - 1`` rows using the
    Uno adapter.  Rows after ``batch_size * k`` are graph padding and stay on
    the base model.  The prompt mapping is intentionally separate: callers
    should use ``base_prompt_mapping`` because draft logits are produced by the
    base vocabulary head.
    """
    if num_input_tokens < 0 or batch_size < 1 or k < 1:
        raise ValueError("Uno draft mapping dimensions must be positive")
    actual = batch_size * k
    if num_input_tokens < actual:
        raise ValueError("draft mapping cannot truncate real query rows")
    per_request = (0,) + (int(uno_lora_id),) * (k - 1)
    return per_request * batch_size + (0,) * (num_input_tokens - actual)


def base_lora_mapping(num_input_tokens: int) -> tuple[int, ...]:
    """Return an all base token mapping of the requested length."""
    if num_input_tokens < 0:
        raise ValueError("base mapping length must be nonnegative")
    return (0,) * num_input_tokens


def base_prompt_mapping(num_requests: int) -> tuple[int, ...]:
    """Return an all base prompt/sampler mapping."""
    if num_requests < 0:
        raise ValueError("base prompt mapping length must be nonnegative")
    return (0,) * num_requests


def _native_token_capacity(wrapper: Any) -> int | None:
    """Return the shared native capacity when its token buffers are known."""
    capacities: list[int] = []
    try:
        for name in _BASE_FIELDS:
            value = getattr(wrapper, name)
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                return None
            capacities.append(int(value.shape[0]))
        for owner in _METAS:
            meta = getattr(wrapper, owner)
            for name in _KERNEL_PREFIX:
                value = getattr(meta, name)
                if not isinstance(value, torch.Tensor) or value.ndim != 1:
                    return None
                capacities.append(int(value.shape[0]))
    except (AttributeError, TypeError, ValueError):
        return None
    return min(capacities) if capacities else None


def _base_plan_size(num_input_tokens: int, native_capacity: int | None) -> int:
    """Round an all-base plan upward without exceeding native storage."""
    if num_input_tokens < 0:
        raise ValueError("base plan length must be nonnegative")
    if native_capacity is None:
        return num_input_tokens
    if num_input_tokens > native_capacity:
        raise ValueError(
            "Uno base physical rows exceed native LoRA metadata capacity "
            f"({native_capacity})"
        )
    if num_input_tokens <= 1:
        return num_input_tokens
    return min(1 << (num_input_tokens - 1).bit_length(), native_capacity)


def _describe(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, torch.Tensor):
        raise AttributeError("expected a tensor metadata buffer")
    if value.ndim != 1:
        raise AttributeError("expected a one-dimensional metadata buffer")
    return (
        value.data_ptr(),
        tuple(value.shape),
        value.stride(),
        value.dtype,
        value.device,
    )


def wrapper_fingerprint(wrapper: Any) -> tuple[Any, ...] | None:
    """Describe the persistent native buffers without reading their values."""
    try:
        parts: list[Any] = []
        for name in _BASE_FIELDS:
            parts.append((name, _describe(getattr(wrapper, name))))
        for owner in _METAS:
            meta = getattr(wrapper, owner)
            for name in (
                *_KERNEL_PREFIX,
                *_KERNEL_RESET,
                "default_num_active_loras_cpu",
            ):
                parts.append(((owner, name), _describe(getattr(meta, name))))
            parts.append(
                (
                    (owner, "captured_lora_counts"),
                    tuple(getattr(meta, "captured_lora_counts", ())),
                )
            )
        return tuple(parts)
    except (AttributeError, TypeError):
        return None


@dataclass(frozen=True)
class _Plan:
    lengths: tuple[int, ...]
    is_prefill: bool
    tensors: tuple[tuple[Any, torch.Tensor], ...]

    def restore(self, wrapper: Any) -> None:
        # All copies are to the original tensors.  In particular, never replace
        # a tensor object: graph kernels retain these exact addresses.
        for name, snapshot in self.tensors:
            if isinstance(name, tuple):
                target = getattr(getattr(wrapper, name[0]), name[1])
            else:
                target = getattr(wrapper, name)
            target[: snapshot.shape[0]].copy_(snapshot, non_blocking=True)
        wrapper.indices_len[:] = self.lengths
        wrapper.is_prefill = self.is_prefill


def snapshot_plan(wrapper: Any) -> _Plan | None:
    """Snapshot the fields written by native Punica metadata preparation."""
    try:
        lengths_raw = tuple(wrapper.indices_len)
        if len(lengths_raw) != len(_BASE_FIELDS) or any(
            value is None or int(value) < 0 for value in lengths_raw
        ):
            return None
        lengths = tuple(int(value) for value in lengths_raw)

        # The native GPU metadata keeps these scalar flags on the CPU.  Refuse
        # an unfamiliar layout rather than accidentally synchronizing while
        # deciding whether it is safe to cache.
        for owner in _METAS:
            meta = getattr(wrapper, owner)
            for name in ("no_lora_flag_cpu", "num_active_loras_cpu"):
                if getattr(meta, name).device.type != "cpu":
                    return None

        tensors: list[tuple[Any, torch.Tensor]] = [
            (name, getattr(wrapper, name)[:length].clone())
            for name, length in zip(_BASE_FIELDS, lengths, strict=True)
        ]
        for owner, length in zip(_METAS, lengths[:2], strict=True):
            meta = getattr(wrapper, owner)
            tensors.extend(
                ((owner, name), getattr(meta, name).clone()) for name in _KERNEL_RESET
            )
            # prepare_tensors returns immediately for all-base rows, leaving
            # its prefix buffers untouched.  Preserve that native behavior by
            # not snapshotting or restoring those stale buffers in that case.
            if not bool(meta.no_lora_flag_cpu[0]):
                tensors.extend(
                    ((owner, name), getattr(meta, name)[:length].clone())
                    for name in _KERNEL_PREFIX
                )
        return _Plan(lengths, bool(wrapper.is_prefill), tuple(tensors))
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


class UnoLoRAPlanCache:
    """Bounded shape and slot keyed cache of native Punica plans.

    A cache belongs to one Punica wrapper.  ``slots`` is part of each key
    because an adapter can move between GPU slots after an out of band load or
    eviction even though the mapping's integer id is unchanged.  Base plans
    are keyed by their physical model-row bucket; their all-base prompt
    metadata is padded to that same bucket, while the caller's actual logits
    count is validated separately.  A cache hit is therefore resident-bucket
    reuse, not a promise that heterogeneous physical buckets avoid native
    metadata preparation.
    """

    def __init__(self, capacity: int = 64):
        if capacity < 1:
            raise ValueError("plan cache capacity must be positive")
        self.capacity = capacity
        self.entries: OrderedDict[tuple[Any, ...], _Plan] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.invalidations = 0
        self._fingerprint: tuple[Any, ...] | None = None

    @staticmethod
    def _is_capturing(wrapper: Any) -> bool:
        token_buffer = getattr(wrapper, "_token_lora_indices", None)
        return (
            isinstance(token_buffer, torch.Tensor)
            and token_buffer.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        )

    def _key(
        self,
        wrapper: Any,
        kind: str,
        shape: tuple[Any, ...],
        slots: tuple[int | None, ...],
    ) -> tuple[Any, ...] | None:
        if self._is_capturing(wrapper):
            return None
        fingerprint = wrapper_fingerprint(wrapper)
        if fingerprint is None:
            logger.warning_once(
                "Uno LoRA plan cache bypassed because the native Punica metadata "
                "layout is unexpected; using native metadata preparation."
            )
            return None
        if self._fingerprint != fingerprint:
            if self._fingerprint is not None:
                self.invalidations += 1
            self.entries.clear()
            self._fingerprint = fingerprint
        return kind, tuple(shape), tuple(slots)

    def restore(
        self,
        wrapper: Any,
        kind: str,
        shape: tuple[Any, ...],
        slots: tuple[int | None, ...],
    ) -> bool:
        """Restore a plan in place; return false when native build is needed."""
        key = self._key(wrapper, kind, shape, slots)
        if key is None:
            self.bypasses += 1
            return False
        plan = self.entries.get(key)
        if plan is None:
            self.misses += 1
            return False
        self.hits += 1
        self.entries.move_to_end(key)
        plan.restore(wrapper)
        return True

    def remember(
        self,
        wrapper: Any,
        kind: str,
        shape: tuple[Any, ...],
        slots: tuple[int | None, ...],
    ) -> None:
        """Remember the result of one native metadata build."""
        key = self._key(wrapper, kind, shape, slots)
        if key is None:
            self.bypasses += 1
            return
        plan = snapshot_plan(wrapper)
        if plan is None:
            logger.warning_once(
                "Uno LoRA plan cache bypassed because native Punica metadata "
                "could not be snapshotted; using native metadata preparation."
            )
            self.bypasses += 1
            return
        self.entries[key] = plan
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def install(
        self,
        wrapper: Any,
        kind: str,
        shape: tuple[Any, ...],
        slots: tuple[int | None, ...],
        build: Callable[[], None],
    ) -> bool:
        """Restore a cached plan or run ``build`` on a shape miss.

        The return value is true only for an in-place cache hit.  The caller's
        ``build`` function is therefore the sole place where the native
        metadata builder can run, and can construct its mapping lazily.
        """
        if self.restore(wrapper, kind, shape, slots):
            return True
        build()
        self.remember(wrapper, kind, shape, slots)
        return False

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "invalidations": self.invalidations,
            "entries": len(self.entries),
        }


class UnoLoRAState:
    """Runner facing adapter lifecycle and plan installation helper.

    ``worker_lora_manager`` is the existing vLLM worker LoRA manager.  Adapter
    loading and activation go through that manager, while per step mapping
    changes bypass only its metadata rebuild and write the existing Punica
    buffers in place.  This class intentionally handles the language model
    wrapper used by the text-only Uno scope.
    """

    def __init__(
        self,
        worker_lora_manager: Any,
        lora_request: Any,
        capacity: int = 64,
        mapping_cls: Any | None = None,
    ):
        self.worker_lora_manager = worker_lora_manager
        self.adapter_manager = getattr(
            worker_lora_manager, "_adapter_manager", worker_lora_manager
        )
        self.lora_request = lora_request
        self.adapter_id = int(lora_request.lora_int_id)
        self.plan_cache = UnoLoRAPlanCache(capacity)
        self.mapping_cls = mapping_cls

    def ensure_adapter(self) -> None:
        """Register and activate the adapter through vLLM's manager."""
        list_adapters = getattr(self.worker_lora_manager, "list_adapters", None)
        if list_adapters is None:
            list_adapters = self.adapter_manager.list_adapters
        if self.adapter_id not in list_adapters():
            add_adapter = getattr(self.worker_lora_manager, "add_adapter", None)
            if add_adapter is None:
                add_adapter = self.adapter_manager.add_adapter
            add_adapter(self.lora_request)
        self.adapter_manager.activate_adapter(self.adapter_id)

    def _wrapper(self) -> Any:
        get_wrapper = getattr(self.adapter_manager, "_get_punica_wrapper", None)
        if get_wrapper is None:
            raise RuntimeError("Uno requires the native Punica LoRA manager")
        wrapper = get_wrapper("language_model")
        has_wrapper_fields = wrapper is not None and all(
            hasattr(wrapper, name) for name in (*_BASE_FIELDS, *_METAS, "indices_len")
        )
        has_meta_fields = has_wrapper_fields and all(
            hasattr(getattr(wrapper, owner), name)
            for owner in _METAS
            for name in (
                *_KERNEL_PREFIX,
                *_KERNEL_RESET,
                "default_num_active_loras_cpu",
                "captured_lora_counts",
            )
        )
        if not has_meta_fields:
            raise RuntimeError("Uno requires a Punica GPU language-model wrapper")
        return wrapper

    def _install(
        self,
        kind: str,
        shape: tuple[Any, ...],
        token_mapping: Callable[[], tuple[int, ...]],
        prompt_mapping: Callable[[], tuple[int, ...]],
        is_prefill: bool = True,
    ) -> bool:
        wrapper = self._wrapper()
        slots = tuple(self.adapter_manager.lora_index_to_id)
        max_loras = int(self.adapter_manager.lora_slots) + 1
        vocab_size = int(self.adapter_manager.vocab_size)

        def build() -> None:
            mapping_cls = self.mapping_cls
            if mapping_cls is None:
                from vllm.lora.layers import LoRAMapping

                mapping_cls = LoRAMapping

            wrapper.update_metadata(
                mapping_cls(
                    token_mapping(),
                    prompt_mapping(),
                    is_prefill=is_prefill,
                ),
                list(slots),
                max_loras,
                vocab_size,
            )

        restored = self.plan_cache.install(wrapper, kind, shape, slots, build)
        # Direct installs deliberately bypass set_adapter_mapping().  Clear its
        # equality cache so a later ordinary request mapping cannot be skipped.
        if hasattr(self.adapter_manager, "_last_mapping"):
            self.adapter_manager._last_mapping = None
        if hasattr(self.adapter_manager, "_last_slot_layout"):
            self.adapter_manager._last_slot_layout = None
        return restored

    def install_draft(self, num_input_tokens: int, batch_size: int, k: int) -> bool:
        """Install one base row plus Uno adapter rows per request."""
        self.ensure_adapter()
        shape = (int(num_input_tokens), int(batch_size), int(k), self.adapter_id)
        return self._install(
            "draft",
            shape,
            lambda: draft_lora_mapping(
                num_input_tokens, batch_size, k, self.adapter_id
            ),
            # A full draft graph also computes logits for padded requests;
            # their sampling slots are -1. Keep base head metadata sized to
            # that physical tensor, including padding.
            lambda: base_prompt_mapping(num_input_tokens),
        )

    def install_base(self, num_input_tokens: int, num_logits_rows: int) -> bool:
        """Install an all-base target mapping without unloading the adapter.

        ``num_input_tokens`` is the number of model rows actually presented to
        the native model, including any graph padding.  ``num_logits_rows`` is
        the unpadded logits/head row count and is used only to validate that
        the physical bucket covers the rows consumed by the head.  The cache
        rounds all-base physical plans upward to a geometric bucket when the
        native token buffers expose a shared capacity; draft plans retain their
        exact lengths.  The prompt metadata deliberately keeps the bucket
        length so target and hook restores share one base cache entry; all
        padded rows remain base rows.
        """
        num_input_tokens = int(num_input_tokens)
        num_logits_rows = int(num_logits_rows)
        if num_logits_rows < 0 or num_logits_rows > num_input_tokens:
            raise ValueError(
                "Uno base logits rows must fit within the physical model rows"
            )
        wrapper = self._wrapper()
        physical_num_tokens = _base_plan_size(
            num_input_tokens, _native_token_capacity(wrapper)
        )
        shape = (physical_num_tokens,)
        return self._install(
            "base",
            shape,
            lambda: base_lora_mapping(physical_num_tokens),
            lambda: base_prompt_mapping(physical_num_tokens),
        )


__all__ = [
    "UnoLoRAPlanCache",
    "UnoLoRAState",
    "base_lora_mapping",
    "base_prompt_mapping",
    "draft_lora_mapping",
    "snapshot_plan",
    "wrapper_fingerprint",
]
