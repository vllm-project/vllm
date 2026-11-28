# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder length bucketing for padded multimodal (audio) encoders.

A padded encoder computes every item over the padded window, so a 2 s clip
batched with a 30 s clip pays for far more frames than it needs. Bucketing
re-groups the batch by true item length into discrete length rungs and computes
each rung over its own window: it removes padding, never content.

It runs in `EncoderRunner.execute_mm_encoder` before the CUDA-graph-or-eager
decision; each rung is a fixed-shape sub-batch, so the CUDA-graph path keeps
working. Losslessness is per encoder class: mask-aware crops are bit-exact.

The feature is opt-in via `additional_config.audio_bucket_seconds` (a seconds
ladder, e.g. `[2, 4, 8, 16, 30]`) and engages only for models that declare an
`EncoderBucketingSpec`; declaring the spec is the opt-in. Setting
`additional_config.audio_bucket_budget` additionally enables compute-budget
merging. With the feature off, no shim is installed and the original forward
runs untouched; a per-batch waste gate keeps bucketing from ever computing more
than the untouched forward.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_CONFIG_KEY = "audio_bucket_seconds"
_CONFIG_KEY_BUDGET = "audio_bucket_budget"

# Fixed-window encoders pin their window length inside `forward`, so rung-sized
# inputs are rejected. Adapted models register a variable-length recipe here at
# import time (see `qwen2_audio.py`), keyed by the wrapped class's exact name so
# subclasses with different semantics never inherit it.
_FORWARD_SHIMS: dict[str, Callable[..., Any]] = {}


def register_forward_shim(class_name: str, shim_fn: Callable[..., Any]) -> None:
    _FORWARD_SHIMS[class_name] = shim_fn


@dataclass(frozen=True)
class LengthBucketingConfig:
    """A discrete ladder of rung lengths in feature frames, sorted ascending."""

    buckets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("LengthBucketingConfig.buckets must be non-empty")
        if tuple(self.buckets) != tuple(sorted(self.buckets)):
            raise ValueError("LengthBucketingConfig.buckets must be sorted ascending")

    def padded_length(self, length: int) -> int | None:
        """Smallest rung `>=` `length`, or `None` when the item exceeds the ladder."""
        for rung in self.buckets:
            if rung >= length:
                return rung
        return None


def _ladder_seconds(
    additional_config: Mapping[str, object] | None,
) -> tuple[float, ...] | None:
    """Read `audio_bucket_seconds` as sorted unique seconds, or `None` when unset."""
    raw = (
        additional_config.get(_CONFIG_KEY)
        if isinstance(additional_config, Mapping)
        else None
    )
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"additional_config.{_CONFIG_KEY} must be a list of seconds, "
            f"got {raw!r}"
        )
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"additional_config.{_CONFIG_KEY} entries must be numbers, "
                f"got {item!r}"
            )
        value = float(item)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"additional_config.{_CONFIG_KEY} entries must be > 0, "
                f"got {item!r}"
            )
        values.append(value)
    seconds = sorted(set(values))
    return tuple(seconds) if seconds else None


def _compute_budget(
    additional_config: Mapping[str, object] | None,
) -> int | None:
    """Read `audio_bucket_budget` as the per-call compute bound, or `None`."""
    raw = (
        additional_config.get(_CONFIG_KEY_BUDGET)
        if isinstance(additional_config, Mapping)
        else None
    )
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"additional_config.{_CONFIG_KEY_BUDGET} must be a positive "
            f"int, got {raw!r}"
        )
    if raw <= 0:
        raise ValueError(
            f"additional_config.{_CONFIG_KEY_BUDGET} must be > 0, "
            f"got {raw!r}"
        )
    return raw


def _rungs_from_seconds(
    seconds: tuple[float, ...], frames_per_second: int, align_to: int = 1
) -> tuple[int, ...]:
    """Convert seconds to mel-frame rungs; `align_to` rounds each rung up."""
    rungs = [max(1, int(round(s * frames_per_second))) for s in seconds]
    if align_to > 1:
        rungs = [((r + align_to - 1) // align_to) * align_to for r in rungs]
    return tuple(rungs)


class EncoderBucketingSpec:
    """Per-model length and crop semantics; declaring a subclass opts the model in.

    Subclasses set `frames_per_second`, `context_margin` (extra frames past the
    published length so a rung never clips what the encoder reads; `0` is valid)
    and optionally `align_to`, and implement the three methods.
    """

    frames_per_second: int | None = None
    context_margin: int | None = None
    align_to: int = 1

    def item_lengths(self, mm_kwargs_batch: Mapping[str, Any]) -> list[int] | None:
        """Per-item true lengths in feature frames, or `None` to skip bucketing."""
        raise NotImplementedError

    def padded_time(self, mm_kwargs_batch: Mapping[str, Any]) -> int:
        """Padded time extent of the batch (the no-bucket compute per item)."""
        raise NotImplementedError

    def crop(
        self,
        mm_kwargs_batch: Mapping[str, Any],
        indices: list[int],
        rung: int,
    ) -> dict[str, Any]:
        """Crop the selected rows to `rung` frames; content is never truncated."""
        raise NotImplementedError


class EncoderBucketing:
    """Groups a batched encoder input into fixed-length rung sub-batches.

    Args:
        spec: The model's length/crop semantics.
        ladder: Rung ladder, or `None` to disable bucketing.
        compute_budget: Compute-budget batching: the per-call compute bound
            (items x padded frames). Oversized groups split into balanced
            chunks; groups below `fill_floor` merge up toward it. `None` emits
            one call per rung (plain bucketing).

    """

    def __init__(
        self,
        spec: EncoderBucketingSpec,
        ladder: LengthBucketingConfig | None,
        compute_budget: int | None = None,
    ):
        self.spec = spec
        self.ladder = ladder
        self.compute_budget = compute_budget
        # Derived, not a knob: a c=8/32/128 sweep moved results under 2%.
        self.fill_floor = max(1, compute_budget // 6) if compute_budget else None

    @property
    def active(self) -> bool:
        """Whether bucketing is enabled (a rung ladder is configured)."""
        return self.ladder is not None

    def bucket(
        self,
        mm_kwargs_batch: dict[str, Any],
        num_items: int,
    ) -> list[tuple[list[int], dict[str, Any]]]:
        """Split a batch into rung sub-batches, or one full batch when not useful."""
        full = [(list(range(num_items)), mm_kwargs_batch)]
        ladder = self.ladder
        if ladder is None or num_items == 0:
            return full
        lengths = self.spec.item_lengths(mm_kwargs_batch)
        if lengths is None or len(lengths) != num_items:
            return full
        time_extent = self.spec.padded_time(mm_kwargs_batch)
        if time_extent <= 0:
            return full
        margin = self.spec.context_margin or 0
        windows: list[int] = []
        for length in lengths:
            rung = ladder.padded_length(length + margin)
            if rung is None:
                # No safe rung: run the untouched forward, never truncate.
                return full
            windows.append(min(rung, time_extent))
        # Waste gate: only bucket when it strictly reduces computed frames.
        if sum(windows) >= num_items * time_extent:
            return full
        if self.compute_budget is not None and (
            num_items * max(windows) <= self.compute_budget
        ):
            # Whole batch fits one call: the unbucketed behavior, no splitting.
            rung = max(windows)
            indices = list(range(num_items))
            return [(indices, self.spec.crop(mm_kwargs_batch, indices, rung))]
        groups: dict[int, list[int]] = {}
        for index, window in enumerate(windows):
            groups.setdefault(window, []).append(index)
        budget = self.compute_budget
        if budget is not None:
            groups = self._fold_rungs_to_floor(groups, budget, self.fill_floor or 1)
            calls: list[tuple[int, list[int]]] = self._split_oversized(groups, budget)
        else:
            calls = list(groups.items())
        return [
            (indices, self.spec.crop(mm_kwargs_batch, indices, rung))
            for rung, indices in calls
        ]

    def _fold_rungs_to_floor(
        self, groups: dict[int, list[int]], budget: int, floor: int
    ) -> dict[int, list[int]]:
        """Fold under-full rungs into the next rung, capped by the budget.

        Groups at or above the floor never fold.
        """
        rungs = sorted(groups)
        folded: dict[int, list[int]] = {}
        i = 0
        while i < len(rungs):
            rung = rungs[i]
            indices = list(groups[rung])
            compute = len(indices) * rung
            j = i + 1
            while j < len(rungs) and compute < floor:
                next_rung = rungs[j]
                merged_compute = (len(indices) + len(groups[next_rung])) * next_rung
                if merged_compute > budget:
                    break
                indices.extend(groups[next_rung])
                rung = next_rung
                compute = merged_compute
                j += 1
            folded[rung] = indices
            i = j
        return folded

    def _split_oversized(
        self, groups: dict[int, list[int]], budget: int
    ) -> list[tuple[int, list[int]]]:
        """Split groups above the budget into balanced chunks (~ budget // rung)."""
        calls: list[tuple[int, list[int]]] = []
        for rung, indices in groups.items():
            n = len(indices)
            max_bs = max(1, budget // rung)
            if n <= max_bs:
                calls.append((rung, indices))
                continue
            num_calls = math.ceil(n / max_bs)
            chunk = math.ceil(n / num_calls)
            for start in range(0, n, chunk):
                calls.append((rung, indices[start : start + chunk]))
        return calls


def _gpu_execution_supported() -> bool:
    """Whether the GPU encoder runner executes on this worker.

    The GPU encoder runner is the bucketer's only consumer.
    """
    from vllm.platforms import current_platform

    return current_platform.is_cuda_alike()


def install_encoder_bucketing(
    model: nn.Module,
    encoders: Mapping[str, nn.Module],
    spec: EncoderBucketingSpec,
    *,
    additional_config: Mapping[str, object] | None = None,
) -> None:
    """Install variable-length encoder shims and register the bucketer.

    Declaring a spec opts the model in; bucketing engages only on GPU execution
    paths and when `additional_config.audio_bucket_seconds` is set. Idempotent.

    Args:
        model: The encoder-owning model to register the bucketer on.
        encoders: `label -> encoder module` map declared by the model.
        spec: The model's length/crop semantics and frame-rate knobs.
        additional_config: The `VllmConfig.additional_config` mapping carrying
            the opt-in bucketing ladder.

    Raises:
        ValueError: If `encoders` is empty or a declared encoder is not a module.
        RuntimeError: If the spec does not set `frames_per_second` or
            `context_margin` when enabled.

    """
    if model is None or getattr(model, "_encoder_bucketing", None) is not None:
        return
    if not encoders:
        raise ValueError(
            "install_encoder_bucketing: declare at least one encoder (label -> module)"
        )
    for label, module in encoders.items():
        if not isinstance(module, nn.Module):
            raise ValueError(
                f"install_encoder_bucketing: declared encoder {label!r} is not "
                f"an nn.Module, got {type(module).__name__}"
            )

    ladder = None
    compute_budget = _compute_budget(additional_config)
    seconds = _ladder_seconds(additional_config)
    if seconds is not None and not _gpu_execution_supported():
        logger.warning(
            "Encoder length bucketing is a GPU-only feature (only the GPU "
            "encoder runner consumes the bucketer); this worker is not a GPU "
            "execution path, so bucketing stays disabled and the original "
            "encoder forward runs unchanged."
        )
        seconds = None
    if seconds is not None:
        frames_per_second = spec.frames_per_second
        if not isinstance(frames_per_second, int):
            raise RuntimeError("EncoderBucketingSpec must set frames_per_second")
        if not isinstance(spec.context_margin, int):
            raise RuntimeError(
                "EncoderBucketingSpec must set context_margin "
                "(0 is a valid explicit margin)"
            )
        ladder = LengthBucketingConfig(
            _rungs_from_seconds(seconds, frames_per_second, spec.align_to)
        )

    manifest: list[tuple[nn.Module, object]] = []
    if ladder is not None:
        for label, module in encoders.items():
            shim_fn = _FORWARD_SHIMS.get(type(module).__name__)
            if shim_fn is not None:
                manifest.append((module, module.forward))
                module.forward = functools.partial(shim_fn, module)
                logger.info(
                    "Bucketed encoder %r: fixed-window class, variable-length "
                    "recipe installed.",
                    label,
                )

    model._encoder_bucketing = EncoderBucketing(
        spec,
        ladder,
        compute_budget=compute_budget,
    )
    model._encoder_bucketing_manifest = manifest
    if ladder is not None:
        logger.info(
            "Encoder length bucketing enabled over rungs %s for modules %s.",
            ladder.buckets,
            list(encoders),
        )
    else:
        logger.info(
            "Encoder length bucketing disabled; enable it with additional_config.%s "
            "(e.g. [2, 4, 8, 16, 30]).",
            _CONFIG_KEY,
        )


def uninstall_encoder_bucketing(model: nn.Module) -> list[str]:
    """Restore shimmed `forward`s and drop the bucketer; returns restored names."""
    manifest = getattr(model, "_encoder_bucketing_manifest", None) or []
    restored: list[str] = []
    for module, original in manifest:
        module.forward = original
        restored.append(type(module).__name__)
    if manifest:
        del model._encoder_bucketing_manifest
    if getattr(model, "_encoder_bucketing", None) is not None:
        del model._encoder_bucketing
    return restored
