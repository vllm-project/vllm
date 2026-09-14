# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply dense or sparse weight updates in checkpoint coordinates.

The model's ``load_weights`` method still handles checkpoint-name mapping,
TP slicing, and packed parameters. Sparse patches use NaN to mark unchanged
checkpoint elements. The supported sparse-loader contract permits only final
same-shaped floating-point ``Tensor.copy_`` writes. Composed, multi-stage, and
custom-write loaders are unsupported.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import NamedTuple

import torch

_DEFAULT_PATCH_CHUNK_BYTES = 512 << 20

__all__ = [
    "CheckpointWeightPatch",
    "load_checkpoint_weight_patches",
]


class CheckpointWeightPatch(NamedTuple):
    """Describe one dense or sparse checkpoint-coordinate update.

    Attributes:
        name: Checkpoint weight name passed to the model loader.
        shape: Full checkpoint tensor shape.
        dtype: Checkpoint tensor dtype.
        values: The flattened full tensor for a dense patch, or the values at
            ``indices`` for a sparse patch.
        indices: Flat indices into the full checkpoint tensor described by
            ``shape``. ``None`` makes ``values`` a dense replacement.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    values: torch.Tensor
    indices: torch.Tensor | None = None


def _load_nan_masked_weights(
    model: torch.nn.Module,
    weights: list[tuple[str, torch.Tensor]],
) -> Iterable[str] | None:
    """Load NaN-masked checkpoint tensors while preserving runtime values at NaNs."""
    # Save the original copy_ so it can be restored after loading.
    original_copy = torch.Tensor.copy_

    # Enforce the sparse-loader contract and preserve destination values where
    # the corresponding source is NaN.
    def copy_non_nan_(
        destination: torch.Tensor,
        source: torch.Tensor,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        if not (
            destination.is_floating_point()
            and source.is_floating_point()
            and destination.shape == source.shape
        ):
            raise NotImplementedError(
                "Sparse checkpoint patches require a same-shaped "
                "floating-point final copy"
            )
        source = source.to(dtype=destination.dtype, device=destination.device)
        return torch.where(
            torch.isnan(source),
            destination,
            source,
            out=destination,
        )

    torch.Tensor.copy_ = copy_non_nan_
    try:
        return model.load_weights(weights)
    finally:
        torch.Tensor.copy_ = original_copy


def _validate_patch_structure(patch: CheckpointWeightPatch) -> int:
    if not patch.name:
        raise ValueError("Checkpoint weight name must be non-empty")
    if not isinstance(patch.dtype, torch.dtype):
        raise TypeError(f"{patch.name}: dtype must be a torch.dtype")
    if any(dim < 0 for dim in patch.shape):
        raise ValueError(f"{patch.name}: shape dimensions must be non-negative")
    if patch.values.ndim != 1:
        raise ValueError(f"{patch.name}: patch values must be a 1D tensor")

    numel = math.prod(patch.shape)
    if patch.indices is None:
        if patch.values.numel() != numel:
            raise ValueError(
                f"{patch.name}: dense patch has {patch.values.numel()} values "
                f"for a {numel}-element tensor"
            )
        return numel

    if not patch.dtype.is_floating_point:
        raise TypeError(
            f"{patch.name}: sparse checkpoint patches require a floating dtype "
            "because NaN is the unchanged-value sentinel"
        )
    if patch.indices.ndim != 1:
        raise ValueError(f"{patch.name}: sparse patch indices must be a 1D tensor")
    if patch.indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{patch.name}: sparse patch indices must be int32 or int64")
    if patch.indices.numel() != patch.values.numel():
        raise ValueError(
            f"{patch.name}: sparse indices and values must have matching lengths"
        )

    return numel


def _load_weight_chunk(
    model: torch.nn.Module,
    weights: list[tuple[str, torch.Tensor]],
    *,
    sparse: bool,
) -> set[str]:
    if not weights:
        return set()
    if sparse:
        loaded_names = _load_nan_masked_weights(model, weights)
    else:
        loaded_names = model.load_weights(weights)
    return set() if loaded_names is None else set(loaded_names)


@torch.no_grad()
def load_checkpoint_weight_patches(
    model: torch.nn.Module,
    patches: Iterable[CheckpointWeightPatch],
    *,
    max_chunk_bytes: int = _DEFAULT_PATCH_CHUNK_BYTES,
    validate_unique_indices: bool = True,
) -> set[str]:
    """Load ordered patches through the model's checkpoint loader.

    A call may contain either dense or sparse patches, but not both. A repeated
    name starts a new loader call, so patches for that weight are applied in input
    order. A later patch may update positions changed by an earlier patch. Each
    sparse patch creates a full checkpoint-shaped tensor whose NaNs mark unchanged
    elements. Each locally applied sparse patch must use one final same-shaped
    floating-point ``Tensor.copy_``. Intermediate or unrelated ``copy_`` calls,
    composed loaders, and custom write paths are unsupported. ``max_chunk_bytes``
    is a batching target; one tensor may exceed it.

    Online checkpoint-format dense updates require the caller to manage vLLM's
    layerwise reload lifecycle. Sparse online updates modify initialized model
    tensors in place and must not use that lifecycle. Patch shapes, dtypes, value
    lengths, and sparse indices are validated before loading begins.

    This function does not roll back a partial update. If a loader fails after a
    write, some destinations may already be changed. Do not serve from the affected
    worker until a known baseline has been restored or the worker has been
    restarted.

    Args:
        model: Model whose native ``load_weights`` method applies the patches.
        patches: Dense replacements or sparse updates in checkpoint coordinates.
        max_chunk_bytes: Target checkpoint tensor bytes per
            ``model.load_weights`` call. A single tensor may exceed this target.
        validate_unique_indices: Whether to reject duplicate sparse indices within
            each patch. Repeated patches may update the same positions. Disable
            only for a trusted producer that already guarantees unique positions
            within each patch.

    Returns:
        Union of the weight names reported by all ``model.load_weights`` calls.
    """

    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")

    patch_list = list(patches)
    patch_numels = [_validate_patch_structure(patch) for patch in patch_list]
    # All patches in one call must use the same dense or sparse representation.
    sparse_flags = {patch.indices is not None for patch in patch_list}
    if len(sparse_flags) > 1:
        raise ValueError("Dense and sparse checkpoint patches cannot be mixed")
    sparse = sparse_flags == {True}

    # Keep validation results on their devices and read one combined result per
    # device, instead of synchronizing the CPU once for every patch.
    checks_by_device: dict[torch.device, list[torch.Tensor]] = {}
    if sparse:
        for patch, numel in zip(patch_list, patch_numels, strict=True):
            assert patch.indices is not None

            # NaN is reserved to mean "leave this checkpoint position unchanged,"
            # so it cannot also be supplied as a new sparse value.
            patch_checks = [torch.isnan(patch.values).any()]
            indices = patch.indices.to(device=patch.values.device)
            if indices.numel():
                # Sparse indices address the flattened full checkpoint tensor.
                patch_checks.append(
                    torch.logical_or(indices < 0, indices >= numel).any()
                )
                if validate_unique_indices:
                    # Equal neighboring indices after sorting indicate duplicates.
                    sorted_indices = torch.sort(indices).values
                    patch_checks.append(
                        (sorted_indices[1:] == sorted_indices[:-1]).any()
                    )

            checks_by_device.setdefault(patch.values.device, []).extend(patch_checks)

        for device_checks in checks_by_device.values():
            if torch.stack(device_checks).any().item():
                invalid_reasons = (
                    "NaN values, out-of-range indices, or duplicate indices"
                    if validate_unique_indices
                    else "NaN values or out-of-range indices"
                )
                raise ValueError(f"Sparse checkpoint patches contain {invalid_reasons}")

    loaded_names: set[str] = set()
    # Collect checkpoint tensors for one model.load_weights() call.
    weight_chunk: list[tuple[str, torch.Tensor]] = []
    chunk_names: set[str] = set()
    chunk_bytes = 0

    for patch, numel in zip(patch_list, patch_numels, strict=True):
        # Count the full checkpoint tensor, including sparse NaN staging.
        checkpoint_tensor_bytes = (
            numel * torch.empty((), dtype=patch.dtype).element_size()
        )
        # Flush before exceeding the batching target. Repeated names also start a
        # new loader call so their patches are applied in input order.
        if weight_chunk and (
            chunk_bytes + checkpoint_tensor_bytes > max_chunk_bytes
            or patch.name in chunk_names
        ):
            loaded_names.update(_load_weight_chunk(model, weight_chunk, sparse=sparse))
            weight_chunk = []
            chunk_names = set()
            chunk_bytes = 0

        if patch.indices is None:
            checkpoint_tensor = patch.values.to(dtype=patch.dtype).view(patch.shape)
        else:
            # NaNs mark unchanged checkpoint positions; index_copy_ fills only
            # the positions supplied by this sparse patch.
            flat = torch.full(
                (numel,),
                float("nan"),
                dtype=patch.dtype,
                device=patch.values.device,
            )
            if patch.values.numel():
                flat.index_copy_(
                    0,
                    patch.indices.to(device=flat.device, dtype=torch.long),
                    patch.values.to(dtype=patch.dtype, device=flat.device),
                )
            checkpoint_tensor = flat.view(patch.shape)

        weight_chunk.append((patch.name, checkpoint_tensor))
        chunk_names.add(patch.name)
        chunk_bytes += checkpoint_tensor_bytes

    # Load the final partial chunk; the helper also accepts an empty chunk.
    loaded_names.update(_load_weight_chunk(model, weight_chunk, sparse=sparse))
    return loaded_names
