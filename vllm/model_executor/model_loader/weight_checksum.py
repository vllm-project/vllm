# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-parameter weight digests and reset, for verifying weight updates."""

import hashlib
from collections.abc import Iterable, Iterator
from math import gcd
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from vllm.model_executor.parameter import SharedWeightParameter

if TYPE_CHECKING:
    from vllm.v1.engine import WeightCheckerModel


def _tensors(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    # Both keep their bytes in inner tensors (e.g. TorchAO's qdata and scale).
    if isinstance(tensor, SharedWeightParameter):
        for index, partition in tensor.partitions.items():
            yield from _tensors(f"{name}.{index}", partition)
    elif is_traceable_wrapper_subclass(tensor):
        for attr in tensor.__tensor_flatten__()[0]:
            yield from _tensors(f"{name}.{attr}", getattr(tensor, attr))
    else:
        yield name, tensor


def _weights(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        yield from _tensors(name, param)


def compute_tensor_digests(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            weight.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()
        ).hexdigest()
        for name, weight in _weights(model)
    }


def zero_weights(model: nn.Module) -> None:
    zero_selected_weights(weight for _, weight in _weights(model))


def zero_selected_weights(weights: Iterable[torch.Tensor]) -> None:
    for weight in weights:
        weight.data.zero_()


class _TensorSpan(NamedTuple):
    device: str
    start: int
    end: int
    name: str
    tensor: torch.Tensor


def _tensor_spans(tensors: Iterable[tuple[str, torch.Tensor]]) -> list[_TensorSpan]:
    spans = []
    for name, tensor in tensors:
        start = tensor.data_ptr()  # Includes the storage offset.
        if start and tensor.numel():
            last = sum(
                (size - 1) * step for size, step in zip(tensor.shape, tensor.stride())
            )
            end = start + (last + 1) * tensor.element_size()
            spans.append(_TensorSpan(str(tensor.device), start, end, name, tensor))
    return sorted(spans, key=lambda span: (span.device, span.start))


def _strided_blocks(tensor: torch.Tensor) -> tuple[int, int]:
    # Collapse contiguous dimensions into a byte block. The remaining strides
    # put each block at a multiple of their gcd from the first one.
    width = itemsize = tensor.element_size()
    step = 0
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size <= 1:
            continue
        stride *= itemsize
        if not step and stride <= width:
            width += (size - 1) * stride
        else:
            step = gcd(step, stride)
    return width, step


def _may_overlap(a: _TensorSpan, b: _TensorSpan) -> bool:
    a_width, a_step = _strided_blocks(a.tensor)
    b_width, b_step = _strided_blocks(b.tensor)
    step = gcd(a_step, b_step)
    if not step:
        return True
    offset = (a.start - b.start) % step
    return offset < b_width or step - offset < a_width


def _possible_aliases(
    weights: Iterable[tuple[str, torch.Tensor]],
    others: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[_TensorSpan, _TensorSpan]]:
    other_spans = _tensor_spans(others)
    if not other_spans:
        return
    first = 0
    for weight in _tensor_spans(weights):
        while first < len(other_spans):
            other = other_spans[first]
            if (other.device, other.end) > (weight.device, weight.start):
                break
            first += 1
        for index in range(first, len(other_spans)):
            other = other_spans[index]
            if other.device != weight.device or other.start >= weight.end:
                break
            if other.end > weight.start and _may_overlap(weight, other):
                yield weight, other


def _tensor_mapping(weight: torch.Tensor):
    return (
        weight.device,
        weight.data_ptr() or id(weight),
        weight.dtype,
        weight.shape,
        weight.stride(),
    )


def select_reset_weights(
    target: nn.Module, draft: nn.Module | None, model: "WeightCheckerModel"
) -> tuple[torch.Tensor, ...] | str:
    """Select unique in-place writes after checking cross-model aliases."""
    target_weights = list(_weights(target))
    draft_weights = (
        [(name, weight) for name, weight in _weights(draft) if weight.numel()]
        if draft is not None
        else []
    )
    if model == "draft" and not draft_weights:
        return "Weight Checker found no resettable draft weights."

    for target_span, draft_span in _possible_aliases(target_weights, draft_weights):
        if _tensor_mapping(target_span.tensor) != _tensor_mapping(draft_span.tensor):
            return (
                f"Cannot reset {model} weights: target parameter {target_span.name!r} "
                f"and draft parameter {draft_span.name!r} have ambiguous "
                "overlapping storage."
            )

    selected = {}
    for owner, weights, other in (
        ("target", target_weights if model != "draft" else [], draft),
        ("draft", draft_weights if model != "target" else [], target),
    ):
        # For both, exact aliases already belong to the target's write plan.
        writes = []
        for name, weight in weights:
            mapping = _tensor_mapping(weight)
            if mapping not in selected:
                selected[mapping] = weight
                writes.append((name, weight))
        if writes and other is not None:
            buffers = (
                item
                for name, buffer in other.named_buffers()
                for item in _tensors(name, buffer)
            )
            if alias := next(_possible_aliases(writes, buffers), None):
                param, buffer = alias
                return (
                    f"Cannot reset {model} weights: {owner} parameter {param.name!r} "
                    f"overlaps buffer {buffer.name!r} in the other model."
                )
    return tuple(selected.values())
