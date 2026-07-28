# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared high-priority CUDA side streams."""

import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, TypeAlias

import torch
import torch.fx as fx
from torch._dynamo.graph_bytecode_inputs import (
    CURRENT_STREAM_INDEX,
    get_external_object_by_index,
    index_to_external_object_weakref,
)

from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream

_streams: dict[int, torch.cuda.Stream] = {}
StreamRole: TypeAlias = Literal["current", "side"]
StreamMapping: TypeAlias = tuple[tuple[int, StreamRole], ...]


def get_side_stream() -> torch.cuda.Stream | None:
    """Return the current device's shared high-priority CUDA stream."""
    if not current_platform.is_cuda():
        return None

    main_stream = current_stream()
    device_index = main_stream.device.index
    assert device_index is not None
    stream = _streams.get(device_index)
    if stream is None:
        _, high_priority = torch.cuda.Stream.priority_range()
        stream = torch.cuda.Stream(device=device_index, priority=high_priority)
        _streams[device_index] = stream
    return stream


def find_stream_mapping(graph: fx.GraphModule) -> StreamMapping:
    """Return the external-object indices used by a side-stream graph."""
    mapping: dict[int, StreamRole] = {}
    side_stream_ids = {id(stream) for stream in _streams.values()}
    for node in graph.graph.nodes:
        if node.target is not get_external_object_by_index:
            continue
        index = node.args[0]
        value = node.meta.get("example_value")
        if not isinstance(index, int):
            continue
        if id(value) in side_stream_ids:
            mapping[index] = "side"
        elif index == CURRENT_STREAM_INDEX:
            mapping[index] = "current"

    if "side" not in mapping.values():
        return ()
    return tuple(sorted(mapping.items()))


def graph_uses_stream_mapping(graph: fx.GraphModule) -> bool:
    return any(
        node.meta.get("custom", {}).get("stream") not in (None, 0)
        or (node.op == "call_function" and str(node.target).startswith("streams."))
        for node in graph.graph.nodes
    )


@contextmanager
def use_stream_mapping(mapping: StreamMapping) -> Iterator[None]:
    """Install one graph's streams and restore the prior registry entries."""
    previous = {
        index: index_to_external_object_weakref.get(index) for index, _ in mapping
    }
    for index, role in mapping:
        stream = current_stream() if role == "current" else get_side_stream()
        assert stream is not None
        index_to_external_object_weakref[index] = weakref.ref(stream)
    try:
        yield
    finally:
        for index, prior in previous.items():
            if prior is None:
                index_to_external_object_weakref.pop(index, None)
            else:
                index_to_external_object_weakref[index] = prior
