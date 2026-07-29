# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared high-priority CUDA side stream."""

from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.graph_bytecode_inputs import CURRENT_STREAM_INDEX

from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_side_stream: torch.cuda.Stream | None = None


def register_side_stream(stream: torch.cuda.Stream) -> None:
    """Publish the side stream so compiled artifacts can resolve it by index.

    Callers own the stream and pass it to their consumers directly; this only
    gives the index resolver below a handle on it. Only one side stream per
    process is supported, since an artifact's stream indices carry no way to
    tell two of them apart.
    """
    global _side_stream
    assert _side_stream is None or _side_stream is stream, (
        "only one side stream per process is supported"
    )
    _side_stream = stream


def graph_uses_side_stream(graph: fx.GraphModule) -> bool:
    return any(
        node.meta.get("custom", {}).get("stream") not in (None, 0)
        or (node.op == "call_function" and str(node.target).startswith("streams."))
        for node in graph.graph.nodes
    )


def _install_stream_index_resolver() -> None:
    """Resolve stream external-object indices without dynamo's bytecode.

    Compiled artifacts reference streams by external-object index from the
    inductor wrapper prologue and from the int args of torch.ops.streams.*.
    Dynamo resolves both through a process-local registry that its generated
    bytecode repopulates on every call; vLLM invokes compiled pieces directly,
    so that registry is empty on AOT reload and is cleared again whenever
    anything else is traced.

    Two rules cover every index. CURRENT_STREAM_INDEX is dynamo's reserved
    index for the current stream, resolved dynamically here because a registry
    entry would pin whichever stream was current at trace time - wrong under
    cudagraph capture. Every other index refers to the side stream, the only
    stream object vLLM graphs capture, so a registry miss falls back to it.
    Violations fail loudly: torch's _get_stream_by_index and
    _get_event_by_index assert the type of the resolved object.
    """
    import torch._dynamo.graph_bytecode_inputs as gbi
    import torch._dynamo.variables.streams as dynamo_streams

    if getattr(gbi.get_external_object_by_index, "_vllm_stream_resolver", False):
        return
    original = gbi.get_external_object_by_index

    def resolver(index: int) -> Any:
        if index == CURRENT_STREAM_INDEX:
            return current_stream()
        try:
            return original(index)
        except AssertionError:
            if _side_stream is None:
                raise
            logger.debug_once(
                "Resolved unregistered external-object index %d to the side "
                "stream; this is expected for AOT-loaded graphs.",
                index,
            )
            return _side_stream

    resolver._vllm_stream_resolver = True  # type: ignore[attr-defined]
    gbi.get_external_object_by_index = resolver
    # variables.streams binds the name at import time, so the streams.* op
    # implementations (_get_stream_by_index) need their own rebind.
    dynamo_streams.get_external_object_by_index = resolver


# Install at import: deserialized artifacts bind the resolver name when their
# generated modules are exec'd, which can happen before any compile runs.
_install_stream_index_resolver()
