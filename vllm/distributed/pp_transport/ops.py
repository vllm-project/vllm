# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent PP copies and opaque cache/state dependency operators."""

from typing import TYPE_CHECKING, Protocol

import torch

from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext


class PPDependencyHandler(Protocol):
    """Model/device implementation registered in static_forward_context."""

    def copy_cache_rows(
        self,
        context: "ForwardContext",
        rows: list[torch.Tensor],
        *,
        receive: bool,
    ) -> None: ...


def _export_dependencies(
    hidden_states: torch.Tensor,
    cache_rows: list[torch.Tensor],
    state_inputs: list[torch.Tensor],
    state_outputs: list[torch.Tensor],
    transport_name: str,
) -> None:
    context = get_forward_context()
    if cache_rows and context.attn_metadata is not None:
        handler: PPDependencyHandler = context.no_compile_layers[transport_name]
        handler.copy_cache_rows(context, cache_rows, receive=False)
    for source, output in zip(state_inputs, state_outputs, strict=True):
        # The ordering activation can be SP-sharded while state rows are full.
        output.copy_(source[: output.shape[0]])


def _import_dependencies(
    hidden_states: torch.Tensor,
    cache_rows: list[torch.Tensor],
    state_inputs: list[torch.Tensor],
    state_outputs: list[torch.Tensor],
    transport_name: str,
) -> None:
    context = get_forward_context()
    if cache_rows and context.attn_metadata is not None:
        handler: PPDependencyHandler = context.no_compile_layers[transport_name]
        handler.copy_cache_rows(context, cache_rows, receive=True)
    for source, output in zip(state_inputs, state_outputs, strict=True):
        output[: source.shape[0]].copy_(source)


def _dependency_fake(*args, **kwargs) -> None:
    return None


# The import's activation mutation is an ordering edge, not a value change:
# downstream attention must observe cache writes before consuming the activation.
# Composite dispatch lets the runtime handler own device-native gather/scatter.
direct_register_custom_op(
    op_name="pp_transport_export",
    op_func=_export_dependencies,
    mutates_args=["cache_rows", "state_outputs"],
    fake_impl=_dependency_fake,
    dispatch_key="CompositeExplicitAutograd",
)
direct_register_custom_op(
    op_name="pp_transport_import",
    op_func=_import_dependencies,
    mutates_args=["hidden_states", "state_outputs"],
    fake_impl=_dependency_fake,
    dispatch_key="CompositeExplicitAutograd",
)

export_pp_dependencies = torch.ops.vllm.pp_transport_export.default
import_pp_dependencies = torch.ops.vllm.pp_transport_import.default


def copy_intermediate_tensors(
    buffers: IntermediateTensors, tensors: IntermediateTensors
) -> IntermediateTensors:
    """Copy capture outputs and retain their individual shapes for graph replay."""
    views = {}
    for key, tensor in tensors.items():
        buffer = buffers[key]
        if (
            tensor.ndim == 0
            or tensor.ndim != buffer.ndim
            or tensor.shape[1:] != buffer.shape[1:]
            or tensor.shape[0] > buffer.shape[0]
            or tensor.dtype != buffer.dtype
        ):
            raise ValueError(
                f"Invalid {key}: {tensor.shape}/{tensor.dtype} "
                f"for buffer {buffer.shape}/{buffer.dtype}"
            )
        buffer[: tensor.shape[0]].copy_(tensor)
        views[key] = buffer[: tensor.shape[0]]
    return IntermediateTensors(views)
