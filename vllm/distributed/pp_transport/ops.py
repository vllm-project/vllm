# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common tensor operations for pipeline-parallel transport."""

from vllm.sequence import IntermediateTensors


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
        view = views[key] = buffer[: tensor.shape[0]]
        view.copy_(tensor)
    return IntermediateTensors(views)
