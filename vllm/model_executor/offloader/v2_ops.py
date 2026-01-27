# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom ops for V2 offloader torch.compile + CUDA graph compatibility.

These ops use mutates_args to create data dependencies that prevent
the compiler from reordering prefetch/sync operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.offloader.v2 import OffloaderV2

# Global reference to the offloader instance, set during OffloaderV2.__init__
_offloader_instance: OffloaderV2 | None = None


def set_offloader_instance(offloader: OffloaderV2 | None) -> None:
    """Set the global offloader instance for custom ops to use."""
    global _offloader_instance
    _offloader_instance = offloader


def sync_offloader_before_capture() -> None:
    """Sync offloader's copy stream before CUDA graph capture.

    Call this before capturing or replaying CUDA graphs. This ensures
    any pre-capture prefetch work is complete before the graph operations.

    Safe to call even if no offloader is active (no-op in that case).
    """
    if _offloader_instance is not None:
        _offloader_instance.sync_before_graph_capture()


# --- wait_prefetch op ---


def _wait_prefetch_impl(
    input_tensor: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Wait for prefetch of layer_idx to complete.

    Synchronizes the compute stream with the copy stream to ensure
    the prefetched weights are ready for use.

    Args:
        input_tensor: Input to the layer (e.g., hidden_states) - returned
            to create data dependency chain.
        layer_idx: Index of the layer to wait for.

    Returns:
        input_tensor unchanged, but creates data dependency for torch.compile.
    """
    if _offloader_instance is not None:
        _offloader_instance._wait_for_layer(layer_idx)
    return input_tensor


def _wait_prefetch_fake(
    input_tensor: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return input_tensor


# --- start_prefetch op ---


def _start_prefetch_impl(
    output_tensor: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Start async prefetch of layer_idx weights.

    Initiates H2D copy on the copy stream for the specified layer.

    Args:
        output_tensor: Output from forward - returned to create ordering
            dependency. This prevents torch.compile from reordering
            this op before the computation that produces output_tensor.
        layer_idx: Index of the layer to prefetch.

    Returns:
        output_tensor unchanged, creating data dependency for torch.compile.
    """
    if _offloader_instance is not None:
        _offloader_instance._start_prefetch(layer_idx)
    return output_tensor


def _start_prefetch_fake(
    output_tensor: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return output_tensor


def join_offloader_after_forward() -> None:
    """Join copy_stream after model forward completes.

    Call this after the model forward pass but before CUDA graph capture
    ends. This ensures copy_stream is rejoined for any prefetches started
    during the forward pass.

    The last layer prefetches a layer that won't have its wait_prefetch
    called until the next forward pass. During capture, this leaves
    copy_stream unjoined, causing cudaErrorStreamCaptureUnjoined.

    Safe to call even if no offloader is active (no-op in that case).
    """
    if _offloader_instance is not None:
        _offloader_instance.join_after_forward()


def register_v2_offloader_ops() -> None:
    """Register custom ops for V2 offloader.

    Must be called before the ops are used. This is typically done
    at module import time.
    """
    direct_register_custom_op(
        op_name="wait_prefetch",
        op_func=_wait_prefetch_impl,
        mutates_args=["input_tensor"],
        fake_impl=_wait_prefetch_fake,
    )

    direct_register_custom_op(
        op_name="start_prefetch",
        op_func=_start_prefetch_impl,
        mutates_args=["output_tensor"],
        fake_impl=_start_prefetch_fake,
    )


# Register ops at module import time
register_v2_offloader_ops()
