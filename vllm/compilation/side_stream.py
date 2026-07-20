# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-stream execution compatible with torch.compile and cudagraphs.

Dynamo cannot trace CUDA stream switches before torch 2.12, so entering and
leaving the side stream are custom ops that must be registered as splitting
ops (CompilationConfig.splitting_ops): piecewise compilation executes them
eagerly between compiled pieces in original graph order
(split_module(keep_original_order=True)), and each compiled piece binds its
launch stream at call entry, so pieces inside the region land on the side
stream. The ops carry no tensor arguments - only the device index, a
compile-time constant - so they are marked side-effectful via
torch.fx.node.has_side_effect to keep FX dead-code elimination from
dropping them; begin/end ordering relies on the order-preserving split.

Once vLLM's minimum torch is >= 2.12 (traceable stream contexts), the
side_stream context manager should be swappable for the native
stream-centric form - stream.wait_stream(current_stream) +
torch.cuda.stream(stream) traced directly - pending validation that traced
stream regions compose with vLLM's piecewise splitting and cudagraph
capture.
"""

import torch
from torch.fx.node import has_side_effect

from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream, direct_register_custom_op


class side_stream:
    """Run the enclosed region on the device's high-priority side stream.

    Entry makes the side stream wait for the current stream; exit only
    switches back, so callers must join via wait_side_stream() before
    consuming side-stream results on the main stream. Streams are created
    lazily inside the (eagerly executed) ops and shared per device; regions
    must not nest on a device (single saved-main-stream slot). See the
    module docstring for torch.compile requirements.
    """

    _streams: dict[int, torch.cuda.Stream] = {}
    _saved: dict[int, torch.cuda.Stream] = {}

    def __init__(self, device: torch.device):
        self.device_index = device.index if device.index is not None else -1

    def __enter__(self) -> None:
        torch.ops.vllm.side_stream_begin(self.device_index)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        torch.ops.vllm.side_stream_end(self.device_index)

    @classmethod
    def get_stream(
        cls, device_index: int, *, create: bool = True
    ) -> torch.cuda.Stream | None:
        """The device's high-priority side stream, lazily created."""
        stream = cls._streams.get(device_index)
        if stream is None and create and current_platform.is_cuda():
            _, high_priority = torch.cuda.Stream.priority_range()
            stream = torch.cuda.Stream(device=device_index, priority=high_priority)
            cls._streams[device_index] = stream
        return stream


def wait_side_stream() -> None:
    """Make the current stream wait for work issued on the side stream.

    No-op if the current device's side stream was never created.
    """
    if not side_stream._streams:
        return
    main_stream = current_stream()
    stream = side_stream._streams.get(main_stream.device.index)
    if stream is not None:
        main_stream.wait_stream(stream)


def side_stream_begin(device_index: int) -> None:
    stream = side_stream.get_stream(device_index)
    if stream is None:
        return

    main_stream = current_stream()
    side_stream._saved[device_index] = main_stream
    stream.wait_stream(main_stream)
    torch.cuda.set_stream(stream)


def side_stream_end(device_index: int) -> None:
    main_stream = side_stream._saved.get(device_index)
    if main_stream is not None:
        torch.cuda.set_stream(main_stream)


def _side_stream_op_fake(device_index: int) -> None:
    return None


# CompositeExplicitAutograd: with no tensor args the dispatcher cannot
# select a backend key, so a plain CUDA registration is unreachable.
direct_register_custom_op(
    op_name="side_stream_begin",
    op_func=side_stream_begin,
    mutates_args=[],
    fake_impl=_side_stream_op_fake,
    dispatch_key="CompositeExplicitAutograd",
    tags=(torch.Tag.maybe_aliasing_or_mutating,),
)

direct_register_custom_op(
    op_name="side_stream_end",
    op_func=side_stream_end,
    mutates_args=[],
    fake_impl=_side_stream_op_fake,
    dispatch_key="CompositeExplicitAutograd",
    tags=(torch.Tag.maybe_aliasing_or_mutating,),
)

# No tensor args -> no mutable schema, so FX would consider the ops pure and
# dead-code-eliminate them (they return None and have no users). Dynamo
# emits the OpOverloadPacket as the node target; cover the overload too.
has_side_effect(torch.ops.vllm.side_stream_begin)
has_side_effect(torch.ops.vllm.side_stream_begin.default)
has_side_effect(torch.ops.vllm.side_stream_end)
has_side_effect(torch.ops.vllm.side_stream_end.default)
