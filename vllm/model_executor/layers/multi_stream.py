import torch
from typing import Any, Callable, List, Optional, Union

from vllm.forward_context import ForwardContext, MultiStreamContext, get_forward_context
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def should_use_multi_stream() -> bool:
    """Utility function to check if multi-stream should be enabled.
    Multi-stream is only enabled when cuda graph is turned on because switch
    stream has extra host overhead.
    """
    return current_platform.is_cuda() \
        and not torch.jit.is_tracing() \
        and torch.cuda.is_current_stream_capturing()


def maybe_run_multi_stream(
        fn0: Callable,
        fn1: Callable,
        args0: torch.Tensor,
        args1: torch.Tensor,
        multi_stream_context: MultiStreamContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Utility function to run two functions in two cuda streams in parallel.
    Multi-stream is only enabled when cuda graph is turned on because switch
    stream has extra host overhead.

    If the current platform is not CUDA, or if multi_stream_context is not
    provided, it will just run the functions sequentially.

    Args:
        fn0: callable for the default stream
        fn1: callable for the second stream, aux_stream
        args0: arguments for fn0
        args1: arguments for fn1
        multi_stream_context: the multi-stream context

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the return values of fn0(args0) and fn1(args1)
    """

    if multi_stream_context is not None and should_use_multi_stream():

        multi_stream_context.event0.record()
        result0 = fn0(args0)

        with torch.cuda.stream(multi_stream_context.aux_stream):
            multi_stream_context.event0.wait()
            result1 = fn1(args1)
            multi_stream_context.event1.record()
        multi_stream_context.event1.wait()

    else:
        result0 = fn0(args0)
        result1 = fn1(args1)

    return (result0, result1)


def maybe_multi_stream_forward(
    layer_name0: str,
    layer_name1: str,
    args0: torch.Tensor,
    args1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    multi_stream_context: MultiStreamContext = forward_context.multi_stream_context

    assert layer_name0 in forward_context.multi_stream_layers, \
        f"Cannot find layer {layer_name0} in multi_stream_layers"
    assert layer_name1 in forward_context.multi_stream_layers, \
        f"Cannot find layer {layer_name1} in multi_stream_layers"

    layer0 = forward_context.multi_stream_layers[layer_name0]
    layer1 = forward_context.multi_stream_layers[layer_name1]

    return maybe_run_multi_stream(layer0, layer1, args0, args1, multi_stream_context)


def maybe_multi_stream_forward_fake(
    layer_name0: str,
    layer_name1: str,
    args0: torch.Tensor,
    args1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()

    layer_name_fake0 = layer_name0 + ".fake"
    layer_name_fake1 = layer_name1 + ".fake"

    assert layer_name_fake0 in forward_context.multi_stream_layers, \
        f"Cannot find layer {layer_name_fake0} in multi_stream_layers. Please add fake layer for {layer_name0}."
    assert layer_name_fake1 in forward_context.multi_stream_layers, \
        f"Cannot find layer {layer_name_fake1} in multi_stream_layers. Please add fake layer for {layer_name1}."

    layer0 = forward_context.multi_stream_layers[layer_name_fake0]
    layer1 = forward_context.multi_stream_layers[layer_name_fake1]

    return (layer0(args0), layer1(args1))


direct_register_custom_op(
    op_name="maybe_multi_stream_forward",
    op_func=maybe_multi_stream_forward,
    mutates_args=["args0", "args1"],
    fake_impl=maybe_multi_stream_forward_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)
