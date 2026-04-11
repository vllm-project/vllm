# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch


def maybe_execute_in_parallel(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: torch.cuda.Stream | None = None,
) -> tuple[Any, Any]:
    """Run two functions potentially in parallel on separate CUDA streams.

    When aux_stream is provided, fn0 runs on the current (default) stream and
    fn1 runs on aux_stream, synchronized via CUDA events.  When aux_stream is
    None, both functions execute sequentially on the current stream.

    This design follows TensorRT-LLM's maybe_execute_in_parallel pattern
    (tensorrt_llm/_torch/modules/multi_stream_utils.py).

    Args:
        fn0: Callable for the default stream.
        fn1: Callable for the auxiliary stream.
        event0: CUDA event recorded before fn0 so aux_stream can wait.
        event1: CUDA event recorded after fn1 so default stream can wait.
        aux_stream: The second CUDA stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        Tuple of (fn0_result, fn1_result).
    """
    if aux_stream is not None:
        event0.record()
        result0 = fn0()
        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)
