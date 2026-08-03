# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import execute_in_parallel, maybe_execute_in_parallel


def _device() -> str:
    return current_platform.device_type


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="multi_stream_utils requires CUDA or ROCm",
)
def test_maybe_execute_in_parallel_matches_sequential():
    x = torch.randn(1024, device=_device())
    aux_stream = torch.cuda.Stream()
    event0 = torch.cuda.Event()
    event1 = torch.cuda.Event()

    parallel = maybe_execute_in_parallel(
        lambda: x * 2.0,
        lambda: x * 3.0,
        event0,
        event1,
        aux_stream,
    )
    sequential = maybe_execute_in_parallel(
        lambda: x * 2.0,
        lambda: x * 3.0,
        event0,
        event1,
        None,
    )

    torch.accelerator.synchronize()
    assert torch.equal(parallel[0], sequential[0])
    assert torch.equal(parallel[1], sequential[1])
    assert torch.equal(parallel[0], x * 2.0)
    assert torch.equal(parallel[1], x * 3.0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="multi_stream_utils requires CUDA or ROCm",
)
def test_execute_in_parallel_matches_sequential():
    x = torch.randn(512, device=_device())
    aux_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    start_event = torch.cuda.Event()
    done_events = [torch.cuda.Event(), torch.cuda.Event()]

    parallel = execute_in_parallel(
        lambda: x + 1.0,
        [lambda: x * 2.0, lambda: x * 3.0],
        start_event,
        done_events,
        aux_streams,
        enable=True,
    )
    sequential = execute_in_parallel(
        lambda: x + 1.0,
        [lambda: x * 2.0, lambda: x * 3.0],
        start_event,
        done_events,
        aux_streams,
        enable=False,
    )

    torch.accelerator.synchronize()
    assert torch.equal(parallel[0], sequential[0])
    assert torch.equal(parallel[1][0], sequential[1][0])
    assert torch.equal(parallel[1][1], sequential[1][1])


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="multi_stream_utils requires CUDA or ROCm",
)
def test_execute_in_parallel_skips_none_aux():
    x = torch.randn(256, device=_device())
    aux_streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]
    start_event = torch.cuda.Event()
    done_events = [torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()]

    default_result, aux_results = execute_in_parallel(
        lambda: x + 10.0,
        [None, lambda: x * 4.0, None],
        start_event,
        done_events,
        aux_streams,
        enable=True,
    )

    torch.accelerator.synchronize()
    assert torch.equal(default_result, x + 10.0)
    assert aux_results[0] is None
    assert torch.equal(aux_results[1], x * 4.0)
    assert aux_results[2] is None


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="multi_stream_utils requires CUDA or ROCm",
)
def test_execute_in_parallel_aux_tensor_on_main_stream():
    x = torch.randn(1024, device=_device())
    aux_stream = torch.cuda.Stream()
    start_event = torch.cuda.Event()
    done_event = torch.cuda.Event()

    _, aux_results = execute_in_parallel(
        lambda: x * 2.0,
        [lambda: x * 5.0],
        start_event,
        [done_event],
        [aux_stream],
        enable=True,
    )

    torch.accelerator.synchronize()
    combined = aux_results[0] + x
    torch.testing.assert_close(combined, x * 6.0)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA-specific launch ordering flag behavior",
)
def test_launch_multi_stream_queue_aux_before_default():
    x = torch.randn(128, device=_device())
    aux_stream = torch.cuda.Stream()
    start_event = torch.cuda.Event()
    done_event = torch.cuda.Event()
    launch_order: list[str] = []

    def default_fn() -> torch.Tensor:
        launch_order.append("default")
        return x + 1.0

    def aux_fn() -> torch.Tensor:
        launch_order.append("aux")
        return x * 2.0

    current_platform.launch_multi_stream(
        default_fn,
        [aux_fn],
        start_event,
        [done_event],
        [aux_stream],
        queue_aux_before_default=True,
    )
    torch.accelerator.synchronize()
    assert launch_order == ["aux", "default"]

    launch_order.clear()
    current_platform.launch_multi_stream(
        default_fn,
        [aux_fn],
        start_event,
        [done_event],
        [aux_stream],
        queue_aux_before_default=False,
    )
    torch.accelerator.synchronize()
    assert launch_order == ["default", "aux"]
