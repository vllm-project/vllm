# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for CUDA/ROCm multi-stream execution helpers."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import execute_in_parallel, maybe_execute_in_parallel


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Multi-stream execution requires CUDA or ROCm",
)
def test_maybe_execute_in_parallel_matches_sequential():
    x = torch.randn(1024, device=current_platform.device_type)
    start_event = torch.cuda.Event()
    done_event = torch.cuda.Event()

    parallel = maybe_execute_in_parallel(
        lambda: x + 1,
        lambda: x * 2,
        start_event,
        done_event,
        torch.cuda.Stream(),
    )
    sequential = maybe_execute_in_parallel(
        lambda: x + 1,
        lambda: x * 2,
        start_event,
        done_event,
        None,
    )

    torch.testing.assert_close(parallel[0], sequential[0])
    torch.testing.assert_close(parallel[1], sequential[1])


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Multi-stream execution requires CUDA or ROCm",
)
def test_execute_in_parallel_matches_sequential():
    x = torch.randn(1024, device=current_platform.device_type)
    start_event = torch.cuda.Event()
    done_events = [torch.cuda.Event() for _ in range(3)]
    aux_streams = [torch.cuda.Stream() for _ in range(3)]
    aux_fns = [lambda: x * 2, None, lambda: x * 3]

    parallel = execute_in_parallel(
        lambda: x + 1,
        aux_fns,
        start_event,
        done_events,
        aux_streams,
        enable=True,
    )
    sequential = execute_in_parallel(
        lambda: x + 1,
        aux_fns,
        start_event,
        done_events,
        aux_streams,
        enable=False,
    )

    torch.testing.assert_close(parallel[0], sequential[0])
    assert parallel[1][1] is sequential[1][1] is None
    torch.testing.assert_close(parallel[1][0], sequential[1][0])
    torch.testing.assert_close(parallel[1][2], sequential[1][2])
