# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for #47415: a WorkerProc init failure used to raise a generic
message and swallow the worker's root-cause traceback (only stderr had it)."""

import multiprocessing
from types import SimpleNamespace

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc


def test_wait_for_ready_surfaces_worker_root_cause():
    reader, writer = multiprocessing.Pipe(duplex=False)
    writer.send(
        {
            "status": WorkerProc.FAILED_STR,
            "exception": "ValueError: To serve ... KV cache is needed",
        }
    )
    handle = SimpleNamespace(ready_pipe=reader, rank=0)

    with pytest.raises(Exception, match="KV cache is needed"):
        WorkerProc.wait_for_ready([handle])


def test_wait_for_ready_falls_back_when_no_exception_sent():
    # Worker died hard (e.g. segfault/OOM-kill) before sending FAILED.
    reader, writer = multiprocessing.Pipe(duplex=False)
    writer.close()  # parent sees EOF
    handle = SimpleNamespace(ready_pipe=reader, rank=0)

    with pytest.raises(Exception, match="See stack trace for root cause"):
        WorkerProc.wait_for_ready([handle])
