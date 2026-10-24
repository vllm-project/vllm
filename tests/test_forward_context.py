# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import vllm.forward_context as forward_context
from vllm.config import VllmConfig


def test_batch_timing_retains_only_recent_samples(monkeypatch):
    config = VllmConfig()
    samples = type(forward_context.batchsize_forward_time)(
        forward_context.batchsize_forward_time.default_factory
    )
    logger = MagicMock()
    monkeypatch.setattr(forward_context, "batchsize_forward_time", samples)
    monkeypatch.setattr(forward_context, "track_batchsize", True)
    monkeypatch.setattr(forward_context, "batchsize_logging_interval", float("inf"))
    monkeypatch.setattr(forward_context, "last_logging_time", 0)
    monkeypatch.setattr(forward_context, "logger", logger)
    monkeypatch.setattr(forward_context.current_platform, "synchronize", lambda: None)
    clock = 0.0

    def now():
        nonlocal clock
        clock += 0.001
        return clock

    monkeypatch.setattr(forward_context.time, "perf_counter", now)
    for _ in range(2048):
        with forward_context.set_forward_context({}, config, num_tokens=16):
            pass
    assert len(samples[16]) == 1024
    monkeypatch.setattr(forward_context, "batchsize_logging_interval", 0)
    with forward_context.set_forward_context({}, config, num_tokens=16):
        pass
    assert logger.info.call_args.args[1] == [(16, 1024, 1.0)]
