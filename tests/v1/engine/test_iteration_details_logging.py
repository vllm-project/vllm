# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import vllm.v1.engine.core as core_module
from vllm.v1.engine.core import EngineCore
from vllm.v1.utils import IterationDetails


def test_iteration_details_logs_result_wait_time(caplog, monkeypatch):
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_logging_iteration_details=True)
    )
    engine._iteration_index = 0

    scheduler_output = MagicMock()
    scheduler_output.total_num_scheduled_tokens = 1

    monkeypatch.setattr(
        core_module,
        "compute_iteration_details",
        lambda _: IterationDetails(1, 2, 3, 4),
    )

    caplog.set_level(logging.INFO, logger=core_module.__name__)

    with engine.log_iteration_details(scheduler_output) as timing:
        assert timing is not None
        timing["result_wait_time"] = 0.01234

    assert "iteration elapsed time:" in caplog.text
    assert "result wait time: 12.34 ms" in caplog.text
