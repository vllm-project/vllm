# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


@pytest.mark.cpu_test
def test_connector_finished_injects_prefill_num_cached_tokens():
    request = MagicMock(spec=Request)
    request.request_id = "req-1"
    request.num_computed_tokens = 32
    request.prefill_num_cached_tokens = 1280
    request.kv_transfer_params = {"do_remote_decode": True}

    scheduler = MagicMock(spec=Scheduler)
    scheduler.connector = MagicMock()
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.return_value = [[1, 2]]
    scheduler.kv_cache_config = MagicMock()
    scheduler.kv_cache_config.kv_cache_groups = [MagicMock()]

    connector_params = {
        "do_remote_prefill": True,
        "remote_block_ids": [1, 2],
    }
    finished_return = (True, connector_params)
    scheduler.connector.request_finished.return_value = finished_return
    scheduler.connector.request_finished_all_groups.return_value = finished_return

    delay_free, params = Scheduler._connector_finished(scheduler, request)

    assert delay_free is True
    assert params is not None
    assert params["num_cached_tokens"] == 1280
    assert params["do_remote_prefill"] is True
