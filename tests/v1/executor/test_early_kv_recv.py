# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys

import pytest

from vllm.v1.executor.early_kv_recv import EarlyKVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


def test_poll_failure_terminates_worker_without_engine_drain():
    # No KV group: polling fails. An ordinary exception must not count as exit(1).
    code = """
from contextlib import suppress
with suppress(Exception):
    from vllm.v1.executor.early_kv_recv import EarlyKVRecvSender
    from vllm.utils.network_utils import get_open_zmq_ipc_path
    EarlyKVRecvSender(get_open_zmq_ipc_path()).poll()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1
    assert "Early KV receive poll failed" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "ordinary_rank,failed",
    [(None, False), (0, False), (1, False), (0, True), (1, True)],
)
def test_early_receives_share_normal_quorum(ordinary_rank, failed):
    aggregator = EarlyKVOutputAggregator(2)

    def aggregate(kv_output=None):
        output = ModelRunnerOutput([], {}, kv_connector_output=kv_output)
        aggregator.aggregate([output])
        assert output.kv_connector_output is not None
        return output.kv_connector_output

    early = set()
    normal = KVConnectorOutput()
    for rank in range(2):
        if rank == ordinary_rank:
            normal = aggregate(
                KVConnectorOutput(
                    finished_recving={"request"},
                    failed_recving={"request"} if failed else set(),
                ),
            )
        else:
            early |= aggregator.add_finished_recving({"request"})
        if rank == 0:
            assert not early and not normal.finished_recving
    assert aggregator.aggregate([None]) is None
    if not normal.finished_recving:
        normal = aggregate()
    assert early | (normal.finished_recving or set()) == {"request"}
    assert normal.failed_recving == ({"request"} if failed else set())
    assert not early if failed else not (early and normal.finished_recving)
    assert aggregate().is_empty()
