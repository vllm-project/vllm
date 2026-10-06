# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`/ready` against injected failures with a single engine.

`/health` stays 200 for all of these; `/ready` must not.
"""

import pytest

from .fault_injection import SCHEDULER_STALL
from .utils import STALL_TIMEOUT_S, ready_probe_server

MODEL = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with ready_probe_server(MODEL, [], tmp_path_factory.mktemp("faults")) as srv:
        yield srv


def test_no_false_positive(server):
    """A healthy engine stays ready when idle and under sustained load."""
    assert server.stays_ready(STALL_TIMEOUT_S + 3)
    with server.busy():
        assert server.stays_ready(STALL_TIMEOUT_S + 3)


def test_scheduler_stall(server):
    """#45388: requests outstanding but zero tokens scheduled."""
    server.set_fault(SCHEDULER_STALL)
    try:
        server.generate_in_background(8)
        assert server.wait_until_unready() is not None
        assert server.status("/health") == 200
    finally:
        server.clear_fault(SCHEDULER_STALL)
    assert server.wait_until_ready()


def test_latent_cuda_error_while_idle(tmp_path):
    """An async CUDA fault that nothing has synchronized on yet is only
    surfaced by the idle GPU probe."""
    with ready_probe_server(MODEL, [], tmp_path) as srv:
        assert srv.status() == 200
        srv.rpc("fault_ima_now", "0")
        assert srv.status("/health") == 200
        assert srv.wait_until_unready() is not None
