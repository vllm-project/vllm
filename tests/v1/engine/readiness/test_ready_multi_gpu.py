# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`/ready` against injected failures across two GPUs (DP=2 and PP=2).

MoE DP ranks step in lockstep waves where idle ranks run dummy batches, and
must not be probed with an ad-hoc dummy batch; dense DP ranks are independent
engines that can be probed like a single engine.
"""

import time

import pytest

from tests.utils import multi_gpu_marks

from .fault_injection import SCHEDULER_STALL
from .utils import STALL_TIMEOUT_S, ready_probe_server

pytestmark = multi_gpu_marks(num_gpus=2)

MOE_MODEL = "ibm-research/PowerMoE-3b"
DENSE_MODEL = "hmellor/tiny-random-LlamaForCausalLM"
MOE_DP_ARGS = [
    "--data-parallel-size",
    "2",
    "--enable-expert-parallel",
    # Weights are dummy; two layers keep startup and memory small.
    "--hf-overrides",
    '{"num_hidden_layers": 2}',
]
DENSE_DP_ARGS = ["--data-parallel-size", "2"]


@pytest.fixture(scope="module")
def moe_server(tmp_path_factory):
    with ready_probe_server(
        MOE_MODEL, MOE_DP_ARGS, tmp_path_factory.mktemp("faults")
    ) as srv:
        yield srv


def test_moe_no_false_positive(moe_server):
    """An idle rank inside a wave keeps counting as progressing, and probes
    racing with wave starts do not flap."""
    with moe_server.busy(dp_ranks=(0,)):
        assert moe_server.stays_ready(STALL_TIMEOUT_S + 3)
    for i in range(10):
        moe_server.generate_in_background(8, dp_rank=i % 2)
        assert moe_server.status() == 200
        time.sleep(0.2 * (i % 3))


def test_moe_scheduler_stall(moe_server):
    """#45388 under DP: ranks hold requests but schedule nothing while the
    wave keeps running dummy batches."""
    moe_server.set_fault(SCHEDULER_STALL)
    try:
        moe_server.generate_in_background(8, dp_rank=0)
        moe_server.generate_in_background(8, dp_rank=1)
        assert moe_server.wait_until_unready() is not None
        assert moe_server.status("/health") == 200
    finally:
        moe_server.clear_fault(SCHEDULER_STALL)
    assert moe_server.wait_until_ready()


def test_moe_collective_mismatch_while_idle(tmp_path):
    """#36594 class: rank 0 is wedged in a DP collective inside a control RPC
    while there is no traffic."""
    with ready_probe_server(MOE_MODEL, MOE_DP_ARGS, tmp_path) as srv:
        assert srv.status() == 200
        srv.rpc("fault_dp_mismatched_collective", "0", background=True)
        assert srv.wait_until_unready() is not None
        assert srv.status("/health") == 200
        # Any API server can name the wedged operation and rank.
        for _ in range(4):
            body = srv.ready_body()
            assert body["reason"] == "probe_timeout"
            stuck = {(op["operation"], op["engine_rank"]) for op in body["in_progress"]}
            assert ("collective_rpc:fault_dp_mismatched_collective", 0) in stuck


def test_dense_latent_cuda_error_while_idle(tmp_path):
    """A latent CUDA fault on an idle dense DP rank is caught by its probe."""
    with ready_probe_server(DENSE_MODEL, DENSE_DP_ARGS, tmp_path) as srv:
        assert srv.status() == 200
        srv.rpc("fault_ima_now", "1")
        assert srv.wait_until_unready() is not None


def test_cuda_error_on_non_output_pp_stage(tmp_path):
    """The probe checks every worker, not only the last PP stage that returns
    model outputs; PP dummy runs do not communicate across stages."""
    args = ["--pipeline-parallel-size", "2"]
    with ready_probe_server(DENSE_MODEL, args, tmp_path) as srv:
        assert srv.status() == 200
        srv.rpc("fault_ima_now_on_worker", "0")
        assert srv.status() == 503
