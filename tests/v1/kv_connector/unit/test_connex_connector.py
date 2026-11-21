# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-level hand-off of ConnexConnector (requires connex-ai)."""

import pytest

pytest.importorskip("connex.adapters.vllm.pd_connector")

from vllm.v1.request import RequestStatus  # noqa: E402

from .utils import (  # noqa: E402
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

BLOCK_SIZE = 16
PARAMS = {
    "remote_ctrl": "10.0.0.1:17100",
    "remote_engine_id": "prefill-0",
    "remote_tp_size": 1,
}


def _scheduler(kv_role):
    return create_scheduler(
        create_vllm_config(
            kv_connector="ConnexConnector",
            kv_role=kv_role,
            block_size=BLOCK_SIZE,
            kv_connector_extra_config={"host": "10.0.0.1", "ctrl_port": 17100},
        )
    )


def test_prefill_holds_blocks_until_released():
    """The prefill keeps the blocks after the request finishes and frees
    them only once the worker reports the transfer as done."""
    scheduler = _scheduler("kv_producer")
    req = create_request(num_tokens=3 * BLOCK_SIZE, do_remote_decode=True)
    req.kv_transfer_params["transfer_id"] = "T"
    scheduler.add_request(req)

    out = scheduler.schedule()
    meta = out.kv_connector_metadata
    assert meta.holds == []
    eco = scheduler.update_from_output(out, create_model_runner_output([req]))
    params = eco[0].outputs[0].kv_transfer_params
    assert params["do_remote_prefill"] and params["transfer_id"] == "T"
    assert params["remote_ctrl"] == "10.0.0.1:17100"
    assert params["remote_num_tokens"] == 3 * BLOCK_SIZE

    out = scheduler.schedule()
    (hold,) = out.kv_connector_metadata.holds
    assert hold[:2] == (req.request_id, "T") and len(hold[2]) == 3
    assert req.request_id in scheduler.requests  # blocks not freed yet

    scheduler.update_from_output(
        out, create_model_runner_output([], finished_sending={req.request_id})
    )
    assert req.request_id not in scheduler.requests


def test_decode_loads_prompt_asynchronously_then_releases():
    scheduler = _scheduler("kv_consumer")
    req = create_request(num_tokens=2 * BLOCK_SIZE + 3, do_remote_prefill=True)
    req.kv_transfer_params = {
        "do_remote_prefill": True,
        "transfer_id": "T",
        **PARAMS,
        "remote_num_tokens": 2 * BLOCK_SIZE + 3,
    }
    scheduler.add_request(req)

    out = scheduler.schedule()
    (pull,) = out.kv_connector_metadata.pulls
    assert pull.transfer_id == "T" and pull.first_block == 0
    assert len(pull.local_blocks) == 3
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    out = scheduler.schedule()
    scheduler.update_from_output(
        out, create_model_runner_output([], finished_recving={req.request_id})
    )
    out = scheduler.schedule()
    assert req.status == RequestStatus.RUNNING

    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_LENGTH_CAPPED)
    out = scheduler.schedule()
    assert out.kv_connector_metadata.releases == [("10.0.0.1:17100", "T", 1)]


def test_resume_with_locally_cached_remote_prefix_runs_without_load():
    """A resumed request whose remote prefix is already in the local prefix
    cache loads nothing: it is scheduled RUNNING right away, and the pull it
    still sends (to acquire the prefill hold) is acquire-only, so the worker
    never reports it in finished_recving (which the scheduler only accepts
    for waiting or finished requests)."""
    from connex.adapters.vllm.pd_connector import _Pull

    if "report" not in _Pull.__dataclass_fields__:
        pytest.skip("connex-ai without acquire-only pulls")
    scheduler = _scheduler("kv_consumer")
    prefix = 2 * BLOCK_SIZE

    # Populate the local prefix cache with the prompt's first two blocks.
    warm = create_request(num_tokens=prefix + 1, common_prefix_len=prefix)
    scheduler.add_request(warm)
    out = scheduler.schedule()
    scheduler.update_from_output(out, create_model_runner_output([warm]))
    scheduler.finish_requests(warm.request_id, RequestStatus.FINISHED_STOPPED)

    # Resume: prompt + generated tokens, remote KV covers the cached prefix.
    req = create_request(num_tokens=prefix + 5, common_prefix_len=prefix)
    req.kv_transfer_params = {
        "do_remote_prefill": True,
        "transfer_id": "T",
        **PARAMS,
        "remote_num_tokens": prefix,
    }
    scheduler.add_request(req)
    out = scheduler.schedule()
    assert req.status == RequestStatus.RUNNING
    assert out.num_scheduled_tokens[req.request_id] == 5  # prefix hit locally
    (pull,) = out.kv_connector_metadata.pulls
    assert pull.local_blocks == [] and pull.report is False

    # The step completes with no receive report, and the request keeps running.
    scheduler.update_from_output(out, create_model_runner_output([req]))
    out = scheduler.schedule()
    assert req.status == RequestStatus.RUNNING
    assert out.kv_connector_metadata.pulls == []
