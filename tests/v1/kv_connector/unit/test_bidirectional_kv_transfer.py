# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for bi-directional KV cache transfer between P and D nodes.

Tests cover the new behaviors added by the bi-directional KV transfer PR:
1. P-node scheduler lifecycle: P pulls KV from D using remote_block_ids,
   eliminating redundant prefill computation in multi-turn conversations.
2. P-node metadata: NixlConnectorMetadata correctly populates recv metadata
   when P pulls KV from D (do_remote_decode=True + remote_block_ids).
3. P-node worker: start_load_kv processes reqs_to_recv for KV pull from D.
4. D-node request_finished: returns kv_transfer_params with remote_block_ids
   and remote_num_tokens so P can pull KV in future turns.
5. Edge cases:
   - No double read after reschedule (_remote_blocks_processed flag)
   - remote_num_tokens bounded by block capacity (num_computed_tokens)
   - kv_recompute_threshold skips small transfers
   - P-node holds blocks for D after finishing
   - Cache MISS first turn falls back to local prefill
   - Partial remote coverage: P pulls partial, computes the rest
   - _remote_blocks_processed flag persists across reschedules

P-node flags: do_remote_prefill=False (prefill locally),
do_remote_decode=True (don't decode locally, send KV to D).
P pulls KV from D when remote_block_ids is not None and
external tokens > 0.
"""

import copy
import time
from unittest.mock import patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlConnector,
    NixlConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
)
from vllm.v1.request import RequestStatus

from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper
from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

pytestmark = pytest.mark.cpu_test

# Common extra config for all bi-directional KV transfer tests.
BIDIR_KV_EXTRA_CONFIG = {"bidirectional_kv_xfer": True, "kv_recompute_threshold": 0}


# Helpers


def _make_p_node_turn2_request(
    request_id, block_size, num_tokens, num_remote_blocks=3, remote_num_tokens=None
):
    """Create a P-node Turn 2 request with remote_block_ids from D."""
    request = create_request(
        request_id=request_id,
        block_size=block_size,
        num_tokens=num_tokens,
        do_remote_decode=True,
    )
    if remote_num_tokens is None:
        remote_num_tokens = num_remote_blocks * block_size
    request.kv_transfer_params["remote_block_ids"] = [list(range(num_remote_blocks))]
    request.kv_transfer_params["remote_num_tokens"] = remote_num_tokens
    request.kv_transfer_params["remote_engine_id"] = "decode-engine"
    request.kv_transfer_params["remote_request_id"] = f"decode-{request_id}"
    request.kv_transfer_params["remote_host"] = "decode-host"
    request.kv_transfer_params["remote_port"] = 5678
    return request


def _make_connector_with_fake_worker(
    hand_shake_latency=0, cycles_before_done=0, do_handshake=True
):
    """Create a NixlConnector with FakeNixlConnectorWorker."""
    vllm_config = create_vllm_config()
    kv_cache_config = make_kv_cache_config(block_size=16, num_blocks=2)
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config,
        connector.engine_id,
        hand_shake_latency=hand_shake_latency,
        kv_cache_config=kv_cache_config,
    )
    worker = connector.connector_worker
    assert isinstance(worker.nixl_wrapper, FakeNixlWrapper)
    worker.nixl_wrapper.set_cycles_before_xfer_done(cycles_before_done)
    worker.kv_cache_layout = "HND"
    if do_handshake:
        remote_agents = worker._nixl_handshake(
            host="localhost",
            port=1234,
            remote_tp_size=1,
            expected_engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
        )
        worker._remote_agents[FakeNixlConnectorWorker.REMOTE_ENGINE_ID] = remote_agents
    return connector, worker


def _make_p_node_recv_metadata(request_id, local_blocks, remote_blocks):
    """Build NixlConnectorMetadata for P-node pulling KV from D."""
    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id=request_id,
        local_block_ids=(local_blocks,),
        kv_transfer_params={
            "do_remote_prefill": False,
            "do_remote_decode": True,
            "remote_block_ids": (remote_blocks,),
            "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            "remote_request_id": f"decode-{request_id}",
            "remote_host": "localhost",
            "remote_port": 1234,
            "remote_tp_size": 1,
        },
    )
    return meta


def _do_load_kv(connector, metadata):
    """Bind metadata and call start_load_kv."""
    connector.bind_connector_metadata(metadata)
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    connector.start_load_kv(ctx)


# 1. P-node scheduler lifecycle tests


def test_multiturn_lifecycle():
    """Full two-turn lifecycle on the P node:
    Turn 1: P prefills locally (do_remote_prefill=False), sends KV to D
    (do_remote_decode=True). Finishes LENGTH_CAPPED with remote_block_ids.
    Turn 2: P receives remote_block_ids from D. P pulls KV from D because
    remote_block_ids is not None and external tokens > 0. Computes only
    new tokens, finishes LENGTH_CAPPED."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size

    t1 = create_request(
        request_id=100, block_size=BS, num_tokens=int(BS * 2.5), do_remote_decode=True
    )
    scheduler.add_request(t1)
    t1_id = t1.request_id
    so = scheduler.schedule()
    mro = create_model_runner_output(reqs=[t1])
    eco = scheduler.update_from_output(so, mro)
    assert t1.status == RequestStatus.FINISHED_LENGTH_CAPPED
    kv = eco[0].outputs[0].kv_transfer_params
    assert kv and sum(len(g) for g in kv["remote_block_ids"]) > 0
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)

    t2 = _make_p_node_turn2_request(200, BS, int(BS * 2.5))
    scheduler.add_request(t2)
    t2_id = t2.request_id
    so = scheduler.schedule()
    assert t2.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_recving={t2_id})
    scheduler.update_from_output(so, mro)
    so = scheduler.schedule()
    mro = create_model_runner_output(reqs=[t2])
    scheduler.update_from_output(so, mro)
    assert t2.status == RequestStatus.FINISHED_LENGTH_CAPPED
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={t1_id, t2_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_first_turn_no_remote_blocks():
    """First turn: P has no remote_block_ids from D yet.
    Standard local prefill, returns kv_transfer_params for future turns."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=3, block_size=BS, num_tokens=int(BS * 2.5), do_remote_decode=True
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    assert req.status != RequestStatus.WAITING_FOR_REMOTE_KVS
    mro = create_model_runner_output(reqs=[req])
    eco = scheduler.update_from_output(so, mro)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert eco[0].outputs[0].kv_transfer_params is not None
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_abort_p_side_during_send():
    """P-side do_remote_decode=True: blocks held until finished_sending."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=42, block_size=BS, num_tokens=int(BS * 2.5), do_remote_decode=True
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    assert req_id in scheduler.requests
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    assert req_id in scheduler.requests
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_abort_p_side_non_length_capped():
    """P-side abort with non-LENGTH_CAPPED → immediate block free."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=44, block_size=BS, num_tokens=int(BS * 2.5), do_remote_decode=True
    )
    req.sampling_params.max_tokens = 100
    req.max_tokens = 100
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    scheduler.finish_requests([req_id], RequestStatus.FINISHED_ABORTED)
    conn = scheduler.connector.connector_scheduler
    assert req_id in conn._reqs_not_processed
    assert req_id not in scheduler.requests
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    assert_scheduler_empty(scheduler)


def test_remote_blocks_exceed_prompt_tokens():
    """D provides more remote tokens than P's prompt needs.
    P caps external tokens to prompt length."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BS * 2.5)
    req = _make_p_node_turn2_request(
        300, BS, NUM_TOKENS, num_remote_blocks=5, remote_num_tokens=5 * BS
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req.num_computed_tokens == NUM_TOKENS
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_recving={req_id})
    scheduler.update_from_output(so, mro)
    so = scheduler.schedule()
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_p_node_pulls_partial_last_block_from_d():
    """D sends remote_block_ids with partially filled last block.
    remote_num_tokens < len(remote_block_ids) * block_size.
    P pulls only remote_num_tokens worth of external tokens."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    num_remote_blocks = 3
    remote_num_tokens = int(BS * 2.5)
    assert remote_num_tokens < num_remote_blocks * BS
    NUM_TOKENS = int(BS * 3.5)
    req = _make_p_node_turn2_request(
        400,
        BS,
        NUM_TOKENS,
        num_remote_blocks=num_remote_blocks,
        remote_num_tokens=remote_num_tokens,
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_recving={req_id})
    scheduler.update_from_output(so, mro)
    so = scheduler.schedule()
    assert len(scheduler.running) == 1
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


# 2. P-node metadata tests


def test_add_new_req_to_recv_populates_remote_meta():
    """add_new_req_to_recv correctly populates RemoteMeta for P-node
    bi-directional KV pull from D."""
    meta = NixlConnectorMetadata()
    kv_params = {
        "remote_block_ids": [[0, 1, 2]],
        "remote_engine_id": "decode-engine",
        "remote_request_id": "decode-req-123",
        "remote_host": "decode-host",
        "remote_port": 5678,
    }
    local_block_ids = ([10, 11, 12],)
    meta.add_new_req_to_recv(
        request_id="test-req",
        local_block_ids=local_block_ids,
        kv_transfer_params=kv_params,
    )
    assert "test-req" in meta.reqs_to_recv
    rm = meta.reqs_to_recv["test-req"]
    assert rm.remote is not None
    assert rm.remote.block_ids == kv_params["remote_block_ids"]
    assert rm.remote.engine_id == "decode-engine"
    assert rm.remote.request_id == "decode-req-123"
    assert rm.remote.host == "decode-host"
    assert rm.remote.port == 5678
    assert rm.local_block_ids == local_block_ids


def test_build_connector_meta_recv_entries():
    """P-node scheduler: do_remote_decode=True + remote_block_ids →
    _reqs_need_recv populated, build_connector_meta produces reqs_to_recv."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = _make_p_node_turn2_request(1, BS, int(BS * 2.5))
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    meta = so.kv_connector_metadata
    assert isinstance(meta, NixlConnectorMetadata)
    assert req_id in meta.reqs_to_recv
    rm = meta.reqs_to_recv[req_id]
    assert rm.remote is not None
    assert rm.remote.engine_id == "decode-engine"


def test_build_connector_meta_clears_reqs_need_recv():
    """After build_connector_meta, _reqs_need_recv is cleared."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = _make_p_node_turn2_request(2, BS, int(BS * 2.5))
    scheduler.add_request(req)
    conn = scheduler.connector.connector_scheduler
    scheduler.schedule()
    assert len(conn._reqs_need_recv) == 0


def test_build_connector_meta_multiple_requests():
    """Multiple P-node requests all included in reqs_to_recv."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    reqs = [_make_p_node_turn2_request(10 + i, BS, int(BS * 2.5)) for i in range(3)]
    for r in reqs:
        scheduler.add_request(r)
    so = scheduler.schedule()
    meta = so.kv_connector_metadata
    assert isinstance(meta, NixlConnectorMetadata)
    assert len(meta.reqs_to_recv) == 3
    for r in reqs:
        assert r.request_id in meta.reqs_to_recv


# 3. P-node worker tests (FakeNixlWrapper)


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker.NixlWrapper",
    FakeNixlWrapper,
)
def test_p_node_pull_kv_from_d(dist_init):
    """P node pulls KV from D via start_load_kv with reqs_to_recv."""
    connector, worker = _make_connector_with_fake_worker()
    meta = _make_p_node_recv_metadata("req-p1", [10, 11, 12], [20, 21, 22])
    _do_load_kv(connector, meta)
    assert "req-p1" in worker._recving_metadata
    _, done_recving = connector.get_finished(finished_req_ids=set())
    assert "req-p1" in done_recving


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker.NixlWrapper",
    FakeNixlWrapper,
)
def test_p_node_pull_then_send_kv(dist_init):
    """Full P-node bi-directional: pull KV from D → prefill →
    send KV back to D via notification."""
    connector, worker = _make_connector_with_fake_worker()
    meta = _make_p_node_recv_metadata("req-p2", [10, 11], [20, 21])
    _do_load_kv(connector, meta)
    _, done_recving = connector.get_finished(finished_req_ids=set())
    assert "req-p2" in done_recving
    worker._reqs_to_send["req-p2"] = time.perf_counter() + 60
    worker._reqs_to_process.add("req-p2")
    notif = f"req-p2:{worker.world_size}".encode()
    orig = worker.nixl_wrapper.get_new_notifs
    worker.nixl_wrapper.get_new_notifs = lambda: {"agent": [notif]}
    done_sending, _ = connector.get_finished(finished_req_ids=set())
    assert "req-p2" in done_sending
    worker.nixl_wrapper.get_new_notifs = orig


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker.NixlWrapper",
    FakeNixlWrapper,
)
def test_p_node_deferred_pull_on_no_handshake(dist_init):
    """P defers KV pull when no prior handshake exists."""
    connector, worker = _make_connector_with_fake_worker(
        hand_shake_latency=0, do_handshake=False
    )
    meta = _make_p_node_recv_metadata("req-p3", [10, 11], [20, 21])
    _do_load_kv(connector, meta)
    assert "req-p3" in worker._recving_metadata
    timeout = 3.0
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        connector.bind_connector_metadata(NixlConnectorMetadata())
        ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
        connector.start_load_kv(ctx)
        _, done = connector.get_finished(finished_req_ids=set())
        if "req-p3" in done:
            return
        time.sleep(0.2)
    raise AssertionError("Transfer did not complete after async handshake")


# 4. D-node request_finished returns kv_transfer_params (new behavior)


def test_d_node_request_finished_returns_kv_params():
    """D-node request_finished returns kv_transfer_params with
    do_remote_decode=True, remote_block_ids, remote_num_tokens
    for P to pull. These params go directly to P node."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=1, block_size=BS, num_tokens=int(BS * 2.5), do_remote_prefill=True
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    scheduler.update_from_output(
        so, create_model_runner_output(reqs=[], finished_recving={req_id})
    )
    so = scheduler.schedule()
    eco = scheduler.update_from_output(
        so, create_model_runner_output(reqs=[req], use_eos=True)
    )
    assert req.status == RequestStatus.FINISHED_STOPPED
    kv = eco[0].outputs[0].kv_transfer_params
    assert kv is not None
    assert kv["do_remote_decode"] is True
    assert kv["do_remote_prefill"] is False
    assert "remote_block_ids" in kv
    assert "remote_num_tokens" in kv
    assert kv["remote_num_tokens"] > 0


def test_d_node_request_finished_delays_block_free():
    """D-node holds blocks (delay_free=True) until P reads them."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=2, block_size=BS, num_tokens=int(BS * 2.5), do_remote_prefill=True
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    scheduler.update_from_output(
        so, create_model_runner_output(reqs=[], finished_recving={req_id})
    )
    so = scheduler.schedule()
    scheduler.update_from_output(
        so, create_model_runner_output(reqs=[req], use_eos=True)
    )
    assert req_id in scheduler.requests
    conn = scheduler.connector.connector_scheduler
    assert req_id in conn._reqs_need_send


def test_d_node_request_finished_remote_num_tokens():
    """D-node kv_transfer_params includes correct remote_num_tokens."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=3, block_size=BS, num_tokens=int(BS * 2.5), do_remote_prefill=True
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    scheduler.update_from_output(
        so, create_model_runner_output(reqs=[], finished_recving={req_id})
    )
    so = scheduler.schedule()
    eco = scheduler.update_from_output(
        so, create_model_runner_output(reqs=[req], use_eos=True)
    )
    kv = eco[0].outputs[0].kv_transfer_params
    assert kv["remote_num_tokens"] > 0
    assert sum(len(g) for g in kv["remote_block_ids"]) > 0


def test_d_node_partial_last_block_remote_num_tokens():
    """D-node: remote_num_tokens < len(remote_block_ids) * block_size
    when last block is partially filled."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=5, block_size=BS, num_tokens=int(BS * 2.5), do_remote_prefill=True
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    scheduler.update_from_output(
        so, create_model_runner_output(reqs=[], finished_recving={req_id})
    )
    so = scheduler.schedule()
    eco = scheduler.update_from_output(
        so, create_model_runner_output(reqs=[req], use_eos=True)
    )
    kv = eco[0].outputs[0].kv_transfer_params
    total_blocks = sum(len(g) for g in kv["remote_block_ids"])
    assert total_blocks == 3
    assert kv["remote_num_tokens"] < total_blocks * BS
    assert kv["remote_num_tokens"] > 0


# 5. Edge case tests


def test_no_double_read_blocks_after_reschedule():
    """Edge case 1: update_state_after_alloc called twice for the same
    bidirectional request (once on initial schedule, once after
    WAITING_FOR_REMOTE_KVS → reschedule). The _remote_blocks_processed
    flag must prevent the request from being added to _reqs_need_recv
    twice, which would cause P to read D's blocks twice."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = _make_p_node_turn2_request(500, BS, int(BS * 2.5))
    scheduler.add_request(req)
    req_id = req.request_id
    conn = scheduler.connector.connector_scheduler

    # First schedule: request enters WAITING_FOR_REMOTE_KVS,
    # _reqs_need_recv populated then cleared by build_connector_meta.
    so = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    meta = so.kv_connector_metadata
    assert isinstance(meta, NixlConnectorMetadata)
    assert req_id in meta.reqs_to_recv
    # _reqs_need_recv should be cleared after build_connector_meta
    assert len(conn._reqs_need_recv) == 0

    # Simulate recv completion
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_recving={req_id})
    scheduler.update_from_output(so, mro)

    # Second schedule after recv: update_state_after_alloc called again.
    # The _remote_blocks_processed flag should prevent re-entry.
    so = scheduler.schedule()
    meta2 = so.kv_connector_metadata
    assert isinstance(meta2, NixlConnectorMetadata)
    # Must NOT be in reqs_to_recv again
    assert req_id not in meta2.reqs_to_recv

    # Clean up
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_remote_num_tokens_bounded_by_blocks():
    """Edge case 2: D-node request_finished must return
    remote_num_tokens <= len(remote_block_ids) * block_size.
    request.num_tokens includes the last sampled token which has no KV
    in the cache, so remote_num_tokens must use num_computed_tokens."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=501,
        block_size=BS,
        num_tokens=int(BS * 2.5),
        do_remote_prefill=True,
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    scheduler.update_from_output(
        so, create_model_runner_output(reqs=[], finished_recving={req_id})
    )
    so = scheduler.schedule()
    eco = scheduler.update_from_output(
        so, create_model_runner_output(reqs=[req], use_eos=True)
    )
    kv = eco[0].outputs[0].kv_transfer_params
    assert kv is not None
    total_blocks = sum(len(g) for g in kv["remote_block_ids"])
    max_tokens_in_blocks = total_blocks * BS
    assert kv["remote_num_tokens"] <= max_tokens_in_blocks, (
        f"remote_num_tokens ({kv['remote_num_tokens']}) exceeds "
        f"block capacity ({max_tokens_in_blocks})"
    )
    assert kv["remote_num_tokens"] > 0


def test_kv_recompute_threshold_skips_small_transfer():
    """Edge case 3: When remote tokens are below kv_recompute_threshold,
    P should skip the remote pull and compute locally instead of
    entering WAITING_FOR_REMOTE_KVS."""
    threshold = 256
    vllm_config = create_vllm_config(
        kv_connector_extra_config={
            "bidirectional_kv_xfer": True,
            "kv_recompute_threshold": threshold,
        },
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size

    # Create request where remote tokens (48) < threshold (256)
    req = _make_p_node_turn2_request(
        502,
        BS,
        int(BS * 2.5),
        num_remote_blocks=3,
        remote_num_tokens=3 * BS,
    )
    scheduler.add_request(req)
    so = scheduler.schedule()
    # Should NOT enter WAITING_FOR_REMOTE_KVS — threshold not met
    assert req.status != RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req.status == RequestStatus.RUNNING

    # Clean up
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req.request_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_p_node_finished_holds_blocks_for_d():
    """Edge case 4: P-node finishes with FINISHED_LENGTH_CAPPED and
    do_remote_decode=True. P must hold blocks (delay_free=True) and
    return kv_transfer_params with do_remote_prefill=True so D can
    read P's blocks."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=503,
        block_size=BS,
        num_tokens=int(BS * 2.5),
        do_remote_decode=True,
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    mro = create_model_runner_output(reqs=[req])
    eco = scheduler.update_from_output(so, mro)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    kv = eco[0].outputs[0].kv_transfer_params
    assert kv is not None
    # P-node finished: should tell D to pull (do_remote_prefill=True)
    assert kv["do_remote_prefill"] is True
    assert kv["do_remote_decode"] is False
    assert "remote_block_ids" in kv
    assert sum(len(g) for g in kv["remote_block_ids"]) > 0
    # Blocks should be held (request still tracked)
    assert req_id in scheduler.requests

    # Clean up: simulate D reading and notifying
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_cache_miss_first_turn_no_remote_pull():
    """Edge case 5: First turn with do_remote_decode=True but no
    remote_block_ids (cache MISS). P should prefill locally with
    num_external_tokens=0 and not enter WAITING_FOR_REMOTE_KVS."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = create_request(
        request_id=504,
        block_size=BS,
        num_tokens=int(BS * 2.5),
        do_remote_decode=True,
    )
    # No remote_block_ids set — this is a cache MISS
    assert req.kv_transfer_params.get("remote_block_ids") is None
    scheduler.add_request(req)
    so = scheduler.schedule()
    # Should NOT wait for remote KVs
    assert req.status != RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req.status == RequestStatus.RUNNING

    # Clean up
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req.request_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_partial_remote_tokens_less_than_prompt():
    """Edge case 6: D's remote_num_tokens covers only part of P's
    prompt. P should pull remote_num_tokens worth of external tokens
    and compute the rest locally."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BS * 4.5)  # 72 tokens
    # D provides only 2 blocks (32 tokens) out of 72
    req = _make_p_node_turn2_request(
        505,
        BS,
        NUM_TOKENS,
        num_remote_blocks=2,
        remote_num_tokens=2 * BS,
    )
    scheduler.add_request(req)
    req_id = req.request_id
    so = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    # num_computed_tokens should reflect the external tokens pulled
    # (capped to remote_num_tokens, not full prompt)
    assert req.num_computed_tokens < NUM_TOKENS

    # Complete the transfer and finish
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_recving={req_id})
    scheduler.update_from_output(so, mro)
    so = scheduler.schedule()
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)


def test_remote_blocks_processed_flag_persists():
    """Edge case 7: After recv completes and request is rescheduled,
    the _remote_blocks_processed flag in kv_transfer_params prevents
    the bidirectional path from re-entering _reqs_need_recv."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config=BIDIR_KV_EXTRA_CONFIG,
    )
    scheduler = create_scheduler(vllm_config)
    BS = vllm_config.cache_config.block_size
    req = _make_p_node_turn2_request(506, BS, int(BS * 2.5))
    scheduler.add_request(req)
    req_id = req.request_id
    conn = scheduler.connector.connector_scheduler

    # First schedule → WAITING_FOR_REMOTE_KVS
    so = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)

    # Recv completes
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_recving={req_id})
    scheduler.update_from_output(so, mro)

    # Verify the flag is set
    assert req.kv_transfer_params.get("_remote_blocks_processed") is True

    # Next schedule: update_state_after_alloc is called again.
    # _reqs_need_recv must NOT contain this request.
    so = scheduler.schedule()
    assert req_id not in conn._reqs_need_recv
    meta = so.kv_connector_metadata
    assert isinstance(meta, NixlConnectorMetadata)
    assert req_id not in meta.reqs_to_recv

    # Clean up
    mro = create_model_runner_output(reqs=[req])
    scheduler.update_from_output(so, mro)
    so = scheduler.schedule()
    scheduler.update_from_output(so, EMPTY_MODEL_RUNNER_OUTPUT)
    so = scheduler.schedule()
    mro = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    mro.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
    scheduler.update_from_output(so, mro)
    assert_scheduler_empty(scheduler)
