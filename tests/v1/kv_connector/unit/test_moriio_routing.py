# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MoRIIO router-authoritative DP-rank routing.

These lock in the redesign that addresses the PR #45043 review feedback
(njhill): the connector must NOT self-derive a DP rank, and the prefill leg
must *return* the rank it actually ran on via ``request_finished`` so the
decode->prefill notify is routed by pure propagation (READ / serial WRITE),
not by an independent hash on each side.

Following test_moriio_unmap.py, we bind the unbound method to a lightweight
stand-in rather than constructing a full scheduler.
"""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)
from vllm.v1.request import RequestStatus

_request_finished = MoRIIOConnectorScheduler.request_finished


def _producer(global_dp_rank: int, dp_size: int, dp_size_local: int):
    """Minimal prefill-side (producer) scheduler stand-in for request_finished."""
    return SimpleNamespace(
        is_producer=True,
        _global_dp_rank=global_dp_rank,
        engine_id="engine-abc",
        host_ip="10.0.0.1",
        handshake_port=6300,
        side_notify_port=7300,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=dp_size,
                data_parallel_size_local=dp_size_local,
                tensor_parallel_size=1,
            )
        ),
    )


def _req(request_id: str, params: dict | None, status=RequestStatus.FINISHED_LENGTH_CAPPED):
    return SimpleNamespace(
        request_id=request_id,
        kv_transfer_params=params,
        status=status,
    )


def test_request_finished_returns_prefill_rank_for_propagation():
    # 2P2D: global rank 11 lives on pod 1 (ranks 8..15), local size 8.
    s = _producer(global_dp_rank=11, dp_size=16, dp_size_local=8)
    req = _req("r1", {"do_remote_decode": True, "transfer_id": "t1"})

    delay_free, params = _request_finished(s, req, block_ids=[])

    # The prefill leg returns the rank IT actually ran on, marked authoritative,
    # so the router forwards it to decode and the notify propagates (no hashing).
    assert params is not None
    assert params["remote_dp_rank"] == 11
    assert params["remote_dp_rank_override"] is True
    assert params["remote_dp_size"] == 16
    assert params["remote_dp_size_local"] == 8
    assert params["transfer_id"] == "t1"
    # Empty block_ids => nothing deferred, blocks freed immediately.
    assert delay_free is False


def test_request_finished_rank_tracks_the_actual_rank():
    # A different rank must be echoed verbatim -- the value is the producer's
    # own rank, never a recomputation of the request id.
    s = _producer(global_dp_rank=4, dp_size=8, dp_size_local=8)
    req = _req("r2", {"do_remote_decode": True, "transfer_id": "t2"})

    _, params = _request_finished(s, req, block_ids=[])

    assert params["remote_dp_rank"] == 4
    assert params["remote_dp_size_local"] == 8


def test_request_finished_no_kv_params_is_noop():
    s = _producer(global_dp_rank=0, dp_size=16, dp_size_local=8)
    req = _req("r3", None)

    delay_free, params = _request_finished(s, req, block_ids=[])

    assert delay_free is False
    assert params is None
