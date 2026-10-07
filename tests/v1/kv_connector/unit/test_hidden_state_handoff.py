# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler flow of the P/D hidden-state handoff."""

import copy

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateRecordSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _config(block_size):
    return KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["hidden_state_record.0"],
                HiddenStateRecordSpec(
                    block_size=1,
                    num_kv_heads=1,
                    head_size=4,
                    head_size_v=0,
                    dtype=torch.float32,
                ),
            ),
        ],
    )


def _make(kv_role="kv_consumer"):
    vllm_config = create_vllm_config(
        kv_connector_extra_config={"hidden_state_handoff": True},
        kv_role=kv_role,
        kv_load_failure_policy="recompute",
    )
    bs = vllm_config.cache_config.block_size
    return vllm_config, create_scheduler(vllm_config, kv_cache_config=_config(bs))


def _recv(scheduler, request, failed=False):
    so = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    out = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    out.kv_connector_output = KVConnectorOutput(
        finished_recving={request.request_id},
        failed_recving={request.request_id} if failed else set(),
    )
    scheduler.update_from_output(so, out)


def test_decode_samples_from_record():
    vllm_config, scheduler = _make()
    bs = vllm_config.cache_config.block_size
    for num_tokens in (bs * 2, bs * 2 + 5):
        request = create_request(
            block_size=bs,
            num_tokens=num_tokens,
            do_remote_prefill=True,
            num_remote_blocks=3,
        )
        scheduler.add_request(request)
        _recv(scheduler, request)
        so = scheduler.schedule()
        rid = request.request_id
        assert so.hidden_state_record_req_ids == {rid}
        assert so.num_scheduled_tokens[rid] == 1
        (new_req,) = so.scheduled_new_reqs
        assert new_req.num_computed_tokens == num_tokens - 1
        # One record block, transferred with the KV.
        assert len(new_req.block_ids) == 2 and len(new_req.block_ids[1]) == 1
        scheduler.update_from_output(so, create_model_runner_output([request]))
        so = scheduler.schedule()
        assert not so.hidden_state_record_req_ids
        assert request.num_computed_tokens == num_tokens + 1
        scheduler.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
        scheduler.update_from_output(so, create_model_runner_output([]))


def test_failed_recv_recomputes():
    vllm_config, scheduler = _make()
    bs = vllm_config.cache_config.block_size
    request = create_request(
        block_size=bs, num_tokens=bs * 2 + 3, do_remote_prefill=True
    )
    scheduler.add_request(request)
    _recv(scheduler, request, failed=True)
    so = scheduler.schedule()
    assert not so.hidden_state_record_req_ids
    assert so.num_scheduled_tokens[request.request_id] > 1


def test_prefiller_returns_record_block():
    vllm_config, scheduler = _make(kv_role="kv_producer")
    bs = vllm_config.cache_config.block_size
    request = create_request(
        block_size=bs, num_tokens=bs * 2 + 3, do_remote_decode=True
    )
    scheduler.add_request(request)
    so = scheduler.schedule()
    assert so.num_scheduled_tokens[request.request_id] == bs * 2 + 3
    outs = scheduler.update_from_output(
        so, create_model_runner_output([request], use_eos=True)
    )
    params = outs[0].outputs[0].kv_transfer_params
    assert len(params["remote_block_ids"]) == 2
    assert len(params["remote_block_ids"][1]) == 1
