# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
    P2pNcclConnectorMetadata,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _make_local_model_dir(tmp_path) -> str:
    # Minimal local HF config to avoid any network calls in tests.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "architectures": ["GPT2LMHeadModel"],
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 32,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    return str(tmp_path)


def _make_p2p_consumer(model_dir: str, send_type: str) -> P2pNcclConnector:
    vllm_config = create_vllm_config(model=model_dir)
    # Unit tests should not require downloading a tokenizer.
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.kv_transfer_config.kv_connector = "P2pNcclConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_consumer"
    vllm_config.kv_transfer_config.kv_parallel_size = 2
    vllm_config.kv_transfer_config.kv_connector_extra_config["send_type"] = send_type
    return P2pNcclConnector(vllm_config, role=KVConnectorRole.SCHEDULER)


class _FakeP2pEngine:
    def __init__(self, send_type: str, recv_store: dict[str, torch.Tensor | None]):
        self.send_type = send_type
        self._recv_store = recv_store

    def has_all_recv_tensors(self, tensor_ids: list[str]) -> bool:
        return all(tid in self._recv_store for tid in tensor_ids)

    def get_recv_tensor(self, tensor_id: str) -> torch.Tensor | None:
        return self._recv_store.get(tensor_id)


def test_p2p_nccl_get_mode_disables_async_loading(tmp_path):
    # prompt_len = 33 => external tokens should be 32 (exclude last token)
    req = create_request(num_tokens=33)
    model_dir = _make_local_model_dir(tmp_path)
    ext_tokens, load_kv_async = _make_p2p_consumer(
        model_dir, "GET"
    ).get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert ext_tokens == 32
    assert load_kv_async is False


def test_p2p_nccl_put_enables_async_loading(tmp_path):
    req = create_request(num_tokens=33)
    model_dir = _make_local_model_dir(tmp_path)
    ext_tokens, load_kv_async = _make_p2p_consumer(
        model_dir, "PUT"
    ).get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert ext_tokens == 32
    assert load_kv_async is True


def test_p2p_nccl_put_async_enables_async_loading(tmp_path):
    req = create_request(num_tokens=33)
    model_dir = _make_local_model_dir(tmp_path)
    ext_tokens, load_kv_async = _make_p2p_consumer(
        model_dir, "PUT_ASYNC"
    ).get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert ext_tokens == 32
    assert load_kv_async is True


@pytest.mark.parametrize("send_type", ["PUT", "PUT_ASYNC"])
def test_p2p_nccl_put_missing_kv_does_not_block_other_request(tmp_path, send_type):
    """
    P2P NCCL (PUT/PUT_ASYNC) PD separation: a request waiting for remote KVs must not
    block another request whose KVs have arrived.

    This mirrors the "req_1 missing prefill KV, req_2 normal, then resume req_1"
    scenario from `测试大纲`, but at the scheduler level (CPU test).
    """

    model_dir = _make_local_model_dir(tmp_path)
    vllm_config = create_vllm_config(model=model_dir)
    # Unit tests should not require downloading a tokenizer.
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.kv_transfer_config.kv_connector = "P2pNcclConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_consumer"
    vllm_config.kv_transfer_config.kv_parallel_size = 2
    vllm_config.kv_transfer_config.kv_connector_extra_config["send_type"] = send_type
    scheduler = create_scheduler(vllm_config)

    # Two decode-side requests: both need remote prefill KVs (external tokens>0).
    # We will report KV arrival for req_2 first, while req_1 remains missing.
    req_1 = create_request(request_id=1, num_tokens=33, block_size=scheduler.block_size)
    req_2 = create_request(request_id=2, num_tokens=33, block_size=scheduler.block_size)

    scheduler.add_request(req_1)
    scheduler.add_request(req_2)

    # STEP 1: both requests should transition to WAITING_FOR_REMOTE_KVS.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 2
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert req_1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_2.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    # STEP 2: only req_2 finishes receiving remote KVs.
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving={req_2.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 3: req_2 should be scheduled (not blocked by req_1 still waiting).
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    assert scheduler_output.scheduled_new_reqs[0].req_id == req_2.request_id
    assert req_1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_2.status == RequestStatus.RUNNING

    # Finish req_2 quickly.
    model_runner_output = create_model_runner_output([req_2], use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()

    # STEP 4: req_1 KVs arrive later; it should resume and finish.
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving={req_1.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0
    assert scheduler_output.scheduled_new_reqs[0].req_id == req_1.request_id
    assert req_1.status == RequestStatus.RUNNING

    model_runner_output = create_model_runner_output([req_1], use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()
    assert_scheduler_empty(scheduler)


@pytest.mark.parametrize("send_type", ["PUT", "PUT_ASYNC"])
def test_p2p_nccl_put_start_load_kv_does_not_pop_recv_store(tmp_path, send_type):
    model_dir = _make_local_model_dir(tmp_path)
    vllm_config = create_vllm_config(model=model_dir)
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.kv_transfer_config.kv_connector = "P2pNcclConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_consumer"
    vllm_config.kv_transfer_config.kv_parallel_size = 2
    vllm_config.kv_transfer_config.kv_connector_extra_config["send_type"] = send_type

    connector = P2pNcclConnector(vllm_config, role=KVConnectorRole.SCHEDULER)

    # Fake request_id needs embedded prefill address for parse_request_id().
    request_id = "cmpl-___prefill_addr_127.0.0.1:1234___decode_addr_127.0.0.1:2345_test"

    # FlashAttention layout: (2, num_blocks, hidden)
    layer = torch.zeros((2, 4, 1), dtype=torch.float32)
    kv_cache = torch.ones((2, 2, 1), dtype=torch.float32)

    recv_store = {f"{request_id}#layer0": kv_cache}
    connector.p2p_nccl_engine = _FakeP2pEngine(
        send_type=send_type, recv_store=recv_store
    )

    meta = P2pNcclConnectorMetadata()
    meta.add_request(
        request_id=request_id,
        token_ids=[1, 2, 3],
        block_ids=[0, 1],
        block_size=vllm_config.cache_config.block_size,
    )
    connector.bind_connector_metadata(meta)

    forward_context = SimpleNamespace(
        attn_metadata=None,
        virtual_engine=0,
        no_compile_layers={"layer0": SimpleNamespace(kv_cache=[layer])},
    )

    connector.start_load_kv(forward_context)

    # Injected into blocks [0, 1].
    assert torch.all(layer[:, 0:2, :] == 1)
    assert torch.all(layer[:, 2:4, :] == 0)

    # The tensor should still exist in recv_store (non-pop semantics).
    assert f"{request_id}#layer0" in recv_store


@pytest.mark.parametrize("send_type", ["PUT", "PUT_ASYNC"])
def test_p2p_nccl_put_start_load_kv_truncates_if_kv_has_more_blocks(
    tmp_path, send_type
):
    model_dir = _make_local_model_dir(tmp_path)
    vllm_config = create_vllm_config(model=model_dir)
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.kv_transfer_config.kv_connector = "P2pNcclConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_consumer"
    vllm_config.kv_transfer_config.kv_parallel_size = 2
    vllm_config.kv_transfer_config.kv_connector_extra_config["send_type"] = send_type

    connector = P2pNcclConnector(vllm_config, role=KVConnectorRole.SCHEDULER)

    request_id = "cmpl-___prefill_addr_127.0.0.1:1234___decode_addr_127.0.0.1:2345_test"

    layer = torch.zeros((2, 4, 1), dtype=torch.float32)
    # kv_cache has 3 blocks, but we only have 2 block_ids available.
    kv_cache = torch.ones((2, 3, 1), dtype=torch.float32)

    recv_store = {f"{request_id}#layer0": kv_cache}
    connector.p2p_nccl_engine = _FakeP2pEngine(
        send_type=send_type, recv_store=recv_store
    )

    meta = P2pNcclConnectorMetadata()
    meta.add_request(
        request_id=request_id,
        token_ids=[1, 2, 3],
        block_ids=[0, 1],
        block_size=vllm_config.cache_config.block_size,
    )
    connector.bind_connector_metadata(meta)

    forward_context = SimpleNamespace(
        attn_metadata=None,
        virtual_engine=0,
        no_compile_layers={"layer0": SimpleNamespace(kv_cache=[layer])},
    )

    connector.start_load_kv(forward_context)

    # Only the overlapping blocks should be injected.
    assert torch.all(layer[:, 0:2, :] == 1)
    assert torch.all(layer[:, 2:4, :] == 0)


@pytest.mark.parametrize("send_type", ["PUT", "PUT_ASYNC"])
def test_p2p_nccl_put_start_load_kv_reloads_if_block_ids_change(tmp_path, send_type):
    model_dir = _make_local_model_dir(tmp_path)
    vllm_config = create_vllm_config(model=model_dir)
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.kv_transfer_config.kv_connector = "P2pNcclConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_consumer"
    vllm_config.kv_transfer_config.kv_parallel_size = 2
    vllm_config.kv_transfer_config.kv_connector_extra_config["send_type"] = send_type

    connector = P2pNcclConnector(vllm_config, role=KVConnectorRole.SCHEDULER)

    request_id = "cmpl-___prefill_addr_127.0.0.1:1234___decode_addr_127.0.0.1:2345_test"

    layer = torch.zeros((2, 4, 1), dtype=torch.float32)
    kv_cache = torch.ones((2, 2, 1), dtype=torch.float32)
    recv_store = {f"{request_id}#layer0": kv_cache}
    connector.p2p_nccl_engine = _FakeP2pEngine(
        send_type=send_type, recv_store=recv_store
    )

    forward_context = SimpleNamespace(
        attn_metadata=None,
        virtual_engine=0,
        no_compile_layers={"layer0": SimpleNamespace(kv_cache=[layer])},
    )

    # Initial load into [0, 1].
    meta_0 = P2pNcclConnectorMetadata()
    meta_0.add_request(
        request_id=request_id,
        token_ids=[1, 2, 3],
        block_ids=[0, 1],
        block_size=vllm_config.cache_config.block_size,
    )
    connector.bind_connector_metadata(meta_0)
    connector.start_load_kv(forward_context)
    assert torch.all(layer[:, 0:2, :] == 1)

    # Simulate "resumed from preemption": new blocks allocated [2, 3].
    meta_1 = P2pNcclConnectorMetadata()
    meta_1.add_request(
        request_id=request_id,
        token_ids=[1, 2, 3],
        block_ids=[2, 3],
        block_size=vllm_config.cache_config.block_size,
    )
    connector.bind_connector_metadata(meta_1)
    connector.start_load_kv(forward_context)
    assert torch.all(layer[:, 2:4, :] == 1)
