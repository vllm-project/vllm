# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MultiConnector wrapping NixlConnector + SimpleCPUOffloadConnector.

Verifies scheduler-level behavior: HMA support detection, load delegation
(first-wins), store-to-all, and connector metadata aggregation.
"""

import uuid
from collections import defaultdict
from unittest.mock import patch

import pytest
import torch

from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnectorMetadata,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

NIXL_WRAPPER_PATCH = (
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker.NixlWrapper"
)


class FakeNixlWrapper:
    """Minimal mock of NixlWrapper for testing without NIXL hardware.

    Duplicated from test_nixl_connector.py to avoid importing that module
    (which has heavy dependencies like ray).
    """

    AGENT_METADATA = b"fake_agent_metadata"
    REMOTE_AGENT_NAME = "remote_agent"

    def __init__(self, agent_name: str, *args, **kwargs):
        self._cycles_before_xfer_done = 0
        self._check_xfer_state_cycles: defaultdict[int, int] = defaultdict(lambda: 0)

    def get_reg_descs(self, caches_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in caches_data]

    def register_memory(self, descs, backends) -> None:
        pass

    def deregister_memory(self, descs) -> None:
        pass

    def get_xfer_descs(self, blocks_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in blocks_data]

    def prep_xfer_dlist(self, agent_name: str, descs: list) -> int:
        return uuid.uuid4().int

    def get_agent_metadata(self) -> bytes:
        return self.AGENT_METADATA

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        return self.REMOTE_AGENT_NAME

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return {}

    def check_xfer_state(self, handle: int) -> str:
        if self._check_xfer_state_cycles[handle] >= self._cycles_before_xfer_done:
            return "DONE"
        self._check_xfer_state_cycles[handle] += 1
        return "PROC"

    def release_xfer_handle(self, handle: int) -> None:
        pass

    def release_dlist_handle(self, handle: int) -> None:
        pass

    def remove_remote_agent(self, agent: str) -> None:
        pass

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass

    def make_prepped_xfer(self, *args, **kwargs) -> int:
        return uuid.uuid4().int

    def transfer(self, handle: int) -> str:
        return "PROC"

    def get_xfer_telemetry(self, handle: int) -> dict:
        return {}

    def set_cycles_before_xfer_done(self, cycles: int):
        pass


BLOCK_SIZE = 16
NUM_KV_HEADS = 1
HEAD_SIZE = 16
DTYPE = torch.float16
_BYTES_PER_BLOCK = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2 * DTYPE.itemsize


def _make_kv_cache_config(
    num_blocks: int = 100,
    swa_enabled: bool = False,
) -> KVCacheConfig:
    """Build KVCacheConfig with non-empty kv_cache_tensors.

    SimpleCPUOffloadConnector requires kv_cache_tensors with real sizes
    (used by _derive_cpu_config).
    """
    fa_layers = ["layer0", "layer2"]
    groups = [
        KVCacheGroupSpec(
            fa_layers,
            FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_SIZE,
                dtype=DTYPE,
            ),
        )
    ]
    tensors = [
        KVCacheTensor(
            size=_BYTES_PER_BLOCK * num_blocks,
            shared_by=fa_layers,
        )
    ]
    if swa_enabled:
        sw_layers = ["layer1", "layer3"]
        groups.append(
            KVCacheGroupSpec(
                sw_layers,
                SlidingWindowSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=NUM_KV_HEADS,
                    head_size=HEAD_SIZE,
                    dtype=DTYPE,
                    sliding_window=128,
                ),
            )
        )
        tensors.append(
            KVCacheTensor(
                size=_BYTES_PER_BLOCK * num_blocks,
                shared_by=sw_layers,
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def _multi_connector_config(swa_enabled: bool = False):
    """Return (vllm_config, kv_cache_config) for a MultiConnector test."""
    kv_cache_config = _make_kv_cache_config(swa_enabled=swa_enabled)
    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_both",
                },
                {
                    "kv_connector": "SimpleCPUOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use": 1 << 30,
                    },
                },
            ],
        },
    )
    return vllm_config, kv_cache_config


@patch(NIXL_WRAPPER_PATCH, FakeNixlWrapper)
def test_nixl_wins_load_over_cpu_offload():
    """When NixlConnector (index 0) has matched tokens from a remote prefill, it should
    win the load: Nixl metadata tracks the recv while CPU offload metadata has no load
     scheduled."""
    vllm_config, kv_cache_config = _multi_connector_config()
    scheduler = create_scheduler(vllm_config, kv_cache_config=kv_cache_config)
    mc = scheduler.connector
    assert isinstance(mc, MultiConnector)

    request = create_request(
        request_id=1,
        num_tokens=BLOCK_SIZE * 3,
        do_remote_prefill=True,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)
    sched_out = scheduler.schedule()

    assert mc._requests_to_connector[request.request_id] == 0

    meta = sched_out.kv_connector_metadata
    assert isinstance(meta, MultiKVConnectorMetadata)
    assert len(meta.metadata) == 2

    nixl_meta = meta.metadata[0]
    assert isinstance(nixl_meta, NixlConnectorMetadata)
    # nixl is tracking the request
    assert request.request_id in nixl_meta.reqs_to_recv

    cpu_meta = meta.metadata[1]
    assert isinstance(cpu_meta, SimpleCPUOffloadMetadata)
    assert not cpu_meta.load_gpu_blocks


@patch(NIXL_WRAPPER_PATCH, FakeNixlWrapper)
def test_cpu_offload_wins_when_nixl_has_no_match():
    """When NixlConnector returns 0 matched tokens and SimpleCPUOffloadConnector has a
    CPU cache hit, the CPU offload connector (index 1) wins the load."""
    vllm_config, kv_cache_config = _multi_connector_config()
    scheduler = create_scheduler(vllm_config, kv_cache_config=kv_cache_config)
    mc = scheduler.connector
    assert isinstance(mc, MultiConnector)

    req1 = create_request(
        request_id=10,
        num_tokens=BLOCK_SIZE * 3,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(req1)
    sched_out1 = scheduler.schedule()

    # build_connector_meta runs before _update_after_schedule, so num_computed_tokens
    # is still 0 during the first schedule and the store sees no confirmed blocks.
    # Simulate one model step so the second schedule triggers the store.
    model_output = create_model_runner_output(reqs=[req1])
    scheduler.update_from_output(sched_out1, model_output)

    sched_out2 = scheduler.schedule()
    meta2 = sched_out2.kv_connector_metadata
    assert isinstance(meta2, MultiKVConnectorMetadata)
    cpu_meta = meta2.metadata[1]
    assert isinstance(cpu_meta, SimpleCPUOffloadMetadata)
    assert cpu_meta.store_event >= 0, "Expected a store event on the second schedule"

    cpu_connector = mc._connectors[1]
    worker_meta = SimpleCPUOffloadWorkerMetadata(
        completed_store_events={
            cpu_meta.store_event: cpu_connector.scheduler_manager._expected_worker_count
        },
    )
    output = KVConnectorOutput(
        finished_recving=set(),
        kv_connector_worker_meta=worker_meta,
    )
    cpu_connector.update_connector_output(output)

    req2 = Request(
        request_id="id-cpu-offload-hit",
        prompt_token_ids=req1.prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    hit_tokens, is_async = mc.get_num_new_matched_tokens(req2, num_computed_tokens=0)
    assert hit_tokens is not None and hit_tokens > 0
    assert mc._requests_to_connector[req2.request_id] == 1
    assert is_async is True


@pytest.mark.parametrize("swa_enabled", [False, True], ids=["fa_only", "fa_sw"])
@patch(NIXL_WRAPPER_PATCH, FakeNixlWrapper)
def test_request_finished_no_async_save(swa_enabled: bool):
    """A normal request (no P/D) produces no async save from either connector.
    MultiConnector returns (False, None) via both request_finished and
    request_finished_all_groups, and cleans up _requests_to_connector."""
    from vllm.v1.request import RequestStatus

    vllm_config, kv_cache_config = _multi_connector_config(swa_enabled=swa_enabled)
    scheduler = create_scheduler(vllm_config, kv_cache_config=kv_cache_config)
    mc = scheduler.connector
    assert isinstance(mc, MultiConnector)

    request = create_request(
        request_id=40,
        num_tokens=BLOCK_SIZE * 2,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)
    scheduler.schedule()
    request.status = RequestStatus.FINISHED_STOPPED

    block_ids = (list(range(2)), list(range(2))) if swa_enabled else (list(range(2)),)
    async_save, txfer_params = mc.request_finished_all_groups(request, block_ids)

    assert async_save is False
    assert txfer_params is None
    assert len(mc._extra_async_saves) == 0
    assert request.request_id not in mc._requests_to_connector
