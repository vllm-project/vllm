# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import filecmp
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorTransferResults,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorLogging,
    KVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorPromMetrics,
    MultiKVConnectorStats,
    MultiKVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlKVConnectorStats,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import (
    KVConnectorOutput,
    KVConnectorWorkerMetadata,
    ModelRunnerOutput,
)
from vllm.v1.request import RequestStatus

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

PROMPT_CONTEXT = "Hi " * 100
PROMPTS = [
    PROMPT_CONTEXT + "Hello, my name is",
    PROMPT_CONTEXT + "The capital of France is",
]

SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=20)


# Test connector with custom stats for testing MultiConnector
class MockConnectorStats(KVConnectorStats):
    """Mock stats class for testing."""

    pass


class MockConnector(KVConnectorBase_V1):
    """Mock connector for testing."""

    _supports_divergent_local_hybrid_hits = False

    def __new__(cls, *args, **kwargs):
        # mock all KVConnectorBase_V1 functions
        mock = MagicMock(spec_set=KVConnectorBase_V1)
        mock.supports_divergent_local_hybrid_hits = (
            cls._supports_divergent_local_hybrid_hits
        )
        # Override just build_kv_connector_stats
        mock.build_kv_connector_stats = cls.build_kv_connector_stats
        mock.get_kv_connector_stats.return_value = None
        mock.get_mem_pool_context.return_value = None
        return mock

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return MockConnectorStats(data=data) if data is not None else None

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def build_connector_meta(self, scheduler_output):
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_tokens) -> None:
        pass


class MockHMAConnector(KVConnectorBase_V1, SupportsHMA):
    """Mock connector that supports HMA for testing."""

    _supports_divergent_local_hybrid_hits = False

    def __new__(cls, *args, **kwargs):
        mock = MagicMock(spec_set=cls)
        mock.supports_divergent_local_hybrid_hits = (
            cls._supports_divergent_local_hybrid_hits
        )
        mock.get_kv_connector_stats.return_value = None
        return mock

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def build_connector_meta(self, scheduler_output):
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_tokens) -> None:
        pass

    def request_finished_all_groups(self, request, block_ids):
        return (False, None)


class MockDivergentHMAConnector(MockHMAConnector):
    _supports_divergent_local_hybrid_hits = True


# Register mock connectors
KVConnectorFactory.register_connector("MockConnector", __name__, MockConnector.__name__)
KVConnectorFactory.register_connector(
    "MockHMAConnector", __name__, MockHMAConnector.__name__
)
KVConnectorFactory.register_connector(
    "MockDivergentHMAConnector",
    __name__,
    MockDivergentHMAConnector.__name__,
)


def test_register_finished_partial_tail_notifies_every_connector(mc):
    connector = mc
    first, second = connector.sub_connectors
    first.register_finished_partial_tail.return_value = True
    second.register_finished_partial_tail.return_value = False
    request = MagicMock()
    block_ids = ([1], [2])
    offloads = [(1, 2, 12)]

    assert connector.register_finished_partial_tail(request, block_ids, offloads)
    first.register_finished_partial_tail.assert_called_once_with(
        request, block_ids, offloads
    )
    second.register_finished_partial_tail.assert_called_once_with(
        request, block_ids, offloads
    )


@pytest.fixture
def mc() -> MultiConnector:
    """MultiConnector using two mocked connectors."""
    mock_connector_config = {
        "kv_connector": "MockConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "tests.v1.kv_connector.unit.test_multi_connector",
    }

    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [mock_connector_config, mock_connector_config],
        },
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )

    mc = MultiConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.WORKER,
        kv_cache_config=kv_cache_config,
    )

    return mc


@pytest.mark.parametrize("chosen", [0, 1])
def test_loaded_groups_follow_the_connector_serving_the_request(mc, chosen):
    request = MagicMock(request_id="request")
    for i, connector in enumerate(mc._connectors):
        connector.get_num_new_matched_tokens.return_value = (
            (16, True) if i == chosen else (0, False)
        )
        connector.get_loaded_kv_cache_group_ids.return_value = (i,)

    assert mc.get_num_new_matched_tokens(request, 0) == (16, True)
    assert mc.get_loaded_kv_cache_group_ids(request) == (chosen,)
    mc._connectors[1 - chosen].get_loaded_kv_cache_group_ids.assert_not_called()


@pytest.fixture
def async_saves(mc):
    """Two save owners whose two worker ranks may finish in different steps."""
    for child in mc.sub_connectors:
        child.get_finished_count.return_value = None
        child.request_finished.return_value = True, None
        child.build_connector_worker_meta.return_value = None
    mc._world_size = 2
    return mc


def _send_step(
    connector, notifications=(), *, receiving=None, failed_receiving=(), scheduler=None
):
    outputs = []
    for rank in range(2):
        metadata = MultiKVConnectorWorkerMetadata(
            metadata=(None, None),
            finished_sending=tuple(
                {"r": {rank}} if (child, rank) in notifications else {}
                for child in range(2)
            ),
        )
        outputs.append(
            ModelRunnerOutput.with_kv_conn_output_only(
                KVConnectorOutput(kv_connector_worker_meta=metadata)
            )
        )
    output = KVOutputAggregator(2).aggregate(outputs).kv_connector_output
    assert output is not None
    output.finished_recving = receiving
    output.failed_recving = set(failed_receiving)
    if scheduler is None:
        connector.update_connector_output(output)
    else:
        scheduler._update_from_kv_xfer_finished(output)
    return output


def _scheduler_with_request(connector, status):
    request = SimpleNamespace(request_id="r", status=status)
    request.is_finished = lambda: RequestStatus.is_finished(request.status)
    scheduler = object.__new__(Scheduler)
    scheduler.connector = connector
    scheduler.requests = {request.request_id: request}
    scheduler.finished_recving_kv_req_ids = set()
    scheduler._kv_fetch_stages = None
    scheduler._free_request_blocks = MagicMock()
    connector.on_new_request(request)
    return scheduler, request


def _start_async_receive(connector, request):
    first, second = connector.sub_connectors
    first.get_num_new_matched_tokens.return_value = 16, True
    second.get_num_new_matched_tokens.return_value = 0, False
    assert connector.get_num_new_matched_tokens(request, 0) == (16, True)
    connector.update_state_after_alloc(request, MagicMock(), 16)


def _finish_with_async_saves(connector):
    request = SimpleNamespace(request_id="r")
    connector.on_new_request(request)
    assert connector.request_finished(request, [1])[0]


def _assert_save_state_cleared(connector):
    assert not connector._partial_tail_owners
    assert not connector._async_load_candidates
    assert not connector._async_load_owners
    assert not connector._requests_to_connector
    assert not connector._async_save_owners
    assert not connector._completed_async_saves
    assert not any(connector._send_workers)


def test_send_completion_requires_each_child_and_distinct_rank(async_saves):
    """A different child or a duplicate rank cannot satisfy a missing ACK."""
    _finish_with_async_saves(async_saves)
    assert _send_step(async_saves, [(0, 0), (1, 1)]).finished_sending is None
    assert _send_step(async_saves, [(0, 0), (1, 0)]).finished_sending is None
    assert _send_step(async_saves, [(0, 1)]).finished_sending == {"r"}
    _assert_save_state_cleared(async_saves)
    assert (
        _send_step(async_saves, [(0, 0), (0, 1), (1, 0), (1, 1)]).finished_sending
        is None
    )
    _assert_save_state_cleared(async_saves)


def test_non_owner_cannot_filter_other_connector_send_completion(async_saves):
    first = async_saves.sub_connectors[0]
    first.request_finished.return_value = False, None
    seen = []

    def filter_completion(output):
        seen.append(output.finished_sending)
        output.finished_sending = None

    first.update_connector_output.side_effect = filter_completion
    _finish_with_async_saves(async_saves)
    assert _send_step(async_saves, [(1, 0), (1, 1)]).finished_sending == {"r"}
    assert seen == [None]
    _assert_save_state_cleared(async_saves)


@pytest.fixture(params=[False, True], ids=["direct-first", "nested-first"])
def nested_mc(request):
    child = {
        "kv_connector": "MockConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": __name__,
    }
    children = [
        child,
        {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"connectors": [child]},
        },
    ]
    if request.param:
        children.reverse()
    config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_connector_extra_config={"connectors": children},
    )
    connector = MultiConnector(
        config,
        KVConnectorRole.SCHEDULER,
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
    )
    nested_first = request.param
    direct = connector.sub_connectors[int(nested_first)]
    nested = connector.sub_connectors[int(not nested_first)]
    for leaf in [direct, *nested.sub_connectors]:
        leaf.get_num_new_matched_tokens.return_value = 0, False
        leaf.get_finished_count.return_value = 1
        leaf.request_finished.return_value = False, None
    return connector, direct, nested, int(nested_first)


def test_nested_nonowner_does_not_swallow_receive_completion(nested_mc):
    """A nested non-loader must not strand the selected child's request."""
    nested_mc, direct, nested, direct_index = nested_mc
    direct.get_num_new_matched_tokens.return_value = 16, True
    request = SimpleNamespace(request_id="r")
    nested_mc.on_new_request(request)
    assert nested_mc.get_num_new_matched_tokens(request, 0) == (16, True)
    nested_mc.update_state_after_alloc(request, MagicMock(), 16)

    leaf_meta = MagicMock(spec=KVConnectorWorkerMetadata)
    nested_meta = MultiKVConnectorWorkerMetadata(metadata=(leaf_meta,))
    child_metadata: list[KVConnectorWorkerMetadata | None] = [nested_meta, nested_meta]
    child_metadata[direct_index] = None
    metadata = MultiKVConnectorWorkerMetadata(metadata=tuple(child_metadata))
    output = KVConnectorOutput(
        finished_recving={"r"}, kv_connector_worker_meta=metadata
    )
    nested_mc.update_connector_output(output)

    assert output.finished_recving == {"r"}
    assert output.finished_sending is None
    assert output.kv_connector_worker_meta is metadata
    assert metadata.metadata[1 - direct_index] is nested_meta
    leaf_output = nested.sub_connectors[0].update_connector_output.call_args.args[0]
    assert leaf_output.kv_connector_worker_meta is leaf_meta
    assert not nested_mc.request_finished(request, [1])[0]


def test_child_output_updates_survive_receive_isolation(nested_mc):
    """Copying completions must not discard child error or auxiliary updates."""
    nested_mc, direct, nested, direct_index = nested_mc
    direct.get_num_new_matched_tokens.return_value = 16, True
    request = SimpleNamespace(request_id="r")
    nested_mc.on_new_request(request)
    assert nested_mc.get_num_new_matched_tokens(request, 0) == (16, True)
    nested_mc.update_state_after_alloc(request, MagicMock(), 16)
    stats, events = MagicMock(), MagicMock()

    def report_errors(output):
        output.finished_recving.clear()
        output.invalid_block_ids = output.invalid_block_ids | {2}
        output.failed_recving = output.failed_recving | {"r"}
        output.kv_connector_stats = stats
        output.kv_cache_events = events
        output.expected_finished_count = 3

    def report_more_errors(output):
        output.invalid_block_ids.add(3)
        output.failed_recving.add("other")

    direct.update_connector_output.side_effect = report_errors
    nested.sub_connectors[0].update_connector_output.side_effect = report_more_errors
    output = KVConnectorOutput(finished_recving={"r"}, invalid_block_ids={1})
    nested_mc.update_connector_output(output)

    assert output.finished_recving == {"r"}
    assert output.invalid_block_ids == {1, 2, 3}
    assert output.failed_recving == {"r", "other"}
    assert output.kv_connector_stats is stats
    assert output.kv_cache_events is events
    assert output.expected_finished_count == 3


@pytest.mark.parametrize("receive_first", [True, False])
def test_cancelled_nested_loader_waits_for_other_save_owner(nested_mc, receive_first):
    """Cancellation releases once, after both the nested load and sibling save."""
    nested_mc, direct, nested, direct_index = nested_mc
    direct.request_finished.return_value = True, None
    nested.sub_connectors[0].get_num_new_matched_tokens.return_value = 16, True
    scheduler, request = _scheduler_with_request(
        nested_mc, RequestStatus.FINISHED_ABORTED
    )
    assert nested_mc.get_num_new_matched_tokens(request, 0) == (16, True)
    nested_mc.update_state_after_alloc(request, MagicMock(), 16)
    assert nested_mc.request_finished(request, [1])[0]

    steps = [{"receiving": {"r"}}, {"notifications": [(direct_index, 0)]}]
    if not receive_first:
        steps.reverse()
    first = _send_step(nested_mc, scheduler=scheduler, **steps[0])
    assert first.finished_recving is None
    assert first.finished_sending is None
    scheduler._free_request_blocks.assert_not_called()

    last = _send_step(nested_mc, scheduler=scheduler, **steps[1])
    assert last.finished_recving is None
    assert last.finished_sending == {"r"}
    scheduler._free_request_blocks.assert_called_once_with(request)

    duplicate = _send_step(
        nested_mc, [(direct_index, 0)], receiving={"r"}, scheduler=scheduler
    )
    assert duplicate.finished_recving is None
    assert duplicate.finished_sending is None
    scheduler._free_request_blocks.assert_called_once_with(request)


def test_scheduler_child_completion_waits_for_other_save_owner(async_saves):
    """A child may finish in update_connector_output, without a worker ACK."""
    _finish_with_async_saves(async_saves)

    def finish_first(output):
        output.finished_sending = {"r"}

    async_saves.sub_connectors[0].update_connector_output.side_effect = finish_first
    assert _send_step(async_saves).finished_sending is None
    assert _send_step(async_saves, [(1, 0), (1, 1)]).finished_sending == {"r"}
    _assert_save_state_cleared(async_saves)


@pytest.mark.parametrize("early_children", [(0,), (0, 1)])
def test_send_completion_before_request_finished_is_reconciled(
    async_saves, early_children
):
    request = SimpleNamespace(request_id="r")
    async_saves.on_new_request(request)
    early = [(child, rank) for child in early_children for rank in range(2)]
    assert _send_step(async_saves, early).finished_sending is None
    pending, _ = async_saves.request_finished(request, [1])
    assert pending == (early_children == (0,))
    if pending:
        assert _send_step(async_saves, [(1, 0), (1, 1)]).finished_sending == {"r"}
    _assert_save_state_cleared(async_saves)


def test_send_completion_uses_child_finished_count(async_saves):
    async_saves.sub_connectors[0].get_finished_count.return_value = 1
    _finish_with_async_saves(async_saves)
    assert _send_step(async_saves, [(0, 0), (1, 0)]).finished_sending is None
    assert _send_step(async_saves, [(1, 1)]).finished_sending == {"r"}


def test_unexpected_send_does_not_create_tracking(async_saves):
    # Unregistered ACKs must not create persistent request state.
    assert _send_step(async_saves, [(0, 0), (1, 1)]).finished_sending is None
    _assert_save_state_cleared(async_saves)


@pytest.mark.parametrize("receiver_saves", [False, True])
@pytest.mark.parametrize("first_completion", ["receive", "send", "both"])
def test_aborted_receive_waits_for_another_send_owner(
    async_saves, receiver_saves, first_completion
):
    """Finishing a cancelled load must not free another child's save source."""
    scheduler, request = _scheduler_with_request(
        async_saves, RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    _start_async_receive(async_saves, request)
    async_saves.sub_connectors[0].request_finished.return_value = (
        receiver_saves,
        None,
    )
    request.status = RequestStatus.FINISHED_ABORTED
    assert async_saves.request_finished(request, [1])[0]

    send_acks = [(1, 0), (1, 1)]
    if receiver_saves:
        send_acks += [(0, 0), (0, 1)]
    completions: dict[str, dict[str, set[str] | list[tuple[int, int]]]] = {
        "receive": {"receiving": {"r"}},
        "send": {"notifications": send_acks},
        "both": {"receiving": {"r"}, "notifications": send_acks},
    }
    output = _send_step(
        async_saves, scheduler=scheduler, **completions[first_completion]
    )
    if first_completion != "both":
        scheduler._free_request_blocks.assert_not_called()
        assert output.finished_recving is None
        assert output.finished_sending is None
        assert "r" in scheduler.requests
        remaining = "send" if first_completion == "receive" else "receive"
        _send_step(async_saves, scheduler=scheduler, **completions[remaining])
    scheduler._free_request_blocks.assert_called_once_with(request)
    assert not scheduler.requests
    _assert_save_state_cleared(async_saves)
    # Duplicate terminal notifications cannot trigger a second block release.
    _send_step(async_saves, scheduler=scheduler, **completions["both"])
    scheduler._free_request_blocks.assert_called_once_with(request)


def test_aborted_receive_only_releases_once_and_allows_request_id_reuse(
    async_saves,
):
    async_saves.sub_connectors[0].request_finished.return_value = False, None
    async_saves.sub_connectors[1].request_finished.return_value = False, None
    for _ in range(2):
        scheduler, request = _scheduler_with_request(
            async_saves, RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        _start_async_receive(async_saves, request)
        request.status = RequestStatus.FINISHED_ABORTED
        assert async_saves.request_finished(request, [1])[0]
        _send_step(async_saves, receiving={"r"}, scheduler=scheduler)
        scheduler._free_request_blocks.assert_called_once_with(request)
        assert not scheduler.requests
        _assert_save_state_cleared(async_saves)


def test_receiving_child_save_is_independent_of_its_receive(async_saves):
    """A receiver may still be saving blocks from an earlier prefill."""
    scheduler, request = _scheduler_with_request(
        async_saves, RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    _start_async_receive(async_saves, request)
    request.status = RequestStatus.FINISHED_ABORTED
    assert async_saves.request_finished(request, [1])[0]
    _send_step(async_saves, [(1, 0), (1, 1)], receiving={"r"}, scheduler=scheduler)
    scheduler._free_request_blocks.assert_not_called()
    _send_step(async_saves, [(0, 0), (0, 1)], scheduler=scheduler)
    scheduler._free_request_blocks.assert_called_once_with(request)
    _assert_save_state_cleared(async_saves)


@pytest.mark.parametrize("failed", [False, True])
def test_normal_receive_keeps_tracking_for_later_saves(async_saves, failed):
    scheduler, request = _scheduler_with_request(
        async_saves, RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    _start_async_receive(async_saves, request)
    output = _send_step(
        async_saves,
        receiving={"r"},
        failed_receiving={"r"} if failed else (),
        scheduler=scheduler,
    )
    assert output.finished_recving == {"r"}
    assert output.failed_recving == ({"r"} if failed else set())
    assert scheduler.finished_recving_kv_req_ids == {"r"}
    scheduler._free_request_blocks.assert_not_called()
    request.status = RequestStatus.FINISHED_STOPPED
    assert async_saves.request_finished(request, [1])[0]
    _send_step(async_saves, [(1, 0), (1, 1)], scheduler=scheduler)
    scheduler._free_request_blocks.assert_not_called()
    _send_step(async_saves, [(0, 0), (0, 1)], scheduler=scheduler)
    scheduler._free_request_blocks.assert_called_once_with(request)
    _assert_save_state_cleared(async_saves)


@pytest.mark.parametrize("allocate_zero_tokens", [False, True])
def test_async_lookup_without_external_allocation_does_not_create_a_receive_wait(
    async_saves, allocate_zero_tokens
):
    scheduler, request = _scheduler_with_request(
        async_saves, RequestStatus.FINISHED_ABORTED
    )
    first, second = async_saves.sub_connectors
    first.get_num_new_matched_tokens.return_value = 16, True
    second.get_num_new_matched_tokens.return_value = 0, False
    assert async_saves.get_num_new_matched_tokens(request, 0) == (16, True)
    if allocate_zero_tokens:
        async_saves.update_state_after_alloc(request, MagicMock(), 0)
    assert async_saves.request_finished(request, [1])[0]
    _send_step(async_saves, [(0, 0), (0, 1), (1, 0), (1, 1)], scheduler=scheduler)
    scheduler._free_request_blocks.assert_called_once_with(request)
    _assert_save_state_cleared(async_saves)


def test_partial_tail_owner_releases_after_its_send_ack(async_saves):
    """Scheduler must hear completion even if only the tail hook delays free."""
    scheduler, request = _scheduler_with_request(
        async_saves, RequestStatus.FINISHED_STOPPED
    )
    first, second = async_saves.sub_connectors
    first.register_finished_partial_tail.return_value = True
    second.register_finished_partial_tail.return_value = False
    for child in async_saves.sub_connectors:
        child.request_finished.return_value = False, None
    partial_tail_delay = async_saves.register_finished_partial_tail(
        request, ([1],), [(0, 1, 12)]
    )
    delay_free, _ = async_saves.request_finished(request, [1])
    assert partial_tail_delay or delay_free
    _send_step(async_saves, [(0, 0)], scheduler=scheduler)
    scheduler._free_request_blocks.assert_not_called()
    _send_step(async_saves, [(0, 1)], scheduler=scheduler)
    scheduler._free_request_blocks.assert_called_once_with(request)
    assert not scheduler.requests
    _assert_save_state_cleared(async_saves)


@pytest.mark.parametrize("early_ranks", [(0,), (0, 1)])
def test_partial_tail_requires_new_acks_after_earlier_saves(async_saves, early_ranks):
    """A newly accepted tail must not reuse earlier full-block send ACKs."""
    scheduler, request = _scheduler_with_request(
        async_saves, RequestStatus.FINISHED_STOPPED
    )
    first, second = async_saves.sub_connectors
    first.register_finished_partial_tail.return_value = True
    second.register_finished_partial_tail.return_value = False
    first.request_finished.return_value = False, None
    _send_step(async_saves, [(0, rank) for rank in early_ranks], scheduler=scheduler)
    assert async_saves.register_finished_partial_tail(request, ([1],), [(0, 1, 12)])
    assert async_saves.request_finished(request, [1])[0]
    _send_step(async_saves, [(0, 1), (1, 0), (1, 1)], scheduler=scheduler)
    scheduler._free_request_blocks.assert_not_called()
    _send_step(async_saves, [(0, 0)], scheduler=scheduler)
    scheduler._free_request_blocks.assert_called_once_with(request)
    _assert_save_state_cleared(async_saves)


@pytest.mark.parametrize("receive_first", [False, True])
def test_partial_tail_and_receive_from_same_child_both_delay_free(
    async_saves, receive_first
):
    scheduler, request = _scheduler_with_request(
        async_saves, RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    _start_async_receive(async_saves, request)
    first, second = async_saves.sub_connectors
    first.register_finished_partial_tail.return_value = True
    second.register_finished_partial_tail.return_value = False
    for child in async_saves.sub_connectors:
        child.request_finished.return_value = False, None
    request.status = RequestStatus.FINISHED_ABORTED
    assert async_saves.register_finished_partial_tail(request, ([1],), [(0, 1, 12)])
    assert async_saves.request_finished(request, [1])[0]
    receive = {"receiving": {"r"}}
    send = {"notifications": [(0, 0), (0, 1)]}
    _send_step(async_saves, scheduler=scheduler, **(receive if receive_first else send))
    scheduler._free_request_blocks.assert_not_called()
    _send_step(async_saves, scheduler=scheduler, **(send if receive_first else receive))
    scheduler._free_request_blocks.assert_called_once_with(request)
    assert not scheduler.requests
    _assert_save_state_cleared(async_saves)


def test_no_async_saves_release_tracking(async_saves):
    request = SimpleNamespace(request_id="r")
    async_saves.on_new_request(request)
    for child in async_saves.sub_connectors:
        child.request_finished.return_value = False, None
    assert not async_saves.request_finished(request, [1])[0]
    _assert_save_state_cleared(async_saves)


def test_worker_preserves_send_identity_and_receive_failures(async_saves, monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1 import multi_connector

    monkeypatch.setattr(
        multi_connector, "get_world_group", lambda: SimpleNamespace(rank=1)
    )
    async_saves.sub_connectors[
        0
    ].get_transfer_results.return_value = KVConnectorTransferResults(
        finished_sending={"r"}
    )
    async_saves.sub_connectors[
        1
    ].get_transfer_results.return_value = KVConnectorTransferResults(
        finished_recving={"recv"}, failed_recving={"recv"}
    )
    result = async_saves.get_transfer_results(set())
    assert not result.finished_sending
    assert result.finished_recving == result.failed_recving == {"recv"}
    metadata = async_saves.build_connector_worker_meta()
    assert isinstance(metadata, MultiKVConnectorWorkerMetadata)
    assert metadata.finished_sending == ({"r": {1}}, {})
    assert async_saves.build_connector_worker_meta() is None


def test_worker_send_metadata_merge_preserves_inputs_and_deduplicates_ranks():
    first = MultiKVConnectorWorkerMetadata(
        metadata=(None, None), finished_sending=({"r": {0}}, {"r": {1}})
    )
    second = MultiKVConnectorWorkerMetadata(
        metadata=(None, None), finished_sending=({"r": {0, 1}}, {})
    )
    result = first.aggregate(second)
    assert result.finished_sending == ({"r": {0, 1}}, {"r": {1}})
    assert first.finished_sending == ({"r": {0}}, {"r": {1}})
    assert second.finished_sending == ({"r": {0, 1}}, {})


def test_multi_connector_mem_pool_context_none(mc: MultiConnector):
    assert mc.get_mem_pool_context() is None


def test_multi_connector_forwards_mem_pool_context(mc: MultiConnector):
    context = MagicMock()
    provider = MagicMock(spec_set=KVConnectorBase_V1)
    provider.get_mem_pool_context.return_value = context
    mc._connectors = [mc._connectors[0], provider]

    assert mc.get_mem_pool_context() is context
    provider.get_mem_pool_context.assert_called_once_with()


def test_multi_connector_rejects_multiple_mem_pool_contexts(mc: MultiConnector):
    providers = [
        MagicMock(spec_set=KVConnectorBase_V1),
        MagicMock(spec_set=KVConnectorBase_V1),
    ]
    for provider in providers:
        provider.get_mem_pool_context.return_value = MagicMock()
    mc._connectors = providers

    with pytest.raises(
        ValueError,
        match="Multiple connectors provide a KV cache memory pool",
    ):
        mc.get_mem_pool_context()


def test_cache_hit_sources_delegate_to_selected_connector(mc: MultiConnector):
    request = MagicMock(request_id="request")
    mc._requests_to_connector[request.request_id] = 1
    sources = {CacheHitSource.DISK: 32}
    mc._connectors[1].get_external_cache_hit_sources.return_value = sources

    assert mc.get_external_cache_hit_sources(request, 32) is sources
    mc._connectors[1].get_external_cache_hit_sources.assert_called_once_with(
        request, 32
    )


# Helper function to compare directories recursively
def _compare_directories(dir1: Path, dir2: Path) -> bool:
    """Compares two directories recursively for identical content."""
    dcmp = filecmp.dircmp(dir1, dir2)
    if dcmp.left_only or dcmp.right_only or dcmp.diff_files:
        print(f"Differences found between {dir1} and {dir2}:")
        print(f"  Left only: {dcmp.left_only}")
        print(f"  Right only: {dcmp.right_only}")
        print(f"  Different files: {dcmp.diff_files}")
        return False
    for sub_dir in dcmp.common_dirs:
        if not _compare_directories(dir1 / sub_dir, dir2 / sub_dir):
            return False
    return True


def test_multi_example_connector_consistency():
    """Tests that MultiConnector with two ExampleConnectors saves
    identical KV cache data to separate storage locations.
    """
    storage_1_path = Path("storage_1/")
    storage_2_path = Path("storage_2/")
    shutil.rmtree(storage_1_path, ignore_errors=True)
    shutil.rmtree(storage_2_path, ignore_errors=True)
    storage_1_path.mkdir()
    storage_2_path.mkdir()

    # Configure MultiConnector with two ExampleConnectors
    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_1_path),
                        "name": "storage1",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_2_path),
                        "name": "storage2",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
            ]
        },
    )

    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        block_size=16,
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
        async_scheduling=False,
    )
    # Run generation - this should trigger saving KV cache
    # Use a single prompt to avoid race conditions depending on the order of scheduling
    _ = llm.generate(PROMPTS[0], SAMPLING_PARAMS)

    # --- Verification ---

    # Check that both storage directories were populated
    local_subdirs = list(storage_1_path.iterdir())
    external_subdirs = list(storage_2_path.iterdir())

    assert len(local_subdirs) > 0, (
        f"Local storage path {storage_1_path} is empty after generation."
    )
    assert len(external_subdirs) > 0, (
        f"External storage path {storage_2_path} is empty after generation."
    )
    assert len(local_subdirs) == len(external_subdirs), (
        f"Mismatch in number of cache entries: "
        f"Local={len(local_subdirs)}, External={len(external_subdirs)}"
    )

    # The subdirectories should correspond to the prompt hashes
    # Since prompts are the same, the hash directories should be the same name
    local_subdir_names = sorted([d.name for d in local_subdirs])
    external_subdir_names = sorted([d.name for d in external_subdirs])
    assert local_subdir_names == external_subdir_names, (
        "Cache directory names do not match between local and external storage"
    )

    # Compare the contents of each corresponding cache directory
    for subdir_name in local_subdir_names:
        print(f"Comparing contents of cache directory: {subdir_name}")
        assert _compare_directories(
            storage_1_path / subdir_name, storage_2_path / subdir_name
        ), (
            f"Contents differ for cache directory '{subdir_name}' between "
            f"{storage_1_path} and {storage_2_path}"
        )

    events = get_connector_events()
    storage1_scheduler_events = _ignore_event_collection(events["storage1-SCHEDULER"])
    storage2_scheduler_events = _ignore_event_collection(events["storage2-SCHEDULER"])
    # Initial events bind the cache manager, query completion counts, and exchange
    # handshake metadata before the request is enqueued.
    assert storage1_scheduler_events[:7] == [
        "bind_kv_cache_manager",
        "get_finished_count",
        "set_xfer_handshake_metadata_pp_aware",
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 0",
        "build_connector_meta",
    ]
    # First three events are from initialization. Layer hooks run before the
    # deferred load starts after the forward pass.
    expected_worker_prefix = [
        "get_mem_pool_context",
        "register_kv_caches",
        "set_host_xfer_buffer_ops",
        "get_handshake_metadata",
        "handle_preemptions",
        "bind_connector_metadata",
        "wait_for_layer_load",
        "save_kv_layer",
    ]
    for connector_name in ("storage1-WORKER", "storage2-WORKER"):
        worker_events = events[connector_name]
        assert worker_events[: len(expected_worker_prefix)] == expected_worker_prefix
        assert worker_events.index("start_load_kv") > worker_events.index(
            "save_kv_layer"
        )
    assert storage2_scheduler_events[:7] == [
        "bind_kv_cache_manager",
        "get_finished_count",
        "set_xfer_handshake_metadata_pp_aware",
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 0",
        "build_connector_meta",
    ]
    # Reset prefix cache or else we'll just get the tokens back from there.
    llm.reset_prefix_cache()

    # Run generation again - this should trigger loading from the first
    # connector.
    _ = llm.generate(PROMPTS[1], SAMPLING_PARAMS)

    events = get_connector_events()
    # get_num_new_matched_tokens will return new tokens from the first
    # connector (first nonzero match is chosen), so update_state_after_alloc
    # will report those external tokens on that one. Other connectors still
    # receive the request's real blocks but with 0 external tokens.
    storage1_scheduler_events = _events_from_request(
        _ignore_event_collection(events["storage1-SCHEDULER"])
    )
    storage2_scheduler_events = _events_from_request(
        _ignore_event_collection(events["storage2-SCHEDULER"])
    )
    assert storage1_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 96",
        "build_connector_meta",
    ]
    assert storage2_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 0",
        "build_connector_meta",
    ]

    # Delete storage1 connector state
    shutil.rmtree(storage_1_path)

    # Reset prefix cache or else we'll just get the tokens back from there.
    llm.reset_prefix_cache()

    # Run generation again - this should trigger loading from the first
    # connector.
    _ = llm.generate(PROMPTS[0], SAMPLING_PARAMS)

    events = get_connector_events()
    # get_num_new_matched_tokens will be called for both connectors but will
    # return 0 from the first connector, while the second connector has a hit.
    # Both connectors receive the request's real blocks, but only the chosen
    # (second) connector reports external tokens.
    storage1_scheduler_events = _events_from_request(
        _ignore_event_collection(events["storage1-SCHEDULER"])
    )
    storage2_scheduler_events = _events_from_request(
        _ignore_event_collection(events["storage2-SCHEDULER"])
    )
    assert storage1_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 0",
        "build_connector_meta",
    ]
    assert storage2_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 96",
        "build_connector_meta",
    ]

    # Clean up
    shutil.rmtree(storage_1_path)
    shutil.rmtree(storage_2_path)


def _ignore_event_collection(events: list[str]) -> list[str]:
    # Filter out per-step polling hooks that the scheduler calls repeatedly
    # and which are not meaningful state transitions for these assertions.
    ignored = {"get_kv_connector_stats", "has_pending_push_work", "take_events"}
    return [event for event in events if event not in ignored]


def _events_from_request(events: list[str]) -> list[str]:
    # The async engine-core can emit a trailing build_connector_meta from the
    # previous generation's final (idle) scheduler step. Depending on timing it
    # may be flushed into this window ahead of on_new_request, so anchor the
    # comparison on the new request's first event to avoid a flaky ordering.
    if "on_new_request" in events:
        return events[events.index("on_new_request") :]
    return events


def get_connector_events() -> dict[str, list[str]]:
    # Read in connector events and reset the files.
    import glob

    event_files = glob.glob(tempfile.gettempdir() + "/connector_*_events.log")
    connector_events = {}
    for fname in event_files:
        name = fname.split("connector_")[1].split("_events.log")[0]
        try:
            with open(fname, "r+") as f:
                connector_events[name] = [line.strip() for line in f if line.strip()]
                f.truncate(0)
        except Exception as e:
            print(f"[ERROR] Could not read connector events for {name}: {e}")

    return connector_events


def test_engine_id_conflict():
    configs = [KVTransferConfig() for _ in range(2)]
    ids = [config.engine_id for config in configs]
    assert ids[0] != ids[1], (
        f"Engine IDs should be different for different configs. Got {ids}"
    )


def test_multi_connector_handle_preemptions_integration():
    """Integration test: verify MultiConnector delegates handle_preemptions
    to all sub-connectors.

    Uses TestExampleConnector which logs all method calls to temp files.
    This test directly calls handle_preemptions on a MultiConnector with
    TestExampleConnector sub-connectors and verifies the calls are logged.
    """
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )

    storage_path = Path(tempfile.mkdtemp())

    try:
        # Configure MultiConnector with two TestExampleConnectors
        connectors_extra_config = {
            "connectors": [
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_path / "s1"),
                        "name": "preempt1",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_path / "s2"),
                        "name": "preempt2",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
            ]
        }

        vllm_config = create_vllm_config(
            block_size=16,
            max_num_batched_tokens=100,
            kv_connector="MultiConnector",
            kv_connector_extra_config=connectors_extra_config,
        )

        # Create scheduler - this initializes the MultiConnector with SCHEDULER role
        scheduler = create_scheduler(vllm_config, num_blocks=10)

        # Clear any events from initialization
        get_connector_events()

        # Directly call handle_preemptions on the scheduler's connector
        # Note: handle_preemptions is normally a worker-side method, but we're
        # testing the delegation behavior of MultiConnector here.
        # The connector attribute contains the KV connector.
        assert scheduler.connector is not None, "Scheduler should have a connector"
        connector_md = scheduler.connector.build_connector_meta(scheduler.schedule())
        scheduler.connector.handle_preemptions(connector_md)

        # Verify both connectors received the handle_preemptions call
        events = get_connector_events()

        # Both SCHEDULER-role connectors should have logged handle_preemptions
        assert "handle_preemptions" in events.get("preempt1-SCHEDULER", []), (
            f"preempt1-SCHEDULER should have handle_preemptions call. "
            f"Got events: {events}"
        )
        assert "handle_preemptions" in events.get("preempt2-SCHEDULER", []), (
            f"preempt2-SCHEDULER should have handle_preemptions call. "
            f"Got events: {events}"
        )

    finally:
        # Cleanup
        shutil.rmtree(storage_path, ignore_errors=True)


class TestMultiConnectorStats:
    """Tests for MultiConnector stats reconstruction and operations."""

    def test_build_kv_connector_stats_with_none(self):
        """Test that build_kv_connector_stats returns empty stats when given None."""
        stats = MultiConnector.build_kv_connector_stats(data=None)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_with_empty_dict(self):
        """Test that build_kv_connector_stats returns empty stats with empty dict."""
        stats = MultiConnector.build_kv_connector_stats(data={})

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_reconstructs_nixl_stats(self):
        """Test that NixlConnector stats are properly reconstructed with
        correct data."""
        serialized_data = {
            "NixlConnector": {
                "transfer_duration": [1.5, 2.3],
                "post_duration": [0.1, 0.2],
                "bytes_transferred": [1024, 2048],
                "num_descriptors": [10, 20],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
                "num_failed_handshakes": [],
                "num_kv_expired_reqs": [],
                "num_notifications_after_expiry": [],
            }
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert "NixlConnector" in stats.data
        nixl_stats = stats.data["NixlConnector"]
        assert isinstance(nixl_stats, NixlKVConnectorStats)
        assert nixl_stats.data["transfer_duration"] == [1.5, 2.3]
        assert nixl_stats.data["post_duration"] == [0.1, 0.2]
        assert nixl_stats.data["bytes_transferred"] == [1024, 2048]
        assert nixl_stats.data["num_descriptors"] == [10, 20]

    def test_build_kv_connector_stats_with_multiple_connectors(self):
        """Test reconstruction with multiple connector types that have custom stats."""
        serialized_data = {
            "NixlConnector": {
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
                "num_failed_handshakes": [],
                "num_kv_expired_reqs": [],
                "num_notifications_after_expiry": [],
            },
            "MockConnector": {"mock_field": [1, 2, 3]},
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        # Both connectors should be reconstructed
        assert len(stats.data) == 2
        assert "NixlConnector" in stats.data
        assert "MockConnector" in stats.data
        assert isinstance(stats.data["NixlConnector"], NixlKVConnectorStats)
        assert isinstance(stats.data["MockConnector"], MockConnectorStats)
        # Verify data is preserved
        assert stats.data["MockConnector"].data == {"mock_field": [1, 2, 3]}
        assert stats.to_dict() == serialized_data

    def test_build_kv_connector_stats_raises_error_for_unknown_connector(self):
        """Test that unknown connectors raise an error."""
        serialized_data = {
            "UnknownConnector": {"some_field": [1, 2, 3]},
            "NixlConnector": {
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
                "num_failed_handshakes": [],
                "num_kv_expired_reqs": [],
                "num_notifications_after_expiry": [],
            },
        }

        with pytest.raises(
            ValueError, match="Connector 'UnknownConnector' is not registered."
        ):
            MultiConnector.build_kv_connector_stats(data=serialized_data)

    def test_build_kv_connector_stats_with_already_instantiated_objects(self):
        """Test that already-instantiated stats objects are preserved (same process)."""
        # This simulates the in-process case where stats are not serialized
        nixl_stats = NixlKVConnectorStats(
            data={
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
                "num_failed_handshakes": [],
                "num_kv_expired_reqs": [],
                "num_notifications_after_expiry": [],
            }
        )
        mock_stats = MockConnectorStats(data={"mock_field": [1, 2, 3]})

        data_with_objects = {
            "NixlConnector": nixl_stats,
            "MockConnector": mock_stats,
        }

        stats = MultiConnector.build_kv_connector_stats(data=data_with_objects)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 2
        # Verify objects are preserved as-is
        assert stats.data["NixlConnector"] is nixl_stats
        assert stats.data["MockConnector"] is mock_stats

    def test_build_kv_connector_stats_with_mixed_objects_and_dicts(self):
        """Test handling mixed already-instantiated and serialized stats."""
        # This can happen during transition or partial serialization
        nixl_stats = NixlKVConnectorStats(
            data={
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
                "num_failed_handshakes": [],
                "num_kv_expired_reqs": [],
                "num_notifications_after_expiry": [],
            }
        )

        mixed_data = {
            "NixlConnector": nixl_stats,  # Already instantiated
            "MockConnector": {"mock_field": [1, 2, 3]},  # Serialized
        }

        stats = MultiConnector.build_kv_connector_stats(data=mixed_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 2
        # Instantiated object preserved
        assert stats.data["NixlConnector"] is nixl_stats
        # Serialized object reconstructed
        assert isinstance(stats.data["MockConnector"], MockConnectorStats)
        assert stats.data["MockConnector"].data == {"mock_field": [1, 2, 3]}

    def test_build_kv_connector_stats_skips_connectors_without_custom_stats(self):
        """Test that connectors without custom stats (return None) are skipped."""
        # ExampleConnector doesn't override build_kv_connector_stats,
        # so it returns None and should be skipped
        serialized_data = {
            "NixlConnector": {
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
                "num_failed_handshakes": [],
                "num_kv_expired_reqs": [],
                "num_notifications_after_expiry": [],
            },
            "ExampleConnector": {"some_field": [1, 2, 3]},
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        # Only NixlConnector should be reconstructed
        assert len(stats.data) == 1
        assert "NixlConnector" in stats.data
        assert isinstance(stats.data["NixlConnector"], NixlKVConnectorStats)
        # ExampleConnector should be skipped (returns None)
        assert "ExampleConnector" not in stats.data

    def test_prom_metrics_passes_flat_data_to_children(self):
        child_metrics = MagicMock()
        metrics = object.__new__(MultiKVConnectorPromMetrics)
        metrics._prom_metrics = {"NixlConnector": child_metrics}
        payload = {"NixlConnector": {"transfer_duration": [1.5]}}

        metrics.observe(payload, engine_idx=2)

        child_metrics.observe.assert_called_once_with({"transfer_duration": [1.5]}, 2)

    def test_aggregate_same_connector(self):
        """Test aggregating stats from the same connector type."""
        stats1 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0],
                        "post_duration": [0.1],
                        "bytes_transferred": [1024],
                        "num_descriptors": [10],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                        "num_failed_handshakes": [],
                        "num_kv_expired_reqs": [],
                        "num_notifications_after_expiry": [],
                    }
                )
            }
        )

        stats2 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [2.0],
                        "post_duration": [0.2],
                        "bytes_transferred": [2048],
                        "num_descriptors": [20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                        "num_failed_handshakes": [],
                        "num_kv_expired_reqs": [],
                        "num_notifications_after_expiry": [],
                    }
                )
            }
        )

        result = stats1.aggregate(stats2)

        assert result is stats1  # Should return self
        assert "NixlConnector" in result.data
        nixl_stats = result.data["NixlConnector"]
        assert nixl_stats.data["transfer_duration"] == [1.0, 2.0]
        assert nixl_stats.data["post_duration"] == [0.1, 0.2]
        assert nixl_stats.data["bytes_transferred"] == [1024, 2048]
        assert nixl_stats.data["num_descriptors"] == [10, 20]

    def test_aggregate_new_connector(self):
        """Test aggregating stats when a new connector type appears."""
        from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
            KVConnectorStats,
        )

        stats1 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0],
                        "post_duration": [0.1],
                        "bytes_transferred": [1024],
                        "num_descriptors": [10],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                        "num_failed_handshakes": [],
                        "num_kv_expired_reqs": [],
                        "num_notifications_after_expiry": [],
                    }
                )
            }
        )

        stats2 = MultiKVConnectorStats(
            data={"ExampleConnector": KVConnectorStats(data={"field": [1, 2]})}
        )

        result = stats1.aggregate(stats2)

        assert "NixlConnector" in result.data
        assert "ExampleConnector" in result.data

    def test_reduce(self):
        """Test that reduce() correctly reduces all nested connector stats."""
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0, 2.0],
                        "post_duration": [0.1, 0.2],
                        "bytes_transferred": [1024, 2048],
                        "num_descriptors": [10, 20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                        "num_failed_handshakes": [],
                        "num_kv_expired_reqs": [],
                        "num_notifications_after_expiry": [],
                    }
                )
            }
        )

        reduced = stats.reduce()

        assert "NixlConnector" in reduced
        assert isinstance(reduced["NixlConnector"], dict)
        # Check that the stats were reduced (should have aggregated values)
        assert "Num successful transfers" in reduced["NixlConnector"]
        assert reduced["NixlConnector"]["Num successful transfers"] == 2

    def test_log_renders_plain_scalars(self):
        """Reduced stats must render as plain numbers in the CLI log, not
        numpy reprs (eg np.float64(2.338))."""
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0, 2.0],
                        "post_duration": [0.1, 0.2],
                        "bytes_transferred": [1024, 2048],
                        "num_descriptors": [10, 20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                        "num_failed_handshakes": [],
                        "num_kv_expired_reqs": [],
                        "num_notifications_after_expiry": [],
                    }
                )
            }
        )

        kv_logging = KVConnectorLogging(kv_transfer_config=None)
        kv_logging.transfer_stats_accumulator = stats
        log_records: list[tuple] = []
        kv_logging.log(log_fn=lambda *args: log_records.append(args))

        assert len(log_records) == 1
        msg = log_records[0][1]
        assert "np.float" not in msg
        assert "'Avg xfer time (ms)': 1500.0" in msg
        # The accumulator is reset after logging.
        assert kv_logging.transfer_stats_accumulator is None

    def test_reset(self):
        """Test that reset() resets all nested connector stats."""
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0, 2.0],
                        "post_duration": [0.1, 0.2],
                        "bytes_transferred": [1024, 2048],
                        "num_descriptors": [10, 20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                        "num_failed_handshakes": [],
                        "num_kv_expired_reqs": [],
                        "num_notifications_after_expiry": [],
                    }
                )
            }
        )

        assert not stats.is_empty()

        stats.reset()

        # After reset, stats should be empty
        assert stats.is_empty()
        nixl_stats = stats.data["NixlConnector"]
        assert len(nixl_stats.data["transfer_duration"]) == 0

    def test_is_empty_with_multiple_connectors(self):
        """Test is_empty() returns correct value with multiple connectors."""
        # All empty
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(data={}),
            }
        )
        # Initialize empty stats
        stats.data["NixlConnector"].reset()
        assert stats.is_empty()

        # One non-empty
        stats.data["NixlConnector"].data["transfer_duration"].append(1.0)
        assert not stats.is_empty()


def test_multi_connector_overrides_all_base_methods():
    """Ensure MultiConnector overrides all public methods from KVConnectorBase_V1."""
    # These are fine to inherit from KVConnectorBase_V1
    # TODO(https://github.com/vllm-project/vllm/pull/31811): Remove
    # get_kv_connector_kv_cache_events from INHERITED_OK once implemented.
    INHERITED_OK = {
        "role",
        "has_connector_metadata",
        "get_kv_connector_kv_cache_events",
    }

    base_members = {
        name for name in dir(KVConnectorBase_V1) if not name.startswith("_")
    } - KVConnectorBase_V1.__abstractmethods__

    missing = [
        name
        for name in sorted(base_members)
        if name not in INHERITED_OK and name not in MultiConnector.__dict__
    ]

    if missing:
        pytest.fail(f"""
MultiConnector does not override these KVConnectorBase_V1 methods: {missing}

MultiConnector wraps other connectors and must delegate all methods.
Please add overrides that delegate to self._connectors.

Options:
  1. Add delegation in MultiConnector (preferred)
  2. Add to INHERITED_OK if the base implementation works correctly
""")


def test_multi_connector_worker_metadata(mc):
    class MockConnectorWorkerMetadata(KVConnectorWorkerMetadata):
        def __init__(self, data: set[str]):
            self.data = data

    class MockConnectorWorkerMetadata0(MockConnectorWorkerMetadata):
        def aggregate(
            self, other: KVConnectorWorkerMetadata
        ) -> KVConnectorWorkerMetadata:
            assert isinstance(other, MockConnectorWorkerMetadata)
            return MockConnectorWorkerMetadata0(data=self.data | other.data)

    class MockConnectorWorkerMetadata1(MockConnectorWorkerMetadata):
        def aggregate(
            self, other: KVConnectorWorkerMetadata
        ) -> KVConnectorWorkerMetadata:
            assert isinstance(other, MockConnectorWorkerMetadata)
            return MockConnectorWorkerMetadata1(data=self.data | other.data)

    # -------------------- test build_worker_connector_meta -------------------

    # both connectors return None
    mc._connectors[0].build_connector_worker_meta.return_value = None
    mc._connectors[1].build_connector_worker_meta.return_value = None
    assert mc.build_connector_worker_meta() is None

    # only first connector returns None
    worker_meta1a = MockConnectorWorkerMetadata1({"1a"})
    mc._connectors[0].build_connector_worker_meta.return_value = None
    mc._connectors[1].build_connector_worker_meta.return_value = worker_meta1a
    mc_worker_meta_none_1a = mc.build_connector_worker_meta()
    assert isinstance(mc_worker_meta_none_1a, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_none_1a.metadata == (None, worker_meta1a)

    # only second connector returns None
    worker_meta0a = MockConnectorWorkerMetadata0({"0a"})
    mc._connectors[0].build_connector_worker_meta.return_value = worker_meta0a
    mc._connectors[1].build_connector_worker_meta.return_value = None
    mc_worker_meta_0a_none = mc.build_connector_worker_meta()
    assert isinstance(mc_worker_meta_0a_none, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_0a_none.metadata == (worker_meta0a, None)

    # both connectors do not return None
    worker_meta0b = MockConnectorWorkerMetadata0({"0b"})
    worker_meta1b = MockConnectorWorkerMetadata1({"1b"})
    mc._connectors[0].build_connector_worker_meta.return_value = worker_meta0b
    mc._connectors[1].build_connector_worker_meta.return_value = worker_meta1b
    mc_worker_meta_0b_1b = mc.build_connector_worker_meta()
    assert isinstance(mc_worker_meta_0b_1b, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_0b_1b.metadata == (worker_meta0b, worker_meta1b)

    # ----------------------------- test aggregate ----------------------------

    # aggregate ({"0a"}, None) and (None, {"1a"}) -> ({"0a"}, {"1a"})
    mc_worker_meta_0a_1a = mc_worker_meta_0a_none.aggregate(mc_worker_meta_none_1a)
    assert isinstance(mc_worker_meta_0a_1a, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_0a_1a.metadata == (worker_meta0a, worker_meta1a)

    # aggregate ({"0a"}, None) and ({"0b"}, None) -> ({"0a", "0b"}, None)
    mc._connectors[0].build_connector_worker_meta.return_value = worker_meta0b
    mc._connectors[1].build_connector_worker_meta.return_value = None
    mc_worker_meta_0b_none = mc.build_connector_worker_meta()
    mc_worker_meta_0a_0b = mc_worker_meta_0a_none.aggregate(mc_worker_meta_0b_none)
    assert isinstance(mc_worker_meta_0a_0b, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_0a_0b.metadata[1] is None
    connector0_md = mc_worker_meta_0a_0b.metadata[0]
    assert isinstance(connector0_md, MockConnectorWorkerMetadata0)
    assert connector0_md.data == {"0a", "0b"}

    # aggregate ({"0a"}, {"1a"}) and ({"0b"}, {"1b"}) -> ({"0a", "0b"}, {"1a", "1b"})
    mc_worker_meta_01a_01b = mc_worker_meta_0a_1a.aggregate(mc_worker_meta_0b_1b)
    assert isinstance(mc_worker_meta_01a_01b, MultiKVConnectorWorkerMetadata)
    metadata = mc_worker_meta_01a_01b.metadata
    assert len(metadata) == 2
    connector0_md, connector1_md = metadata
    assert isinstance(connector0_md, MockConnectorWorkerMetadata0)
    assert isinstance(connector1_md, MockConnectorWorkerMetadata1)
    assert connector0_md.data == {"0a", "0b"}
    assert connector1_md.data == {"1a", "1b"}

    # ---------------------- test update_connector_output ---------------------

    def verify_worker_metadata(expected_metadata: MockConnectorWorkerMetadata | None):
        def _verify_worker_metadata(connector_output: KVConnectorOutput):
            worker_meta = connector_output.kv_connector_worker_meta
            if expected_metadata is None:
                assert worker_meta is None
                return

            assert isinstance(worker_meta, MockConnectorWorkerMetadata)
            assert type(worker_meta) is type(expected_metadata)
            assert expected_metadata.data == worker_meta.data

        return _verify_worker_metadata

    def assert_update_connector_output_called(mc: MultiConnector):
        for c in mc._connectors:
            c.update_connector_output.assert_called_once()
            c.update_connector_output.reset_mock()

    # no worker meta
    kv_connector_output = KVConnectorOutput()
    mc._connectors[0].update_connector_output.side_effect = verify_worker_metadata(None)
    mc._connectors[1].update_connector_output.side_effect = verify_worker_metadata(None)
    mc.update_connector_output(kv_connector_output)
    assert_update_connector_output_called(mc)

    # multi worker meta
    kv_connector_output.kv_connector_worker_meta = mc_worker_meta_01a_01b
    mc._connectors[0].update_connector_output.side_effect = verify_worker_metadata(
        connector0_md
    )
    mc._connectors[1].update_connector_output.side_effect = verify_worker_metadata(
        connector1_md
    )
    mc.update_connector_output(kv_connector_output)
    assert_update_connector_output_called(mc)
    assert kv_connector_output.kv_connector_worker_meta == mc_worker_meta_01a_01b


def _make_multi_connector(connector_names: list[str]) -> MultiConnector:
    """Build a MultiConnector wrapping the given registered connectors."""
    connectors = [
        {
            "kv_connector": name,
            "kv_role": "kv_both",
            "kv_connector_module_path": "tests.v1.kv_connector.unit.test_multi_connector",  # noqa: E501
        }
        for name in connector_names
    ]
    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_connector_extra_config={"connectors": connectors},
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )
    return MultiConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.WORKER,
        kv_cache_config=kv_cache_config,
    )


def test_multi_connector_hma_support_detection():
    """At runtime, _all_support_hma is True only when every sub-connector
    implements SupportsHMA. Test all combinations of HMA / non-HMA
    sub-connectors.
    """
    assert supports_hma(MultiConnector)

    # -- All non-HMA connectors => _all_support_hma is False --
    mc_none = _make_multi_connector(["MockConnector", "MockConnector"])
    assert not supports_hma(mc_none._connectors[0])
    assert not supports_hma(mc_none._connectors[1])
    assert mc_none._all_support_hma is False

    # -- All HMA connectors => _all_support_hma is True --
    mc_all = _make_multi_connector(["MockHMAConnector", "MockHMAConnector"])
    assert supports_hma(mc_all._connectors[0])
    assert supports_hma(mc_all._connectors[1])
    assert mc_all._all_support_hma is True

    # -- Mixed: first HMA, second non-HMA => _all_support_hma is False --
    mc_mixed1 = _make_multi_connector(["MockHMAConnector", "MockConnector"])
    assert supports_hma(mc_mixed1._connectors[0])
    assert not supports_hma(mc_mixed1._connectors[1])
    assert mc_mixed1._all_support_hma is False

    # -- Mixed: first non-HMA, second HMA => _all_support_hma is False --
    mc_mixed2 = _make_multi_connector(["MockConnector", "MockHMAConnector"])
    assert not supports_hma(mc_mixed2._connectors[0])
    assert supports_hma(mc_mixed2._connectors[1])
    assert mc_mixed2._all_support_hma is False


def test_divergent_local_hybrid_hit_capability_is_conservative():
    all_supported = _make_multi_connector(
        ["MockDivergentHMAConnector", "MockDivergentHMAConnector"]
    )
    assert all_supported.supports_divergent_local_hybrid_hits is True

    mixed = _make_multi_connector(["MockDivergentHMAConnector", "MockHMAConnector"])
    assert mixed.supports_divergent_local_hybrid_hits is False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires GPU to instantiate LLM"
)
def test_multi_connector_mixed_hma_disables_hybrid_kv_cache(monkeypatch):
    """When MultiConnector wraps a mix of HMA (NixlConnector) and non-HMA
    (MockConnector) sub-connectors, verify that:
    1. The scheduler's MultiConnector has _all_support_hma == False.
    2. vLLM auto-disables the hybrid KV cache manager (no preference expressed by user)
    """
    from unittest.mock import patch

    from tests.v1.kv_connector.unit.test_nixl_connector import FakeNixlWrapper

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_consumer",
                },
                {
                    "kv_connector": "MockConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": (
                        "tests.v1.kv_connector.unit.test_multi_connector"
                    ),
                },
            ],
        },
    )

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
        FakeNixlWrapper,
    ):
        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            max_model_len=128,
            max_num_seqs=1,
            max_num_batched_tokens=128,
            kv_transfer_config=kv_transfer_config,
        )
        try:
            # HMA should be auto-disabled when user has not expressed a preference.
            assert (
                llm.llm_engine.vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
                is True
            )
            # The scheduler-side MultiConnector should detect the mixed
            # HMA support among its sub-connectors.
            scheduler = llm.llm_engine.engine_core.engine_core.scheduler
            mc = scheduler.connector
            assert isinstance(mc, MultiConnector)
            assert mc._all_support_hma is False
        finally:
            llm.llm_engine.engine_core.shutdown()
