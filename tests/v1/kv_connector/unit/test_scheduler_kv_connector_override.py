# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest

import vllm.plugins as plugins_module
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request


class DummyConnectorMetadata(KVConnectorMetadata):
    def __init__(
        self,
        block_hashes_by_req: dict[str, list[BlockHash]],
        block_ids_by_req: dict[str, tuple[list[int], ...]],
    ):
        self.block_hashes_by_req = block_hashes_by_req
        self.block_ids_by_req = block_ids_by_req


class DummyKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, role, kv_cache_config)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return (0, False)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        pass

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        block_hashes_by_req = getattr(scheduler_output, "block_hashes_by_req", None)
        assert block_hashes_by_req is not None, (
            "DummyKVConnector expected 'block_hashes_by_req' on scheduler_output"
        )
        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None
        block_ids_by_req = {}
        for req_id in block_state.req_ids:
            block_ids = block_state.get_block_ids(req_id)
            assert block_ids is not None
            block_ids_by_req[req_id] = block_ids
        return DummyConnectorMetadata(
            block_hashes_by_req=block_hashes_by_req,
            block_ids_by_req=block_ids_by_req,
        )

    def start_load_kv(self, kv_caches, finished_req_ids):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


def _my_plugin():
    """Registers the dummy KV connector and overrides _build_kv_connector_meta"""
    if "DummyKVConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "DummyKVConnector",
            __name__,
            DummyKVConnector.__name__,
        )

    def _custom_build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        block_hashes_by_req: dict[str, list[BlockHash]] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            block_hashes_by_req[req_id] = request.block_hashes

        scheduler_output.block_hashes_by_req = block_hashes_by_req  # type: ignore[attr-defined]
        return connector.build_connector_meta(scheduler_output)

    Scheduler._build_kv_connector_meta = _custom_build_kv_connector_meta


@pytest.fixture
def _load_plugin():
    """Load the fake plugin through the real load_general_plugins() path."""
    ep = MagicMock()
    ep.name = "dummy_kv_connector_plugin"
    ep.value = f"{__name__}:_my_plugin"
    ep.load.return_value = _my_plugin

    # Reset the global guard so load_general_plugins() actually runs.
    plugins_module.plugins_loaded = False
    with patch("importlib.metadata.entry_points", return_value=[ep]):
        plugins_module.load_general_plugins()
        yield
    # Reset again so other tests are not affected.
    plugins_module.plugins_loaded = False


def test_connector_receives_block_hashes(_load_plugin):
    block_size = 16
    num_tokens = 48  # 3 full blocks worth of tokens
    scheduler = create_scheduler(
        use_kv_connector="DummyKVConnector", block_size=block_size
    )
    requests = create_requests(
        num_requests=3, num_tokens=num_tokens, block_size=block_size
    )
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()

    # Exercise worker serialization without opening sockets or shared memory.
    MessageQueue(n_reader=0, n_local_reader=0).enqueue(output)
    meta = output.kv_connector_metadata
    assert isinstance(meta, DummyConnectorMetadata)
    assert len(meta.block_hashes_by_req) == 3

    for req in requests:
        assert req.request_id in meta.block_hashes_by_req
        # Each request has num_tokens / block_size = 3 full block hashes.
        assert len(meta.block_hashes_by_req[req.request_id]) == (
            num_tokens // block_size
        )
        assert meta.block_hashes_by_req[req.request_id] == req.block_hashes

    expected_block_ids = {
        req.req_id: req.block_ids for req in output.scheduled_new_reqs
    }
    assert meta.block_ids_by_req == expected_block_ids
    assert output.kv_connector_block_state is None


def _make_model_runner_output(scheduler: Scheduler) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in scheduler.running],
        req_id_to_index={req.request_id: i for i, req in enumerate(scheduler.running)},
        sampled_token_ids=[[1000]] * len(scheduler.running),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_connector_block_state_covers_steps_without_new_blocks(_load_plugin):
    """A store save lands on the step that fills a block, which allocates none.

    The connector must still be able to look up that request's current table.
    """
    block_size = 16
    scheduler = create_scheduler(
        use_kv_connector="DummyKVConnector", block_size=block_size
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=3 * block_size - 1, block_size=block_size
    )
    scheduler.add_request(request)

    output = scheduler.schedule()
    scheduler.update_from_output(output, _make_model_runner_output(scheduler))

    # The first decode token fills the third block: no allocation this step.
    output = scheduler.schedule()
    assert output.scheduled_cached_reqs.req_ids == [request.request_id]
    [new_block_ids] = output.scheduled_cached_reqs.new_block_ids
    assert not new_block_ids

    meta = output.kv_connector_metadata
    assert isinstance(meta, DummyConnectorMetadata)
    block_ids = meta.block_ids_by_req.get(request.request_id)
    assert block_ids is not None
    assert block_ids == scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert len(block_ids[0]) == 3


def test_connector_block_state_resolves_only_offered_requests():
    tables = {"a": ([1, 2],), "b": ([3],)}
    resolve_block_ids = MagicMock(side_effect=tables.__getitem__)
    block_state = KVConnectorBlockState(
        req_ids={"a"},
        resolve_block_ids=resolve_block_ids,
        boundary_state_offloads={},
    )

    assert block_state.get_block_ids("b") is None
    resolve_block_ids.assert_not_called()
    assert block_state.get_block_ids("a") == ([1, 2],)
    resolve_block_ids.assert_called_once_with("a")

    tables["a"] = ([4, 5],)
    assert block_state.get_block_ids("a") == ([4, 5],)
