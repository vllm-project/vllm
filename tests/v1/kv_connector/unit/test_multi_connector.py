# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import filecmp
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    maybe_update_block_size,
)
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorTransferResults,
    KVLoadRange,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorLogging,
    KVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
    MultiKVConnectorPromMetrics,
    MultiKVConnectorStats,
    MultiKVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlKVConnectorStats,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput, KVConnectorWorkerMetadata
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


def test_register_finished_partial_tail_notifies_every_connector():
    connector = object.__new__(MultiConnector)
    first = MagicMock(spec_set=KVConnectorBase_V1)
    second = MagicMock(spec_set=KVConnectorBase_V1)
    first.register_finished_partial_tail.return_value = True
    second.register_finished_partial_tail.return_value = False
    connector._connectors = [first, second]
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


class RangeTransport(KVConnectorBase_V1, SupportsHMA):
    """Deterministic external transport; MultiConnector itself is real."""

    def __init__(self, config, role, kv_cache_config):
        super().__init__(config, role, kv_cache_config)
        self.options = config.kv_transfer_config.kv_connector_extra_config
        self.loads = []
        self.lookups = []
        self.completions = KVConnectorTransferResults()

    @property
    def supports_load_range(self):
        return self.options.get("range", True)

    @property
    def load_range_alignment(self):
        return self.options.get("alignment")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.lookups.append(num_computed_tokens)
        endpoint = self.options.get("endpoint", 0)
        return (
            None if endpoint is None else max(0, endpoint - num_computed_tokens),
            self.options.get("async", True),
        )

    def get_loaded_kv_cache_group_ids(self, request):
        return tuple(self.options.get("groups", [0]))

    def get_loadable_window_group_ids(self, request, end_token):
        if "windows" in self.options:
            return tuple(self.options["windows"].get(str(end_token), []))
        if end_token <= self.options.get("endpoint", 0):
            return tuple(g for g in self.options.get("groups", [0]) if g != 0)
        return ()

    def get_external_cache_hit_sources(self, request, num_external_tokens):
        return {CacheHitSource.DISK: num_external_tokens} if num_external_tokens else {}

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.loads.append(num_external_tokens)

    def update_state_after_alloc_for_range(self, request, blocks, load_range):
        self.loads.append(load_range)

    def get_transfer_results(self, finished_req_ids):
        return self.completions

    def get_block_ids_with_load_errors(self):
        return set(self.options.get("load_errors", []))

    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def request_finished_all_groups(self, request, block_ids):
        return False, None


KVConnectorFactory.register_connector("RangeTransport", __name__, "RangeTransport")


def _range_config(options, policy="range_aware"):
    extra = {
        "connectors": [
            {
                "kv_connector": "RangeTransport",
                "kv_role": "kv_both",
                "kv_connector_extra_config": option,
            }
            for option in options
        ]
    }
    if policy is not None:
        extra["load_policy"] = policy
    return create_vllm_config(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
        max_num_batched_tokens=256,
        max_model_len=256,
    )


def _range_cache(kind="full"):
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    groups = [KVCacheGroupSpec(["full"], full)]
    if kind != "full":
        spec = (
            SlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.bfloat16,
                sliding_window=32,
                bounded_replay=True,
            )
            if kind == "bounded"
            else SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.bfloat16,
                sliding_window=32,
            )
        )
        groups.append(KVCacheGroupSpec(["swa"], spec))
    return KVCacheConfig(num_blocks=1000, kv_cache_tensors=[], kv_cache_groups=groups)


def _range_connector(options, policy="range_aware", role=KVConnectorRole.SCHEDULER):
    return MultiConnector(_range_config(options, policy), role, _range_cache())


@pytest.mark.parametrize(
    "options,local,adopted,expected",
    [
        (
            [{"endpoint": 64, "range": False}, {"endpoint": 96}],
            0,
            96,
            [64, KVLoadRange(64, 96)],
        ),
        (
            [{"endpoint": 64}, {"endpoint": 96}],
            0,
            96,
            [KVLoadRange(0, 64, False), KVLoadRange(64, 96)],
        ),
        ([{"endpoint": 64}, {"endpoint": 96}], 0, 0, [0, 0]),
        (
            [{"endpoint": 64, "range": False}, {"endpoint": 96}],
            32,
            64,
            [32, KVLoadRange(64, 96)],
        ),
        (
            [{"endpoint": 60}, {"endpoint": 96}],
            0,
            96,
            [KVLoadRange(0, 48, False), KVLoadRange(48, 96)],
        ),
        ([{"endpoint": 60, "range": False}, {"endpoint": 96}], 0, 96, [0, 96]),
        ([{"endpoint": 64, "async": False}, {"endpoint": 96}], 0, 96, [0, 96]),
        ([{"endpoint": 12, "range": False}, {"endpoint": 96}], 0, 96, [0, 96]),
        (
            [{"endpoint": 60, "alignment": 24}, {"endpoint": 96, "alignment": 4}],
            0,
            96,
            [KVLoadRange(0, 48, False), KVLoadRange(48, 96)],
        ),
        (
            [{"endpoint": 48, "alignment": 64}, {"endpoint": 80}, {"endpoint": 112}],
            16,
            96,
            [0, KVLoadRange(16, 80, False), KVLoadRange(80, 112)],
        ),
        (
            [{"endpoint": 48, "alignment": 64}, {"endpoint": 80, "alignment": 64}],
            16,
            64,
            [0, 64],
        ),
        (
            [{"endpoint": 64, "range": False}, {"endpoint": 96}, {"endpoint": 128}],
            0,
            128,
            [64, KVLoadRange(64, 96, False), KVLoadRange(96, 128)],
        ),
        (
            [
                {"endpoint": 64, "range": False},
                {"endpoint": 96, "range": False},
                {"endpoint": 128},
            ],
            0,
            128,
            [64, 0, KVLoadRange(64, 128)],
        ),
        (
            [{"endpoint": 64, "range": False}, {"endpoint": 70}],
            0,
            70,
            [64, KVLoadRange(64, 70)],
        ),
        ([{"endpoint": 96}, {"endpoint": 96}], 0, 96, [96, 0]),
        ([{"endpoint": 96}, {"endpoint": 64}], 0, 96, [96, 0]),
        (
            [{"endpoint": 64, "range": False}, {"endpoint": 96, "range": False}],
            0,
            96,
            [0, 96],
        ),
        (
            [{"endpoint": 64}, {"endpoint": 96}],
            0,
            80,
            [KVLoadRange(0, 64, False), KVLoadRange(64, 80)],
        ),
        ([{"endpoint": 64}, {"endpoint": 96}], 0, 64, [KVLoadRange(0, 64), 0]),
        ([{"endpoint": 64}, {"endpoint": 96}], 0, 32, [KVLoadRange(0, 32), 0]),
    ],
)
@pytest.mark.cpu_test
def test_range_assignments(options, local, adopted, expected):
    connector = _range_connector(options)
    request = create_request(num_tokens=160)
    matched, _ = connector.get_num_new_matched_tokens(request, local)
    assert matched == max(o["endpoint"] for o in options) - local
    blocks = KVCacheBlocks(([],))
    connector.update_state_after_alloc(request, blocks, adopted)
    assert [c.loads for c in connector.sub_connectors] == [[e] for e in expected]
    # A later no-load update must not repeat a transfer or discard replay state.
    connector.update_state_after_alloc(request, blocks, 0)
    assert [c.loads for c in connector.sub_connectors] == [[e, 0] for e in expected]
    if adopted:
        assert connector.get_external_cache_hit_sources(request, adopted) == {
            CacheHitSource.DISK: adopted
        }


@pytest.mark.cpu_test
def test_range_lookup_waits_for_all_sources():
    connector = _range_connector([{"endpoint": None}, {"endpoint": 96}])
    request = create_request(num_tokens=160)
    assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
    assert [c.lookups for c in connector.sub_connectors] == [[0], [0]]


@pytest.mark.cpu_test
def test_range_load_errors_are_unioned():
    connector = _range_connector([{"load_errors": [1, 2]}, {"load_errors": [2, 3]}])
    assert connector.get_block_ids_with_load_errors() == {1, 2, 3}


@pytest.mark.parametrize("alignment", [0, -1])
@pytest.mark.cpu_test
def test_range_invalid_alignment(alignment):
    connector = _range_connector(
        [{"endpoint": 64}, {"endpoint": 96, "alignment": alignment}]
    )
    with pytest.raises(ValueError, match="load_range_alignment must be positive"):
        connector.get_num_new_matched_tokens(create_request(num_tokens=160), 0)


@pytest.mark.cpu_test
def test_unknown_range_policy():
    with pytest.raises(ValueError, match="Unknown MultiConnector load_policy"):
        _range_connector([{"endpoint": 64}], policy="range-aware")


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("range_capable", [False, True])
@pytest.mark.cpu_test
def test_recurrent_load_uses_single_source(wrapped, range_capable):
    spec = MambaSpec(block_size=16, shapes=((1, 1),), dtypes=(torch.float32,))
    if wrapped:
        spec = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"mamba": spec})
    cache = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba"], spec)],
    )
    config = _range_config([{"endpoint": 64, "range": range_capable}, {"endpoint": 96}])
    connector = MultiConnector(config, KVConnectorRole.SCHEDULER, cache)
    request = create_request(num_tokens=160)
    assert connector.get_num_new_matched_tokens(request, 0) == (96, True)
    connector.update_state_after_alloc(request, KVCacheBlocks(([],)), 96)
    assert [c.loads for c in connector.sub_connectors] == [[0], [96]]


@pytest.mark.cpu_test
def test_range_hybrid_block_alignment():
    cache = _range_cache("swa")
    cache.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    connector = MultiConnector(
        _range_config([{"endpoint": 384}, {"endpoint": 512, "groups": [0, 1]}]),
        KVConnectorRole.SCHEDULER,
        cache,
    )
    request = create_request(num_tokens=600)
    assert connector.get_num_new_matched_tokens(request, 0) == (512, True)
    connector.update_state_after_alloc(request, KVCacheBlocks(([], [])), 512)
    assert [c.loads for c in connector.sub_connectors] == [
        [KVLoadRange(0, 256, False)],
        [KVLoadRange(256, 512, window_group_ids=(1,), window_end_token=512)],
    ]


@pytest.mark.parametrize("groups", [[0], [0, 1]])
@pytest.mark.cpu_test
def test_legacy_window_loader_cannot_race_terminal_source(groups):
    connector = MultiConnector(
        _range_config(
            [
                {"endpoint": 64, "range": False, "groups": groups},
                {"endpoint": 96, "groups": [0, 1]},
            ]
        ),
        KVConnectorRole.SCHEDULER,
        _range_cache("swa"),
    )
    request = create_request(num_tokens=160)
    assert connector.get_num_new_matched_tokens(request, 0) == (96, True)
    connector.update_state_after_alloc(request, KVCacheBlocks(([], [])), 96)
    assert [c.loads for c in connector.sub_connectors] == [[0], [96]]


@pytest.mark.parametrize("policy", [None, "range_aware"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("kind", ["full", "swa", "bounded"])
@pytest.mark.parametrize(
    "endpoint,window_loaded", [(128, False), (127, False), (128, True), (127, True)]
)
@pytest.mark.cpu_test
def test_range_scheduler_replay(policy, is_async, kind, endpoint, window_loaded):
    """Complete async receives before checking the real scheduler's replay."""
    groups = [0, 1] if window_loaded or kind == "swa" else [0]
    config = _range_config(
        [
            {
                "endpoint": 64,
                "async": is_async,
                "groups": [0, 1] if kind == "swa" else [0],
            },
            {"endpoint": endpoint, "async": is_async, "groups": groups},
        ],
        policy,
    )
    register_all_kvcache_specs(config)
    scheduler = create_scheduler(
        config, num_blocks=1000, kv_cache_config=_range_cache(kind)
    )
    request = create_request(num_tokens=160)
    scheduler.add_request(request)
    output = scheduler.schedule()
    adopted = endpoint if policy else 64
    restores = bool(policy and window_loaded)
    if kind == "bounded" and not restores:
        adopted = adopted // 16 * 16
    if is_async:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler.update_from_output(
            output,
            create_model_runner_output([], finished_recving={request.request_id}),
        )
        output = scheduler.schedule()
    expected = adopted - 32 if kind == "bounded" and not restores else adopted
    new = next(r for r in output.scheduled_new_reqs if r.req_id == request.request_id)
    assert new.num_computed_tokens == expected
    assert new.replay_start == (expected if kind == "bounded" and not restores else 0)
    assert output.num_scheduled_tokens[request.request_id] == 160 - expected
    assert scheduler.connector.get_external_cache_hit_sources(request, adopted) == {
        CacheHitSource.DISK: adopted
    }


@pytest.mark.parametrize("adopted", [32, 64, 80, 96])
@pytest.mark.cpu_test
def test_truncation_reselects_window_before_allocation(adopted):
    connector = MultiConnector(
        _range_config(
            [{"endpoint": 64, "groups": [0, 1]}, {"endpoint": 96, "groups": [0, 1]}]
        ),
        KVConnectorRole.SCHEDULER,
        _range_cache("swa"),
    )
    request = create_request(num_tokens=160)
    connector.get_num_new_matched_tokens(request, 0)
    assert connector.get_loaded_kv_cache_group_ids_for_prefix(request, adopted) == (
        0,
        1,
    )
    connector.update_state_after_alloc(request, KVCacheBlocks(([], [])), adopted)
    assert connector.get_loaded_kv_cache_group_ids(request) == (0, 1)
    loads = [c.loads[0] for c in connector.sub_connectors]
    owner = 0 if adopted <= 64 else 1
    assert loads[owner].window_group_ids == (1,)
    assert loads[owner].window_end_token == adopted
    assert loads[1 - owner] == 0 or loads[1 - owner].window_group_ids == ()
    assert connector.get_external_cache_hit_sources(request, adopted) == {
        CacheHitSource.DISK: adopted
    }


@pytest.mark.parametrize("owner", [0, 2, None, "sync_only"])
@pytest.mark.parametrize("kind", ["swa", "bounded"])
@pytest.mark.cpu_test
def test_window_owner_is_independent_of_prefix_ranges(owner, kind):
    options: list[dict[str, Any]] = [{"endpoint": n} for n in (64, 96, 32)]
    if owner == "sync_only":
        options[2].update({"async": False, "windows": {"96": [1]}})
    elif owner is not None:
        options[owner]["windows"] = {"96": [1]}
        if owner == 0:
            options[1]["windows"] = {"96": [1]}
    connector = MultiConnector(
        _range_config(options), KVConnectorRole.SCHEDULER, _range_cache(kind)
    )
    request = create_request(num_tokens=160)
    assert connector.get_num_new_matched_tokens(request, 0) == (96, True)
    connector.update_state_after_alloc(request, KVCacheBlocks(([], [])), 96)
    loads = [c.loads[0] for c in connector.sub_connectors]
    if owner is None or owner == "sync_only":
        assert loads == [0, 96, 0]
        assert connector.get_loaded_kv_cache_group_ids(request) == (0,)
    else:
        assert loads[0].start_token == 0 and loads[0].end_token == 64
        assert loads[1] == KVLoadRange(64, 96)
        assert loads[owner].window_group_ids == (1,)
        assert loads[owner].window_end_token == 96
        if owner == 2:
            assert loads[2].num_tokens == 0
        assert connector.get_loaded_kv_cache_group_ids(request) == (0, 1)
        metadata = connector.build_connector_meta(SchedulerOutput.make_empty())
        assert metadata.pending_async_loads == {
            request.request_id: (0, 1, 2) if owner == 2 else (0, 1)
        }
    assert connector.get_external_cache_hit_sources(request, 96) == {
        CacheHitSource.DISK: 96
    }


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.cpu_test
def test_scheduler_uses_window_only_source_without_replay(is_async):
    config = _range_config(
        [
            {"endpoint": 64, "async": is_async},
            {"endpoint": 128, "async": is_async},
            {"endpoint": 32, "async": is_async, "windows": {"128": [1]}},
        ]
    )
    register_all_kvcache_specs(config)
    scheduler = create_scheduler(
        config, num_blocks=1000, kv_cache_config=_range_cache("bounded")
    )
    request = create_request(num_tokens=160)
    scheduler.add_request(request)
    output = scheduler.schedule()
    if is_async:
        scheduler.update_from_output(
            output,
            create_model_runner_output([], finished_recving={request.request_id}),
        )
        output = scheduler.schedule()
    new = next(r for r in output.scheduled_new_reqs if r.req_id == request.request_id)
    assert new.num_computed_tokens == 128
    assert new.replay_start == 0
    assert output.num_scheduled_tokens[request.request_id] == 32
    assert scheduler.connector.sub_connectors[2].loads[0] == KVLoadRange(
        128, 128, window_group_ids=(1,), window_end_token=128
    )


@pytest.mark.cpu_test
def test_window_missing_at_shortened_endpoint_falls_back_before_allocation():
    connector = MultiConnector(
        _range_config([{"endpoint": 64, "windows": {"96": [1]}}, {"endpoint": 96}]),
        KVConnectorRole.SCHEDULER,
        _range_cache("bounded"),
    )
    request = create_request(num_tokens=160)
    connector.get_num_new_matched_tokens(request, 0)
    assert connector.get_loaded_kv_cache_group_ids(request) == (0, 1)
    assert connector.get_loaded_kv_cache_group_ids_for_prefix(request, 80) == (0,)
    connector.update_state_after_alloc(request, KVCacheBlocks(([], [])), 80)
    assert [c.loads for c in connector.sub_connectors] == [[0], [80]]


@pytest.mark.parametrize("first_source", [0, 1])
@pytest.mark.parametrize("failed_source", [None, 0, 1])
@pytest.mark.parametrize(
    "abort_timing", ["never", "before", "with_first", "after_first"]
)
@pytest.mark.cpu_test
def test_range_receive_barrier(first_source, failed_source, abort_timing):
    """Cancellation and a failed source cannot release another writer's blocks."""
    worker = _range_connector([{}, {}, {}], role=KVConnectorRole.WORKER)
    worker.bind_connector_metadata(
        MultiKVConnectorMetadata(
            metadata=tuple(KVConnectorMetadata() for _ in range(3)),
            pending_async_loads={"req": (0, 1)},
        )
    )
    children = worker.sub_connectors
    children[2].completions = KVConnectorTransferResults(
        finished_recving={"req"}, failed_recving={"req"}
    )
    if abort_timing == "before":
        assert worker.get_transfer_results({"req"}) == KVConnectorTransferResults()
    children[first_source].completions = KVConnectorTransferResults(
        finished_recving={"req"},
        failed_recving={"req"} if failed_source == first_source else set(),
    )
    finished = {"req"} if abort_timing == "with_first" else set()
    assert worker.get_transfer_results(finished) == KVConnectorTransferResults()
    if abort_timing == "after_first":
        assert worker.get_transfer_results({"req"}) == KVConnectorTransferResults()
    assert worker.get_transfer_results(set()) == KVConnectorTransferResults()
    last = 1 - first_source
    children[last].completions = KVConnectorTransferResults(
        finished_recving={"req"},
        failed_recving={"req"} if failed_source == last else set(),
    )
    assert worker.get_transfer_results(set()) == KVConnectorTransferResults(
        finished_recving={"req"},
        failed_recving={"req"} if failed_source is not None else set(),
    )
    for child in children:
        child.completions = KVConnectorTransferResults()
    assert worker.get_transfer_results({"req"}) == KVConnectorTransferResults()


@pytest.mark.cpu_test
def test_range_three_source_metadata_and_completion():
    options = [{"endpoint": n} for n in (64, 96, 128)]
    scheduler = _range_connector(options)
    request = create_request(num_tokens=160)
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, KVCacheBlocks(([],)), 128)
    metadata = scheduler.build_connector_meta(SchedulerOutput.make_empty())
    assert metadata.pending_async_loads == {request.request_id: (0, 1, 2)}
    worker = _range_connector(options, role=KVConnectorRole.WORKER)
    worker.bind_connector_metadata(metadata)
    for index, child in enumerate(worker.sub_connectors):
        child.completions = KVConnectorTransferResults(
            finished_recving={request.request_id}
        )
        assert worker.get_finished(set()) == (
            None,
            {request.request_id} if index == 2 else None,
        )
    assert worker.get_finished(set()) == (None, None)


@pytest.mark.cpu_test
def test_default_first_hit_and_async_saves():
    worker = _range_connector([{"endpoint": 64}, {"endpoint": 96}], policy=None)
    request = create_request(num_tokens=160)
    assert worker.get_num_new_matched_tokens(request, 0) == (64, True)
    worker.update_state_after_alloc(request, KVCacheBlocks(([],)), 64)
    assert [c.loads for c in worker.sub_connectors] == [[64], [0]]
    worker.bind_connector_metadata(
        MultiKVConnectorMetadata(
            metadata=(KVConnectorMetadata(), KVConnectorMetadata()),
            extra_async_saves={"saved": 1},
        )
    )
    first, second = worker.sub_connectors
    first.completions = KVConnectorTransferResults(
        finished_sending={"saved"},
        finished_recving={"failed"},
        failed_recving={"failed"},
    )
    second.completions = KVConnectorTransferResults(
        finished_sending={"saved"},
        finished_recving={"loaded"},
    )
    assert worker.get_transfer_results(set()) == KVConnectorTransferResults(
        finished_sending={"saved"},
        finished_recving={"failed", "loaded"},
        failed_recving={"failed"},
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

    block_size = maybe_update_block_size(16)
    expected_num_blocks = 2 if block_size == 64 else 7
    expected_num_matched_tokens = (expected_num_blocks - 1) * block_size
    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        block_size=block_size,
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
        f"update_state_after_alloc num_blocks=[{expected_num_blocks}] 0",
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
        f"update_state_after_alloc num_blocks=[{expected_num_blocks}] 0",
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
    assert storage1_scheduler_events[:5] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        f"get_loaded_kv_cache_group_ids_for_prefix {expected_num_matched_tokens}",
        (
            "update_state_after_alloc "
            f"num_blocks=[{expected_num_blocks}] {expected_num_matched_tokens}"
        ),
        "build_connector_meta",
    ]
    assert storage2_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        f"update_state_after_alloc num_blocks=[{expected_num_blocks}] 0",
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
        f"update_state_after_alloc num_blocks=[{expected_num_blocks}] 0",
        "build_connector_meta",
    ]
    assert storage2_scheduler_events[:5] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        f"get_loaded_kv_cache_group_ids_for_prefix {expected_num_matched_tokens}",
        (
            "update_state_after_alloc "
            f"num_blocks=[{expected_num_blocks}] {expected_num_matched_tokens}"
        ),
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

    @pytest.mark.parametrize("with_exporter", [False, True])
    def test_prom_metrics_skips_stats_only_connectors(self, with_exporter):
        """Stats-only children must not prevent supported metrics from exporting."""
        child_metrics = MagicMock()
        metrics = object.__new__(MultiKVConnectorPromMetrics)
        metrics._prom_metrics = (
            {"MooncakeStoreConnector": child_metrics} if with_exporter else {}
        )
        payload = {"MooncakeConnector": {"bytes_transferred": [1024]}}
        if with_exporter:
            payload["MooncakeStoreConnector"] = {"load_get_bytes": [2048]}

        metrics.observe(payload, engine_idx=2)

        if with_exporter:
            child_metrics.observe.assert_called_once_with({"load_get_bytes": [2048]}, 2)

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


def test_multi_connector_overrides_all_base_methods(mc):
    """Ensure MultiConnector overrides all public methods from KVConnectorBase_V1."""
    # These are fine to inherit from KVConnectorBase_V1
    # TODO(https://github.com/vllm-project/vllm/pull/31811): Remove
    # get_kv_connector_kv_cache_events from INHERITED_OK once implemented.
    INHERITED_OK = {
        "role",
        "has_connector_metadata",
        "get_kv_connector_kv_cache_events",
        "supports_load_range",
        "load_range_alignment",
        "update_state_after_alloc_for_range",
        # MultiConnector coordinates range loaders, but is not itself one.
        "get_loadable_window_group_ids",
    }

    assert not mc.supports_load_range
    assert mc.get_loadable_window_group_ids(create_request(num_tokens=160), 96) == ()

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
