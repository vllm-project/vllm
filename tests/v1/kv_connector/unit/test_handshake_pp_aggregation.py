# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    WorkerRanks,
)
from vllm.v1.engine import core as engine_core_module

pytestmark = pytest.mark.cpu_test


class _Metadata(KVConnectorHandshakeMetadata):
    pass


@dataclass
class _AgentMetadata(KVConnectorHandshakeMetadata):
    agent: str


class _FakeExecutor:
    handshake_metadata_src: list[dict[Any, KVConnectorHandshakeMetadata] | None] | None
    last_instance: "_FakeExecutor | None" = None

    def __init__(
        self,
        vllm_config: Any,
    ) -> None:
        del vllm_config
        self.handshake_metadata = self.handshake_metadata_src
        self.handshake_calls = 0
        _FakeExecutor.last_instance = self

    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[Any, KVConnectorHandshakeMetadata] | None] | None:
        self.handshake_calls += 1
        return self.handshake_metadata

    def init_kv_output_aggregator(self, connector: KVConnectorBase_V1) -> None:
        pass


def _run_engine_core_handshake(
    monkeypatch: pytest.MonkeyPatch,
    connector: KVConnectorBase_V1,
    *,
    handshake_metadata: list[dict[Any, KVConnectorHandshakeMetadata] | None] | None,
) -> _FakeExecutor:
    class _FakeScheduler:
        def __init__(self, **kwargs: Any) -> None:
            self.connector = connector
            self.ec_connector = None

        def get_kv_connector(self) -> KVConnectorBase_V1:
            return connector

    _FakeExecutor.handshake_metadata_src = handshake_metadata
    _FakeExecutor.last_instance = None

    monkeypatch.setattr("vllm.plugins.load_general_plugins", lambda: None)
    monkeypatch.setattr(
        engine_core_module.EngineCore,
        "_initialize_kv_caches",
        lambda self, vllm_config: SimpleNamespace(kv_cache_groups=[object()]),
    )
    monkeypatch.setattr(
        engine_core_module,
        "StructuredOutputManager",
        lambda vllm_config: object(),
    )
    monkeypatch.setattr(
        engine_core_module,
        "resolve_kv_cache_block_sizes",
        lambda kv_cache_config, vllm_config: (16, 16),
    )
    monkeypatch.setattr(engine_core_module, "freeze_gc_heap", lambda: None)
    monkeypatch.setattr(
        engine_core_module, "maybe_attach_gc_debug_callback", lambda: None
    )
    monkeypatch.setattr(engine_core_module, "enable_envs_cache", lambda: None)
    monkeypatch.setattr(engine_core_module, "get_hash_fn_by_name", lambda name: None)
    monkeypatch.setattr(engine_core_module, "init_none_hash", lambda hash_fn: None)
    # Normally set by init_none_hash; read when building the prefix hash config.
    monkeypatch.setattr(
        engine_core_module.kv_cache_utils, "NONE_HASH", b"", raising=False
    )
    monkeypatch.setattr(
        engine_core_module, "get_request_block_hasher", lambda *args: None
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank_local=0),
        scheduler_config=SimpleNamespace(
            get_scheduler_cls=lambda: _FakeScheduler,
            enable_chunked_prefill=False,
            async_scheduling=False,
            prefill_schedule_interval=1,
        ),
        speculative_config=None,
        ec_transfer_config=None,
        is_mm_encoder_only=False,
        max_concurrent_batches=1,
        num_hash_lookahead_tokens=0,
        model_config=SimpleNamespace(
            runner_type="generate",
            is_diffusion=False,
            supports_multimodal_inputs=False,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            prefix_caching_hash_algo="builtin",
        ),
    )

    engine_core_module.EngineCore(vllm_config, _FakeExecutor, log_stats=False)
    assert _FakeExecutor.last_instance is not None
    return _FakeExecutor.last_instance


class _LegacyConnector(KVConnectorBase_V1):
    def __init__(self, pcp_size: int = 1, dcp_size: int = 1) -> None:
        self._vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                prefill_context_parallel_size=pcp_size,
                decode_context_parallel_size=dcp_size,
            )
        )
        self.legacy_metadata: dict[int, KVConnectorHandshakeMetadata] | None = None

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        pass

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        raise NotImplementedError

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        self.legacy_metadata = metadata


class _PPAwareConnector(_LegacyConnector):
    def __init__(self, pcp_size: int = 1, dcp_size: int = 1) -> None:
        super().__init__(pcp_size, dcp_size)
        self.pp_aware_metadata: (
            dict[tuple[int, int], KVConnectorHandshakeMetadata] | None
        ) = None

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        self.pp_aware_metadata = metadata


def test_engine_unwraps_handshake_metadata_for_legacy_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine core unwraps worker metadata to `{tp_rank: metadata}` for a
    connector that has not opted into PP-aware handshake (single-PP producer,
    all `pp_rank == 0`)."""
    metadata_0 = _Metadata()
    metadata_1 = _Metadata()
    connector = _LegacyConnector()

    executor = _run_engine_core_handshake(
        monkeypatch,
        connector,
        handshake_metadata=[
            {WorkerRanks(pp_rank=0, pcp_rank=0, tp_rank=0): metadata_0},
            None,
            {WorkerRanks(pp_rank=0, pcp_rank=0, tp_rank=1): metadata_1},
        ],
    )

    assert executor.handshake_calls == 1
    assert connector.legacy_metadata == {0: metadata_0, 1: metadata_1}


def test_engine_rejects_pp_producer_for_legacy_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connector that has not opted into PP-aware handshake must not silently
    drop metadata from `pp_rank > 0`; engine core init raises instead."""
    connector = _LegacyConnector()

    with pytest.raises(ValueError, match="does not support PP-disaggregated"):
        _run_engine_core_handshake(
            monkeypatch,
            connector,
            handshake_metadata=[
                {WorkerRanks(pp_rank=0, pcp_rank=0, tp_rank=0): _Metadata()},
                {WorkerRanks(pp_rank=1, pcp_rank=0, tp_rank=0): _Metadata()},
            ],
        )


def test_engine_passes_handshake_metadata_through_for_pp_aware_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PP-aware connector receives every stage's metadata keyed by
    `(pp_rank, tp_rank)`."""
    metadata_0 = _Metadata()
    metadata_1 = _Metadata()
    connector = _PPAwareConnector()

    executor = _run_engine_core_handshake(
        monkeypatch,
        connector,
        handshake_metadata=[
            {WorkerRanks(pp_rank=0, pcp_rank=0, tp_rank=0): metadata_0},
            {WorkerRanks(pp_rank=1, pcp_rank=0, tp_rank=0): metadata_1},
        ],
    )

    assert executor.handshake_calls == 1
    assert connector.legacy_metadata is None
    assert connector.pp_aware_metadata == {
        (0, 0): metadata_0,
        (1, 0): metadata_1,
    }


@pytest.mark.parametrize(
    ("dcp_size", "agents", "expected"),
    [
        # PCP x DCP shards: PCP rank 1 maps past the TP ranks, as before.
        (2, ("pcp0", "pcp1"), {(0, 0): "pcp0", (0, 1): "pcp1"}),
        # Replicated PCP: identical reports share one key.
        (1, ("same", "same"), {(0, 0): "same"}),
    ],
)
def test_engine_keys_pcp_ranks_by_tp_rank_for_pp_aware_connector(
    monkeypatch: pytest.MonkeyPatch,
    dcp_size: int,
    agents: tuple[str, str],
    expected: dict[tuple[int, int], str],
) -> None:
    """A connector keyed by `(pp_rank, tp_rank)` receives PCP ranks under the
    keys that workers reported before they reported their ranks."""
    connector = _PPAwareConnector(pcp_size=2, dcp_size=dcp_size)

    _run_engine_core_handshake(
        monkeypatch,
        connector,
        handshake_metadata=[
            {WorkerRanks(pp_rank=0, pcp_rank=pcp_rank, tp_rank=0): _AgentMetadata(a)}
            for pcp_rank, a in enumerate(agents)
        ],
    )

    assert connector.pp_aware_metadata == {
        key: _AgentMetadata(agent) for key, agent in expected.items()
    }


def test_engine_rejects_different_pcp_metadata_for_pp_aware_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without DCP, PCP ranks share a `(pp_rank, tp_rank)` key. A connector
    keyed that way gets an error rather than only the last rank's metadata."""
    connector = _PPAwareConnector(pcp_size=2, dcp_size=1)

    with pytest.raises(ValueError, match="PCP ranks reported different"):
        _run_engine_core_handshake(
            monkeypatch,
            connector,
            handshake_metadata=[
                {WorkerRanks(0, 0, 0): _AgentMetadata("nixl")},
                {WorkerRanks(0, 1, 0): _AgentMetadata("store")},
            ],
        )


def test_engine_passes_platform_worker_keys_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-tree platform workers that key metadata by their own tuples keep
    the previous merge, so their connectors receive those keys unchanged."""
    metadata_0 = _Metadata()
    metadata_1 = _Metadata()
    connector = _PPAwareConnector(pcp_size=2)

    _run_engine_core_handshake(
        monkeypatch,
        connector,
        handshake_metadata=[{(0, 0, 0): metadata_0}, {(0, 1, 0): metadata_1}],
    )

    assert connector.pp_aware_metadata == {(0, 0, 0): metadata_0, (0, 1, 0): metadata_1}
