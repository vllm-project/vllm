# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
)
from vllm.v1.engine import core as engine_core_module

pytestmark = pytest.mark.cpu_test


class _Metadata(KVConnectorHandshakeMetadata):
    pass


class _FakeExecutor:
    handshake_metadata_src: (
        list[dict[tuple[int, int], KVConnectorHandshakeMetadata] | None] | None
    )
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
    ) -> list[dict[tuple[int, int], KVConnectorHandshakeMetadata] | None] | None:
        self.handshake_calls += 1
        return self.handshake_metadata

    def init_kv_output_aggregator(self, connector: KVConnectorBase_V1) -> None:
        pass


def _run_engine_core_handshake(
    monkeypatch: pytest.MonkeyPatch,
    connector: KVConnectorBase_V1,
    *,
    handshake_metadata: (
        list[dict[tuple[int, int], KVConnectorHandshakeMetadata] | None] | None
    ),
) -> _FakeExecutor:
    class _FakeScheduler:
        def __init__(self, **kwargs: Any) -> None:
            self.connector = connector

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
    monkeypatch.setattr(
        engine_core_module,
        "MULTIMODAL_REGISTRY",
        SimpleNamespace(engine_receiver_cache_from_config=lambda vllm_config: None),
    )
    monkeypatch.setattr(engine_core_module, "freeze_gc_heap", lambda: None)
    monkeypatch.setattr(
        engine_core_module, "maybe_attach_gc_debug_callback", lambda: None
    )
    monkeypatch.setattr(engine_core_module, "enable_envs_cache", lambda: None)
    monkeypatch.setattr(engine_core_module, "get_hash_fn_by_name", lambda name: None)
    monkeypatch.setattr(engine_core_module, "init_none_hash", lambda hash_fn: None)
    monkeypatch.setattr(
        engine_core_module, "get_request_block_hasher", lambda *args: None
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank_local=0),
        scheduler_config=SimpleNamespace(
            get_scheduler_cls=lambda: _FakeScheduler,
            enable_chunked_prefill=False,
            async_scheduling=False,
        ),
        speculative_config=None,
        ec_transfer_config=None,
        max_concurrent_batches=1,
        model_config=SimpleNamespace(runner_type="generate"),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            prefix_caching_hash_algo="builtin",
        ),
    )

    engine_core_module.EngineCore(vllm_config, _FakeExecutor, log_stats=False)
    assert _FakeExecutor.last_instance is not None
    return _FakeExecutor.last_instance


class _LegacyConnector(KVConnectorBase_V1):
    def __init__(self) -> None:
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
    def __init__(self) -> None:
        super().__init__()
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
    """Engine core always asks workers for `(pp_rank, tp_rank)`-keyed metadata,
    then unwraps to `{tp_rank: metadata}` for a connector that has not opted
    into PP-aware handshake (single-PP producer, all `pp_rank == 0`)."""
    metadata_0 = _Metadata()
    metadata_1 = _Metadata()
    connector = _LegacyConnector()

    executor = _run_engine_core_handshake(
        monkeypatch,
        connector,
        handshake_metadata=[
            {(0, 0): metadata_0},
            None,
            {(0, 1): metadata_1},
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
            handshake_metadata=[{(0, 0): _Metadata()}, {(1, 0): _Metadata()}],
        )


def test_engine_passes_handshake_metadata_through_for_pp_aware_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PP-aware connector receives the full `(pp_rank, tp_rank)`-keyed dict
    unchanged."""
    metadata_0 = _Metadata()
    metadata_1 = _Metadata()
    connector = _PPAwareConnector()

    executor = _run_engine_core_handshake(
        monkeypatch,
        connector,
        handshake_metadata=[{(0, 0): metadata_0}, {(1, 0): metadata_1}],
    )

    assert executor.handshake_calls == 1
    assert connector.legacy_metadata is None
    assert connector.pp_aware_metadata == {
        (0, 0): metadata_0,
        (1, 0): metadata_1,
    }
