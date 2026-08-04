# SPDX-License-Identifier: Apache-2.0
"""Regression test for LMCache#3318.

The vLLM v1 adapter previously asserted
``len(slot_mapping) == len(token_ids)`` inside ``wait_for_save``. When the
state desynced (e.g. upstream allocation failure or preemption-induced
mismatch) the assertion fired as an unhandled ``AssertionError`` and
killed the entire EngineCore process for every connected user.

The fix replaces the assert with a logged ``continue`` so the engine
stays alive and only the affected request's save is dropped. This test
locks in that behavior by feeding ``wait_for_save`` a request whose
``slot_mapping`` and ``token_ids`` lengths disagree and asserting:

1. ``wait_for_save`` does not raise.
2. A warning is emitted naming the request id and both lengths.
3. ``lmcache_engine.store`` is not called for the desynced request
   (the save is dropped, not silently corrupted).
4. ``lookup_unpin`` is still called so the pin count stays balanced.
"""

# Standard
from types import SimpleNamespace
import logging

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    SaveSpec,
)


class _FakeParent:
    def __init__(self, metadata: LMCacheConnectorMetadata) -> None:
        self._connector_metadata = metadata

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata:
        return self._connector_metadata


class _FakeEngine:
    """Records calls to ``lookup_unpin`` and ``store`` so the test can
    assert which paths fired."""

    def __init__(self) -> None:
        self.unpinned: list[str] = []
        self.store_calls: list[str] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store(self, *args, **kwargs) -> None:
        self.store_calls.append(kwargs.get("req_id", "<unknown>"))


def _make_desync_request(
    req_id: str, token_ids_len: int, slot_mapping_len: int
) -> SimpleNamespace:
    """Build a request whose ``token_ids`` and ``slot_mapping`` lengths
    disagree, simulating a state desync."""
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(range(token_ids_len)),
        slot_mapping=torch.arange(slot_mapping_len, dtype=torch.long),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
        disagg_spec=None,
        is_last_prefill=True,
        request_configs=None,
    )


def _make_connector(
    requests: list[SimpleNamespace],
) -> tuple[LMCacheConnectorV1Impl, _FakeEngine]:
    metadata = LMCacheConnectorMetadata(requests=requests)  # type: ignore[arg-type]
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    # ``lmcache_engine`` is a read-only property backed by ``self._manager``;
    # inject the fake engine through the manager so the property resolves to it.
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=engine
    )
    connector.kv_role = "kv_producer"
    connector.use_layerwise = False
    connector.enable_blending = False
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector.config = SimpleNamespace(pd_bidirectional=False)
    return connector, engine


def test_wait_for_save_skips_desynced_request_and_keeps_engine_alive() -> None:
    """Length mismatch must drop only the affected request's save, log a
    warning, and let ``wait_for_save`` return normally.

    Regression for https://github.com/LMCache/LMCache/issues/3318.
    """
    # lmcache's ``init_logger`` sets ``propagate = False`` on the adapter
    # logger so its records do not reach pytest's ``caplog`` (which
    # attaches to the root logger). Toggling ``propagate`` is fragile --
    # any lazy import that re-runs ``init_logger`` resets it. Attach a
    # local handler directly to the named logger instead so we capture
    # the warning regardless of how lmcache configures propagation.
    captured_records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    adapter_logger = logging.getLogger("lmcache.integration.vllm.vllm_v1_adapter")
    # ``init_logger`` sets the logger level from ``LMCACHE_LOG_LEVEL`` (default
    # INFO). If a prior import set it above WARNING, ``logger.warning`` would be
    # filtered before reaching our handler. Force WARNING for the duration of
    # the test and restore the original level in ``finally``.
    original_level = adapter_logger.level
    adapter_logger.setLevel(logging.WARNING)
    adapter_logger.addHandler(handler)
    try:
        desync_req = _make_desync_request(
            "req-desync", token_ids_len=4, slot_mapping_len=3
        )
        connector, engine = _make_connector([desync_req])

        connector.wait_for_save()

        # 1. lookup_unpin still ran (pin balance preserved)
        assert engine.unpinned == ["req-desync"]

        # 2. store was NOT called for the desynced request (save dropped)
        assert engine.store_calls == []

        # 3. A warning was emitted naming the request and both lengths
        warnings = [r for r in captured_records if r.levelno == logging.WARNING]
        assert any(
            "req-desync" in r.getMessage()
            and "slot_mapping=3" in r.getMessage()
            and "token_ids=4" in r.getMessage()
            for r in warnings
        ), (
            "Expected desync warning naming req-desync; "
            f"got {[r.getMessage() for r in warnings]}"
        )
    finally:
        adapter_logger.removeHandler(handler)
        adapter_logger.setLevel(original_level)
