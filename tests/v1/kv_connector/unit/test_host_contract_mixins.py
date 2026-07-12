# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the device-neutral KVConnector host-contract mixins added to
vLLM core. Validate the generic behavior in isolation (no device, no specific
connector):
- worker mixin: init is a no-op without a connector; handshake returns None.
- runner mixin: register is a no-op without a connector; the sync/writeback
  hooks default to no-op and are overridable.
"""

import types


def test_worker_mixin_init_noop_without_connector(monkeypatch):
    import vllm.v1.worker.kv_connector_worker_mixin as m

    calls: dict[str, object] = {}
    monkeypatch.setattr(
        m,
        "ensure_kv_transfer_initialized",
        lambda cfg, kvc: calls.setdefault("init", (cfg, kvc)),
    )
    m.KVConnectorWorkerMixin.maybe_initialize_kv_transfer("CFG", "KVC")
    assert calls["init"] == ("CFG", "KVC")


def test_worker_mixin_handshake_none_without_connector(monkeypatch):
    import vllm.v1.worker.kv_connector_worker_mixin as m

    monkeypatch.setattr(m, "has_kv_transfer_group", lambda: False)
    assert m.KVConnectorWorkerMixin.get_kv_connector_handshake_metadata() is None


def test_runner_mixin_register_noop_without_connector(monkeypatch):
    import vllm.v1.worker.kv_connector_model_runner_mixin as m

    monkeypatch.setattr(m, "has_kv_transfer_group", lambda: False)
    # Must not raise and must not call get_kv_transfer_group.
    called = {"n": 0}
    monkeypatch.setattr(
        m, "get_kv_transfer_group", lambda: called.__setitem__("n", called["n"] + 1)
    )
    m.KVConnectorModelRunnerMixin.register_kv_caches_with_connector({"l0": object()})
    assert called["n"] == 0


def test_runner_mixin_register_calls_connector_when_present(monkeypatch):
    import vllm.v1.worker.kv_connector_model_runner_mixin as m

    monkeypatch.setattr(m, "has_kv_transfer_group", lambda: True)
    seen: dict[str, object] = {}
    conn = types.SimpleNamespace(
        register_kv_caches=lambda kv: seen.setdefault("kv", kv)
    )
    monkeypatch.setattr(m, "get_kv_transfer_group", lambda: conn)
    kv = {"l0": object()}
    m.KVConnectorModelRunnerMixin.register_kv_caches_with_connector(kv)
    assert seen["kv"] is kv


def test_runner_mixin_sync_hooks_default_noop():
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorModelRunnerMixin as R,
    )

    class _Runner(R):
        pass

    r = _Runner()
    # Default hooks are no-ops (return None, don't raise) regardless of args.
    assert r.sync_kv_caches_before_save(object()) is None
    assert r.writeback_kv_caches_after_load(object()) is None


def test_runner_mixin_sync_hooks_overridable():
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorModelRunnerMixin as R,
    )

    calls = []

    class _Runner(R):
        def sync_kv_caches_before_save(self, so):
            calls.append(("sync", so))

        def writeback_kv_caches_after_load(self, so):
            calls.append(("writeback", so))

    r = _Runner()
    r.sync_kv_caches_before_save("A")
    r.writeback_kv_caches_after_load("B")
    assert calls == [("sync", "A"), ("writeback", "B")]
