class DummyKVTransferConfig:
    def __init__(self, extra=None):
        self.kv_connector_extra_config = extra or {}

    def get_from_extra_config(self, key, default):
        return self.kv_connector_extra_config.get(key, default)


def test_env_overrides_backends(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        _resolve_nixl_backends,
    )

    cfg = DummyKVTransferConfig(extra={"backends": ["UCX"]})
    monkeypatch.setenv("VLLM_NIXL_DISAGGREGATION_BACKEND", "libfabric")

    assert _resolve_nixl_backends(cfg) == ["LIBFABRIC"]


def test_fallback_to_config_when_env_missing(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        _resolve_nixl_backends,
    )

    cfg = DummyKVTransferConfig(extra={"backends": ["UCX", "LIBFABRIC"]})
    monkeypatch.delenv("VLLM_NIXL_DISAGGREGATION_BACKEND", raising=False)

    assert _resolve_nixl_backends(cfg) == ["UCX", "LIBFABRIC"]
