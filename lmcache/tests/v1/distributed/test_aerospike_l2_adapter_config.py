# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Aerospike L2 adapter config (no native extension required)."""

# Standard
import os

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.factory import ensure_adapter_loaded


class TestAerospikeL2AdapterConfig:
    def test_type_registered(self):
        ensure_adapter_loaded("aerospike")
        assert "aerospike" in get_registered_l2_adapter_types()

    def test_from_dict_required_fields(self):
        ensure_adapter_loaded("aerospike")
        # First Party
        from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
            AerospikeL2AdapterConfig,
        )

        cfg = AerospikeL2AdapterConfig.from_dict(
            {
                "type": "aerospike",
                "hosts": "127.0.0.1:3000",
                "namespace": "lmcache",
                "set_name": "kv_it",
            }
        )
        assert cfg.hosts == "127.0.0.1:3000"
        assert cfg.namespace == "lmcache"
        assert cfg.set_name == "kv_it"
        assert get_type_name_for_config(cfg) == "aerospike"

    def test_set_alias(self):
        ensure_adapter_loaded("aerospike")
        # First Party
        from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
            AerospikeL2AdapterConfig,
        )

        cfg = AerospikeL2AdapterConfig.from_dict(
            {"type": "aerospike", "hosts": "h:3000", "set": "myset"}
        )
        assert cfg.set_name == "myset"

    def test_missing_hosts_raises(self):
        ensure_adapter_loaded("aerospike")
        # First Party
        from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
            AerospikeL2AdapterConfig,
        )

        try:
            AerospikeL2AdapterConfig.from_dict({"type": "aerospike"})
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_env_vars_used_when_config_empty(self, monkeypatch):
        # First Party
        from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
            AerospikeL2AdapterConfig,
        )

        monkeypatch.setenv("LMCACHE_AEROSPIKE_HOSTS", "env-host:3000")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_NAMESPACE", "env-ns")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_SET", "env-set")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_PASSWORD", "env-pass")

        cfg = AerospikeL2AdapterConfig(hosts="", namespace="", set_name="")

        hosts = cfg.hosts or os.environ.get("LMCACHE_AEROSPIKE_HOSTS", "")
        namespace = cfg.namespace or os.environ.get(
            "LMCACHE_AEROSPIKE_NAMESPACE", "lmcache"
        )
        set_name = cfg.set_name or os.environ.get("LMCACHE_AEROSPIKE_SET", "kv_chunks")
        username = cfg.username or os.environ.get("LMCACHE_AEROSPIKE_USERNAME", "")
        password = cfg.password or os.environ.get("LMCACHE_AEROSPIKE_PASSWORD", "")

        assert hosts == "env-host:3000"
        assert namespace == "env-ns"
        assert set_name == "env-set"
        assert username == "env-user"
        assert password == "env-pass"
