# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for KV-cache layout selection in KV transfer connectors.

Focus:
- Raw-copy connectors should require HND for non-MLA models.
- The global layout decision should respect connector requirements.
"""

from unittest.mock import MagicMock

import pytest


def _make_config(*, use_mla: bool | None):
    """Minimal VllmConfig-like mock."""
    cfg = MagicMock()
    if use_mla is None:
        cfg.model_config = None
    else:
        cfg.model_config = MagicMock()
        cfg.model_config.use_mla = use_mla

    # Required by get_kv_connector_cache_layout
    cfg.kv_transfer_config = MagicMock()
    return cfg


# ---------------------------------------------------------------------------
# Connector-level behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "connector_path",
    [
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector.P2pNcclConnector",
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnector",
    ],
)
class TestRawCopyConnectorLayout:
    @staticmethod
    def _import_connector(connector_path: str):
        module_path, class_name = connector_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def test_non_mla_requires_hnd(self, connector_path):
        connector = self._import_connector(connector_path)
        cfg = _make_config(use_mla=False)
        assert connector.get_required_kvcache_layout(cfg) == "HND"

    def test_mla_requires_no_specific_layout(self, connector_path):
        connector = self._import_connector(connector_path)
        cfg = _make_config(use_mla=True)
        assert connector.get_required_kvcache_layout(cfg) is None

    def test_no_model_config_requires_no_specific_layout(self, connector_path):
        connector = self._import_connector(connector_path)
        cfg = _make_config(use_mla=None)
        assert connector.get_required_kvcache_layout(cfg) is None


# ---------------------------------------------------------------------------
# System-level decision path
# ---------------------------------------------------------------------------


class TestKvConnectorCacheLayoutDecision:
    def test_falls_back_to_nhd_when_connector_has_no_requirement(self, monkeypatch):
        from vllm.distributed.kv_transfer.kv_connector import utils as kv_utils
        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_kv_connector_cache_layout,
        )

        cfg = _make_config(use_mla=True)

        monkeypatch.setattr(kv_utils, "get_current_vllm_config", lambda: cfg)

        class _NoRequirementConnector:
            @classmethod
            def get_required_kvcache_layout(cls, _cfg):
                return None

        monkeypatch.setattr(
            kv_utils.KVConnectorFactory,
            "get_connector_class",
            lambda _kv_transfer_cfg: _NoRequirementConnector,
        )

        assert get_kv_connector_cache_layout() == "NHD"

    def test_uses_hnd_when_raw_copy_connector_requires_it(self, monkeypatch):
        from vllm.distributed.kv_transfer.kv_connector import utils as kv_utils
        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_kv_connector_cache_layout,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p import (
            p2p_nccl_connector,
        )

        p2p_nccl_connector_cls = p2p_nccl_connector.P2pNcclConnector

        cfg = _make_config(use_mla=False)

        monkeypatch.setattr(kv_utils, "get_current_vllm_config", lambda: cfg)

        monkeypatch.setattr(
            kv_utils.KVConnectorFactory,
            "get_connector_class",
            lambda _kv_transfer_cfg: p2p_nccl_connector_cls,
        )

        assert get_kv_connector_cache_layout() == "HND"
