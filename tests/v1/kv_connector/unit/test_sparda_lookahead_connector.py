# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.sparda_lookahead_connector import (
    SparDALookaheadConnector,
    SparDALookaheadMetadata,
)

pytestmark = pytest.mark.cpu_test


def _config(*, extra_config=None):
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="SparDALookaheadConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config or {},
        )
    )


def test_connector_is_registered():
    cls = KVConnectorFactory.get_connector_class_by_name("SparDALookaheadConnector")

    assert cls is SparDALookaheadConnector


def test_wait_for_layer_load_schedules_next_layer():
    connector = SparDALookaheadConnector(
        _config(),
        KVConnectorRole.WORKER,
        kv_cache_config=None,
    )
    connector.bind_connector_metadata(
        SparDALookaheadMetadata(
            layer_names=("model.layers.0.self_attn", "model.layers.1.self_attn")
        )
    )

    connector.start_load_kv(forward_context=None)
    connector.wait_for_layer_load("model.layers.0.self_attn")

    assert connector.loaded_layers == frozenset({"model.layers.0.self_attn"})
    assert connector.scheduled_layers == (
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    )


def test_prefetch_distance_skips_ahead():
    connector = SparDALookaheadConnector(
        _config(extra_config={"prefetch_distance": 2}),
        KVConnectorRole.WORKER,
        kv_cache_config=None,
    )
    connector.bind_connector_metadata(
        SparDALookaheadMetadata(
            layer_names=(
                "model.layers.0.self_attn",
                "model.layers.1.self_attn",
                "model.layers.2.self_attn",
            ),
            prefetch_distance=2,
        )
    )

    connector.start_load_kv(forward_context=None)
    connector.wait_for_layer_load("model.layers.0.self_attn")

    assert connector.scheduled_layers == (
        "model.layers.0.self_attn",
        "model.layers.2.self_attn",
    )


def test_prefetch_plan_is_attached_to_next_layer():
    connector = SparDALookaheadConnector(
        _config(),
        KVConnectorRole.WORKER,
        kv_cache_config=None,
    )
    connector.bind_connector_metadata(
        SparDALookaheadMetadata(
            layer_names=("model.layers.0.self_attn", "model.layers.1.self_attn"),
            prefetch_plan={"model.layers.0.self_attn": (4, 7)},
        )
    )

    connector.start_load_kv(forward_context=None)
    connector.wait_for_layer_load("model.layers.0.self_attn")

    assert connector.scheduled_prefetches["model.layers.1.self_attn"] == (4, 7)


def test_piecewise_cudagraph_required_by_default():
    assert SparDALookaheadConnector.requires_piecewise_for_cudagraph({})
    assert not SparDALookaheadConnector.requires_piecewise_for_cudagraph(
        {"enable_layer_prefetch": False}
    )


def test_rejects_invalid_prefetch_distance():
    with pytest.raises(ValueError, match="prefetch_distance"):
        SparDALookaheadConnector(
            _config(extra_config={"prefetch_distance": 0}),
            KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
