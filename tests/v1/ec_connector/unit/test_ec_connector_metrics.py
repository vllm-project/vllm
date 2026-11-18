# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the connector-agnostic EC connector metrics plumbing.

Covers only the base infra: no concrete connector type implements these
hooks yet, so every assertion here is about the documented no-op defaults
and the container contracts, mirroring the equivalent KV connector tests.
"""

from unittest.mock import Mock

import pytest

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnector,
)
from vllm.distributed.ec_transfer.ec_connector.metrics import (
    ECConnectorLogging,
    ECConnectorProm,
    ECConnectorStats,
)
from vllm.v1.outputs import ECConnectorOutput

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def example_connector(tmp_path):
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.get_from_extra_config = Mock(return_value=str(tmp_path))
    config.ec_transfer_config.is_ec_producer = True
    config.ec_transfer_config.is_ec_consumer = False
    return ECExampleConnector(vllm_config=config, role=ECConnectorRole.SCHEDULER)


def test_base_stats_methods_raise_not_implemented():
    """The base container has no aggregation logic of its own."""
    stats = ECConnectorStats()

    with pytest.raises(NotImplementedError):
        stats.reset()
    with pytest.raises(NotImplementedError):
        stats.aggregate(stats)
    with pytest.raises(NotImplementedError):
        stats.reduce()
    with pytest.raises(NotImplementedError):
        stats.is_empty()


def test_connector_hooks_default_to_none(example_connector):
    """No connector type overrides these yet, so every hook is a no-op."""
    assert example_connector.get_ec_connector_stats() is None
    assert ECExampleConnector.build_ec_connector_stats() is None
    assert (
        ECExampleConnector.build_prom_metrics(
            vllm_config=Mock(spec=VllmConfig),
            metric_types={},
            labelnames=[],
            per_engine_labelvalues={},
        )
        is None
    )


def test_ec_connector_output_is_empty_reflects_stats_field():
    output = ECConnectorOutput()
    assert output.is_empty()

    output.ec_connector_stats = ECConnectorStats(data={"x": 1})
    assert not output.is_empty()


def test_logging_and_prom_are_no_ops_without_ec_transfer_config():
    """A normal run with no EC connector configured must stay silent."""
    logging = ECConnectorLogging(ec_transfer_config=None)
    logging.log()  # Should not raise, nothing to log.

    vllm_config = Mock(spec=VllmConfig)
    vllm_config.ec_transfer_config = None
    prom = ECConnectorProm(vllm_config, labelnames=[], per_engine_labelvalues={})
    assert prom.prom_metrics is None
    prom.observe({})  # Should not raise.
