# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the V2 GPU model runner's EC connector wrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.gpu.ec_connector import NO_OP_EC_CONNECTOR, ActiveECConnector

pytestmark = pytest.mark.cpu_test

WORKER_META = object()


def _scheduler_output() -> SimpleNamespace:
    return SimpleNamespace(
        ec_connector_metadata=ECConnectorMetadata(), finished_req_ids=frozenset()
    )


def _connector(
    encoder_cache: dict | None = None,
    is_producer: bool = True,
    is_consumer: bool = False,
) -> tuple[ActiveECConnector, MagicMock]:
    fake = MagicMock(spec=ECConnectorBase)
    fake.is_producer = is_producer
    fake.is_consumer = is_consumer
    fake.get_finished.return_value = (None, None)
    fake.build_connector_worker_meta.return_value = WORKER_META
    with patch("vllm.v1.worker.gpu.ec_connector.get_ec_transfer", return_value=fake):
        return ActiveECConnector(SimpleNamespace(), encoder_cache or {}), fake


@pytest.mark.parametrize(
    ("is_producer", "is_consumer"), [(True, False), (True, True), (False, True)]
)
def test_saves_newly_added_caches_for_every_producer(is_producer, is_consumer):
    """An ec_both node is also a producer: it must offload what it just computed."""
    encoder_cache = {"mm_old": None}
    connector, fake = _connector(encoder_cache, is_producer, is_consumer)

    with connector.maybe_get_output(_scheduler_output()):
        encoder_cache["mm_new"] = None

    saved = [call.kwargs["mm_hash"] for call in fake.save_caches.call_args_list]
    assert saved == (["mm_new"] if is_producer else [])
    assert fake.start_load_caches.called == is_consumer


def test_worker_meta_is_reported_on_context_exit():
    """Reported in the finally block, so is_empty() sees it only after the exit."""
    connector, fake = _connector()

    with connector.maybe_get_output(_scheduler_output()) as output:
        assert output.ec_connector_worker_meta is None

    assert output.ec_connector_worker_meta is WORKER_META
    assert fake.clear_connector_metadata.called


def test_no_forward_reports_without_running_the_model():
    connector, _ = _connector()

    output = connector.no_forward(_scheduler_output())

    assert output.ec_connector_output.ec_connector_worker_meta is WORKER_META

    empty = NO_OP_EC_CONNECTOR.no_forward(_scheduler_output())
    assert empty is EMPTY_MODEL_RUNNER_OUTPUT
