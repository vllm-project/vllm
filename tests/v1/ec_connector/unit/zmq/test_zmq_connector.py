# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ECZmqConnector shell.

The shell is what the engine talks to, so the test that matters walks one whole
engine step across the four connector objects a 1E1PD deployment has: the
producer's scheduler and worker, and the consumer's worker and scheduler.
"""

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.ec_transfer.ec_connector.zmq.connector import ECZmqConnector
from vllm.v1.outputs import ECConnectorOutput

pytestmark = pytest.mark.cpu_test

_EMBEDDING = torch.arange(64, dtype=torch.float16).reshape(8, 8)


class _Feature:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.modality = "image"
        self.data = None


class _Request:
    def __init__(self, *mm_hashes: str, request_id="req0"):
        self.mm_features = [_Feature(mm_hash) for mm_hash in mm_hashes]
        self.ec_transfer_params = None
        self.request_id = request_id


def test_the_connector_is_registered_under_its_name(vllm_config):
    config = vllm_config()

    connector_cls = ECConnectorFactory.get_connector_class(config.ec_transfer_config)

    assert connector_cls is ECZmqConnector


def test_each_role_builds_only_its_own_delegate(make_connector, vllm_config):
    scheduler_side = make_connector(ECConnectorRole.SCHEDULER, vllm_config())
    worker_side = make_connector(ECConnectorRole.WORKER, vllm_config())

    assert scheduler_side.connector_worker is None
    assert scheduler_side.connector_scheduler is not None
    assert worker_side.connector_scheduler is None
    assert worker_side.connector_worker is not None


def test_an_unknown_role_is_rejected(vllm_config):
    with pytest.raises(ValueError, match="Unknown ECConnectorRole"):
        ECZmqConnector(vllm_config(), role="not-a-role")


def test_a_producer_reports_its_items_in_the_response(make_connector, vllm_config):
    """The caller keys the consumer's request on what is reported here."""
    connector = make_connector(ECConnectorRole.SCHEDULER, vllm_config("ec_producer"))

    delay_free, params = connector.request_finished(_Request("mm0", "mm1"))

    assert delay_free is False
    assert params == {"ec_items": [{"mm_hash": "mm0"}, {"mm_hash": "mm1"}]}


def test_a_consumer_reports_nothing(make_connector, vllm_config):
    connector = make_connector(ECConnectorRole.SCHEDULER, vllm_config("ec_consumer"))

    assert connector.request_finished(_Request("mm0")) == (False, None)


def test_an_embedding_travels_from_producer_step_to_consumer_cache(
    make_connector, vllm_config, until
):
    consumer_config = vllm_config("ec_consumer")
    consumer_port = consumer_config.ec_transfer_config.ec_port
    producer_config = vllm_config(
        "ec_producer",
        extra_config={
            "ec_zmq_consumers": [{"host": "127.0.0.1", "port": consumer_port}]
        },
    )
    producer_sched = make_connector(ECConnectorRole.SCHEDULER, producer_config)
    producer_worker = make_connector(ECConnectorRole.WORKER, producer_config)
    consumer_worker = make_connector(ECConnectorRole.WORKER, consumer_config)
    consumer_sched = make_connector(ECConnectorRole.SCHEDULER, consumer_config)
    request = _Request("mm0")

    # Producer step: the scheduler routes the item, the worker pushes it.
    producer_sched.update_state_after_alloc(request, 0)
    producer_worker.bind_connector_metadata(
        producer_sched.build_connector_meta(scheduler_output=None)
    )
    producer_encoder_cache = {"mm0": _EMBEDDING}
    producer_worker.save_caches(producer_encoder_cache, "mm0")
    producer_worker.clear_connector_metadata()

    # Consumer step: the worker reports the arrival to its scheduler.
    worker_meta = None

    def arrived() -> bool:
        nonlocal worker_meta
        worker_meta = worker_meta or consumer_worker.build_connector_worker_meta()
        return worker_meta is not None

    assert until(arrived)
    consumer_sched.update_connector_output(
        ECConnectorOutput(ec_connector_worker_meta=worker_meta)
    )
    assert consumer_sched.has_cache_item("mm0") is True

    # Consumer step: the scheduler schedules the load, the worker performs it.
    consumer_sched.update_state_after_alloc(request, 0)
    consumer_worker.bind_connector_metadata(
        consumer_sched.build_connector_meta(scheduler_output=None)
    )
    consumer_encoder_cache: dict[str, torch.Tensor] = {}
    consumer_worker.start_load_caches(consumer_encoder_cache)
    consumer_worker.clear_connector_metadata()

    assert torch.equal(consumer_encoder_cache["mm0"], _EMBEDDING)

    # The producer's push completed, so the engine may quiesce again.
    finished_sending, _ = producer_worker.get_finished(set())
    assert finished_sending == {"mm0"}
    producer_sched.update_connector_output(
        ECConnectorOutput(finished_sending=finished_sending)
    )
    assert producer_sched.has_pending_push_work() is False
