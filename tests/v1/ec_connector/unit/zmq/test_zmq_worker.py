# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the worker side of the ECZmqConnector.

These run a real producer and a real consumer over real TCP sockets, because
the behaviour worth guarding is exactly what crosses the process boundary: the
embedding a consumer ends up with, who sends it, and what happens when a
delivery cannot be completed.
"""

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.zmq.common import (
    ECZmqConnectorMetadata,
    ZmqDst,
)

pytestmark = pytest.mark.cpu_test

_EMBEDDING = torch.arange(64, dtype=torch.float16).reshape(8, 8)


def _dst(worker) -> ZmqDst:
    """The destination that addresses `worker`'s receive sockets."""
    options = worker._options
    return ZmqDst(
        host="127.0.0.1",
        port=options.recv_port_base,
        num_ranks=options.num_recv_ranks,
    )


def _send(producer, consumer, mm_hash: str = "mm0", embedding=_EMBEDDING) -> None:
    meta = ECZmqConnectorMetadata(sends={mm_hash: [_dst(consumer)]})
    producer.save_caches({mm_hash: embedding}, mm_hash, meta)


def test_embedding_reaches_the_consumer_encoder_cache(make_worker, until):
    consumer = make_worker("ec_consumer")
    producer = make_worker("ec_producer")

    _send(producer, consumer)

    assert until(lambda: consumer.build_worker_meta() is not None)
    encoder_cache: dict[str, torch.Tensor] = {}
    consumer.start_load_caches(encoder_cache, ECZmqConnectorMetadata(loads=["mm0"]))

    assert torch.equal(encoder_cache["mm0"], _EMBEDDING)


def test_arrival_is_reported_once_per_rank(make_worker, until):
    """The scheduler counts these reports, so a rank must report exactly once."""
    consumer = make_worker("ec_consumer")
    producer = make_worker("ec_producer")

    _send(producer, consumer)

    assert until(lambda: consumer._staging.used_bytes > 0)
    assert consumer.build_worker_meta().staged == {"mm0": 1}
    assert consumer.build_worker_meta() is None


def test_producer_reports_completed_sends(make_worker, until):
    """`has_pending_push_work` depends on these, so every send must report."""
    consumer = make_worker("ec_consumer")
    producer = make_worker("ec_producer")

    _send(producer, consumer)

    assert until(lambda: producer.get_finished() == {"mm0"})


def test_every_consumer_rank_gets_its_own_copy(make_worker, vllm_config, until):
    """All TP ranks read the encoder cache, so all of them must be fed."""
    config = vllm_config("ec_consumer", tensor_parallel_size=2)
    rank0 = make_worker(tp_rank=0, tp_size=2, vllm_config=config)
    rank1 = make_worker(tp_rank=1, tp_size=2, vllm_config=config)
    producer = make_worker("ec_producer")

    _send(producer, rank0)

    for consumer in (rank0, rank1):
        assert until(lambda c=consumer: c._staging.used_bytes > 0)
        encoder_cache: dict[str, torch.Tensor] = {}
        consumer.start_load_caches(encoder_cache, ECZmqConnectorMetadata(loads=["mm0"]))
        assert torch.equal(encoder_cache["mm0"], _EMBEDDING)


def test_only_the_first_rank_sends(make_worker, until):
    """Every rank holds the same encoder output; sending it once is enough."""
    consumer = make_worker("ec_consumer")
    producer = make_worker("ec_producer", tp_rank=1, tp_size=2)

    _send(producer, consumer)

    assert not until(lambda: consumer._staging.used_bytes > 0, timeout=0.5)
    assert producer.get_finished() is None


def test_nothing_is_sent_without_a_destination(make_worker, until):
    consumer = make_worker("ec_consumer")
    producer = make_worker("ec_producer")

    producer.save_caches({"mm0": _EMBEDDING}, "mm0", ECZmqConnectorMetadata())

    assert not until(lambda: consumer._staging.used_bytes > 0, timeout=0.5)
    assert producer.get_finished() is None


def test_a_missing_encoder_output_still_completes_the_send(make_worker):
    """Otherwise the scheduler would wait for a push that can never happen."""
    consumer = make_worker("ec_consumer")
    producer = make_worker("ec_producer")

    meta = ECZmqConnectorMetadata(sends={"mm0": [_dst(consumer)]})
    producer.save_caches({}, "mm0", meta)

    assert producer.get_finished() == {"mm0"}


def test_load_of_an_absent_embedding_is_not_fatal(make_worker):
    """A load whose staged entry expired must not take the engine down."""
    consumer = make_worker("ec_consumer")
    encoder_cache: dict[str, torch.Tensor] = {}

    consumer.start_load_caches(encoder_cache, ECZmqConnectorMetadata(loads=["mm0"]))

    assert encoder_cache == {}


def test_load_leaves_an_already_cached_hash_alone(make_worker, until):
    consumer = make_worker("ec_consumer")
    producer = make_worker("ec_producer")
    _send(producer, consumer)
    assert until(lambda: consumer._staging.used_bytes > 0)

    local = torch.ones_like(_EMBEDDING)
    encoder_cache = {"mm0": local}
    consumer.start_load_caches(encoder_cache, ECZmqConnectorMetadata(loads=["mm0"]))

    assert encoder_cache["mm0"] is local


def test_a_producer_only_worker_does_not_listen(make_worker):
    producer = make_worker("ec_producer")

    assert producer._recv_socket is None
    assert producer.build_worker_meta() is None


def test_an_oversized_embedding_is_rejected_not_retried_forever(make_worker, until):
    """A payload larger than the whole budget can never be staged."""
    consumer = make_worker("ec_consumer", extra_config={"ec_zmq_staging_bytes": 8})
    producer = make_worker("ec_producer")

    _send(producer, consumer)

    assert until(lambda: producer.get_finished() == {"mm0"})
    assert not until(lambda: consumer._staging.used_bytes > 0, timeout=0.5)
