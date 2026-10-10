# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ECZmqConnector staging area.

Staging is what keeps a received embedding alive between the receive thread and
the next engine step. Its job is to bound memory, hand each embedding over
exactly once, and report arrivals in a way the scheduler can count per rank.
"""

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.zmq.worker.staging import (
    EmbeddingStaging,
)

pytestmark = pytest.mark.cpu_test

_EMBEDDING = torch.zeros(8, 8, dtype=torch.float16)  # 128 bytes
_NBYTES = _EMBEDDING.numel() * _EMBEDDING.element_size()


def _staging(capacity_bytes=1024, ttl_s=60.0) -> EmbeddingStaging:
    return EmbeddingStaging(capacity_bytes=capacity_bytes, ttl_s=ttl_s)


def test_put_then_pop_hands_the_embedding_over_once():
    staging = _staging()

    assert staging.try_put("mm0", _EMBEDDING) is True
    assert staging.used_bytes == _NBYTES
    assert torch.equal(staging.pop("mm0"), _EMBEDDING)
    assert staging.pop("mm0") is None
    assert staging.used_bytes == 0


def test_budget_is_enforced_and_recovers_after_a_pop():
    staging = _staging(capacity_bytes=_NBYTES)

    assert staging.try_put("mm0", _EMBEDDING) is True
    assert staging.try_put("mm1", _EMBEDDING) is False

    staging.pop("mm0")

    assert staging.try_put("mm1", _EMBEDDING) is True


def test_duplicate_put_is_reported_once():
    """Two reports from one rank would make an item look ready on all ranks."""
    staging = _staging()

    staging.try_put("mm0", _EMBEDDING)
    staging.try_put("mm0", _EMBEDDING)

    assert staging.drain_arrivals() == {"mm0": 1}
    assert staging.used_bytes == _NBYTES


def test_arrivals_are_drained_only_once():
    staging = _staging()
    staging.try_put("mm0", _EMBEDDING)

    assert staging.drain_arrivals() == {"mm0": 1}
    assert staging.drain_arrivals() == {}


def test_expired_entries_are_dropped():
    """An embedding nobody comes for must not pin memory forever."""
    staging = _staging(ttl_s=0.0)
    staging.try_put("mm0", _EMBEDDING)

    staging.expire()

    assert staging.pop("mm0") is None
    assert staging.used_bytes == 0


def test_expiry_makes_room_for_new_arrivals():
    staging = _staging(capacity_bytes=_NBYTES, ttl_s=0.0)
    staging.try_put("mm0", _EMBEDDING)

    assert staging.try_put("mm1", _EMBEDDING) is True
    assert staging.pop("mm0") is None
    assert torch.equal(staging.pop("mm1"), _EMBEDDING)


def test_clear_releases_everything():
    staging = _staging()
    staging.try_put("mm0", _EMBEDDING)

    staging.clear()

    assert staging.used_bytes == 0
    assert staging.drain_arrivals() == {}
