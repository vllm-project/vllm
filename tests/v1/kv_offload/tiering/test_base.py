# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import LazyTransferJob

_CTX = ReqContext(req_id="test")


def _key(n: int):
    return make_offload_key(n.to_bytes(8, "big"), 0)


def _make_job(alloc_fn, keys=None):
    if keys is None:
        keys = [_key(1), _key(2)]
    return LazyTransferJob(
        job_id=1,
        _keys=keys,
        _chunk_ids=np.array([], dtype=np.int64),
        is_promotion=True,
        req_context=_CTX,
        primary_alloc_fn=alloc_fn,
    )


class TestLazyTransferJob:
    def test_not_materialized_before_materialize(self):
        """is_materialized() is False until materialize() is called."""
        job = _make_job(lambda keys, ctx: (list(keys), np.array([0, 1])))
        assert not job.is_materialized()

    def test_keys_raises_before_materialization(self):
        """Accessing .keys on an unmaterialized job raises AssertionError."""
        job = _make_job(lambda keys, ctx: (list(keys), np.array([0, 1])))
        with pytest.raises(AssertionError):
            _ = job.keys

    def test_chunk_ids_raises_before_materialization(self):
        """Accessing .chunk_ids on an unmaterialized job raises AssertionError."""
        job = _make_job(lambda keys, ctx: (list(keys), np.array([0, 1])))
        with pytest.raises(AssertionError):
            _ = job.chunk_ids

    def test_successful_materialization(self):
        """After materialize() with a returning alloc_fn, keys/chunk_ids are set."""
        keys = [_key(1), _key(2)]
        chunk_ids = np.array([3, 7], dtype=np.int64)
        job = _make_job(lambda k, ctx: (list(k), chunk_ids), keys=keys)

        job.materialize()

        assert job.is_materialized()
        assert job.lazy_success is True
        assert list(job.keys) == keys
        assert np.array_equal(job.chunk_ids, chunk_ids)

    def test_alloc_fn_receives_original_keys_and_ctx(self):
        """primary_alloc_fn is called with exactly _keys and req_context."""
        keys = [_key(10), _key(20)]
        received = {}

        def capturing_alloc(k, ctx):
            received["keys"] = list(k)
            received["ctx"] = ctx
            return (list(k), np.array([0, 1], dtype=np.int64))

        job = _make_job(capturing_alloc, keys=keys)
        job.materialize()

        assert received["keys"] == keys
        assert received["ctx"] is _CTX

    def test_allocation_failure_sets_lazy_success_false(self):
        """When alloc_fn returns None, lazy_success=False and keys/chunk_ids
        are empty.
        """
        job = _make_job(lambda keys, ctx: None)

        job.materialize()

        assert job.is_materialized()
        assert job.lazy_success is False
        assert list(job.keys) == []
        assert len(job.chunk_ids) == 0

    def test_materialize_is_idempotent(self):
        """Calling materialize() twice invokes alloc_fn exactly once."""
        call_count = [0]

        def counting_alloc(keys, ctx):
            call_count[0] += 1
            return (list(keys), np.array([0, 1], dtype=np.int64))

        job = _make_job(counting_alloc)
        job.materialize()
        job.materialize()

        assert call_count[0] == 1
        assert job.is_materialized()
