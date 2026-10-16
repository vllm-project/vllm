# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.manager import (
    CascadeTransferJob,
    PromotionTransferJob,
)

_CTX = ReqContext(req_id="test")


def _key(n: int):
    return make_offload_key(n.to_bytes(8, "big"), 0)


def _make_deferred_job(alloc_fn, keys=None):
    if keys is None:
        keys = [_key(1), _key(2)]
    return PromotionTransferJob(
        job_id=1,
        req_context=_CTX,
        keys=keys,
        chunk_ids=None,
        alloc_fn=alloc_fn,
    )


class TestPromotionTransferJob:
    def test_keys_raise_before_get(self):
        """Accessing .keys/.chunk_ids before get() raises ValueError."""
        job = _make_deferred_job(lambda keys, ctx: (list(keys), np.array([0, 1])))
        with pytest.raises(ValueError):
            _ = job.keys
        with pytest.raises(ValueError):
            _ = job.chunk_ids

    def test_successful_get(self):
        """get() returns the alloc_fn claim; properties expose it."""
        keys = [_key(1), _key(2)]
        chunk_ids = np.array([3, 7], dtype=np.int32)
        job = _make_deferred_job(lambda k, ctx: (list(k), chunk_ids), keys=keys)

        assert job.get() == (keys, chunk_ids)
        assert list(job.keys) == keys
        assert np.array_equal(job.chunk_ids, chunk_ids)

    def test_alloc_fn_receives_requested_keys_and_ctx(self):
        """alloc_fn is called with exactly the requested keys and req_context."""
        keys = [_key(10), _key(20)]
        received = {}

        def capturing_alloc(k, ctx):
            received["keys"] = list(k)
            received["ctx"] = ctx
            return (list(k), np.array([0, 1], dtype=np.int32))

        job = _make_deferred_job(capturing_alloc, keys=keys)
        job.get()

        assert received["keys"] == keys
        assert received["ctx"] is _CTX

    def test_allocation_failure_memoized(self):
        """A None alloc_fn result is memoized: get() stays None, alloc_fn runs
        once, and properties keep raising."""
        calls = []

        def failing_alloc(keys, ctx):
            calls.append(1)
            return None

        job = _make_deferred_job(failing_alloc)

        assert job.get() is None
        assert job.get() is None
        assert len(calls) == 1
        with pytest.raises(ValueError):
            _ = job.keys

    def test_get_is_idempotent(self):
        """Calling get() twice invokes alloc_fn exactly once."""
        call_count = [0]

        def counting_alloc(keys, ctx):
            call_count[0] += 1
            return (list(keys), np.array([0, 1], dtype=np.int32))

        job = _make_deferred_job(counting_alloc)
        first = job.get()
        assert job.get() == first
        assert call_count[0] == 1

    def test_malformed_deferred_job_rejected(self):
        """A deferred job without alloc_fn fails construction."""
        with pytest.raises(AssertionError):
            PromotionTransferJob(
                job_id=1,
                req_context=_CTX,
                keys=[_key(1)],
                chunk_ids=None,
            )

    def test_materialized_at_construction(self):
        """Equal-length keys/chunk_ids make the job materialized; get() returns
        the pre-set claim without an alloc_fn."""
        keys = [_key(1), _key(2)]
        chunk_ids = np.array([4, 5], dtype=np.int32)
        job = PromotionTransferJob(
            job_id=1, req_context=_CTX, keys=keys, chunk_ids=chunk_ids
        )

        assert job.get() == (keys, chunk_ids)
        assert list(job.keys) == keys
        assert np.array_equal(job.chunk_ids, chunk_ids)


class TestCascadeTransferJob:
    def test_claim_available_immediately(self):
        """Cascades are materialized at construction; get() is a claim readback."""
        keys = [_key(1)]
        chunk_ids = np.array([0], dtype=np.int32)
        job = CascadeTransferJob(
            job_id=1, req_context=_CTX, keys=keys, chunk_ids=chunk_ids
        )

        assert job.get() == (keys, chunk_ids)
        assert job.get() == (keys, chunk_ids)
        assert list(job.keys) == keys
        assert np.array_equal(job.chunk_ids, chunk_ids)
