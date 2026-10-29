# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the write-admission half of ``TouchOnHitFsTier``.

Run inside the engine container (it needs the installed ``vllm`` package and
the tier bind-mounted at ``/opt/kv_tier``)::

    MSYS_NO_PATHCONV=1 docker exec myia_vllm-medium-swift15-27b \
        sh -c 'cd /opt/kv_tier && python3 -m unittest -v test_lru_fs_tier'

These tests deliberately build the tier with ``object.__new__`` instead of
``__init__``: the real constructor needs an ``OffloadingSpec``, a live memoryview
and a thread pool, none of which exist in a unit test. Everything the override
touches is set explicitly below, so the test also serves as a checklist of the
state ``submit_store`` depends on.
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

import lru_fs_tier


class FakeMapper:
    def __init__(self, base: str) -> None:
        self.base = base

    def get_file_name(self, key: str) -> str:
        return os.path.join(self.base, key + ".bin")


class FakePool:
    def __init__(self) -> None:
        self.enqueued: list[tuple] = []

    def enqueue_store(self, job_id, n_tasks, tasks) -> None:
        self.enqueued.append((job_id, n_tasks, list(tasks)))


def make_tier(base: str, skip_rewrite: bool = True) -> lru_fs_tier.TouchOnHitFsTier:
    tier = object.__new__(lru_fs_tier.TouchOnHitFsTier)
    tier._ok = 0
    tier._miss = 0
    tier._err = 0
    tier._warned = 0
    tier._skip_rewrite = skip_rewrite
    tier._skipped_keys = 0
    tier._written_keys = 0
    tier.file_mapper = FakeMapper(base)
    tier._pool = FakePool()
    tier._store_job_keys = {}
    tier._job_block_counts = {}
    tier.events = []
    return tier


def make_job(keys, chunk_ids=None, job_id=7) -> TransferJob:
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array(chunk_ids if chunk_ids is not None else list(range(len(keys)))),
        is_promotion=False,
        req_context=mock.MagicMock(),
    )


class SubmitStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = self._tmp.name
        patcher = mock.patch.object(
            FileSystemTierManager, "submit_store", autospec=True
        )
        self.super_store = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _create(self, key: str) -> None:
        with open(os.path.join(self.base, key + ".bin"), "wb") as fh:
            fh.write(b"x")

    # -- all blocks present -------------------------------------------------

    def test_all_present_writes_nothing_but_finishes_the_job(self) -> None:
        tier = make_tier(self.base)
        for k in ("a", "b"):
            self._create(k)
        job = make_job(["a", "b"])

        tier.submit_store(job)

        self.super_store.assert_not_called()
        self.assertEqual(len(tier._pool.enqueued), 1)
        job_id, n_tasks, tasks = tier._pool.enqueued[0]
        self.assertEqual(job_id, job.job_id)
        self.assertEqual(n_tasks, len(tasks))
        tasks[0]()  # the no-op must be callable and must not raise
        self.assertEqual(tier._job_block_counts[job.job_id], 0)
        # The blocks ARE in the tier: the manager must still be told so.
        self.assertEqual(tier._store_job_keys[job.job_id], ["a", "b"])
        self.assertEqual(tier._skipped_keys, 2)
        self.assertEqual(tier._written_keys, 0)

    # -- nothing present ----------------------------------------------------

    def test_nothing_present_defers_to_upstream_unchanged(self) -> None:
        tier = make_tier(self.base)
        job = make_job(["a", "b"])

        tier.submit_store(job)

        self.super_store.assert_called_once()
        passed = self.super_store.call_args.args[1]
        self.assertIs(passed, job, "the untouched job object should reach upstream")
        self.assertEqual(tier._pool.enqueued, [])
        self.assertEqual(tier._skipped_keys, 0)
        self.assertEqual(tier._written_keys, 2)

    # -- partial ------------------------------------------------------------

    def test_partial_keeps_only_the_missing_blocks(self) -> None:
        tier = make_tier(self.base)
        self._create("b")
        job = make_job(["a", "b", "c"], chunk_ids=[10, 20, 30])

        tier.submit_store(job)

        self.super_store.assert_called_once()
        passed = self.super_store.call_args.args[1]
        self.assertEqual(list(passed.keys), ["a", "c"])
        self.assertEqual(list(passed.chunk_ids), [10, 30])
        self.assertEqual(passed.job_id, job.job_id, "job_id must survive the filter")
        self.assertEqual(tier._skipped_keys, 1)
        self.assertEqual(tier._written_keys, 2)
        # The caller's job must not be mutated under it.
        self.assertEqual(list(job.keys), ["a", "b", "c"])

    # -- fail-open ----------------------------------------------------------

    def test_probe_failure_falls_back_to_full_store(self) -> None:
        tier = make_tier(self.base)
        job = make_job(["a"])
        with mock.patch.object(
            lru_fs_tier.TouchOnHitFsTier,
            "_existing_flags",
            side_effect=OSError("boom"),
        ):
            tier.submit_store(job)

        self.super_store.assert_called_once()
        self.assertIs(self.super_store.call_args.args[1], job)
        self.assertEqual(tier._err, 1)
        self.assertEqual(tier._skipped_keys, 0)

    # -- kill switch --------------------------------------------------------

    def test_kill_switch_restores_write_through(self) -> None:
        tier = make_tier(self.base, skip_rewrite=False)
        self._create("a")
        job = make_job(["a"])

        with mock.patch.object(
            lru_fs_tier.TouchOnHitFsTier, "_existing_flags"
        ) as probe:
            tier.submit_store(job)
            probe.assert_not_called()

        self.super_store.assert_called_once()
        self.assertIs(self.super_store.call_args.args[1], job)
        self.assertEqual(tier._skipped_keys, 0)

    # -- degenerate ---------------------------------------------------------

    def test_empty_job_defers_to_upstream(self) -> None:
        tier = make_tier(self.base)
        tier.submit_store(make_job([]))
        self.super_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
