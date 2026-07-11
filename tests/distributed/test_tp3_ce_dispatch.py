# SPDX-License-Identifier: Apache-2.0

from vllm.distributed.parallel_state import _should_use_tp3_ce


def test_tp3_ce_requires_known_large_logical_batch():
    args = (3, 2, 5120, 3)
    assert not _should_use_tp3_ce(None, *args)
    assert not _should_use_tp3_ce(1365, *args)
    assert _should_use_tp3_ce(1366, *args)


def test_tp3_ce_rejects_unsupported_layouts():
    assert not _should_use_tp3_ce(4096, 1, 1, 5120, 3)
    assert not _should_use_tp3_ce(4096, 1, 2, 4096, 3)
    assert not _should_use_tp3_ce(4096, 1, 2, 5120, 2)
