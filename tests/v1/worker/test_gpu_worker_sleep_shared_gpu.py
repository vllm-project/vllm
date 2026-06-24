# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for Worker.sleep() on a shared GPU.

Worker.sleep() must not crash when device-global free memory drops between
the before/after reads of ``torch.cuda.mem_get_info`` -- for example when a
sibling PP/TP rank grows its NCCL buffers, or another model resident on the
same physical device allocates, concurrently with this process releasing its
own pool. Freed bytes are measured from the allocator's own process-scoped
pool, and a negative result is logged rather than asserted.
"""

from unittest.mock import MagicMock, patch

from vllm.v1.worker.gpu_worker import Worker


def _make_bare_worker() -> Worker:
    # Bypass __init__ (which requires a GPU + full VllmConfig); we only exercise
    # the sleep() accounting path.
    worker = object.__new__(Worker)
    worker._sleep_saved_buffers = {}
    return worker


def test_sleep_survives_shared_gpu_free_memory_drop():
    """device-global free DROPS across sleep (shared GPU) -> no crash.

    Pre-fix this raised ``AssertionError: Memory usage increased after
    sleeping.`` because freed_bytes was a device-global delta that went
    negative. Post-fix freed_bytes is the allocator pool delta (process
    scoped), so a concurrent foreign allocation can never make sleep crash.
    """
    worker = _make_bare_worker()

    allocator = MagicMock()
    # This process genuinely released 4 GiB from its own pool.
    allocator.get_current_usage.side_effect = [8 * 1024**3, 4 * 1024**3]

    total = 80 * 1024**3
    # Device-global free DROPS across the sleep call: a sibling rank /
    # co-located model grabbed memory while we freed ours. mem_get_info is
    # called twice by the pre-fix code (before + after) and once by the
    # post-fix code (after only); supply enough values for both. The
    # pre-fix device-global delta (40 - 45 = -5 GiB) is what triggered the
    # AssertionError this test guards against.
    mem_info_returns = iter(
        [
            (45 * 1024**3, total),  # pre-fix free_before (post-fix ignores)
            (40 * 1024**3, total),  # free_after_sleep
        ]
    )

    with (
        patch(
            "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
            return_value=allocator,
        ),
        patch(
            "vllm.v1.worker.gpu_worker.torch.cuda.mem_get_info",
            side_effect=lambda *a, **k: next(mem_info_returns),
        ),
    ):
        # Must NOT raise (pre-fix: AssertionError).
        worker.sleep(level=1)

    allocator.sleep.assert_called_once()
    # freed_bytes is computed from the process-scoped pool, not device-global.
    assert allocator.get_current_usage.call_count == 2


def test_sleep_reports_positive_freed_bytes_from_pool():
    """Normal case: allocator pool shrinks -> positive freed_bytes, no warning."""
    worker = _make_bare_worker()

    allocator = MagicMock()
    allocator.get_current_usage.side_effect = [10 * 1024**3, 2 * 1024**3]

    total = 80 * 1024**3
    mem_info_returns = iter([(60 * 1024**3, total), (50 * 1024**3, total)])

    with (
        patch(
            "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
            return_value=allocator,
        ),
        patch(
            "vllm.v1.worker.gpu_worker.torch.cuda.mem_get_info",
            side_effect=lambda *a, **k: next(mem_info_returns),
        ),
        patch("vllm.v1.worker.gpu_worker.logger") as mock_logger,
    ):
        worker.sleep(level=1)

    # Freed 8 GiB from the pool; no "increased after sleeping" warning.
    mock_logger.warning.assert_not_called()
    mock_logger.info.assert_called_once()
