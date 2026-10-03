# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side watchdog for wedged device kernels.

A device kernel that stops making forward progress blocks its worker in a
CUDA synchronization indefinitely (e.g. the FlashMLA sparse-decode MLA
kernel spinning in the ptxas-lowered `tcgen05.alloc.cta_group::2` TMEM
allocation protocol, vllm-project/vllm#51035). The executor-side RPC
timeout only observes the output rank, so a wedged non-output rank used to
stay silent until an unrelated collective asserted elsewhere. Arming this
watchdog around model execution gives the wedged rank its own diagnostic.
"""

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from vllm.logger import init_logger

logger = init_logger(__name__)


@contextmanager
def wedged_kernel_watchdog(timeout_seconds: float, context: str) -> Iterator[None]:
    """Log a diagnostic if the wrapped section outlives `timeout_seconds`.

    Re-warns every `timeout_seconds` while the section is still running so
    the evidence survives log rotation. A non-positive timeout disables the
    watchdog.
    """
    if timeout_seconds <= 0:
        yield
        return

    done = threading.Event()

    def _watch() -> None:
        start = time.monotonic()
        while not done.wait(timeout_seconds):
            logger.error(
                "%s has been running for %.0f s without completing: a device "
                "kernel may be wedged (spinning with no forward progress, "
                "e.g. the FlashMLA sparse-decode MLA TMEM allocation stall, "
                "see vllm-project/vllm#51035). Check nvidia-smi for 100%% "
                "utilization at idle-level power, and attach cuda-gdb to "
                "pid %d to capture the resident kernel.",
                context,
                time.monotonic() - start,
                os.getpid(),
            )

    thread = threading.Thread(target=_watch, daemon=True, name="WedgedKernelWatchdog")
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()
