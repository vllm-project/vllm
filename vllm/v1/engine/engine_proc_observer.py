# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time

import psutil

from vllm.logger import init_logger

logger = init_logger(__name__)


class EngineProcObserver:
    """For tracking EngineCore orphaned status in background process."""

    @staticmethod
    def track_processes(pids_with_create_time: list[tuple[int, float]],
                        parent_pid: int, alive_check_interval: int):
        """
        Check every alive_check_interval seconds
        whether any EngineCore has been orphaned.
        """

        while True:
            if not psutil.pid_exists(parent_pid):
                logger.info(
                    "EngineCores have been orphaned... Proceeding to terminate."
                )

                for pid, create_time in pids_with_create_time:
                    try:
                        if psutil.Process(pid).create_time() == create_time:
                            os.kill(pid, 9)
                    except psutil.NoSuchProcess:
                        # Process already terminated, which is fine.
                        pass
                    except Exception as e:
                        logger.warning(
                            "Failed to kill orphaned process %d: %s", pid, e)

                return

            time.sleep(alive_check_interval)
