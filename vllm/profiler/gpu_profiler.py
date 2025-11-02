# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.cuda.profiler as cuda_profiler

from vllm.logger import init_logger

logger = init_logger(__name__)


class CudaProfilerWrapper:
    def __init__(self) -> None:
        self._profiler_running = False

    def start(self) -> None:
        try:
            cuda_profiler.start()
            self._profiler_running = True
            logger.info_once("Started CUDA profiler")
        except Exception as e:
            logger.warning_once("Failed to start CUDA profiler: %s", e)

    def stop(self) -> None:
        if self._profiler_running:
            try:
                cuda_profiler.stop()
                logger.info_once("Stopped CUDA profiler")
            except Exception as e:
                logger.warning_once("Failed to stop CUDA profiler: %s", e)
            finally:
                self._profiler_running = False

    def shutdown(self) -> None:
        """Ensure profiler is stopped when shutting down."""
        self.stop()
