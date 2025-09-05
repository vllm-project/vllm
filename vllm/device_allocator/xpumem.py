# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import torch

from vllm.logger import init_logger
from vllm.utils import GiB_bytes
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class XpuMemAllocator:
    """
    A singleton class that provides same interface as CuMemAllocator for
    sleep mode support
    """
    instance: "XpuMemAllocator" = None

    @staticmethod
    def get_instance() -> "XpuMemAllocator":
        """
        XpuMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        if XpuMemAllocator.instance is None:
            XpuMemAllocator.instance = XpuMemAllocator()
        return XpuMemAllocator.instance

    def __init__(self):
        self.model_on_cpu = False
        self.sleep_level = 1
        self.saved_kv_cache_config = None
        self.sleep_saved_buffers = {}

    def sleep(
        self,
        offload_tags: Optional[Union[tuple[str, ...], str]] = None,
        level: int = 1,
        model_runner: Optional[GPUModelRunner] = None,
    ) -> None:
        """Put the worker into sleep mode to reduce memory usage.

        Unlike GPU workers that use custom memory allocators, XPU workers
        use a simpler approach of moving model to CPU and clearing KV cache.

        :param offload_tags: The tags of the memory allocation that will be
            offloaded. Not used for xpu
        :param level: sleep level(1 or 2)
        :param model_runner: perform sleep/wake_up for non-cuda platforms
        """
        if self.model_on_cpu:
            logger.warning("Worker is already in sleep mode")
            return
        self.sleep_level = level

        free_bytes_before_sleep, total_bytes = torch.xpu.mem_get_info()
        logger.warning(
            "Entering sleep mode: free bytes before sleep: %.2f GiB, "
            "total bytes: %.2f GiB", free_bytes_before_sleep / GiB_bytes,
            total_bytes / GiB_bytes)

        # Move model to CPU (if model is loaded)
        if hasattr(model_runner, 'model') and model_runner.model is not None:
            logger.info("Moving model to CPU for sleep mode")
            model_runner.model.to("cpu")
            model = model_runner.model
            if self.sleep_level == 2:
                self.sleep_saved_buffers = {
                    name: buffer.cpu().clone()
                    for name, buffer in model.named_buffers()
                }
            self.model_on_cpu = True
        else:
            logger.info("Model not loaded yet, skipping model move to CPU")

        free_bytes_after_model_sleep, total_bytes = torch.xpu.mem_get_info()
        logger.warning(
            "Free bytes after model sleep: %.2f GiB, "
            "total bytes: %.2f GiB", free_bytes_after_model_sleep / GiB_bytes,
            total_bytes / GiB_bytes)

        # Clear KV cache
        if hasattr(model_runner, 'kv_caches'):
            logger.info("Clearing KV cache for sleep mode")
            self.saved_kv_cache_config = getattr(model_runner,
                                                 'kv_cache_config', None)
            # Clear KV cache tensors
            if hasattr(model_runner, 'kv_caches'):
                del model_runner.kv_caches
                for layer_name in (model_runner.vllm_config.compilation_config.
                                   static_forward_context):
                    model_runner.vllm_config.compilation_config.static_forward_context[
                        layer_name].kv_cache = None
                model_runner.kv_caches = []
        # Clear XPU cache
        torch.xpu.empty_cache()
        free_bytes_after_sleep, total_bytes = torch.xpu.mem_get_info()
        logger.warning(
            "Free bytes after kv cache cleanup: %.2f GiB, "
            "total bytes: %.2f GiB", free_bytes_after_sleep / GiB_bytes,
            total_bytes / GiB_bytes)

        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total_bytes - free_bytes_after_sleep

        logger.warning(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self,
                tags: Optional[list[str]] = None,
                model_runner: Optional[GPUModelRunner] = None) -> None:
        """Wake up the worker from sleep mode.

        Moves the model back to XPU and optionally reinitializes KV cache.
        For sleep_level 2, will do model weights reload

        Args:
            tags: Optional list of tags (kept for interface compatibility)
            level: sleep level
            model_runner: perform sleep/wake_up for xpu
        """
        if tags is None:
            tags = ["weights", "kv_cache"]

        free_bytes_before_wakeup, total_bytes = torch.xpu.mem_get_info()
        logger.warning(
            "free bytes before wake up: %.2f GiB, "
            "total bytes: %.2f GiB", free_bytes_before_wakeup / GiB_bytes,
            total_bytes / GiB_bytes)

        if "weights" in tags:
            if not self.model_on_cpu:
                logger.warning("Worker is not in sleep mode")
                return

            logger.info(
                "Waking up worker: moving model back to XPU or reload it")

            # Move model back to XPU or reload it
            if hasattr(model_runner,
                       'model') and model_runner.model is not None:

                if self.model_on_cpu:
                    model_runner.model.to("xpu")
                    self.model_on_cpu = False
                if self.sleep_level == 2:
                    model_runner.reload_weights()
            else:

                logger.info("Model not loaded yet, skipping model move to XPU")
        free_bytes_after_model_wakeup, total_bytes = torch.xpu.mem_get_info()
        logger.warning(
            "free bytes after model wake up: %.2f GiB, "
            "total bytes: %.2f GiB", free_bytes_after_model_wakeup / GiB_bytes,
            total_bytes / GiB_bytes)

        if "kv_cache" in tags:
            # If KV cache was cleared, it will need to be reinitialized
            # by the engine when needed
            if self.saved_kv_cache_config is not None:
                model_runner.initialize_kv_cache_tensors(
                    self.saved_kv_cache_config)
            self.saved_kv_cache_config = None

        if len(self.sleep_saved_buffers):
            model = model_runner.model
            for name, buffer in model.named_buffers():
                if name in self.sleep_saved_buffers:
                    buffer.data.copy_(self.sleep_saved_buffers[name].data)
            self.sleep_saved_buffers = {}
        logger.info("Worker wake up completed")
        free_bytes_after_kv_wakeup, total_bytes = torch.xpu.mem_get_info()
        logger.warning(
            "free bytes after kv wake up: %.2f GiB, "
            "total bytes: %.2f GiB", free_bytes_after_kv_wakeup / GiB_bytes,
            total_bytes / GiB_bytes)
