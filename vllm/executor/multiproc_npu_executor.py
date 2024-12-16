import os

import torch  # noqa

try:
    import torch_npu  # noqa
except ImportError:
    print("Failed to import torch_npu")

from vllm.executor.multiproc_gpu_executor import (
    MultiprocessingGPUExecutor, MultiprocessingGPUExecutorAsync)
from vllm.executor.npu_executor import NPUExecutor
from vllm.logger import init_logger
from vllm.utils import update_environment_variables

logger = init_logger(__name__)


class MultiprocessingNPUExecutor(MultiprocessingGPUExecutor, NPUExecutor):
    """Python multiprocessing-based multi-NPU executor"""

    def _check_executor_parameters(self):
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        # Set ASCEND_RT_VISIBLE_DEVICES for the driver, inherited by workers
        if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
            update_environment_variables({
                "ASCEND_RT_VISIBLE_DEVICES":
                (",".join(map(str, range(world_size))))
            })

        npu_device_count = torch.npu.device_count()
        # Use confusing message for more common TP-only case.
        assert tensor_parallel_size <= npu_device_count, (
            f"please set tensor_parallel_size ({tensor_parallel_size}) "
            f"to less than max local Ascend npu count ({npu_device_count})")

        assert world_size <= npu_device_count, (
            f"please ensure that world_size ({world_size}) "
            f"is less than than max local Ascend npu count "
            f"({npu_device_count})")


class MultiprocessingNPUExecutorAsync(MultiprocessingNPUExecutor,
                                      MultiprocessingGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
