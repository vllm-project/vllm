# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.sampler import CPUSampler
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    @property
    def sampler_cls(self) -> type[Sampler]:
        return super().sampler_cls if HAS_TRITON else CPUSampler

    # TBD: Whether need to move this to Worker?
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        self.profile_run()
        logger.info("Warming up done.")
