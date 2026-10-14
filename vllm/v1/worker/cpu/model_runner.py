# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.worker.cpu.block_table import CPUBlockTables
from vllm.v1.worker.cpu.sampler import CPUSampler
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    @property
    def sampler_cls(self) -> type[CPUSampler]:
        return CPUSampler

    @property
    def block_tables_cls(self) -> type[CPUBlockTables]:
        return CPUBlockTables

    # TBD: Whether need to move this to Worker?
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        self.profile_run()
        logger.info("Warming up done.")
