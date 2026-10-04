# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.worker.cpu.sample.gumbel import warm_up as warm_up_gumbel
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    # TBD: Whether need to move this to Worker?
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        self.profile_run()
        # profile_run's dummy batch samples greedily, which never reaches the
        # sampler's compiled graph.
        warm_up_gumbel(self.max_num_reqs, self.vocab_size, self.device)
        logger.info("Warming up done.")
