# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.model_state import (
    CPUDefaultModelState,
    CPUMambaHybridModelState,
)
from vllm.v1.worker.cpu.sampler import CPUSampler
from vllm.v1.worker.cpu.utils import CPUKVBlockZeroer
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.utils import KVBlockZeroer

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    @property
    def sampler_cls(self) -> type[Sampler]:
        return super().sampler_cls if HAS_TRITON else CPUSampler

    @property
    def model_state_cls(self) -> type[ModelState]:
        model_state_cls = super().model_state_cls
        if not HAS_TRITON:
            if model_state_cls is DefaultModelState:
                return CPUDefaultModelState
            if model_state_cls is MambaHybridModelState:
                return CPUMambaHybridModelState
        return model_state_cls

    @property
    def kv_block_zeroer_cls(self) -> type[KVBlockZeroer]:
        return super().kv_block_zeroer_cls if HAS_TRITON else CPUKVBlockZeroer

    # TBD: Whether need to move this to Worker?
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        self.profile_run()
        logger.info("Warming up done.")
