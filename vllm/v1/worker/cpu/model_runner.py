# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.cpu.block_table import CPUBlockTables
from vllm.v1.worker.cpu.model_state import (
    CPUDefaultModelState,
    CPUEncoderOnlyModelState,
    CPUMambaHybridModelState,
)
from vllm.v1.worker.cpu.sampler import CPUSampler, CPUWatermarkSampler
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker
from vllm.v1.worker.cpu.utils import CPUKVBlockZeroer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_decoder import EncoderDecoderModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.utils import KVBlockZeroer

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    @property
    def block_tables_cls(self) -> type[BlockTables]:
        return super().block_tables_cls if HAS_TRITON else CPUBlockTables

    @property
    def sampler_cls(self) -> type[Sampler]:
        return super().sampler_cls if HAS_TRITON else CPUSampler

    @property
    def watermark_sampler_cls(self) -> type[GPUWatermarkSampler]:
        return super().watermark_sampler_cls if HAS_TRITON else CPUWatermarkSampler

    @property
    def model_state_cls(self) -> type[ModelState]:
        model_state_cls = super().model_state_cls
        if not HAS_TRITON:
            if model_state_cls is DefaultModelState:
                return CPUDefaultModelState
            if model_state_cls is MambaHybridModelState:
                return CPUMambaHybridModelState
            if model_state_cls is EncoderOnlyModelState:
                return CPUEncoderOnlyModelState
            if model_state_cls is EncoderDecoderModelState:
                return model_state_cls
            raise ValueError(
                "CPU Model Runner V2 without Triton does not support model state "
                f"{model_state_cls.__name__}."
            )
        return model_state_cls

    @property
    def structured_outputs_worker_cls(self) -> type[StructuredOutputsWorker]:
        if HAS_TRITON:
            return super().structured_outputs_worker_cls
        return CPUStructuredOutputsWorker

    @property
    def kv_block_zeroer_cls(self) -> type[KVBlockZeroer]:
        return super().kv_block_zeroer_cls if HAS_TRITON else CPUKVBlockZeroer

    # TBD: Whether need to move this to Worker?
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        self.profile_run()
        logger.info("Warming up done.")
