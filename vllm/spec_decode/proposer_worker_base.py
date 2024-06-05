from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposer
from vllm.worker.worker_base import WorkerBase


class ProposerWorkerBase(WorkerBase, SpeculativeProposer):
    """Interface for proposer workers"""

    @abstractmethod
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[Optional[List[SamplerOutput]], bool]:
        raise NotImplementedError

    def set_include_gpu_probs_tensor(self):
        """Implementation optional"""
        pass


class NonLLMProposerWorkerBase(ProposerWorkerBase, ABC):
    """Proposer worker which does not use a model with kvcache"""

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """get_spec_proposals is used to get the proposals"""
        return []

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """This is never called on the proposer, only the target model"""
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        return 0
