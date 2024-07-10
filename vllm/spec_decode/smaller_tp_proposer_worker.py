from typing import List, Optional, Set, Tuple

import torch

from vllm.distributed.parallel_state import (get_tp_group,
                                             init_model_parallel_group,
                                             patch_tensor_parallel_group)
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase

logger = init_logger(__name__)


class SmallerTpProposerWorker(ProposerWorkerBase):
    """Class which allows a speculative draft model to run with smaller tensor
    parallel degree than target model.
    This reduces the communication overhead of small draft models.

    To implement this feature, this class differs behavior based on is_dummy
    flag, where dummy means worker that does not participate draft generation.
    Participating workers use a smaller tp group by patching vLLM's tensor
    parallel group temporarily during forward passes of draft models.
    """

    @classmethod
    def maybe_wrap_worker(cls, worker, draft_tensor_parallel_size: int,
                          target_tensor_parallel_size: int):
        """Wrap the worker in a SmallerTpProposerWorker if necessary.
        """
        if draft_tensor_parallel_size == target_tensor_parallel_size:
            return worker

        # gpu ranks that will generate draft tokens together
        draft_ranks = list(range(draft_tensor_parallel_size))

        logger.info("Wrapping {%s} in {%s}", type(worker), cls)
        return cls(worker, draft_ranks)

    def __init__(self, worker: MultiStepWorker, draft_ranks: List[int]):
        """Create a SmallerTpProposerWorker.

        Args:
            worker (MultiStepWorker): an actual worker wrapped with this class
            draft_ranks (List[int]): if this value is given, only the GPU ranks
            written in this value participate in draft generation
        """
        self._worker = worker
        self._draft_ranks = draft_ranks

        # init during init_device
        self._is_dummy = False
        self._tp_group = None

    def _patch_tensor_parallel_group(self):
        """Temporarily patch the global tp group state with its own tp group
        state.
        """
        return patch_tensor_parallel_group(self._tp_group)

    def init_device(self) -> None:
        self._is_dummy = get_tp_group().rank not in self._draft_ranks

        # dummy workers do nothing
        if self._is_dummy:
            return

        # creates tp process group containing only a subset of gpu ranks
        local_rank = get_tp_group().local_rank
        tp_backend = torch.distributed.get_backend(get_tp_group().device_group)
        self._tp_group = init_model_parallel_group([self._draft_ranks],
                                                   local_rank, tp_backend)

        with self._patch_tensor_parallel_group():
            self._worker.init_device()

    def set_include_gpu_probs_tensor(self) -> None:
        if self._is_dummy:
            return

        # Need include_gpu_probs_tensor for multi_step_worker
        self._worker.set_include_gpu_probs_tensor()

    def load_model(self) -> None:
        if self._is_dummy:
            return

        with self._patch_tensor_parallel_group():
            self._worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        if self._is_dummy:
            # this case is not used now
            return -1, -1

        with self._patch_tensor_parallel_group():
            return self._worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        if self._is_dummy:
            return

        with self._patch_tensor_parallel_group():
            self._worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        # Do not check _is_dummy, as it's always called by get_spec_proposals
        return self._worker.sampler_output(
            execute_model_req, sample_len,
            seq_ids_with_bonus_token_in_last_step)

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """
        if self._is_dummy:
            return SpeculativeProposals(None, None, None)

        with self._patch_tensor_parallel_group():
            return self._worker.get_spec_proposals(
                execute_model_req, seq_ids_with_bonus_token_in_last_step)

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        if self._is_dummy:
            return []

        with self._patch_tensor_parallel_group():
            return self._worker.execute_model(execute_model_req)

    def get_cache_block_size_bytes(self) -> int:
        if self._is_dummy:
            # by returning zero, target worker can use the entire kv cache space
            return 0

        return self._worker.get_cache_block_size_bytes()

    @property
    def vocab_size(self) -> int:
        return self._worker.vocab_size
