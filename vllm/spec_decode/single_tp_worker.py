import copy
import weakref
from typing import List, Tuple, Set, Optional, Union
from datetime import timedelta

import torch
import torch.distributed

from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeProposer
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker import Worker
from vllm.lora.request import LoRARequest

from vllm.distributed.parallel_state import patch_tensor_parallel_group, _ENABLE_CUSTOM_ALL_REDUCE
from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class SingleTpWorker(ProposerWorkerBase):
    """Class which allows a speculative draft model to run with tensor parallel
    degree of 1, while target model runs with larger tensor parallel degree.
    This reduces the overhead of small draft models.

    This is implemented by changing vLLM's tensor parallel group to a group of
    size 1 during forward passes.
    """

    @classmethod
    def maybe_wrap_worker(cls, worker, draft_parallel_config: ParallelConfig,
                          target_parallel_config: ParallelConfig,
                          rank: int, local_rank: int):
        """Wrap the worker in a SingleTpWorker if necessary.
        """
        draft_tp = draft_parallel_config.tensor_parallel_size
        target_tp = target_parallel_config.tensor_parallel_size
        if draft_tp > target_tp:
            raise ValueError("{cls} only supports draft_tp smaller than target_tp."
                             f"{draft_tp=} {target_tp=}")

        ranks = list(range(draft_tp))

        if draft_tp == target_tp:
            return worker

        logger.info(f"{rank=}, {ranks=}")
        if rank in ranks:
            logger.info(f"Wrapping {type(worker)} in {cls}")
            return cls(worker, ranks, local_rank)
        else:
            logger.info(f"dummy worker that would not participate in draft generation")
            return DummyProposerWorker(worker)

    def __init__(
        self,
        worker: Union[Worker, ProposerWorkerBase],
        ranks: List[int],
        local_rank: int
    ):
        self._worker = worker
        self._ranks = ranks
        self._local_rank = local_rank
        self._tp_group = None
        self._tp_cpu_group = None
        self._tp_pynccl_comm = None
        self._tp_ca_comm = None

    def _patch_tensor_parallel_group(self):
        return patch_tensor_parallel_group(self._tp_group, self._tp_cpu_group,
                                    self._tp_pynccl_comm, self._tp_ca_comm)

    def init_device(self):
        """Initialize the model on all ranks.

        This also creates a single-rank process group containing only the
        self process.
        """
        self._tp_group = torch.distributed.new_group(
            ranks=self._ranks, timeout=timedelta(seconds=10))
        self._tp_cpu_group = torch.distributed.new_group(
            ranks=self._ranks, timeout=timedelta(seconds=10), backend="gloo")

        if len(self._ranks) > 1:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            self._tp_pynccl_comm = PyNcclCommunicator(
                group=self._tp_cpu_group,
                device=self._local_rank,
            )
            if _ENABLE_CUSTOM_ALL_REDUCE:
                from vllm.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce)
                self._tp_ca_comm = CustomAllreduce(
                    group=self._tp_cpu_group,
                    device=self._local_rank,
                )

        logger.info(f"init_device. ranks: {self._ranks}")

        with self._patch_tensor_parallel_group():
            self._worker.init_device()

    def set_include_gpu_probs_tensor(self):
        self._worker.set_include_gpu_probs_tensor()

    def load_model(self):
        logger.info("SingleTPWorker.load_model()")
        with self._patch_tensor_parallel_group():
            self._worker.load_model()

    def determine_num_available_blocks(self):
        """Profile the model on all ranks.
        """
        with self._patch_tensor_parallel_group():
            return self._worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int):
        """Initialize the cache engine on all ranks.
        """
        with self._patch_tensor_parallel_group():
            self._worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[List[SamplerOutput], bool]:
        return self._worker.sampler_output(execute_model_req, sample_len)

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """
        with self._patch_tensor_parallel_group():
            return self._worker.get_spec_proposals(execute_model_req)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        with self._patch_tensor_parallel_group():
            return self._worker.execute_model(execute_model_req)

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        return self._worker.get_cache_block_size_bytes()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    @property
    def max_model_len(self) -> int:
        return self._worker.max_model_len

    @property
    def vocab_size(self) -> int:
        return self._worker.vocab_size


class DummyProposerWorker(ProposerWorkerBase):

    def __init__(
        self,
        worker: Union[Worker, ProposerWorkerBase],
    ):
        self._worker = worker

    def init_device(self):
        pass

    def set_include_gpu_probs_tensor(self):
        pass

    def load_model(self):
        pass

    def determine_num_available_blocks(self):
        pass

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int):
        pass

    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[List[SamplerOutput], bool]:
        return None

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> SpeculativeProposals:
        return None

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        return None

    def get_cache_block_size_bytes(self) -> int:
        return 0

    def add_lora(self, lora_request: LoRARequest) -> bool:
        pass

    def remove_lora(self, lora_id: int) -> bool:
        pass

    def list_loras(self) -> Set[int]:
        pass

    @property
    def max_model_len(self) -> int:
        return self._worker.max_model_len

    @property
    def vocab_size(self) -> int:
        return self._worker.vocab_size
