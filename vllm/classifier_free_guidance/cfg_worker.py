import copy

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.config import ParallelConfig, ClassifierFreeGuidanceConfig
from vllm.logger import init_logger
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase
from vllm.sequence import ExecuteModelRequest, SamplerOutput, SequenceGroupMetadata, SequenceData

from vllm.classifier_free_guidance.cfg_model_runner import CFGModelRunner
from vllm.classifier_free_guidance.separated_worker import SeparatedWorker

logger = init_logger(__name__)


def create_cfg_worker(*args, **kwargs) -> "CFGWorker":

    assert "classifier_free_guidance_config" in kwargs
    classifier_free_guidance_config: ClassifierFreeGuidanceConfig = kwargs.get("classifier_free_guidance_config")
    assert classifier_free_guidance_config is not None
    kwargs.pop("classifier_free_guidance_config")

    kwargs["model_runner_cls"] = CFGModelRunner
    root_worker = SeparatedWorker(*args, **kwargs)

    guidance_model_config = classifier_free_guidance_config.guidance_model_config
    guidance_parallel_config = classifier_free_guidance_config.guidance_parallel_config
    kwargs.update(
        model_config=guidance_model_config,
        parallel_config=guidance_parallel_config,
    )
    guidance_worker = SeparatedWorker(*args, **kwargs)

    return CFGWorker(
        root_worker=root_worker, 
        guidance_worker=guidance_worker,
    )


class CFGWorker(LoraNotSupportedWorkerBase):
    def __init__(
        self,
        root_worker: WorkerBase,
        guidance_worker: WorkerBase,
    ):
        self.root_worker = root_worker
        self.guidance_worker = guidance_worker

    def init_device(self):
        self.root_worker.init_device()
        self.guidance_worker.init_device()

    def load_model(self):
        self.root_worker.load_model()
        # TODO(zhaoyinglia): guidance_worker shares weight with root_worker
        self.guidance_worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_gpu_blocks, num_cpu_blocks = (
            self.root_worker.determine_num_available_blocks())

        root_cache_block_size_bytes = (
            self.root_worker.get_cache_block_size_bytes())
        guidance_cache_block_size_bytes = (
            self.guidance_worker.get_cache_block_size_bytes())

        new_num_gpu_blocks = int(
            num_gpu_blocks * root_cache_block_size_bytes /
            (guidance_cache_block_size_bytes + root_cache_block_size_bytes))
        return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(
        self, 
        num_gpu_blocks: int,
        num_cpu_blocks: int
        ):
        self.root_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                          num_cpu_blocks=num_cpu_blocks)
        self.guidance_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:

        # get root models's logits
        scores = self.root_worker.execute_model_part(execute_model_req)
        # prepare negative request with shallow copy
        negative_seq_group_metadata_list: List[SequenceGroupMetadata] = []
        negative_excute_model_req = execute_model_req.clone(negative_seq_group_metadata_list)
        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            negative_seq_group_metadata = copy.copy(seq_group_metadata)
            negative_seq_data: Dict[int, SequenceData] = {}
            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                negative_seq_data[seq_id] = copy.copy(seq_data)
                negative_seq_data[seq_id].prompt_token_ids = seq_data.negative_prompt_token_ids
                negative_seq_data[seq_id].negative_prompt_token_ids = []
                negative_seq_data[seq_id].output_token_ids = seq_data.output_token_ids[:]

            negative_seq_group_metadata.seq_data = negative_seq_data
            negative_seq_group_metadata_list.append(negative_seq_group_metadata)
        negative_excute_model_req.seq_group_metadata_list = negative_seq_group_metadata_list

        # get unconditional logits
        unconditional_logits = self.guidance_worker.execute_model_part(negative_excute_model_req)

        # do logist_processor
        scores = self.root_worker.compute_logits(scores)

        # do classifier free guidance logist process
        for seq_group in self.root_worker.model_input.sampling_metadata.seq_groups:
            seq_ids = seq_group.seq_ids
            guidance_scale = seq_group.sampling_params.guidance_scale
            if guidance_scale == 1.0:
                break
            for seq_id, logits_row_idx in zip(seq_ids, seq_group.sample_indices):
                logits_row = torch.nn.functional.log_softmax(scores[logits_row_idx], dim=-1)
                unconditional_logits_row = torch.nn.functional.log_softmax(unconditional_logits[logits_row_idx], dim=-1)
                scores[logits_row_idx] = guidance_scale * (logits_row - unconditional_logits_row) + unconditional_logits_row

        # do sample
        output = self.root_worker.do_sample(scores)

        # output is List[SamplerOutput]
        return output

    def get_cache_block_size_bytes(self):
        raise NotImplementedError
