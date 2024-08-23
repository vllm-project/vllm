import copy

from typing import Dict, List, Optional, Tuple

import torch

from vllm.config import ParallelConfig, ClassifierFreeGuidanceConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
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
        is_driver_worker=kwargs["is_driver_worker"],
        parallel_config=kwargs["parallel_config"],
    )


class CFGWorker(LoraNotSupportedWorkerBase):
    def __init__(
        self,
        root_worker: WorkerBase,
        guidance_worker: WorkerBase,
        is_driver_worker: bool,
        parallel_config: ParallelConfig,
    ):
        self.root_worker = root_worker
        self.guidance_worker = guidance_worker
        self.is_driver_worker = is_driver_worker
        self.parallel_config = parallel_config
        assert self.parallel_config.pipeline_parallel_size == 1

    def init_device(self):
        self.root_worker.init_device()
        self.guidance_worker.init_device()

    def load_model(self):
        self.root_worker.load_model()
        self.guidance_worker.share_model(self.root_worker)

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

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.root_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                          num_cpu_blocks=num_cpu_blocks)
        self.guidance_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:

        # prepare negative request with shallow copy
        if execute_model_req is not None:
            negative_seq_group_metadata_list: List[SequenceGroupMetadata] = []
            negative_excute_model_req = execute_model_req.clone(negative_seq_group_metadata_list)
            for seq_group_metadata in execute_model_req.seq_group_metadata_list:
                negative_seq_group_metadata = copy.copy(seq_group_metadata)
                negative_seq_data: Dict[int, SequenceData] = {}
                negative_block_tables: Dict[int, List[int]] = {}
                assert len(seq_group_metadata.seq_data) == 1
                for seq_id in seq_group_metadata.seq_data.keys():
                    negative_seq_data[seq_id] = seq_group_metadata.negative_seq_data
                    negative_block_tables[seq_id] = seq_group_metadata.negative_block_table

                if negative_seq_group_metadata.is_prompt:
                    negative_seq_group_metadata.token_chunk_size = list(negative_seq_data.values())[0].get_len()

                negative_seq_group_metadata.seq_data = negative_seq_data
                negative_seq_group_metadata.block_tables = negative_block_tables
                negative_seq_group_metadata.negative_seq_data = None
                negative_seq_group_metadata.negative_block_table = None
                negative_seq_group_metadata_list.append(negative_seq_group_metadata)
            negative_excute_model_req.seq_group_metadata_list = negative_seq_group_metadata_list
        else:
            negative_excute_model_req = None

        inputs = self.root_worker.prepare_input(execute_model_req)
        negative_inputs = self.guidance_worker.prepare_input(negative_excute_model_req)
        if inputs is None:
            assert negative_inputs is None
            return None

        # get root models's logits
        condition_logits = self.root_worker.execute_model_part(inputs)
        # get unconditional logits
        unconditional_logits = self.guidance_worker.execute_model_part(negative_inputs)

        # do classifier free guidance logist process
        model_input, _ = inputs
        if condition_logits is not None:
            for seq_group in model_input.sampling_metadata.seq_groups:
                seq_ids = seq_group.seq_ids
                guidance_scale = seq_group.sampling_params.guidance_scale
                if guidance_scale == 1.0:
                    break
                for seq_id, logits_row_idx in zip(seq_ids, seq_group.sample_indices):
                    logits_row = torch.nn.functional.log_softmax(condition_logits[logits_row_idx], dim=-1)
                    unconditional_logits_row = torch.nn.functional.log_softmax(unconditional_logits[logits_row_idx], dim=-1)
                    condition_logits[logits_row_idx] = guidance_scale * (logits_row - unconditional_logits_row) + unconditional_logits_row

        # do logist_processor
        scores = self.root_worker.compute_logits(condition_logits, model_input)
        if not self.is_driver_worker:
            return []

        # do sample
        output = self.root_worker.do_sample(scores, model_input)

        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            get_pp_group().send_tensor_dict(output.tensors, all_gather_group=get_tp_group())
            return [None]

        # output is List[SamplerOutput]
        return output

    def get_cache_block_size_bytes(self):
        raise NotImplementedError
