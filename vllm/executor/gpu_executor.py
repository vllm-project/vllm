import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest, PoolerOutput, SamplerOutput, SequenceGroupMetadata, SequenceData
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


def create_worker(worker_module_name, worker_class_name, **kwargs):
    wrapper = WorkerWrapperBase(
        worker_module_name=worker_module_name,
        worker_class_name=worker_class_name,
    )
    wrapper.init_worker(**kwargs)
    return wrapper.worker


class GPUExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        print("[zyl] gpu executor _create_worker:")
        self.driver_worker = self._create_worker()
        print("[zyl] gpu executor init_device:")
        self.driver_worker.init_device()
        print("[zyl] gpu executor load_model:")
        self.driver_worker.load_model()
        print("[zyl] driver_worker:", self.driver_worker)

        # print("[zyl] gpu executor _create_worker:")
        # self.cfg_worker = self._create_worker()
        # print("[zyl] gpu executor init_device:")
        # self.cfg_worker.init_device()
        # print("[zyl] gpu executor load_model:")
        # self.cfg_worker.load_model()
        # print("[zyl] cfg_worker:", self.cfg_worker)

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            multimodal_config=self.multimodal_config,
            speculative_config=self.speculative_config,
            prompt_adapter_config=self.prompt_adapter_config,
            classifier_free_guidance_config=self.classifier_free_guidance_config,
            is_driver_worker=(not self.parallel_config)
            or (rank % self.parallel_config.tensor_parallel_size == 0),
        )

    def _get_create_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict:
        worker_kwargs = self._get_worker_kwargs(local_rank, rank,
                                                distributed_init_method)
        if self.speculative_config is None and self.classifier_free_guidance_config is None:
            worker_kwargs.update(worker_module_name="vllm.worker.worker",
                                 worker_class_name="Worker")
        elif self.speculative_config is not None:
            worker_kwargs.update(
                worker_module_name="vllm.spec_decode.spec_decode_worker",
                worker_class_name="create_spec_worker")
        else:
            worker_kwargs.update(
                worker_module_name="vllm.classifier_free_guidance.cfg_worker",
                worker_class_name="create_cfg_worker")
        return worker_kwargs

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):
        return create_worker(**self._get_create_worker_kwargs(
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method))

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()
        # num_gpu_blocks, num_cpu_blocks = self.driver_worker.determine_num_available_blocks()

        # driver_cache_block_size_bytes = self.driver_worker.get_cache_block_size_bytes()
        # cfg_cache_block_size_bytes = self.cfg_worker.get_cache_block_size_bytes()

        # new_num_gpu_blocks = int(
        #     num_gpu_blocks * driver_cache_block_size_bytes /
        #     (cfg_cache_block_size_bytes + driver_cache_block_size_bytes))

        # return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        # self.cfg_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
        # print("[zyl] gpu_executor execute_model_req:", execute_model_req)
        # for i, seq_group_metadata in enumerate(execute_model_req.seq_group_metadata_list):
        #     seq_data = next(iter(seq_group_metadata.seq_data.values()))
        #     seq_len = seq_data.get_len()
        #     print("[zyl] driver seq_data:", seq_data)
        #     print("[zyl] driver seq_len:", seq_len)
        # print("[zyl] gpu executor driver_worker.execute_model:", self.driver_worker.execute_model)
        output = self.driver_worker.execute_model(execute_model_req)
        # print("[zyl] gpu_executor output:", output)
        # print("[zyl] gpu_executor sampled_token_ids:", output[0].sampled_token_ids)
        # exit(0)
        return output

    # def execute_model(
    #     self, execute_model_req: ExecuteModelRequest
    # ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
    #     # print("[zyl] gpu_executor execute_model_req:", execute_model_req)
    #     # for i, seq_group_metadata in enumerate(execute_model_req.seq_group_metadata_list):
    #     #     seq_data = next(iter(seq_group_metadata.seq_data.values()))
    #     #     seq_len = seq_data.get_len()
    #     #     print("[zyl] driver seq_data:", seq_data)
    #     #     print("[zyl] driver seq_len:", seq_len)
    #     # print("[zyl] gpu_executor self.driver_worker:", self.driver_worker)
    #     logits_driver, model_input_driver = self.driver_worker.execute_model(execute_model_req, do_no_processor=False)
    #     # print("[zyl] gpu_executor logits_driver:", logits_driver)

    #     # cfg_seq_group_metadata_list: List[SequenceGroupMetadata] = []
    #     # cfg_execute_model_req = execute_model_req.clone(cfg_seq_group_metadata_list)
    #     # for seq_group_metadata in execute_model_req.seq_group_metadata_list:
    #     #     new_seq_group_metadata = copy.copy(seq_group_metadata)
    #     #     new_seq_data: Dict[int, SequenceData] = {}
    #     #     for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
    #     #         if len(old_seq_data.output_token_ids) == 0:
    #     #             new_seq_data[seq_id] = copy.copy(old_seq_data)
    #     #             new_seq_data[seq_id].prompt_token_ids = old_seq_data.prompt_token_ids[-1:]
    #     #             new_seq_data[seq_id].output_token_ids = ()
    #     #         else:
    #     #             new_seq_data[seq_id] = copy.copy(old_seq_data)
    #     #             new_seq_data[seq_id].prompt_token_ids = old_seq_data.prompt_token_ids[-1:]
    #     #             new_seq_data[seq_id].output_token_ids = old_seq_data.output_token_ids[:]
    #     #     new_seq_group_metadata.seq_data = new_seq_data
    #     #     cfg_seq_group_metadata_list.append(new_seq_group_metadata)
    #     # cfg_execute_model_req.seq_group_metadata_list = cfg_seq_group_metadata_list


    #     # cfg_seq_group_metadata_list: List[SequenceGroupMetadata] = []
    #     # cfg_execute_model_req = execute_model_req.clone(cfg_seq_group_metadata_list)
    #     # for seq_group_metadata in execute_model_req.seq_group_metadata_list:
    #     #     new_seq_group_metadata = copy.copy(seq_group_metadata)
    #     #     new_seq_data: Dict[int, SequenceData] = {}
    #     #     for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
    #     #         # if seq_group_metadata.is_prompt:
    #     #         new_seq_data[seq_id] = copy.copy(old_seq_data)
    #     #         new_seq_data[seq_id].prompt_token_ids = old_seq_data.negative_prompt_token_ids
    #     #         new_seq_data[seq_id].negative_prompt_token_ids = []
    #     #         new_seq_data[seq_id].output_token_ids = old_seq_data.output_token_ids[:]

    #     #     new_seq_group_metadata.seq_data = new_seq_data
    #     #     cfg_seq_group_metadata_list.append(new_seq_group_metadata)
    #     # cfg_execute_model_req.seq_group_metadata_list = cfg_seq_group_metadata_list


    #     cfg_seq_group_metadata_list: List[SequenceGroupMetadata] = []
    #     cfg_execute_model_req = execute_model_req.clone(cfg_seq_group_metadata_list)
    #     for seq_group_metadata in execute_model_req.seq_group_metadata_list:
    #         new_seq_group_metadata = copy.deepcopy(seq_group_metadata)
    #         new_seq_data: Dict[int, SequenceData] = {}
    #         for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
    #             # if seq_group_metadata.is_prompt:
    #             new_seq_data[seq_id] = copy.deepcopy(old_seq_data)
    #             new_seq_data[seq_id].prompt_token_ids = old_seq_data.negative_prompt_token_ids
    #             new_seq_data[seq_id].negative_prompt_token_ids = []
    #             new_seq_data[seq_id].output_token_ids = old_seq_data.output_token_ids[:]

    #         new_seq_group_metadata.seq_data = new_seq_data
    #         cfg_seq_group_metadata_list.append(new_seq_group_metadata)
    #     cfg_execute_model_req.seq_group_metadata_list = cfg_seq_group_metadata_list


    #     # print("[zyl] gpu_executor cfg_execute_model_req:", cfg_execute_model_req)
    #     # for i, seq_group_metadata in enumerate(cfg_execute_model_req.seq_group_metadata_list):
    #     #     seq_data = next(iter(seq_group_metadata.seq_data.values()))
    #     #     seq_len = seq_data.get_len()
    #     #     print("[zyl] cfg seq_data:", seq_data)
    #     #     print("[zyl] cfg seq_len:", seq_len)
    #     logits_cfg, _ = self.cfg_worker.execute_model(cfg_execute_model_req, do_no_processor=True)
    #     # print("[zyl] gpu_executor logits_cfg:", logits_cfg)

    #     logits = logits_cfg + 5.0 * (logits_driver - logits_cfg)
    #     # print("[zyl] gpu_executor logits:", logits)
    #     output: SamplerOutput = self.driver_worker.model_runner.model.sample(logits=logits, sampling_metadata=model_input_driver.sampling_metadata)
    #     # print("[zyl] gpu_executor output:", output)
    #     # print("[zyl] gpu_executor sampled_token_ids:", output.sampled_token_ids)
    #     # exit(0)
    #     return [output]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        assert prompt_adapter_request.prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return self.driver_worker.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return self.driver_worker.remove_prompt_adapter(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
                "prompt_adapter_id must be greater than 0."
        return self.driver_worker.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.driver_worker.list_prompt_adapters()

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output
