from typing import Any, Dict, List, Optional, Set, Tuple

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class GPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.

        If speculative decoding is enabled, we instead create the speculative
        worker.
        """
        if self.speculative_config is None:
            self._init_non_spec_worker()
        else:
            self._init_spec_worker()

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
            vision_language_config=self.vision_language_config,
            is_driver_worker=rank == 0,
        )

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):
        wrapper = WorkerWrapperBase(
            worker_module_name="vllm.worker.worker",
            worker_class_name="Worker",
        )
        wrapper.init_worker(**self._get_worker_kwargs(local_rank, rank,
                                                      distributed_init_method))
        return wrapper.worker

    def _init_non_spec_worker(self):
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _init_spec_worker(self):
        """Initialize a SpecDecodeWorker, using a draft model for proposals.
        """
        assert self.speculative_config is not None

        from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker

        target_worker = self._create_worker()

        draft_worker_kwargs = self._get_worker_kwargs()
        # Override draft-model specific worker args.
        draft_worker_kwargs.update(
            model_config=self.speculative_config.draft_model_config,
            parallel_config=self.speculative_config.draft_parallel_config,
            ngram_prompt_lookup_max=self.speculative_config.
            ngram_prompt_lookup_max,
            ngram_prompt_lookup_min=self.speculative_config.
            ngram_prompt_lookup_min,
            # TODO allow draft-model specific load config.
            #load_config=self.load_config,
        )

        spec_decode_worker = SpecDecodeWorker.create_worker(
            scorer_worker=target_worker,
            draft_worker_kwargs=draft_worker_kwargs,
            disable_by_batch_size=self.speculative_config.
            speculative_disable_by_batch_size,
        )

        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = spec_decode_worker

        # Load model handled in spec decode worker.
        self.driver_worker.init_device()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output
