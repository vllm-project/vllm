"""An OpenVINO worker class."""
from typing import Any, Dict, List, Optional, Tuple

import openvino as ov
import torch
import torch.distributed

from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.openvino_model_runner import OpenVINOModelRunner
from vllm.worker.worker_base import LoraNotSupportedWorkerBase

logger = init_logger(__name__)


class OpenVINOCacheEngine:
    """Manages the KV cache for OpenVINO backend.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        assert device_config.device_type == "openvino"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        if device_config.device.type == "cpu" and \
            cache_config.cache_dtype == ov.Type.u8:
            # Scale, zero point and quantized data will be stored together.
            # The layout for per token per head:
            # |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)| # noqa: E501
            # so, we have to extend head_size by 8, which is sizeof(float)
            # for scale and sizeof(float) for zeropoint
            self.head_size += 8
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for OpenVINO backend, because we want to reuse KV cache management
        # in the scheduler.
        self.num_cpu_blocks = cache_config.num_gpu_blocks

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.head_size,
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the cache.
        self.kv_cache: List[Tuple[ov.Tensor,
                                  ov.Tensor]] = self._allocate_kv_cache(
                                      self.num_cpu_blocks)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
    ) -> List[Tuple[ov.Tensor, ov.Tensor]]:
        """Allocates KV cache."""
        k_block_shape = v_block_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)[1:]
        kv_cache: List[Tuple[ov.Tensor, ov.Tensor]] = []
        for _ in range(self.num_layers):
            key_blocks = ov.Tensor(self.cache_config.cache_dtype,
                                   k_block_shape)
            value_blocks = ov.Tensor(self.cache_config.cache_dtype,
                                     v_block_shape)
            kv_cache.append((key_blocks, value_blocks))
        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError(
            "Swap is not supported in OpenVINOCacheEngine.")

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError(
            "Swap is not supported in OpenVINOCacheEngine.")

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.kv_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: ov.Type,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        if cache_dtype == ov.Type.u8:
            # Scale, zero point and quantized data will be stored together.
            # The layout for per token per head:
            # |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)| # noqa: E501
            # so, we have to extend head_size by 8, which is sizeof(float)
            # for scale and sizeof(float) for zeropoint
            head_size += 8

        key_cache_block = block_size * num_kv_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = cache_dtype.size
        return dtype_size * total


class OpenVINOWorker(LoraNotSupportedWorkerBase):
    """A worker class that executes the model on OpenVINO backend.

    Each worker is associated with a single OpenVINO device. The worker is
    responsible for maintaining the KV cache and executing the model on the
    OpenVINO backend.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        kv_cache_dtype: Optional[ov.Type] = ov.Type.undefined,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.parallel_config.rank = rank
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.multimodal_config = multimodal_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()
        self.model_runner = OpenVINOModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=self.load_config,
            lora_config=self.lora_config,
            multimodal_config=self.multimodal_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: OpenVINOCacheEngine
        self.kv_cache: List[Tuple[ov.Tensor, ov.Tensor]]

    def init_device(self) -> None:
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of blocks available for the KV cache.

        This determines how many KV blocks can fit into the configured
        KV cache space.

        Note that since vLLM assumes a block resides on GPU if it can be
        modified, we return num_gpu_blocks=num_cpu_blocks and num_cpu_blocks=0.
        This allows us to reuse the scheduler of vLLM without generalizing it
        to different devices.
        """
        # For OpenVINO backend, the block number will be calculated based on the
        # openvino_kvcache_space_bytes.
        cache_block_size = self.get_cache_block_size_bytes()
        num_cpu_blocks = int(self.cache_config.openvino_kvcache_space_bytes //
                             cache_block_size)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        # Note: To reuse the cache management procedure,
        # use cpu cache as 'gpu cache'.
        num_gpu_blocks = num_cpu_blocks
        num_cpu_blocks = 0
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache. Currently, swappable CPU memory is not
        supported.

        Since this worker does not support GPUs, we use the num_gpu_blocks to
        determine how many non-swappable CPU blocks to allocate.
        """
        assert (num_cpu_blocks == 0
                ), f"{type(self)} does not support swappable cache"

        # Note: To reuse the cache management procedure,
        # use cpu cache as 'gpu cache'.
        num_cpu_blocks = num_gpu_blocks

        self._validate_num_cpu_blocks(num_cpu_blocks)
        self.cache_config.num_gpu_blocks = num_cpu_blocks
        self.cache_config.num_cpu_blocks = 0

        # Initialize the cache.
        self._init_cache_engine()

    def _validate_num_cpu_blocks(self, num_cpu_blocks: int) -> None:
        """Raise errors if the num_cpu_blocks is invalid."""
        if num_cpu_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `VLLM_OPENVINO_KVCACHE_SPACE` when "
                "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_cpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`VLLM_OPENVINO_KVCACHE_SPACE` or decreasing `max_model_len` "
                "when initializing the engine.")

    def _init_cache_engine(self) -> None:
        self.cache_engine = OpenVINOCacheEngine(
            self.cache_config,
            self.model_config,
            self.parallel_config,
            self.device_config,
        )
        self.kv_cache = self.cache_engine.kv_cache
        self.model_runner.block_size = self.cache_engine.block_size

        assert self.kv_cache is not None

        # Populate the cache to warmup the memory
        for key_cache, value_cache in self.kv_cache:
            key_cache.data[:] = 0
            value_cache.data[:] = 0

    def cache_copy(
        self,
        blocks_to_copy: List[Tuple[int, int]],
    ) -> None:
        self.cache_engine.copy(blocks_to_copy)  # type: ignore

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        if execute_model_req is None:
            seq_group_metadata_list = None
        else:
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups: int = len(seq_group_metadata_list)
            assert execute_model_req is not None
            blocks_to_copy = execute_model_req.blocks_to_copy
            assert len(execute_model_req.blocks_to_swap_in) == 0
            assert len(execute_model_req.blocks_to_swap_out) == 0
            data: Dict[str, Any] = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_copy": execute_model_req.blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_copy = data["blocks_to_copy"]

        self.cache_copy(blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.kv_cache)

        # OpenVINO worker only supports single-step execution.
        return [output]

    def init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        parallel_config = self.parallel_config
        rank = self.rank
        distributed_init_method = self.distributed_init_method
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            backend="gloo",
        )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cpu())

        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Return the size in bytes of a single KV cache block."""
        return OpenVINOCacheEngine.get_cache_block_size(
            self.cache_config.block_size,
            self.cache_config.cache_dtype,
            self.model_config,
            self.parallel_config,
        )
