from typing import Optional

import torch
from transformers import AutoConfig, PretrainedConfig

_GiB = 1 << 30


class ModelConfig:

    def __init__(
        self,
        model: str,
        download_dir: Optional[str],
        use_np_weights: bool,
        use_dummy_weights: bool,
        dtype: str,
        seed: int,
    ) -> None:
        self.model = model
        self.download_dir = download_dir
        self.use_np_weights = use_np_weights
        self.use_dummy_weights = use_dummy_weights
        self.seed = seed

        self.hf_config: PretrainedConfig = AutoConfig.from_pretrained(model)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size}).")

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_num_heads(self, parallel_config: "ParallelConfig") -> int:
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class CacheConfig:

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GiB

        # Will be set after profiling.
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None


class ParallelConfig:

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        use_ray: bool,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.use_ray = use_ray

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")


class SchedulerConfig:

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ) -> None:
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str,
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    dtype = dtype.lower()
    if dtype == "default":
        if config_dtype == torch.float32:
            # Following the common practice, we use float16 for float32 models.
            torch_dtype = torch.float16
        else:
            torch_dtype = config_dtype
    else:
        if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {dtype}")
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            pass
        else:
            # Casting between float16 and bfloat16 is not allowed.
            raise ValueError(
                f"Cannot use {torch_dtype} for {config_dtype} model.")

    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")
    return torch_dtype
