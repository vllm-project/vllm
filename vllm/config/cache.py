# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal, Optional, get_args

from pydantic import SkipValidation, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils import GiB_bytes, get_cpu_memory

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig
else:
    ParallelConfig = Any

logger = init_logger(__name__)

BlockSize = Literal[1, 8, 16, 32, 64, 128]
CacheDType = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"]
MambaDType = Literal["auto", "float32"]
PrefixCachingHashAlgo = Literal["builtin", "sha256", "sha256_cbor_64bit"]


@config
@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: SkipValidation[BlockSize] = None  # type: ignore
    """Size of a contiguous cache block in number of tokens. This is ignored on
    neuron devices and set to `--max-model-len`. On CUDA devices, only block
    sizes up to 32 are supported. On HPU devices, block size defaults to 128.

    This config has no static default. If left unspecified by the user, it will
    be set in `Platform.check_and_update_config()` based on the current
    platform."""
    gpu_memory_utilization: float = 0.9
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    swap_space: float = 4
    """Size of the CPU swap space per GPU (in GiB)."""
    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc)."""
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    num_gpu_blocks_override: Optional[int] = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: Optional[int] = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    enable_prefix_caching: Optional[bool] = None
    """Whether to enable prefix caching. Disabled by default for V0. Enabled by
    default for V1."""
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "builtin"
    """Set the hash algorithm for prefix caching:\n
    - "builtin" is Python's built-in hash.\n
    - "sha256" is collision resistant but with certain overheads.
    This option uses Pickle for object serialization before hashing.\n
    - "sha256_cbor_64bit" provides a reproducible, cross-language compatible
    hash. It serializes objects using canonical CBOR and hashes them with
    SHA-256. The resulting hash consists of the lower 64 bits of the SHA-256
    digest."""
    cpu_offload_gb: float = 0
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    """
    calculate_kv_scales: bool = False
    """This enables dynamic calculation of `k_scale` and `v_scale` when
    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model
    checkpoint if available. Otherwise, the scales will default to 1.0."""
    cpu_kvcache_space_bytes: Optional[int] = None
    """(CPU backend only) CPU key-value cache space."""
    mamba_page_size_padded: Optional[int] = None
    """ Optional override for mamba page size; used by hybrid mamba/attention
    models to ensure exact alignment with attention page size."""

    mamba_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (both the conv as well as the
    ssm state). If set to 'auto', the data type will be inferred from the model
    config."""
    mamba_ssm_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (ssm state only, conv state will
    still be controlled by mamba_cache_dtype). If set to 'auto', the data type
    for the ssm state will be determined by mamba_cache_dtype."""

    # Will be set after profiling.
    num_gpu_blocks: Optional[int] = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: Optional[int] = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    kv_sharing_fast_prefill: bool = False
    """This feature is work in progress and no prefill optimization takes place
    with this flag enabled currently.

    In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254),
    some layers can skip tokens corresponding to prefill. This flag enables
    attention metadata for eligible layers to be overriden with metadata
    necessary for implementating this optimization in some models (e.g. Gemma3n)
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.cache_dtype)
        factors.append(self.mamba_cache_dtype)
        factors.append(self.mamba_ssm_cache_dtype)
        # `cpu_offload_gb` does not use `torch.compile` yet.
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        self.swap_space_bytes = self.swap_space * GiB_bytes

        self._verify_cache_dtype()
        self._verify_prefix_caching()

    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        if self.cpu_offload_gb < 0:
            raise ValueError("CPU offload space must be non-negative"
                             f", but got {self.cpu_offload_gb}")

        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

        if self.kv_sharing_fast_prefill:
            logger.warning_once(
                "--kv-sharing-fast-prefill is currently work in progress "
                "and not functional yet (i.e. no prefill savings)")

        return self

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in get_args(CacheDType):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor.")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def _verify_prefix_caching(self) -> None:
        if not self.enable_prefix_caching:
            return

        if self.sliding_window is not None and not envs.VLLM_USE_V1:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. "
                "Run with --disable-sliding-window to use prefix caching.")

        if (self.enable_prefix_caching and self.prefix_caching_hash_algo
                not in get_args(PrefixCachingHashAlgo)):
            raise ValueError(
                "Unknown prefix caching hash algorithm: "
                f"{self.prefix_caching_hash_algo}. Must be one of "
                f"{get_args(PrefixCachingHashAlgo)}.")

    def verify_with_parallel_config(
        self,
        parallel_config: ParallelConfig,
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (f"{cpu_memory_usage / GiB_bytes:.2f} GiB out of the "
               f"{total_cpu_memory / GiB_bytes:.2f} GiB total CPU memory "
               "is allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. %s", msg)
