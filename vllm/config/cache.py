# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, SkipValidation, field_validator

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import format_gib, get_cpu_memory

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig
else:
    ParallelConfig = Any

logger = init_logger(__name__)

BlockSize = Literal[1, 8, 16, 32, 64, 128, 256]
CacheDType = Literal[
    "auto",
    "bfloat16",
    "int8",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
]
MambaDType = Literal["auto", "float32", "float16"]
MambaCacheMode = Literal["all", "align", "none"]
PrefixCachingHashAlgo = Literal["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"]
KVOffloadingBackend = Literal["native", "lmcache"]


@config
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: SkipValidation[BlockSize] = None  # type: ignore
    """Size of a contiguous cache block in number of tokens. On CUDA devices,
    only block sizes up to 32 are supported.

    This config has no static default. If left unspecified by the user, it will
    be set in `Platform.check_and_update_config()` based on the current
    platform."""
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    swap_space: float = Field(default=4, ge=0)
    """Size of the CPU swap space per GPU (in GiB)."""
    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc).
    Some models (namely DeepSeekV3.2) default to fp8, set to bfloat16 to use
    bfloat16 instead, this is an invalid option for models that do not default
    to fp8.
    """
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    num_gpu_blocks_override: int | None = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: int | None = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    enable_prefix_caching: bool = True
    """Whether to enable prefix caching."""
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"
    """Set the hash algorithm for prefix caching:\n
    - "sha256" uses Pickle for object serialization before hashing. This is the
    current default, as SHA256 is the most secure choice to avoid potential
    hash collisions.\n
    - "sha256_cbor" provides a reproducible, cross-language compatible hash. It
    serializes objects using canonical CBOR and hashes them with SHA-256.\n
    - "xxhash" uses Pickle serialization with xxHash (128-bit) for faster,
    non-cryptographic hashing. Requires the optional ``xxhash`` package.
    IMPORTANT: Use of a hashing algorithm that is not considered 
    cryptographically secure theoretically increases the risk of hash collisions,
    which can cause undefined behavior or even leak private information in
    multi-tenant environments. Even if collisions are still very unlikely, it is
    important to consider your security risk tolerance against the performance
    benefits before turning this on.\n
    - "xxhash_cbor" combines canonical CBOR serialization with xxHash for
    reproducible hashing. Requires the optional ``xxhash`` package."""
    cpu_offload_gb: float = Field(default=0, ge=0)
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
    cpu_kvcache_space_bytes: int | None = None
    """(CPU backend only) CPU key-value cache space."""
    mamba_page_size_padded: int | None = None
    """ Optional override for mamba page size; used by hybrid mamba/attention
    models to ensure exact alignment with attention page size."""
    mamba_block_size: int | None = Field(default=None, gt=0)
    """Size of a contiguous cache block in number of tokens for mamba cache.
    Can be set only when prefix caching is enabled.
    Value must be a multiple of 8 to align with causal_conv1d kernel."""
    mamba_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (both the conv as well as the
    ssm state). If set to 'auto', the data type will be inferred from the model
    config."""
    mamba_ssm_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (ssm state only, conv state will
    still be controlled by mamba_cache_dtype). If set to 'auto', the data type
    for the ssm state will be determined by mamba_cache_dtype."""
    mamba_cache_mode: MambaCacheMode = "none"
    """The cache strategy for Mamba layers.
    - "none": set when prefix caching is disabled.
    - "all": cache the mamba state of all tokens at position i * block_size. This is 
           the default behavior (for models that support it) when prefix caching is
           enabled.
    - "align": only cache the mamba state of the last token of each scheduler step and
           when the token is at position i * block_size.
    """

    # Will be set after profiling.
    num_gpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    kv_sharing_fast_prefill: bool = False
    """This feature is work in progress and no prefill optimization takes place
    with this flag enabled currently.

    In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254),
    some layers can skip tokens corresponding to prefill. This flag enables
    attention metadata for eligible layers to be overridden with metadata
    necessary for implementing this optimization in some models (e.g. Gemma3n)
    """

    kv_cache_memory_bytes: int | None = None
    """Size of KV Cache per GPU in bytes. By default, this is set to None
    and vllm can automatically infer the kv cache size based on
    gpu_memory_utilization. However, users may want to manually specify
    the kv cache memory size. kv_cache_memory_bytes allows more fine-grain
    control of how much memory gets used when compared with using
    gpu_memory_utilization. Note that kv_cache_memory_bytes
    (when not-None) ignores gpu_memory_utilization"""

    kv_offloading_size: float | None = None
    """Size of the KV cache offloading buffer in GiB. When TP > 1, this is
    the total buffer size summed across all TP ranks. By default, this is set
    to None, which means no KV offloading is enabled. When set, vLLM will
    enable KV cache offloading to CPU using the kv_offloading_backend."""

    kv_offloading_backend: KVOffloadingBackend = "native"
    """The backend to use for KV cache offloading. Supported backends include
    'native' (vLLM native CPU offloading), 'lmcache'.
    KV offloading is only activated when kv_offloading_size is set."""

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
        ignored_factors = {
            # Runtime/derived knobs that don't affect compiled graph shape
            "gpu_memory_utilization",
            "swap_space",
            "is_attention_free",
            "num_gpu_blocks_override",
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
            "cpu_kvcache_space_bytes",
            "mamba_page_size_padded",
            # Post-init/derived counters
            "num_gpu_blocks",
            "num_cpu_blocks",
            # WIP feature toggle not impacting compiled graph shape
            "kv_sharing_fast_prefill",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    @field_validator("cache_dtype", mode="after")
    @classmethod
    def _validate_cache_dtype(cls, cache_dtype: CacheDType) -> CacheDType:
        if cache_dtype.startswith("fp8") or cache_dtype == "int8":
            msg = (
                f"Using {cache_dtype} data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts performance."
            )
            if cache_dtype == "int8":
                msg += " This is an experimental feature and may impact accuracy."
            else:
                msg += (
                    " Meanwhile, it may cause accuracy drop without a proper "
                    "scaling factor."
                )
            logger.info(msg)
        return cache_dtype

    def verify_with_parallel_config(
        self,
        parallel_config: ParallelConfig,
    ) -> None:
        swap_space_bytes = math.ceil(self.swap_space * GiB_bytes)
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = swap_space_bytes * num_gpus_per_node

        msg = (
            f"{format_gib(cpu_memory_usage)} GiB out of the "
            f"{format_gib(total_cpu_memory)} GiB total CPU memory "
            "is allocated for the swap space."
        )
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. %s", msg)
