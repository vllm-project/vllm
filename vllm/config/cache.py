# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：KV Cache 的配置，以及 profile 之后回填进来的实际容量
# [CN] 职责：block_size / cache_dtype / gpu_memory_utilization / prefix caching 策略，
# [CN]       以及显存画像完成后写回的 num_gpu_blocks、kv_cache_size_tokens 等结果值。
# [CN] 链路：CacheConfig -> 显存画像(v1/worker/utils.py) -> num_gpu_blocks
# [CN]       -> KVCacheManager 分配 -> 每个 layer 的 KV cache 张量
# [CN] 三条主线：
# [CN]   1. block_size    —— KV cache 的最小分配粒度，同时是 attention kernel 的 tiling 单位
# [CN]   2. cache_dtype   —— KV 量化方式，既决定显存占用也决定能走哪个 kernel
# [CN]   3. prefix caching —— 前缀哈希复用，哈希算法与匹配粒度都在这里配
# [CN] 易错点：gpu_memory_utilization 是**按实例**算的比例，不看别的进程占了多少；
# [CN]         两个实例共用一张卡时两边都要设 0.5，而不是各自 0.92 然后祈祷不 OOM。

from collections.abc import Callable
from dataclasses import field
from functools import cache
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from vllm.config.utils import config, get_from_deprecated_env_if_set
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

logger = init_logger(__name__)

# [CN] 旧名 -> 新名的兼容表。NHD/HND 是早期叫法，内部统一用带.号的结构化名。

_LAYOUT_COMPAT_ALIASES = {
    "NHD": "LBNHC",
    "HND": "LBHNC",
}


# [CN] 名字解析带缓存且不可回退到这里重复查：未知 layout 直接抛錯，
# [CN] 因为 layout 拼错只会表现为「显存里排布不对、结果悄悄错」，很难查。

@cache
def _layout_from_name(layout_name: str) -> KVCacheLayout:
    layout_name = _LAYOUT_COMPAT_ALIASES.get(layout_name, layout_name)
    try:
        return KVCacheLayout[layout_name]
    except KeyError:
        raise ValueError(
            f"Unknown KV cache layout {layout_name!r}. "
            f"Valid layouts: {[m.name for m in KVCacheLayout]}"
        ) from None


# [CN] 允许的 KV cache dtype 全集。分成三类：
# [CN]   auto      —— 跟随模型 dtype
# [CN]   fp8/nvfp4 —— 张量级量化，省显存但有精度损失，需要合适的 scale
# [CN]   *_per_token_head / turboquant_* —— 更细粒度的 per-head scale 量化
# [CN] 注意 fp8 在不同平台语义不同（ROCm 指 e4m3、Gaudi 用 fp8_inc），跨平台配的时候要看清。

CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "nvfp4_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
    "nvfp4_4over6",
]


# [CN] 兼容读取已废弃的环境变量：读到就报错式提示迁移路径。返回 None 表示「用户没通过环境变量设」。

def _get_prefix_cache_retention_interval() -> int | None:
    env_value = get_from_deprecated_env_if_set(
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
        "v0.29",
        "prefix_cache_retention_interval",
    )
    return 0 if env_value is None else int(env_value)


# [CN] Mamba 侧没有 KV cache，只有 conv state 与 ssm state，因此 dtype 范围比 attention 小得多。

MambaDType = Literal["auto", "float32", "float16", "bfloat16"]
MambaCacheMode = Literal["all", "align", "none"]
PrefixCachingHashAlgo = Literal["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"]
KVOffloadingBackend = Literal["native", "lmcache"]


@config
# [CN] 注意本类的字段分成两拨：能由用户配的（有 docstring 明文），
# [CN] 和 init=False 的推导值（profile / 分配后回填）。后者不进 __init__ 签名。

class CacheConfig:
    """Configuration for the KV cache."""

    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16

    # [CN] KV cache 的分配粒度。默认 16，用户显式指定时会置 user_specified_block_size=True
    # [CN] —— 这个布尔值会被 selector 用来决定「是否把 block_size 当作后端选择条件」。

    block_size: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """Size of a contiguous cache block in number of tokens.
    Accepts None (meaning "use default"). After construction, always int."""
    user_specified_block_size: bool = field(default=False, init=False)
    """Whether block_size was explicitly provided. Derived automatically."""
    user_specified_mamba_block_size: bool = field(default=False, init=False)
    """Whether mamba_block_size was explicitly provided. Derived automatically."""
    # [CN] 物理 KV 排布，None 表示尚未决定。由 EngineCore 在所有后端选定后一次性协商定，
    # [CN] 再通过 RPC / KVCacheConfig 同步给各个 worker ——**定稿后不可改**。
    # [CN] 之所以不能提前定：单个后端选择时看不到同伴，无法知道全局哪种 layout 更优。

    kv_cache_layout: str | None = field(default=None, init=False)
    """Resolved physical KV cache layout name (a ``KVCacheLayout`` member).

    ``None`` means the layout has not been resolved yet. The engine core
    resolves it once (``resolve_kv_cache_layout``) before memory profiling, and
    every worker process adopts the resolved name — via the
    ``set_kv_cache_layout`` RPC, or ``KVCacheConfig.kv_cache_layout`` for
    workers spawned after resolution — before the KV cache is allocated. Once
    set the value is final and read with ``get_resolved_kv_cache_layout``,
    which raises on ``None``. Tests and standalone tools may pre-set a value,
    which resolution then honors as-is."""
    # [CN] 前缀哈希的计算粒度。可以比物理 block 更细（比如 32 vs 1024），只要每个 KV group 的
    # [CN] block_size 都能被它整除 —— 这样能在物理块内部获得更细的命中边界。
    # [CN] 它只影响「能不能命中」，不影响「多久存一次」，等价于代码里的 hash_block_size。

    prefix_match_unit: int | None = Field(default=None, gt=0)
    """The finest token boundary (in tokens) a prefix-cache hit can land on.

    Prefix-cache keys are computed every `prefix_match_unit` tokens. It can
    be set finer than the physical KV cache block sizes (e.g. 32 vs a
    1024-token hybrid-model block) as long as every KV cache group's
    `block_size` is divisible by it, enabling cache hits at boundaries
    inside a physical block. It controls matching granularity only, not how
    often states are stored.

    This equals to the `hash_block_size` used throughout the KV cache code.
    """
    # [CN] 本实例可用的显存比例。它是 per-instance 的：多个 vLLM 实例共用一张 GPU 时，
    # [CN] 各设各的比例，互不可见，因此需要人工留够余量。

    gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.92. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    # [CN] KV cache 存储 dtype。量化后显存占用下降、带宽压力减小，但若缺乏合适的 scale 会掉精度。

    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc).
    Some models (namely DeepSeekV3.2) default to fp8, set to bfloat16 to use
    bfloat16 instead, this is an invalid option for models that do not default
    to fp8.
    "nvfp4_4over6" uses the NVFP4 layout and selects between max/6 and max/4
    scales per 16 values by minimizing squared reconstruction error.
    """
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    # [CN] 手动指定 GPU block 数，跳过画像结果。主要用于**复现抢占行为**的测试，生产不要设。

    num_gpu_blocks_override: int | None = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: int | None = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    # [CN] 前缀复用开关。同一段 prompt（含系统提示）的二次请求可以跳过 prefill。

    enable_prefix_caching: bool = True
    """Whether to enable prefix caching."""
    # [CN] 哈希算法选择。sha256 是默认值（安全性最好），xxhash 更快但非密码学强度 ——
    # [CN] 多租户环境里碰撞理论上可能导致串号，属于安全/性能的取舍。
    # [CN] 带 _cbor 后缀的版本用规范化的 CBOR 序列化，保证跨语言可复现。

    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"
    """Set the hash algorithm for prefix caching:

    - "sha256" uses Pickle for object serialization before hashing. This is the current
      default, as SHA256 is the most secure choice to avoid potential hash collisions.
    - "sha256_cbor" provides a reproducible, cross-language compatible hash. It
      serializes objects using canonical CBOR and hashes them with SHA-256.
    - "xxhash" uses Pickle serialization with xxHash (128-bit) for faster,
      non-cryptographic hashing. Requires the optional ``xxhash`` package.
      IMPORTANT: Use of a hashing algorithm that is not considered  cryptographically
      secure theoretically increases the risk of hash collisions, which can cause
      undefined behavior or even leak private information in multi-tenant environments.
      Even if collisions are still very unlikely, it is important to consider your
      security risk tolerance against the performance benefits before turning this on.
    - "xxhash_cbor" combines canonical CBOR serialization with xxHash for
      reproducible hashing. Requires the optional ``xxhash`` package."""
    # [CN] 检查点保留间隔，只对 sliding-window 和 Mamba 组有意义 ——
    # [CN] 这两类的中间状态本来就不是每步都能复用，需要定期打点。
    # [CN] 0 = 只保留语义检查点（如分支汇合处）；None = 密集保留。

    prefix_cache_retention_interval: int | None = Field(
        default_factory=_get_prefix_cache_retention_interval, ge=0
    )
    """Token interval between retained sliding-window and Mamba prefix-cache
    checkpoints. ``0`` retains only semantic checkpoints, including the latest
    replay boundary and shared-prefix junctions. Positive values additionally
    retain periodic checkpoints at the specified interval, which must be a
    multiple of the scheduler block size. ``None`` retains checkpoints densely.
    Applies only to sliding-window and Mamba cache groups."""
    # [CN] 让部分层跳过量化（按层序号或 attention 类型指定）。常见动机是
    # [CN] 某几层对精度特别敏感，逐层回退到高精度比整体不量化划算得多。

    kv_cache_dtype_skip_layers: list[str] = field(default_factory=list)
    """Layer patterns to skip KV cache quantization. Accepts layer indices
    (e.g., '0', '2', '4') or attention type names (e.g., 'sliding_window')."""
    mamba_page_size_padded: int | None = None
    """ Optional override for mamba page size; used by hybrid mamba/attention
    models to ensure exact alignment with attention page size."""
    skip_page_size_padded: int | None = None
    """Optional override for the page size of layers skipped from KV cache
    quantization (``--kv-cache-dtype-skip-layers``); set during block-size
    alignment so unquantized skip layers pad up to the quantized primary's
    page."""
    # [CN] Mamba 的块大小。必须是 8 的倍数以对齐 causal_conv1d kernel 的要求，
    # [CN] 且只有在开启 prefix caching 时才有意义（否则没有 state 可复用）。

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
    # [CN] Mamba 状态缓存策略。'align'（只存调度步末/整块边界）是开启前缀缓存时的默认：
    # [CN] 相比 'all'（每个 block 边界都存），写入频率低得多，而命中率损失有限。

    mamba_cache_mode: MambaCacheMode = "none"
    """The cache strategy for Mamba layers:

    - "none": set when prefix caching is disabled.
    - "all": cache the mamba state of all tokens at position i * block_size.
    - "align": only cache the mamba state of the last token of each scheduler step and
      when the token is at position i * block_size. This is the default when prefix
      caching is enabled.
    """
    replayssm_buffer_len: int = Field(default=16, gt=0)
    """ReplaySSM logical history length B for Mamba2. Triton uses B physical
    rows and FlashInfer uses B+1. Kimi-K3 speculative decode does not use B.
    Default 16."""
    # [CN] ReplaySSM：缓存最近的 SSM 输入、跳过每步全状态写回，只在需要时（flush）落盘。
    # [CN] 用「一些重算」换「每步都写整份 state」的带宽开销。

    use_replayssm: bool = False
    """Use the ReplaySSM Mamba2 decode kernel: cache recent SSM inputs and skip
    the per-step full-state store, writing the checkpoint back only on flush.
    Requires mamba_cache_mode 'none' or 'align' (prefix caching) and the Triton
    or FlashInfer mamba backend; standard (non-speculative) decode only. In align
    mode flushes are most efficient when mamba_block_size is a multiple of
    replayssm_buffer_len, but this is not required."""
    use_kda_recoverssm: bool = field(default=False, init=False)
    """Whether Kimi-K3 KDA uses RecoverSSM speculative decode."""

    # [CN] 以下三个由画 Result 象阶段回填：只有跑一遍 dummy forward 才知道还剩多少显存可用。

    # Will be set after profiling.
    num_gpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    # Set after KV cache initialization.
    # [CN] 不用 num_gpu_blocks * block_size 来算容量：混合模型里一个请求会同时占用多个 KV group，
    # [CN] 简单相乘会高估并发能力，所以这里记的是 group-aware 的实际容量。

    kv_cache_size_tokens: int | None = field(default=None, init=False)
    """Per-DP-engine KV cache capacity in tokens (group-aware). Uses
    group-aware capacity since num_gpu_blocks * block_size can be wrong
    for hybrid models where requests occupy multiple KV cache groups."""
    kv_cache_max_concurrency: float | None = field(default=None, init=False)
    """Per-DP-engine maximum concurrency at max_model_len tokens."""

    kv_sharing_fast_prefill: bool = False
    """In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254),
    some layers can skip tokens corresponding to prefill. This flag enables
    attention metadata for eligible layers to be overridden with metadata
    necessary for implementing this optimization in some models (e.g. Gemma3n)
    NOTE: KV cache sharing is not supported for MRv2 (v2 model runner).
    """

    # [CN] 直接指定 KV cache 字节数，比 gpu_memory_utilization 更精确。
    # [CN] 一旦给了它就**完全忽略** gpu_memory_utilization —— 两者不是叠加关系。

    kv_cache_memory_bytes: int | None = None
    """Size of KV Cache per GPU in bytes. By default, this is set to None
    and vllm can automatically infer the kv cache size based on
    gpu_memory_utilization. However, users may want to manually specify
    the kv cache memory size. kv_cache_memory_bytes allows more fine-grain
    control of how much memory gets used when compared with using
    gpu_memory_utilization. Note that kv_cache_memory_bytes
    (when not-None) ignores gpu_memory_utilization"""

    # [CN] CPU 侧卸载缓冲区大小（GiB）。TP>1 时这个值是所有 TP rank 的**总和**。
    # [CN] 只有设了它才会真正启用卸载，等于 None 就是关闭。

    kv_offloading_size: float | None = None
    """Size of the KV cache offloading buffer in GiB. When TP > 1, this is
    the total buffer size summed across all TP ranks. By default, this is set
    to None, which means no KV offloading is enabled. When set, vLLM will
    enable KV cache offloading to CPU using the kv_offloading_backend."""

    kv_offloading_backend: KVOffloadingBackend = "native"
    """The backend to use for KV cache offloading. Supported backends include
    'native' (vLLM native CPU offloading), 'lmcache'.
    KV offloading is only activated when kv_offloading_size is set."""

    # [CN] 同其他 config：只哈希影响计算图形状的字段。加字段时务必在这两个集合里做选择。

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
        # [CN] 显式列出**不参与**哈希的字段，比列出参与的更可靠：
        # [CN] 忘了排除会造成「配置没变但缓存失效」，只是慢；忘了包含会造成「变了但命中旧图」，是错的。

        ignored_factors = {
            # Runtime/derived knobs that don't affect compiled graph shape
            "gpu_memory_utilization",
            "kv_cache_memory_bytes",
            "is_attention_free",
            "num_gpu_blocks_override",
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
            "prefix_cache_retention_interval",
            # Prefix-caching implementation detail (doesn't affect compiled graph).
            "prefix_match_unit",
            "mamba_page_size_padded",
            "skip_page_size_padded",
            "user_specified_block_size",
            "user_specified_mamba_block_size",
            "_block_size_resolved",
            # Post-init/derived counters
            "num_gpu_blocks",
            "num_cpu_blocks",
            "kv_cache_size_tokens",
            "kv_cache_max_concurrency",
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

    # [CN] 幂等守卫。pydantic 在嵌套模型里会重复触发 validator，没有它会误置 user_specified_* 标记。

    _block_size_resolved: bool = field(default=False, init=False)
    """Guard against pydantic re-running _apply_block_size_default."""

    @field_validator("block_size", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        if value is None:
            return value
        return handler(value)

    @model_validator(mode="after")
    # [CN] 把 block_size 的「用户没设」与「设了默认值」区分开 —— 这个区分一路影响到后端选择。

    def _apply_block_size_default(self) -> "CacheConfig":
        # Pydantic re-runs validators when CacheConfig is nested inside
        # another pydantic model (e.g. VllmConfig). Guard against that.
        # [CN] 重复进入直接返回，避免第二次把 user_specified_* 当成「用户显式设置」。

        if self._block_size_resolved:
            return self
        self._block_size_resolved = True
        if self.block_size is None:
            self.block_size = self.DEFAULT_BLOCK_SIZE
        else:
            self.user_specified_block_size = True
        if self.mamba_block_size is not None:
            self.user_specified_mamba_block_size = True
        return self

    @field_validator("mamba_cache_mode", mode="after")
    @classmethod
    def _validate_mamba_cache_mode(cls, mode: MambaCacheMode) -> MambaCacheMode:
        if mode == "all":
            logger.warning_once(
                "Mamba cache mode 'all' is deprecated and will be removed in an "
                "upcoming release. If this is a problem, please open an issue "
                "at https://github.com/vllm-project/vllm/issues."
            )
        return mode

    @field_validator("cache_dtype", mode="after")
    @classmethod
    def _validate_cache_dtype(cls, cache_dtype: CacheDType) -> CacheDType:
        if kv_cache_uses_per_token_head_scales(cache_dtype):
            logger.info(
                "Using %s data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Dynamic per-token-head scales will be computed at runtime.",
                str(cache_dtype),
            )
        elif is_quantized_kv_cache(cache_dtype):
            logger.info(
                "Using %s data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor",
                str(cache_dtype),
            )
        return cache_dtype

    # [CN] 读取定稿 layout。宁可在未解析时抛错，也不给一个「猜的」默认值：
    # [CN] layout 错了不会崩，只会让 attention 读到排布错误的 KV。

    def get_resolved_kv_cache_layout(self) -> KVCacheLayout:
        if self.kv_cache_layout is None:
            raise ValueError(
                "KV cache layout has not been resolved yet; it is resolved once "
                "by the engine core (resolve_kv_cache_layout) unless explicitly "
                "set by the user."
            )
        return _layout_from_name(self.kv_cache_layout)
