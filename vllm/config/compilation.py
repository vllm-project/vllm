# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from collections import Counter
from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict, Field, TypeAdapter, field_validator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.config.utils import (
    Range,
    config,
    get_hash_factors,
    hash_factors,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import is_torch_equal_or_newer

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = object

logger = init_logger(__name__)


class CompilationMode(enum.IntEnum):
    """The compilation approach used for torch.compile-based compilation of the
    model."""

    NONE = 0
    """No torch.compile compilation is applied, model runs in fully eager pytorch mode.
    The model runs as-is."""
    STOCK_TORCH_COMPILE = 1
    """The standard `torch.compile` compilation pipeline."""
    DYNAMO_TRACE_ONCE = 2
    """Single Dynamo trace through the model, avoiding recompilation."""
    VLLM_COMPILE = 3
    """Custom vLLM Inductor-based backend with caching, piecewise compilation,
    shape specialization, and custom passes."""


class CUDAGraphMode(enum.Enum):
    """Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def decode_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    def has_mode(self, mode: "CUDAGraphMode") -> bool:
        assert not mode.separate_routine()
        if self.separate_routine():
            return mode.value in self.value
        return self == mode

    def requires_piecewise_compilation(self) -> bool:
        return self.has_mode(CUDAGraphMode.PIECEWISE)

    def max_cudagraph_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    def has_full_cudagraphs(self) -> bool:
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    def has_piecewise_cudagraphs(self) -> bool:
        return self.requires_piecewise_compilation()

    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)

    def valid_runtime_modes(self) -> bool:
        return self in [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]

    def __str__(self) -> str:
        return self.name


@config
@dataclass(config=ConfigDict(extra="forbid"))
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config.

    You must pass PassConfig to VLLMConfig constructor via the CompilationConfig
    constructor. VLLMConfig's post_init does further initialization.
    If used outside of the VLLMConfig, some fields may be left in an
    improper state.
    """

    # New flags
    fuse_norm_quant: bool = Field(default=None)
    """Fuse the custom RMSNorm + quant ops."""
    fuse_act_quant: bool = Field(default=None)
    """Fuse the custom SiluMul + quant ops."""
    fuse_attn_quant: bool = Field(default=None)
    """Fuse the custom attention + quant ops."""
    eliminate_noops: bool = Field(default=None)
    """Eliminate no-op ops."""
    enable_sp: bool = Field(default=None)
    """Enable sequence parallelism."""
    fuse_gemm_comms: bool = Field(default=None)
    """Enable async TP."""
    fuse_allreduce_rms: bool = Field(default=None)
    """Enable flashinfer allreduce fusion."""

    # ROCm/AITER specific fusions
    fuse_act_padding: bool = Field(default=None)
    """Fuse the custom RMSNorm + padding ops."""
    fuse_rope_kvcache: bool = Field(default=None)
    """Fuse the QK rope + KV cache ops."""

    fi_allreduce_fusion_max_size_mb: float | None = None
    """The threshold of the communicated tensor sizes under which
    vllm should use flashinfer fused allreduce. Specified as a
    float in MB.
    Unspecified will fallback to default values
    which are compute capability and world size dependent.
        FI_ALLREDUCE_FUSION_MAX_SIZE_MB = {
            90: {
                2: 64,  # 64MB
                4: 2,  # 2MB
                8: 1,  # 1MB
            },
            100: {
                2: 64,  # 64MB
                4: 32,  # 32MB
                8: 1,  # 1MB
            },
        }, where key is the device capability"""
    enable_qk_norm_rope_fusion: bool = False
    """Enable fused Q/K RMSNorm + RoPE pass."""

    # TODO(luka) better pass enabling system.

    def flashinfer_max_size(self, world_size: int) -> int | None:
        """
        Returns the max communication size in bytes for flashinfer
        allreduce fusion for the given world size. Returns None if world size
        is not supported by configs as it's not supported by flashinfer.
        """

        MiB = 1024 * 1024
        FI_SUPPORTED_WORLD_SIZES = [2, 4, 8]
        if world_size not in FI_SUPPORTED_WORLD_SIZES:
            return None
        max_size_mb = self.fi_allreduce_fusion_max_size_mb
        if max_size_mb is None:
            max_size_mb = self.default_fi_allreduce_fusion_max_size_mb().get(world_size)

        return int(max_size_mb * MiB) if max_size_mb is not None else None

    @staticmethod
    def default_fi_allreduce_fusion_max_size_mb() -> dict[int, float]:
        from vllm.compilation.collective_fusion import FI_ALLREDUCE_FUSION_MAX_SIZE_MB
        from vllm.platforms import current_platform

        if not current_platform.is_cuda():
            return {}
        return FI_ALLREDUCE_FUSION_MAX_SIZE_MB.get(
            current_platform.get_device_capability().to_int(), {}
        )

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """

        return hash_factors(get_hash_factors(self, set()))

    @field_validator(
        "fuse_norm_quant",
        "fuse_act_quant",
        "fuse_attn_quant",
        "eliminate_noops",
        "enable_sp",
        "fuse_gemm_comms",
        "fuse_allreduce_rms",
        "fuse_act_padding",
        "fuse_rope_kvcache",
        mode="wrap",
    )
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        if value is None:
            return value
        return handler(value)

    def __post_init__(self) -> None:
        # Handle deprecation and defaults

        if not self.eliminate_noops:
            if self.fuse_norm_quant or self.fuse_act_quant:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm/SiluMul + quant (fp8) fusion might not work"
                )
            if self.fuse_attn_quant:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Attention + quant (fp8) fusion might not work"
                )
            if self.fuse_allreduce_rms:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Allreduce + rms norm + quant (fp8) fusion might not work"
                )
            if self.fuse_act_padding:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm + padding fusion might not work"
                )
        if self.enable_qk_norm_rope_fusion and not current_platform.is_cuda_alike():
            logger.warning_once(
                "QK Norm + RoPE fusion enabled but the current platform is not "
                "CUDA or ROCm. The fusion will be disabled."
            )
            self.enable_qk_norm_rope_fusion = False
        if self.fuse_act_padding and not current_platform.is_rocm():
            logger.warning_once(
                "Padding fusion enabled but the current platform is not ROCm. "
                "The fusion will be disabled."
            )
            self.fuse_act_padding = False
        if self.fuse_rope_kvcache and not current_platform.is_rocm():
            logger.warning_once(
                "KV cache fusion enabled but the current platform is not ROCm. "
                "The fusion will be disabled."
            )
            self.fuse_rope_kvcache = False


class DynamicShapesType(str, enum.Enum):
    """Types of dynamic shapes handling in torch.compile().
    see  Dynamic shapes and vllm guard dropping in torch_compile.md
    for more details."""

    BACKED = "backed"
    """Use backed dynamic shapes. torch.compile() guards on backed dynamic
    shapes and may add guards. Symbols are specialized to 0, 1, or >=2 even
    without encountering branching on those ranges."""

    UNBACKED = "unbacked"
    """Use unbacked dynamic shapes. Guaranteed not to be guarded on and not
    0/1 specialized, but may throw data dependent errors when branches require
    their value without explicit unbacked handling."""

    BACKED_SIZE_OBLIVIOUS = "backed_size_oblivious"
    """Experimental flag that treats backed symbols as unbacked when explicit
    unbacked handling is defined."""


@config
@dataclass(config=ConfigDict(extra="forbid"))
class DynamicShapesConfig:
    """Configuration to control/debug torch compile dynamic shapes."""

    type: DynamicShapesType = DynamicShapesType.BACKED
    """Controls the type of dynamic shapes handling to use with torch.compile().

    - BACKED: Default PyTorch behavior with potential guards ignored.
    - UNBACKED: No guards guaranteed (most sound) but may throw
      data dependent errors.
    - BACKED_SIZE_OBLIVIOUS: Experimental safer alternative to
      backed/unbacked.
    """

    evaluate_guards: bool = False
    """
    A debug mode to detect and fail if Dynamo ever specializes a dynamic shape by
    guarding on it. When True, dynamic shape guards are not dropped from dynamo.
    And a failure will be triggered if a recompilation ever happens due to that.
    This mode requires VLLM_USE_BYTECODE_HOOK to be 0.
    Enabling this allow observing the dynamic shapes guards in the tlparse
    artifacts also.
    When type is backed, aot_compile must be disabled for this mode to work.
    until this change picked up https://github.com/pytorch/pytorch/pull/169239.
    """

    assume_32_bit_indexing: bool = False
    """
    whether all tensor sizes can use 32 bit indexing.
    `True` requires PyTorch 2.10+
    """

    def compute_hash(self) -> str:
        """
        Provide a hash for DynamicShapesConfig
        """

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, {})
        return hash_factors(factors)


@config
@dataclass(config=ConfigDict(extra="forbid"))
class CompilationConfig:
    """Configuration for compilation.

    You must pass CompilationConfig to VLLMConfig constructor.
    VLLMConfig's post_init does further initialization. If used outside of the
    VLLMConfig, some fields will be left in an improper state.

    It has three parts:

    - Top-level Compilation control:
        - [`mode`][vllm.config.CompilationConfig.mode]
        - [`debug_dump_path`][vllm.config.CompilationConfig.debug_dump_path]
        - [`cache_dir`][vllm.config.CompilationConfig.cache_dir]
        - [`backend`][vllm.config.CompilationConfig.backend]
        - [`custom_ops`][vllm.config.CompilationConfig.custom_ops]
        - [`splitting_ops`][vllm.config.CompilationConfig.splitting_ops]
        - [`compile_mm_encoder`][vllm.config.CompilationConfig.compile_mm_encoder]
    - CudaGraph capture:
        - [`cudagraph_mode`][vllm.config.CompilationConfig.cudagraph_mode]
        - [`cudagraph_capture_sizes`]
        [vllm.config.CompilationConfig.cudagraph_capture_sizes]
        - [`max_cudagraph_capture_size`]
        [vllm.config.CompilationConfig.max_cudagraph_capture_size]
        - [`cudagraph_num_of_warmups`]
        [vllm.config.CompilationConfig.cudagraph_num_of_warmups]
        - [`cudagraph_copy_inputs`]
        [vllm.config.CompilationConfig.cudagraph_copy_inputs]
    - Inductor compilation:
        - [`compile_sizes`][vllm.config.CompilationConfig.compile_sizes]
        - [`compile_ranges_split_points`]
            [vllm.config.CompilationConfig.compile_ranges_split_points]
        - [`inductor_compile_config`]
        [vllm.config.CompilationConfig.inductor_compile_config]
        - [`inductor_passes`][vllm.config.CompilationConfig.inductor_passes]
        - custom inductor passes

    Why we have different sizes for cudagraph and inductor:
    - cudagraph: a cudagraph captured for a specific size can only be used
        for the same size. We need to capture all the sizes we want to use.
    - inductor: a graph compiled by inductor for a general shape can be used
        for different sizes. Inductor can also compile for specific sizes,
        where it can have more information to optimize the graph with fully
        static shapes. However, we find the general shape compilation is
        sufficient for most cases. It might be beneficial to compile for
        certain small batchsizes, where inductor is good at optimizing.
    """

    # Top-level Compilation control
    level: int = Field(default=None)
    """
    Level is deprecated and will be removed in the next release,
    either 0.12.0 or 0.11.2 whichever is soonest.
    Please use mode. Currently all levels are mapped to mode.
    """
    # Top-level Compilation control
    mode: CompilationMode = Field(default=None)
    """The compilation approach used for torch.compile-based compilation of the
    model.

    - None: If None, we will select the default compilation mode.
      For V1 engine this is 3.
    - 0: NONE: No torch.compile compilation is applied, model runs in fully
         eager pytorch mode. The model runs as-is.
    - 1: STOCK_TORCH_COMPILE: The standard `torch.compile` compilation pipeline.
    - 2: DYNAMO_TRACE_ONCE: Single Dynamo trace through the model, avoiding
         recompilation by removing guards.
         Requires no dynamic-shape-dependent control-flow.
    - 3: VLLM_COMPILE: Custom vLLM Inductor-based backend with caching,
         piecewise compilation, shape specialization, and custom passes."""
    debug_dump_path: Path | None = None
    """The path to dump the debug information."""
    cache_dir: str = ""
    """The directory to store the compiled graph, to accelerate Inductor
    compilation. By default, it will use model-related information to generate
    a cache directory."""
    compile_cache_save_format: Literal["binary", "unpacked"] = field(
        default_factory=lambda: envs.VLLM_COMPILE_CACHE_SAVE_FORMAT
    )
    """Format for saving torch compile cache:\n
    - "binary": saves as binary file (multiprocess safe)\n
    - "unpacked": saves as directory structure for inspection/debugging
    (NOT multiprocess safe)\n
    Defaults to `VLLM_COMPILE_CACHE_SAVE_FORMAT` if not specified.
    """
    backend: str = ""
    """The backend for compilation. It needs to be a string:

    - "" (empty string): use the default backend ("inductor" on CUDA-alike
    platforms).
    - "eager"/"openxla"/...: use the specified backend registered in PyTorch.
    - "full.module.name": a qualified name which can be used to import the

    backend function.
    We use string to avoid serialization issues when using compilation in a
    distributed setting. When the compilation mode is 1 or 2, the backend is
    used for the compilation directly (it sees the whole graph). When the
    compilation mode is 3, the backend supports both whole graph and piecewise
    compilation, available backends include eager, inductor, and custom backends,
    the latter of which can be defined via `get_compile_backend`. Furthermore,
    compilation is only piecewise if splitting ops is set accordingly and
    use_inductor_graph_partition is off. Note that the default options for
    splitting ops are sufficient for piecewise compilation.
    """
    custom_ops: list[str] = field(default_factory=list)
    """Fine-grained control over which custom ops to enable/disable. Use 'all'
    to enable all, 'none' to disable all. Also specify a list of custom op
    names to enable (prefixed with a '+'), or disable (prefixed with a '-').
    Examples:

    - 'all,-op1' to enable all except op1
    - 'none,+op1,+op2' to enable only op1 and op2

    By default, all custom ops are enabled when running without Inductor and
    disabled when running with Inductor: mode>CompilationMode.NONE and
    backend="inductor".
    Inductor generates (fused) Triton kernels for disabled custom ops."""
    splitting_ops: list[str] | None = None
    """A list of ops to exclude from cudagraphs, used in piecewise compilation.

    The behavior depends on use_inductor_graph_partition:

    - When use_inductor_graph_partition=False (default):
        These ops are used for Dynamo FX-level graph splitting. The graph is
        split at these ops before Inductor compilation, creating separate
        subgraphs for cudagraph capture.

    - When use_inductor_graph_partition=True:
        These ops are used to register Inductor partition rules. The graph
        partitioning happens at Inductor codegen time after all passes and
        fusions are finished, allowing compilation and custom passes to operate
        on the full graph while still excluding these ops from cudagraphs.

    If None, defaults to attention ops for piecewise cudagraphs.
    If empty list [], no ops are excluded (suitable for full cudagraphs)."""
    compile_mm_encoder: bool = False
    """Whether or not to compile the multimodal encoder.
    Currently, this only works for `Qwen2_5_vl` and `mLLaMa4` models
    on selected platforms. Disabled by default until more models
    are supported/tested to work."""

    # Inductor capture
    compile_sizes: list[int | str] | None = None
    """Sizes to compile for inductor. In addition
    to integers, it also supports "cudagraph_capture_sizes" to
    specify the sizes for cudagraph capture."""

    compile_ranges_split_points: list[int] | None = None
    """Split points that represent compile ranges for inductor.
    The compile ranges are
    [1, split_points[0]],
    [split_points[0] + 1, split_points[1]], ...,
    [split_points[-1] + 1, max_num_batched_tokens].
    Compile sizes are also used single element ranges,
    the range is represented as [compile_sizes[i], compile_sizes[i]].

    If a range overlaps with the compile size, graph for compile size
    will be prioritized, i.e. if we have a range [1, 8] and a compile size 4,
    graph for compile size 4 will be compiled and used instead of the graph
    for range [1, 8].
    """

    inductor_compile_config: dict = field(default_factory=dict)
    """Additional configurations for inductor.
    - None: use default configurations."""

    inductor_passes: dict[str, str] = field(default_factory=dict)
    """Additional passes for inductor. It is a dictionary
    from pass name to pass function qualified name. We use function
    name because the config uses JSON format. If we pass the config
    from Python, functions can also be passed directly via Python object
    constructor, e.g. `CompilationConfig(inductor_passes={"a": func})`."""

    # CudaGraph compilation
    cudagraph_mode: CUDAGraphMode = Field(default=None)
    """
    The mode of the cudagraph:

    - NONE, no cudagraph capture.
    - PIECEWISE.
    - FULL.
    - FULL_DECODE_ONLY.
    - FULL_AND_PIECEWISE. (v1 default)

    PIECEWISE mode build piecewise cudagraph only, keeping the cudagraph
    incompatible ops (i.e. some attention ops) outside the cudagraph
    for general flexibility.

    FULL mode: Capture full cudagraph for all batches. Can be good for small
    models or workloads with small prompts; not supported by many backends.
    Generally for performance FULL_AND_PIECEWISE is better.

    FULL_DECODE_ONLY mode: Capture full cudagraph for decode batches only.
    Mixed prefill-decode batches are run without cudagraphs. Can be good for
    decode instances in a P/D setup where prefill is not as important so we
    can save some memory.

    FULL_AND_PIECEWISE mode: Capture full cudagraph for decode batches and
    piecewise cudagraph for prefill and mixed prefill-decode batches.
    This is the most performant mode for most models and is the default.

    Currently, the cudagraph mode is only used for the v1 engine.
    Note that the cudagraph logic is generally orthogonal to the
    compilation logic. While piecewise cudagraphs require piecewise
    compilation (mode=VLLM_COMPILE and non-empty splitting_ops), full
    cudagraphs are supported with and without compilation.

    Warning: This flag is new and subject to change in addition
    more modes may be added.
    """
    cudagraph_num_of_warmups: int = 0
    """Number of warmup runs for cudagraph.
    It means the first several runs will be treated as warmup runs.
    Only after that, the execution will be recorded, and the recorded
    cudagraph will be used for subsequent runs."""
    cudagraph_capture_sizes: list[int] | None = None
    """Sizes to capture cudagraph.
    - None (default): capture sizes are inferred from vllm config.
    - list[int]: capture sizes are specified as given."""
    cudagraph_copy_inputs: bool = False
    """Whether to copy input tensors for
    cudagraph. If the caller can guarantee that the same input buffers
    are always used, it can set this to False. Otherwise, it should
    set this to True, and the compiler will copy the input to an
    internally managed buffer. Default is False.
    Note that this flag is only effective when cudagraph_mode is PIECEWISE.
    """
    cudagraph_specialize_lora: bool = True
    """Whether to create separate cuda graphs for cases with and without active
    LoRA adapters. When set to False, the LoRA-enabled cuda graph will be used
    for all cases, incurring the overhead of running LoRA ops even when no
    adapters are active. Setting this to True will remove this overhead at the
    cost of increased startup time and slightly higher memory usage.
    When `enable_lora` is False, this option has no effect.
    """

    use_inductor_graph_partition: bool = Field(default=None)
    """Use inductor graph partition to split the graph at cudagraph_unsafe ops.
    This partition happens at inductor codegen time after all passes and fusions
    are finished. It generates a single `call` function which wraps
    cudagraph-safe ops into partition functions and leave cudagraph-unsafe ops
    outside the partition functions. For a graph with N cudagraph-unsafe ops
    (e.g., Attention), there would be N+1 partitions. To mark an op as
    cudagraph unsafe, we can add `tags=(torch._C.Tag.cudagraph_unsafe)` when
    register the custom op.

    This config supports both full cudagraph and piecewise cudagraph without
    compiling twice. For piecewise cudagraph, it applies vLLM CUDAGraph wrapper
    to each partition. For N+1 partitions, there would be N+1
    CUDAGraph wrapper instances.

    For full CUDAGraph, we always apply a single CUDAGraph wrapper outside the
    inductor `call` function in the model runner. The top-level full cudagraph
    capture ignores all partitioning.
    """

    pass_config: PassConfig = field(default_factory=PassConfig)
    """Custom inductor passes, see PassConfig for more details"""

    max_cudagraph_capture_size: int = field(default=None)
    """The maximum cudagraph capture size.

    If cudagraph_capture_sizes is specified, this will be set to the largest
    size in that list (or checked for consistency if specified). If
    cudagraph_capture_sizes is not specified, the list of sizes is generated
    automatically following the pattern:

        [1, 2, 4] + list(range(8, 256, 8)) + list(
        range(256, max_cudagraph_capture_size + 1, 16))

    If not specified, max_cudagraph_capture_size is set to min(max_num_seqs*2,
    512) by default. This voids OOM in tight memory scenarios with small
    max_num_seqs, and prevents capture of many large graphs (>512) that would
    greatly increase startup time with limited performance benefit.
    """

    dynamic_shapes_config: DynamicShapesConfig = field(
        default_factory=DynamicShapesConfig
    )
    """Configuration for dynamic shapes options"""

    local_cache_dir: str = field(default=None, init=False)  # type: ignore
    """local cache dir for each rank"""

    # keep track of enabled and disabled custom ops
    enabled_custom_ops: Counter[str] = field(default_factory=Counter, init=False)
    """custom ops that are enabled"""
    disabled_custom_ops: Counter[str] = field(default_factory=Counter, init=False)
    """custom ops that are disabled"""
    traced_files: set[str] = field(default_factory=set, init=False)
    """files that are traced for compilation"""
    compilation_time: float = field(default=0.0, init=False)
    """time taken for compilation"""

    static_forward_context: dict[str, Any] = field(default_factory=dict, init=False)
    """Per-model forward context
    Map from layer name to layer objects that need to be accessed outside
    model code, e.g., Attention, FusedMOE when dp_size>1."""

    static_all_moe_layers: list[str] = field(default_factory=list, init=False)
    """The names of all the MOE layers in the model
    """

    # Attention ops; used for piecewise cudagraphs
    # Use PyTorch operator format: "namespace::name"
    _attention_ops: ClassVar[list[str]] = [
        "vllm::unified_attention",
        "vllm::unified_attention_with_output",
        "vllm::unified_mla_attention",
        "vllm::unified_mla_attention_with_output",
        "vllm::mamba_mixer2",
        "vllm::mamba_mixer",
        "vllm::short_conv",
        "vllm::linear_attention",
        "vllm::plamo2_mamba_mixer",
        "vllm::gdn_attention_core",
        "vllm::kda_attention",
        "vllm::sparse_attn_indexer",
        "vllm::rocm_aiter_sparse_attn_indexer",
    ]

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # Opt-out: default-include declared fields; keep a tiny exclude set;
        # normalize types; keep SHA-256. For nested opaque configs, include a
        # stable identifier (e.g., pass_config.compute_hash()) instead of object id.

        ignored_factors = {
            # Paths/dirs and runtime/metrics that don’t affect compiled graph
            "debug_dump_path",
            "cache_dir",
            "local_cache_dir",
            "traced_files",
            "compilation_time",
            "static_forward_context",
            "pass_config",  # handled separately below
            "dynamic_shapes_config",  # handled separately below
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)

        factors["pass_config"] = self.pass_config.compute_hash()
        factors["dynamic_shapes_config"] = self.dynamic_shapes_config.compute_hash()
        return hash_factors(factors)

    def __repr__(self) -> str:
        exclude = {
            "static_forward_context": True,
            "enabled_custom_ops": True,
            "disabled_custom_ops": True,
            "compilation_time": True,
            "traced_files": True,
            "inductor_compile_config": {
                "post_grad_custom_post_pass": True,
            },
        }

        # exclude default attr in pass_config
        pass_config_exclude = {}
        for attr, default_val in vars(PassConfig()).items():
            if getattr(self.pass_config, attr) == default_val:
                pass_config_exclude[attr] = True
        if pass_config_exclude:
            exclude["pass_config"] = pass_config_exclude

        config = TypeAdapter(CompilationConfig).dump_python(
            self, exclude=exclude, exclude_unset=True
        )

        return str(config)

    __str__ = __repr__

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode_before(cls, value: Any) -> Any:
        """
        Enable parsing the `mode` field from string mode names.
        Accepts both integers (0-3) and string names, like NONE, STOCK_TORCH_COMPILE,
        DYNAMO_TRACE_ONCE, VLLM_COMPILE.
        """
        if isinstance(value, str):
            # Convert string mode name to integer value
            mode_name = value.upper()

            if mode_name not in CompilationMode.__members__:
                raise ValueError(
                    f"Invalid compilation mode: {value}. "
                    f"Valid modes are: {', '.join(CompilationMode.__members__.keys())}"
                )

            return CompilationMode[mode_name]
        return value

    @field_validator("cudagraph_mode", mode="before")
    @classmethod
    def validate_cudagraph_mode_before(cls, value: Any) -> Any:
        """Enable parsing of the `cudagraph_mode` enum type from string."""
        if isinstance(value, str):
            return CUDAGraphMode[value.upper()]
        return value

    @field_validator("pass_config", mode="before")
    @classmethod
    def validate_pass_config_before(cls, value: Any) -> Any:
        """Enable parsing of the `pass_config` field from a dictionary."""
        if isinstance(value, dict):
            return PassConfig(**value)
        return value

    @field_validator("compile_cache_save_format")
    @classmethod
    def validate_compile_cache_save_format(cls, value: str) -> str:
        if value not in ("binary", "unpacked"):
            raise ValueError(
                f"compile_cache_save_format must be 'binary' or 'unpacked', "
                f"got: {value}"
            )
        return value

    @field_validator(
        "level",
        "mode",
        "cudagraph_mode",
        "max_cudagraph_capture_size",
        "use_inductor_graph_partition",
        mode="wrap",
    )
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        if value is None:
            return value
        return handler(value)

    def __post_init__(self) -> None:
        if self.level is not None:
            logger.warning(
                "Level is deprecated and will be removed in the next release,"
                "either 0.12.0 or 0.11.2 whichever is soonest."
                "Use mode instead."
                "If both level and mode are given,"
                "only mode will be used."
            )
            if self.mode is None:
                self.mode = self.level

        count_none = self.custom_ops.count("none")
        count_all = self.custom_ops.count("all")
        assert count_none + count_all <= 1, "Can only specify 'none' or 'all'"

        # TODO(zou3519/luka): There are 2 issues with auto-functionalization V2:
        # 1. A bug in PyTorch, fixed in 2.7:
        #    https://github.com/pytorch/pytorch/issues/147924
        # 2. Custom passes (fusion) rely on auto-functionalization V1 and don't
        #    work with V2. Addressing this will take extra engineering effort
        #    and it is not yet a priority. RFC here:
        #    https://github.com/vllm-project/vllm/issues/14703

        KEY = "enable_auto_functionalized_v2"
        if KEY not in self.inductor_compile_config:
            self.inductor_compile_config[KEY] = False

        for k, v in self.inductor_passes.items():
            if not isinstance(v, str):
                assert callable(v), f"pass {k} should be callable or a qualified name"
                self.inductor_compile_config[k] = (
                    v if isinstance(v, InductorPass) else CallableInductorPass(v)
                )
                continue

            # resolve function from qualified name
            names = v.split(".")
            module = ".".join(names[:-1])
            func_name = names[-1]
            func = __import__(module).__dict__[func_name]
            self.inductor_compile_config[k] = (
                func if isinstance(func, InductorPass) else CallableInductorPass(func)
            )

        if self.pass_config.enable_qk_norm_rope_fusion:
            # TODO(zhuhaoran): support rope native forward match and remove this.
            # Linked issue: https://github.com/vllm-project/vllm/issues/28042
            self.custom_ops.append("+rotary_embedding")
        if self.pass_config.fuse_rope_kvcache:
            if envs.VLLM_ROCM_USE_AITER_TRITON_ROPE:
                logger.warning(
                    "Cannot use VLLM_ROCM_USE_AITER_TRITON_ROPE with "
                    "fuse_rope_kvcache. Disabling fuse_rope_kvcache."
                )
                self.pass_config.fuse_rope_kvcache = False
            # TODO(Rohan138): support rope native forward match and remove this.
            # Linked issue: https://github.com/vllm-project/vllm/issues/28042
            self.custom_ops.append("+rotary_embedding")

        if (
            is_torch_equal_or_newer("2.9.0.dev")
            and "combo_kernels" not in self.inductor_compile_config
            and "benchmark_combo_kernel" not in self.inductor_compile_config
            # (fixme @boyuan) combo kernel does not support cpu yet.
            and not current_platform.is_cpu()
        ):
            # use horizontal fusion, which is useful for fusing qk-norm and
            # qk-rope when query and key have different shapes.
            self.inductor_compile_config["combo_kernels"] = True
            self.inductor_compile_config["benchmark_combo_kernel"] = True

        if self.use_inductor_graph_partition and not is_torch_equal_or_newer(
            "2.9.0.dev"
        ):
            raise ValueError(
                "use_inductor_graph_partition is only "
                "supported with torch>=2.9.0.dev. Set "
                "use_inductor_graph_partition=False instead."
            )

        for op in self.custom_ops:
            if op[0] not in {"+", "-"} and op not in {"all", "none"}:
                raise ValueError(
                    f"Invalid syntax '{op}' for custom op, "
                    "must be 'all', 'none', '+op' or '-op' "
                    "(where 'op' is the registered op name)"
                )

        # Currently only eager and inductor backend are supported.
        # for piecewise compilation. Custom backends are not suppported for
        # piecewise compilation. Update when more backends are supported.
        if self.mode == CompilationMode.VLLM_COMPILE and self.backend not in [
            "",
            "eager",
            "inductor",
        ]:
            raise ValueError(
                f"Invalid backend for piecewise compilation: {self.backend}"
            )

        if self.backend == "":
            self.backend = current_platform.get_compile_backend()

    def init_backend(self, vllm_config: "VllmConfig") -> str | Callable:
        """
        Initialize the backend for the compilation config from a vllm config.
        Arguments:
            vllm_config: The vllm config to initialize the backend from.
        Returns:
            The backend for the compilation config.
        """
        if self.mode is None:
            raise ValueError(
                "No compilation mode is set. This method should only be "
                "called via vllm config where the level is set if none is "
                "provided."
            )
        if self.mode == CompilationMode.NONE:
            raise ValueError("No compilation mode is set.")

        from torch._dynamo.backends.registry import list_backends

        torch_backends = list_backends(exclude_tags=tuple())
        if self.mode in [
            CompilationMode.STOCK_TORCH_COMPILE,
            CompilationMode.DYNAMO_TRACE_ONCE,
        ]:
            if self.backend in torch_backends:
                return self.backend
            return resolve_obj_by_qualname(self.backend)

        assert self.mode == CompilationMode.VLLM_COMPILE
        if self.backend not in ["eager", "inductor"]:
            logger.info("Using OOT custom backend for compilation.")

        from vllm.compilation.backends import VllmBackend

        # TODO[@lucaskabela]: See if we can forward prefix
        # https://github.com/vllm-project/vllm/issues/27045
        return VllmBackend(vllm_config)

    def post_init_cudagraph_sizes(self) -> None:
        """To complete the initialization after cudagraph related
        configs are set. This includes:
        - initialize compile_sizes
        """

        computed_compile_sizes = []
        if self.compile_sizes is not None:
            # de-duplicate the sizes provided by the config
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == "cudagraph_capture_sizes", (
                        "Unrecognized size type in compile_sizes, "
                        f"expect 'cudagraph_capture_sizes', got {x}"
                    )
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes  # type: ignore

        # make sure the sizes are in ascending order
        self.cudagraph_capture_sizes.sort()
        if self.cudagraph_capture_sizes:
            assert self.cudagraph_capture_sizes[-1] == self.max_cudagraph_capture_size

    def set_splitting_ops_for_v1(
        self, all2all_backend: str, data_parallel_size: int = 1
    ):
        # To compatible with OOT hardware plugin platform (for example vllm-ascend)
        # which currently only supports sequence parallelism in eager mode.
        if self.mode != CompilationMode.VLLM_COMPILE:
            if self.splitting_ops is None:
                self.splitting_ops = []
            return

        # NOTE: this function needs to be called only when mode is
        # CompilationMode.VLLM_COMPILE
        assert self.mode == CompilationMode.VLLM_COMPILE, (
            "set_splitting_ops_for_v1 should only be called when "
            "mode is CompilationMode.VLLM_COMPILE"
        )

        if self.pass_config.fuse_attn_quant and not self.use_inductor_graph_partition:
            self.set_splitting_ops_for_attn_fusion()
        else:
            if self.splitting_ops is None:
                # NOTE: When using full cudagraph, instead of setting an empty
                # list and capture the full cudagraph inside the flattened fx
                # graph, we keep the piecewise fx graph structure but capture
                # the full cudagraph outside the fx graph. This reduces some
                # cpu overhead when the runtime batch_size is not cudagraph
                # captured. see https://github.com/vllm-project/vllm/pull/20059
                # for details. Make a copy to avoid mutating the class-level
                # list via reference.
                self.splitting_ops = list(self._attention_ops)
            elif len(self.splitting_ops) == 0:
                if (
                    self.cudagraph_mode == CUDAGraphMode.PIECEWISE
                    or self.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
                ):
                    logger.warning_once(
                        "Using piecewise cudagraph with empty splitting_ops"
                    )
                if self.cudagraph_mode == CUDAGraphMode.PIECEWISE:
                    logger.warning_once(
                        "Piecewise compilation with empty splitting_ops does not "
                        "contain piecewise cudagraph. Setting cudagraph_"
                        "mode to NONE. Hint: If you are using attention "
                        "backends that support cudagraph, consider manually "
                        "setting cudagraph_mode to FULL or FULL_DECODE_ONLY "
                        "to enable full cudagraphs."
                    )
                    self.cudagraph_mode = CUDAGraphMode.NONE
                elif self.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
                    logger.warning_once(
                        "Piecewise compilation with empty splitting_ops does "
                        "not contain piecewise cudagraph. Setting "
                        "cudagraph_mode to FULL."
                    )
                    self.cudagraph_mode = CUDAGraphMode.FULL
                self.splitting_ops = []

        # Disable CUDA graphs for DeepEP high-throughput since its not CG compatible
        if (
            all2all_backend == "deepep_high_throughput"
            and data_parallel_size > 1
            and self.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # TODO: Piecewise Cuda graph might be enabled
            # if torch compile cache key issue fixed
            # See https://github.com/vllm-project/vllm/pull/25093
            logger.info(
                "DeepEP: Disabling CUDA Graphs since DeepEP high-throughput kernels "
                "are optimized for prefill and are incompatible with CUDA Graphs. "
                "In order to use CUDA Graphs for decode-optimized workloads, "
                "use --all2all-backend with another option, such as "
                "deepep_low_latency, pplx, or allgather_reducescatter."
            )
            self.cudagraph_mode = CUDAGraphMode.NONE

    def set_splitting_ops_for_attn_fusion(self):
        assert self.pass_config.fuse_attn_quant
        if self.splitting_ops is None:
            self.splitting_ops = []
            if self.cudagraph_mode.has_piecewise_cudagraphs():
                logger.warning_once(
                    "fuse_attn_quant is incompatible with piecewise "
                    "cudagraph when use_inductor_graph_partition is off. "
                    "In this case, splitting_ops will be set to empty "
                    "list, and cudagraph_mode will be set to FULL. "
                    "Please ensure you are using attention backends that "
                    "support cudagraph or set cudagraph_mode to NONE "
                    "explicitly if encountering any problems."
                )
                self.cudagraph_mode = CUDAGraphMode.FULL

        assert not self.splitting_ops_contain_attention(), (
            "attention ops should not be in splitting_ops when fuse_attn_quant is True"
        )

    def splitting_ops_contain_attention(self) -> bool:
        return self.splitting_ops is not None and all(
            op in self.splitting_ops for op in self._attention_ops
        )

    def is_attention_compiled_piecewise(self) -> bool:
        if not self.splitting_ops_contain_attention():
            return False

        if not self.use_inductor_graph_partition:
            # Dynamo-level FX split case
            return self.mode == CompilationMode.VLLM_COMPILE

        # Inductor partition case
        return self.backend == "inductor" and self.mode != CompilationMode.NONE

    def custom_op_log_check(self):
        """
        This method logs the enabled/disabled custom ops and checks that the
        passed custom_ops field only contains relevant ops.
        It is called at the end of set_current_vllm_config,
        after the custom ops have been instantiated.
        """

        if len(self.enabled_custom_ops) + len(self.disabled_custom_ops) == 0:
            logger.debug("No custom ops found in model.")
            return

        logger.debug("enabled custom ops: %s", self.enabled_custom_ops)
        logger.debug("disabled custom ops: %s", self.disabled_custom_ops)

        all_ops_in_model = self.enabled_custom_ops | self.disabled_custom_ops
        for op in self.custom_ops:
            if op in {"all", "none"}:
                continue

            assert op[0] in {"+", "-"}, (
                "Invalid custom op syntax (should be checked during init)"
            )

            # check if op name exists in model
            op_name = op[1:]
            if op_name not in all_ops_in_model:
                from vllm.model_executor.custom_op import op_registry

                # Does op exist at all or is it just not present in this model?
                # Note: Only imported op classes appear in the registry.
                missing_str = (
                    "doesn't exist (or wasn't imported/registered)"
                    if op_name not in op_registry
                    else "not present in model"
                )

                enable_str = "enabling" if op[0] == "+" else "disabling"
                logger.warning_once(
                    "Op '%s' %s, %s with '%s' has no effect",
                    op_name,
                    missing_str,
                    enable_str,
                    op,
                )

    def is_custom_op_enabled(self, op: str) -> bool:
        if "all" in self.custom_ops:
            return f"-{op}" not in self.custom_ops

        assert "none" in self.custom_ops
        return f"+{op}" in self.custom_ops

    def adjust_cudagraph_sizes_for_spec_decode(
        self, uniform_decode_query_len: int, tensor_parallel_size: int
    ):
        multiple_of = uniform_decode_query_len
        if tensor_parallel_size > 1 and self.pass_config.enable_sp:
            multiple_of = max(uniform_decode_query_len, tensor_parallel_size)
            if (
                multiple_of % uniform_decode_query_len != 0
                or multiple_of % tensor_parallel_size != 0
            ):
                raise ValueError(
                    f"Can't determine cudagraph shapes that are both a "
                    f"multiple of {uniform_decode_query_len} "
                    f"(num_speculative_tokens + 1) required by spec-decode "
                    f"and {tensor_parallel_size} (tensor_parallel_size) "
                    f"required by sequence parallelism please adjust "
                    f"num_speculative_tokens or disable sequence parallelism"
                )

        if not self.cudagraph_capture_sizes or multiple_of <= 1:
            return

        assert self.max_cudagraph_capture_size is not None
        rounded_sizes = sorted(
            set(
                round_up(size, multiple_of)
                for size in self.cudagraph_capture_sizes
                if round_up(size, multiple_of) <= self.max_cudagraph_capture_size
            )
        )

        if len(rounded_sizes) == 0 and multiple_of <= self.max_cudagraph_capture_size:
            # if one valid but would be round_down use that
            rounded_sizes = [multiple_of]

        if len(rounded_sizes) == 0:
            raise ValueError(
                f"No valid cudagraph sizes after rounding to multiple of {multiple_of} "
                f"(num_speculative_tokens + 1 or tp if sequence parallelism is enabled)"
                f" please adjust num_speculative_tokens ({uniform_decode_query_len - 1}"
                f") or max_cudagraph_capture_size ({self.max_cudagraph_capture_size})"
                f" or cudagraph_capture_sizes ({self.cudagraph_capture_sizes})"
            )

        self.max_cudagraph_capture_size = rounded_sizes[-1]
        self.cudagraph_capture_sizes = rounded_sizes

    def get_compile_ranges(self) -> list[Range]:
        """Get the compile ranges for the compilation config."""
        if self.compile_ranges_split_points is None:
            return []
        split_points = sorted(set(self.compile_ranges_split_points))
        return [
            Range(start=s + 1, end=e)
            for s, e in zip([0] + split_points[:-1], split_points)
        ]
