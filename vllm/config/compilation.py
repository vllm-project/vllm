# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import TypeAdapter, field_validator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import is_torch_equal_or_newer

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = object

logger = init_logger(__name__)


class CompilationMode:
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
@dataclass
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config."""

    enable_fusion: bool = False
    """Whether to enable the custom fusion (RMSNorm/SiluMul+quant) pass."""
    enable_attn_fusion: bool = False
    """Whether to enable the custom attention+quant fusion pass."""
    enable_noop: bool = False
    """Whether to enable the custom no-op elimination pass."""
    enable_sequence_parallelism: bool = False
    """Whether to enable sequence parallelism."""
    enable_async_tp: bool = False
    """Whether to enable async TP."""
    enable_fi_allreduce_fusion: bool = False
    """Whether to enable flashinfer allreduce fusion."""
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
    """Whether to enable the fused Q/K RMSNorm + RoPE pass."""

    # TODO(luka) better pass enabling system.

    def flashinfer_max_size(self, world_size: int) -> int | None:
        """
        Returns the max communication size in bytes for flashinfer
        allreduce fusion for the given world size. Returns None if world size
        is not supported by configs as it's not supported by flashinfer.
        """

        MiB = 1024 * 1024
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

    def uuid(self):
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """
        return InductorPass.hash_dict(asdict(self))

    def __post_init__(self) -> None:
        if not self.enable_noop:
            if self.enable_fusion:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm/SiluMul + quant (fp8) fusion might not work"
                )
            if self.enable_attn_fusion:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Attention + quant (fp8) fusion might not work"
                )
            if self.enable_fi_allreduce_fusion:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Allreduce + rms norm + quant (fp8) fusion might not work"
                )
        if self.enable_qk_norm_rope_fusion and not current_platform.is_cuda():
            logger.warning_once(
                "QK Norm + RoPE fusion enabled but the current platform is not "
                "CUDA. The fusion will be disabled."
            )
            self.enable_qk_norm_rope_fusion = False


@config
@dataclass
class CompilationConfig:
    """Configuration for compilation. It has three parts:

    - Top-level Compilation control:
        - [`mode`][vllm.config.CompilationConfig.mode]
        - [`debug_dump_path`][vllm.config.CompilationConfig.debug_dump_path]
        - [`cache_dir`][vllm.config.CompilationConfig.cache_dir]
        - [`backend`][vllm.config.CompilationConfig.backend]
        - [`custom_ops`][vllm.config.CompilationConfig.custom_ops]
        - [`splitting_ops`][vllm.config.CompilationConfig.splitting_ops]
        - [`compile_mm_encoder`][vllm.config.CompilationConfig.compile_mm_encoder]
    - CudaGraph capture:
        - [`use_cudagraph`][vllm.config.CompilationConfig.use_cudagraph]
        - [`cudagraph_mode`][vllm.config.CompilationConfig.cudagraph_mode]
        - [`cudagraph_capture_sizes`]
        [vllm.config.CompilationConfig.cudagraph_capture_sizes]
        - [`max_cudagraph_capture_size`]
        [vllm.config.CompilationConfig.max_cudagraph_capture_size]
        - [`cudagraph_num_of_warmups`]
        [vllm.config.CompilationConfig.cudagraph_num_of_warmups]
        - [`cudagraph_copy_inputs`]
        [vllm.config.CompilationConfig.cudagraph_copy_inputs]
        - [`full_cuda_graph`][vllm.config.CompilationConfig.full_cuda_graph]
    - Inductor compilation:
        - [`use_inductor`][vllm.config.CompilationConfig.use_inductor]
        - [`compile_sizes`][vllm.config.CompilationConfig.compile_sizes]
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
    level: int | None = None
    """
    Level is deprecated and will be removed in the next release,
    either 0.12.0 or 0.11.2 whichever is soonest.
    Please use mode. Currently all levels are mapped to mode.
    """
    # Top-level Compilation control
    mode: int | None = None
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
    compilation mode is 3, the backend is used for the piecewise compilation
    (it sees a part of the graph). The backend can not be custom for compilation
    mode 3, i.e. the backend must be either eager or inductor. Furthermore,
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
    disabled when running with Inductor: mode>=VLLM_COMPILE and use_inductor=True.
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
    compile_mm_encoder: bool = True
    """Whether or not to compile the multimodal encoder.
    Currently, this only works for `Qwen2_5_vl`."""

    # Inductor capture
    use_inductor: bool | None = None
    """
    Whether to use inductor compilation.

    This flag is deprecated and will be removed in the next release 0.12.0.
    Please use the 'backend' option instead.

    - False: inductor compilation is not used. graph runs in eager
        (custom_ops enabled by default).
    - True: inductor compilation is used (custom_ops disabled by default).
        One graph for symbolic shape and one graph per size in compile_sizes
        are compiled using configurations in inductor_compile_config.

    This setting is ignored if mode<VLLM_COMPILE.

    For future compatibility:
    If use_inductor is True, backend="inductor" otherwise backend="eager".
    """
    compile_sizes: list[int | str] | None = None
    """Sizes to compile for inductor. In addition
    to integers, it also supports "cudagraph_capture_sizes" to
    specify the sizes for cudagraph capture."""
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
    cudagraph_mode: CUDAGraphMode | None = None
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
    use_cudagraph: bool = True
    """Whether to use cudagraph inside compilation:

    - False: cudagraph inside compilation is not used.\n
    - True: cudagraph inside compilation is used. It requires
        that all input buffers have fixed addresses, and all
        splitting ops write their outputs to input buffers.

    Warning: This flag is deprecated and will be removed in the next major or
    minor release, i.e. v0.11.0 or v1.0.0. Please use cudagraph_mode=FULL_AND
    _PIECEWISE instead.
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
    full_cuda_graph: bool | None = False
    """whether to use a full cuda graph for the entire forward pass rather than
    splitting certain operations such as attention into subgraphs. Thus this
    flag cannot be used together with splitting_ops. This may provide
    performance benefits for smaller models.
    Warning: This flag is deprecated and will be removed in the next major or
    minor release, i.e. v0.11.0 or v1.0.0. Please use cudagraph_mode=
    FULL_AND_PIECEWISE instead.
    """
    cudagraph_specialize_lora: bool = True
    """Whether to create separate cuda graphs for cases with and without active
    LoRA adapters. When set to False, the LoRA-enabled cuda graph will be used
    for all cases, incurring the overhead of running LoRA ops even when no
    adapters are active. Setting this to True will remove this overhead at the
    cost of increased startup time and slightly higher memory usage.
    When `enable_lora` is False, this option has no effect.
    """

    use_inductor_graph_partition: bool = False
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

    max_cudagraph_capture_size: int | None = field(default=None)
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
    local_cache_dir: str = field(default=None, init=False)  # type: ignore
    """local cache dir for each rank"""
    bs_to_padded_graph_size: list[int] = field(
        default=None,  # type: ignore
        init=False,
    )
    """optimization:
    Intuitively, bs_to_padded_graph_size should be dict[int, int].
    since we know all keys are in a range [0, max_cudagraph_capture_size],
    we can optimize it to list[int] for better lookup performance."""

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
    ]

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
        factors.append(self.mode)
        factors.append(self.backend)
        factors.append(self.custom_ops)
        factors.append(self.splitting_ops)
        factors.append(self.use_inductor)
        factors.append(self.use_inductor_graph_partition)
        factors.append(self.inductor_compile_config)
        factors.append(self.inductor_passes)
        factors.append(self.pass_config.uuid())
        factors.append(self.compile_cache_save_format)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __repr__(self) -> str:
        exclude = {
            "static_forward_context": True,
            "enabled_custom_ops": True,
            "disabled_custom_ops": True,
            "compilation_time": True,
            "bs_to_padded_graph_size": True,
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

    @field_validator("cudagraph_mode", mode="before")
    @classmethod
    def validate_cudagraph_mode_before(cls, value: Any) -> Any:
        """
        enable parse the `cudagraph_mode` enum type from string
        """
        if isinstance(value, str):
            return CUDAGraphMode[value.upper()]
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

        if is_torch_equal_or_newer("2.6"):
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

        if isinstance(self.pass_config, dict):
            self.pass_config = PassConfig(**self.pass_config)

        if self.pass_config.enable_qk_norm_rope_fusion:
            # TODO(zhuhaoran): support rope native forward match and remove this.
            # Linked issue: https://github.com/vllm-project/vllm/issues/28042
            self.custom_ops.append("+rotary_embedding")

        if (
            is_torch_equal_or_newer("2.9.0.dev")
            and "combo_kernels" not in self.inductor_compile_config
            and "benchmark_combo_kernel" not in self.inductor_compile_config
        ):
            # use horizontal fusion, which is useful for fusing qk-norm and
            # qk-rope when query and key have different shapes.
            self.inductor_compile_config["combo_kernels"] = True
            self.inductor_compile_config["benchmark_combo_kernel"] = True

        # migrate the deprecated flags
        if not self.use_cudagraph:
            logger.warning(
                "use_cudagraph is deprecated, use cudagraph_mode=NONE instead."
            )
            if (
                self.cudagraph_mode is not None
                and self.cudagraph_mode != CUDAGraphMode.NONE
            ):
                raise ValueError(
                    "use_cudagraph and cudagraph_mode are mutually"
                    " exclusive, prefer cudagraph_mode since "
                    "use_cudagraph is deprecated."
                )
            self.cudagraph_mode = CUDAGraphMode.NONE
        if self.full_cuda_graph:
            logger.warning(
                "full_cuda_graph is deprecated, use cudagraph_mode=FULL instead."
            )
            if (
                self.cudagraph_mode is not None
                and not self.cudagraph_mode.has_full_cudagraphs()
            ):
                raise ValueError(
                    "full_cuda_graph and cudagraph_mode are "
                    "mutually exclusive, prefer cudagraph_mode "
                    "since full_cuda_graph is deprecated."
                )
            self.cudagraph_mode = CUDAGraphMode.FULL

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

        if self.use_inductor is not None:
            logger.warning_once(
                "The 'use_inductor' flag is deprecated and will be "
                "removed in the next release (v0.12.0). "
                "Please use the 'backend' option instead.",
            )
            self.backend = "inductor" if self.use_inductor else "eager"

        if self.backend == "":
            self.backend = current_platform.simple_compile_backend

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
                "No compilation mode is set. This method should only be \
                called via vllm config where the level is set if none is \
                provided."
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
            raise ValueError(
                f"Invalid backend for piecewise compilation: {self.backend}"
            )

        from vllm.compilation.backends import VllmBackend

        # TODO[@lucaskabela]: See if we can forward prefix
        # https://github.com/vllm-project/vllm/issues/27045
        return VllmBackend(vllm_config)

    def post_init_cudagraph_sizes(self) -> None:
        """To complete the initialization after cudagraph related
        configs are set. This includes:
        - initialize compile_sizes
        - pre-compute the mapping bs_to_padded_graph_size
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

        # pre-compute the mapping from batch size to padded graph size
        self.bs_to_padded_graph_size = [
            0 for i in range(self.max_cudagraph_capture_size + 1)
        ]
        for end, start in zip(
            self.cudagraph_capture_sizes + [self.max_cudagraph_capture_size + 1],
            [0] + self.cudagraph_capture_sizes,
        ):
            for bs in range(start, end):
                if bs == start:
                    self.bs_to_padded_graph_size[bs] = start
                else:
                    self.bs_to_padded_graph_size[bs] = end

    def set_splitting_ops_for_v1(self):
        # NOTE: this function needs to be called only when mode is
        # CompilationMode.VLLM_COMPILE
        assert self.mode == CompilationMode.VLLM_COMPILE, (
            "set_splitting_ops_for_v1 should only be called when "
            "mode is CompilationMode.VLLM_COMPILE"
        )

        if self.use_inductor_graph_partition:
            self.set_splitting_ops_for_inductor_graph_partition()
            return

        if self.pass_config.enable_attn_fusion:
            # here use_inductor_graph_partition is False
            self.set_splitting_ops_for_attn_fusion()
            return

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
            logger.warning_once("Using piecewise compilation with empty splitting_ops")
            if self.cudagraph_mode == CUDAGraphMode.PIECEWISE:
                logger.warning_once(
                    "Piecewise compilation with empty splitting_ops do not"
                    "contains piecewise cudagraph. Setting cudagraph_"
                    "mode to NONE. Hint: If you are using attention backends "
                    "that support cudagraph, consider manually setting "
                    "cudagraph_mode to FULL or FULL_DECODE_ONLY to enable "
                    "full cudagraphs."
                )
                self.cudagraph_mode = CUDAGraphMode.NONE
            elif self.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
                logger.warning_once(
                    "Piecewise compilation with empty splitting_ops do not "
                    "contains piecewise cudagraph. Setting cudagraph_mode "
                    "to FULL."
                )
                self.cudagraph_mode = CUDAGraphMode.FULL
            self.splitting_ops = []

    def set_splitting_ops_for_inductor_graph_partition(self):
        assert self.use_inductor_graph_partition
        if self.splitting_ops is None:
            self.splitting_ops = list(self._attention_ops)

    def set_splitting_ops_for_attn_fusion(self):
        assert self.pass_config.enable_attn_fusion
        # For dynamo-partition (non-inductor) attention fusion,
        # set splitting_ops to empty to avoid splitting at attention ops
        self.splitting_ops = []
        if self.cudagraph_mode.has_piecewise_cudagraphs():
            logger.warning_once(
                "enable_attn_fusion is incompatible with piecewise "
                "cudagraph when use_inductor_graph_partition is off. "
                "In this case, splitting_ops will be set to empty "
                "list, and cudagraph_mode will be set to FULL. "
                "Please ensure you are using attention backends that "
                "support cudagraph or set cudagraph_mode to NONE "
                "explicitly if encountering any problems."
            )
            self.cudagraph_mode = CUDAGraphMode.FULL

        assert not self.splitting_ops_contain_attention(), (
            "attention ops should not be in splitting_ops "
            "when enable_attn_fusion is True"
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
        return self.backend == "inductor" and self.mode > CompilationMode.NONE

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
                from vllm.model_executor.custom_op import CustomOp

                # Does op exist at all or is it just not present in this model?
                # Note: Only imported op classes appear in the registry.
                missing_str = (
                    "doesn't exist (or wasn't imported/registered)"
                    if op_name not in CustomOp.op_registry
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
