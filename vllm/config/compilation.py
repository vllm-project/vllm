# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import hashlib
from collections import Counter
from dataclasses import asdict, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Union

from pydantic import TypeAdapter, field_validator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils import is_torch_equal_or_newer, resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = object

logger = init_logger(__name__)


class CompilationLevel:
    # constants for the levels of the compilation process
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3


class CUDAGraphMode(enum.Enum):
    """ Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def decode_mode(self) -> 'CUDAGraphMode':
        return CUDAGraphMode(self.value[0]) if \
            self.separate_routine() else self

    def mixed_mode(self) -> 'CUDAGraphMode':
        return CUDAGraphMode(self.value[1]) if \
            self.separate_routine() else self

    def requires_piecewise_compilation(self) -> bool:
        return (self.decode_mode() == CUDAGraphMode.PIECEWISE
                or self.mixed_mode() == CUDAGraphMode.PIECEWISE)

    def max_cudagraph_mode(self) -> 'CUDAGraphMode':
        return CUDAGraphMode(max(
            self.value)) if self.separate_routine() else self

    def has_full_cudagraphs(self) -> bool:
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)


@config
@dataclass
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config."""

    enable_fusion: bool = field(default_factory=lambda: not envs.VLLM_USE_V1)
    """Whether to enable the custom fusion (RMSNorm/SiluMul+quant) pass."""
    enable_attn_fusion: bool = False
    """Whether to enable the custom attention+quant fusion pass."""
    enable_noop: bool = field(default_factory=lambda: not envs.VLLM_USE_V1)
    """Whether to enable the custom no-op elimination pass."""
    enable_sequence_parallelism: bool = False
    """Whether to enable sequence parallelism."""
    enable_async_tp: bool = False
    """Whether to enable async TP."""
    enable_fi_allreduce_fusion: bool = False
    """Whether to enable flashinfer allreduce fusion."""
    fi_allreduce_fusion_max_token_num: int = 16384
    """Max number of tokens to used in flashinfer allreduce fusion."""

    # TODO(luka) better pass enabling system.

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
                    "RMSNorm/SiluMul + quant (fp8) fusion might not work")
            if self.enable_attn_fusion:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Attention + quant (fp8) fusion might not work")


@config
@dataclass
class CompilationConfig:
    """Configuration for compilation. It has three parts:

    - Top-level Compilation control:
        - [`level`][vllm.config.CompilationConfig.level]
        - [`debug_dump_path`][vllm.config.CompilationConfig.debug_dump_path]
        - [`cache_dir`][vllm.config.CompilationConfig.cache_dir]
        - [`backend`][vllm.config.CompilationConfig.backend]
        - [`custom_ops`][vllm.config.CompilationConfig.custom_ops]
        - [`splitting_ops`][vllm.config.CompilationConfig.splitting_ops]
    - CudaGraph capture:
        - [`use_cudagraph`][vllm.config.CompilationConfig.use_cudagraph]
        - [`cudagraph_mode`][vllm.config.CompilationConfig.cudagraph_mode]
        - [`cudagraph_capture_sizes`]
        [vllm.config.CompilationConfig.cudagraph_capture_sizes]
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
    level: Optional[int] = None
    """The level of compilation:

    - None: If None, we will select the default compilation level.
      For V1 engine this is 3, for V0 engine this is 0.
    - 0: no compilation.
    - 1: dynamo as is.
    - 2: dynamo once.
    - 3: piecewise compilation."""
    debug_dump_path: str = ""
    """The path to dump the debug information."""
    cache_dir: str = ""
    """The directory to store the compiled graph, to accelerate Inductor
    compilation. By default, it will use model-related information to generate
    a cache directory."""
    backend: str = ""
    """The backend for compilation. It needs to be a string:

    - "" (empty string): use the default backend.
    - "eager"/"openxla"/...: use the specified backend registered in PyTorch.
    - "full.module.name": a qualified name which can be used to import the

    backend function.
    We use string to avoid serialization issues when using compilation in a
    distributed setting. When the compilation level is 1 or 2, the backend is
    used for the compilation directly (it sees the whole graph). When the
    compilation level is 3, the backend is used for the piecewise compilation
    (it sees a part of the graph)."""
    custom_ops: list[str] = field(default_factory=list)
    """Fine-grained control over which custom ops to enable/disable. Use 'all'
    to enable all, 'none' to disable all. Also specify a list of custom op
    names to enable (prefixed with a '+'), or disable (prefixed with a '-').
    Examples:

    - 'all,-op1' to enable all except op1
    - 'none,+op1,+op2' to enable only op1 and op2

    By default, all custom ops are enabled when running without Inductor and
    disabled when running with Inductor: level>=PIECEWISE and use_inductor=True.
    Inductor generates (fused) Triton kernels for disabled custom ops."""
    splitting_ops: Optional[list[str]] = None
    """A list of ops to split the full graph into subgraphs, used in piecewise
    compilation."""

    # Inductor capture
    use_inductor: bool = True
    """Whether to use inductor compilation:

    - False: inductor compilation is not used. graph runs in eager
        (custom_ops enabled by default).
    - True: inductor compilation is used (custom_ops disabled by default).
        One graph for symbolic shape and one graph per size in compile_sizes
        are compiled using configurations in inductor_compile_config.

    This setting is ignored if level<PIECEWISE."""
    compile_sizes: Optional[list[Union[int, str]]] = None
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
    cudagraph_mode: Optional[CUDAGraphMode] = None
    """
    The mode of the cudagraph:

    - NONE, no cudagraph capture.
    - PIECEWISE. (v1 default)
    - FULL.
    - FULL_DECODE_ONLY.
    - FULL_AND_PIECEWISE.

    PIECEWISE mode build piecewise cudagraph only, keeping the cudagraph
    incompatible ops (i.e. some attention ops) outside the cudagraph
    for general flexibility.
    This is the default mode.

    FULL mode: Capture full cudagraph for all batches. Can be good for small
    models or workloads with small prompts; not supported by many backends.
    Generally for performance FULL_AND_PIECEWISE is better.
    
    FULL_DECODE_ONLY mode: Capture full cudagraph for decode batches only.
    Mixed prefill-decode batches are run without cudagraphs. Can be good for
    decode instances in a P/D setup where prefill is not as important so we
    can save some memory.
    
    FULL_AND_PIECEWISE mode: Capture full cudagraph for decode batches and
    piecewise cudagraph for prefill and mixed prefill-decode batches.
    This is like the most performant mode for most models.

    Currently, the cudagraph mode is only used for the v1 engine.
    Note that the cudagraph logic is generally orthogonal to the 
    compilation logic. While piecewise cudagraphs require piecewise 
    compilation (level=PIECEWISE and non-empty splitting_ops), full
    cudagraphs are supported with and without compilation.
    
    Warning: This flag is new and subject to change in addition 
    more modes may be added.
    """
    use_cudagraph: bool = True
    """Whether to use cudagraph inside compilation.
    - False: cudagraph inside compilation is not used.
    - True: cudagraph inside compilation is used. It requires
        that all input buffers have fixed addresses, and all
        splitting ops write their outputs to input buffers.
    In the vLLM V1 Engine, this flag only applies for
    CompilationLevel.PIECEWISE (aka -O3).
    Note that this is orthogonal to the cudagraph capture logic
    outside of compilation.
    Warning: This flag is deprecated and will be removed in the next major or
    minor release, i.e. v0.11.0 or v1.0.0. Please use cudagraph_mode instead.
    """
    cudagraph_num_of_warmups: int = 0
    """Number of warmup runs for cudagraph.
    It means the first several runs will be treated as warmup runs.
    Only after that, the execution will be recorded, and the recorded
    cudagraph will be used for subsequent runs."""
    cudagraph_capture_sizes: Optional[list[int]] = None
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
    full_cuda_graph: Optional[bool] = False
    """whether to use a full cuda graph for the entire forward pass rather than
    splitting certain operations such as attention into subgraphs. Thus this
    flag cannot be used together with splitting_ops. This may provide
    performance benefits for smaller models.
    Warning: This flag is deprecated and will be removed in the next major or
    minor release, i.e. v0.11.0 or v1.0.0. Please use cudagraph_mode instead.
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

    max_capture_size: int = field(default=None, init=False)  # type: ignore
    """not configurable, computed after init"""
    local_cache_dir: str = field(default=None, init=False)  # type: ignore
    """local cache dir for each rank"""
    bs_to_padded_graph_size: list[int] = field(
        default=None,  # type: ignore
        init=False)
    """optimization:
    Intuitively, bs_to_padded_graph_size should be dict[int, int].
    since we know all keys are in a range [0, max_capture_size],
    we can optimize it to list[int] for better lookup performance."""

    # keep track of enabled and disabled custom ops
    enabled_custom_ops: Counter[str] = field(default_factory=Counter,
                                             init=False)
    """custom ops that are enabled"""
    disabled_custom_ops: Counter[str] = field(default_factory=Counter,
                                              init=False)
    """custom ops that are disabled"""
    traced_files: set[str] = field(default_factory=set, init=False)
    """files that are traced for compilation"""
    compilation_time: float = field(default=0.0, init=False)
    """time taken for compilation"""

    static_forward_context: dict[str, Any] = field(default_factory=dict,
                                                   init=False)
    """Per-model forward context
    Map from layer name to layer objects that need to be accessed outside
    model code, e.g., Attention, FusedMOE when dp_size>1."""

    # Attention ops; used for piecewise cudagraphs
    _attention_ops: ClassVar[list[str]] = [
        "vllm.unified_attention",
        "vllm.unified_attention_with_output",
        "vllm.mamba_mixer2",
        "vllm.mamba_mixer",
        "vllm.short_conv",
        "vllm.linear_attention",
        "vllm.plamo2_mamba_mixer",
        "vllm.gdn_attention",
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
        factors.append(self.level)
        factors.append(self.backend)
        factors.append(self.custom_ops)
        factors.append(self.splitting_ops)
        factors.append(self.use_inductor)
        factors.append(self.inductor_compile_config)
        factors.append(self.inductor_passes)
        factors.append(self.pass_config.uuid())
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

        return TypeAdapter(CompilationConfig).dump_json(
            self,
            exclude=exclude,  # type: ignore[arg-type]
            exclude_unset=True).decode()

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

    def __post_init__(self) -> None:
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
            KEY = 'enable_auto_functionalized_v2'
            if KEY not in self.inductor_compile_config:
                self.inductor_compile_config[KEY] = False

        for k, v in self.inductor_passes.items():
            if not isinstance(v, str):
                assert callable(v), (
                    f"pass {k} should be callable or a qualified name")
                self.inductor_compile_config[k] = v if isinstance(
                    v, InductorPass) else CallableInductorPass(v)
                continue

            # resolve function from qualified name
            names = v.split(".")
            module = ".".join(names[:-1])
            func_name = names[-1]
            func = __import__(module).__dict__[func_name]
            self.inductor_compile_config[k] = func if isinstance(
                func, InductorPass) else CallableInductorPass(func)

        if isinstance(self.pass_config, dict):
            self.pass_config = PassConfig(**self.pass_config)

        # migrate the deprecated flags
        if not self.use_cudagraph:
            logger.warning("use_cudagraph is deprecated, use "
                           "cudagraph_mode=NONE instead.")
            if self.cudagraph_mode is not None:
                raise ValueError(
                    "use_cudagraph and cudagraph_mode are mutually"
                    " exclusive, prefer cudagraph_mode since "
                    "use_cudagraph is deprecated.")
            self.cudagraph_mode = CUDAGraphMode.NONE
        if self.full_cuda_graph:
            logger.warning("full_cuda_graph is deprecated, use "
                           "cudagraph_mode=FULL instead.")
            if self.cudagraph_mode is not None:
                raise ValueError("full_cuda_graph and cudagraph_mode are "
                                 "mutually exclusive, prefer cudagraph_mode "
                                 "since full_cuda_graph is deprecated.")
            self.cudagraph_mode = CUDAGraphMode.FULL

        if (self.use_inductor_graph_partition
                and not is_torch_equal_or_newer("2.9.0.dev")):
            raise ValueError("use_inductor_graph_partition is only "
                             "supported with torch>=2.9.0.dev. Set "
                             "use_inductor_graph_partition=False instead.")

    def init_backend(self, vllm_config: "VllmConfig") -> Union[str, Callable]:
        if self.level == CompilationLevel.NO_COMPILATION:
            raise ValueError("No compilation level is set.")

        from torch._dynamo.backends.registry import list_backends
        torch_backends = list_backends(exclude_tags=tuple())
        if self.level in [
                CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE
        ]:
            if self.backend == "":
                return "eager"
            if self.backend in torch_backends:
                return self.backend
            return resolve_obj_by_qualname(self.backend)

        # TODO: pass user-specified backend to piecewise compilation
        # merge with the config use_inductor
        assert self.level == CompilationLevel.PIECEWISE

        from vllm.compilation.backends import VllmBackend
        return VllmBackend(vllm_config)

    def init_with_cudagraph_sizes(self,
                                  cudagraph_capture_sizes: list[int]) -> None:
        """To complete the initialization of config,
        we need to know the cudagraph sizes."""

        if self.cudagraph_capture_sizes is None:
            self.cudagraph_capture_sizes = cudagraph_capture_sizes
        else:
            # de-duplicate the sizes provided by the config
            dedup_sizes = list(set(self.cudagraph_capture_sizes))
            if len(dedup_sizes) < len(self.cudagraph_capture_sizes):
                logger.info(("cudagraph sizes specified by model runner"
                             " %s is overridden by config %s"),
                            cudagraph_capture_sizes, dedup_sizes)
            self.cudagraph_capture_sizes = dedup_sizes

        computed_compile_sizes = []
        if self.compile_sizes is not None:
            # de-duplicate the sizes provided by the config
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == "cudagraph_capture_sizes", \
                    "Unrecognized size type in compile_sizes, " \
                    f"expect 'cudagraph_capture_sizes', got {x}"
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes  # type: ignore

        # sort to make sure cudagraph capture sizes are in descending order
        self.cudagraph_capture_sizes.sort(reverse=True)
        self.max_capture_size = self.cudagraph_capture_sizes[
            0] if self.cudagraph_capture_sizes else 0

        # pre-compute the mapping from batch size to padded graph size
        self.bs_to_padded_graph_size = [
            0 for i in range(self.max_capture_size + 1)
        ]
        for end, start in zip(self.cudagraph_capture_sizes,
                              self.cudagraph_capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.bs_to_padded_graph_size[bs] = start
                else:
                    self.bs_to_padded_graph_size[bs] = end
        self.bs_to_padded_graph_size[
            self.max_capture_size] = self.max_capture_size

    def set_splitting_ops_for_v1(self):
        # NOTE: this function needs to be called only when level is
        # CompilationLevel.PIECEWISE
        assert self.level == CompilationLevel.PIECEWISE, (
            "set_splitting_ops_for_v1 should only be called when "
            "level is CompilationLevel.PIECEWISE")

        use_inductor_graph_partition_msg = (
            "When use_inductor_graph_partition=True, splitting_ops "
            "are ignored and set to an empty list. Instead, "
            "\"tags=(torch._C.Tag.cudagraph_unsafe, ),\" is "
            "used to annotate custom ops for graph partition.")

        if self.splitting_ops is None:
            if self.use_inductor_graph_partition:
                # When using inductor graph partition, we set splitting_ops
                # to be empty and rely on torch._C.Tag.cudagraph_unsafe to
                # annotate custom ops as splitting ops.
                logger.warning_once(use_inductor_graph_partition_msg)
                self.splitting_ops = []
            else:
                # NOTE: When using full cudagraph, instead of setting an empty
                # list and capture the full cudagraph inside the flattened fx
                # graph, we keep the piecewise fx graph structure but capture
                # the full cudagraph outside the fx graph. This reduces some
                # cpu overhead when the runtime batch_size is not cudagraph
                # captured. see https://github.com/vllm-project/vllm/pull/20059
                # for details. make a copy to avoid mutating the class-level
                # list via reference.
                self.splitting_ops = list(self._attention_ops)
        elif len(self.splitting_ops) == 0:
            logger.warning_once(
                "Using piecewise compilation with empty "
                "splitting_ops and use_inductor_graph_partition"
                f"={self.use_inductor_graph_partition}.")
            if (self.cudagraph_mode == CUDAGraphMode.PIECEWISE
                    and not self.use_inductor_graph_partition):
                logger.warning_once(
                    "When compilation level is piecewise with empty "
                    "splitting_ops, PIECEWISE cudagraph_mode will be "
                    "treated as FULL cudagraph_mode. Please ensure you are "
                    "using attention backends that support cudagraph or set "
                    "cudagraph_mode to NONE explicitly if encountering "
                    "any problems.")
                self.cudagraph_mode = CUDAGraphMode.FULL
            self.splitting_ops = []
        elif self.use_inductor_graph_partition:
            logger.warning_once(use_inductor_graph_partition_msg)
            self.splitting_ops = []

    def splitting_ops_contain_attention(self) -> bool:
        return self.splitting_ops is not None and all(
            op in self.splitting_ops for op in self._attention_ops)

    def is_attention_compiled_piecewise(self) -> bool:
        use_fx_graph_piecewise_compilation = (
            self.level == CompilationLevel.PIECEWISE
            and self.splitting_ops_contain_attention())

        inductor_used = (self.level == CompilationLevel.PIECEWISE
                         and self.use_inductor) or (
                             self.level >= CompilationLevel.DYNAMO_AS_IS
                             and self.backend == "inductor")
        use_inductor_piecewise_compilation = (
            inductor_used and self.use_inductor_graph_partition
            and not self.splitting_ops_contain_attention())

        return use_fx_graph_piecewise_compilation or \
            use_inductor_piecewise_compilation
