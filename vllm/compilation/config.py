import copy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

import vllm.envs as envs
from vllm.logger import init_logger

from .compile_context import get_compile_context

logger = init_logger(__name__)


class CompilationConfig(BaseModel):
    """
    Configuration for compilation.
    It has two parts:
    - CudaGraph capture:
        - use_cudagraph: whether to use cudagraph inside compilation.
            - False: cudagraph inside compilation is not used.
            - True: cudagraph inside compilation is used. It requires
                that all input buffers have fixed addresses.
            Note that this is orthogonal to the cudagraph capture out
            side of compilation.
            TODO: move outside cudagraph logic into compilation.
            torch.compile will handle cudagraph capture logic in the future.
        - cudagraph_capture_sizes: sizes to capture cudagraph.
            - None: capture sizes are inferred from compilation context.
            - List[int]: capture sizes are specified.
        - cudagraph_num_of_warmups: number of warmup runs for cudagraph.
            It means the first several runs will be treated as warmup runs.
            Only after that, the execution will be recorded, and the recorded
            cudagraph will be used for subsequent runs.
    - Inductor compilation:
        - use_inductor: whether to use inductor compilation.
            - False: inductor compilation is not used. graph runs in eager.
            - True: inductor compilation is used. one graph for symbolic shape
                is compiled. In addition, compile for different sizes specified
                in inductor_compile_sizes, using configurations
                in inductor_compile_config.
        - inductor_compile_sizes: sizes to compile for inductor.
        - inductor_specialize_for_cudagraph_no_more_than: an optional integer
            to specialize inductor for cudagraph sizes no more than the
            specified size. It is useful when we want to specialize inductor
            with a subset of cudagraph sizes.
        - inductor_compile_config: additional configurations for inductor.
            - None: use default configurations.
        - inductor_passes: additional passes for inductor. It is a dictionary
            from pass name to pass function qualified name. We use function
            name because the config uses json format. If we pass the config
            from Python, functions can also be passed directly via Python object
            constructor, e.g. `CompilationConfig(inductor_passes={"a": func})`
    
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
    use_inductor: bool = True
    inductor_specialize_for_cudagraph_no_more_than: Optional[int] = None
    inductor_compile_sizes: Optional[List[int]] = Field(default_factory=dict)
    inductor_compile_config: Dict = Field(default_factory=dict)
    inductor_passes: Dict[str, str] = Field(default_factory=dict)

    use_cudagraph: bool = False
    non_cudagraph_ops: List[str] = Field(default_factory=list)
    cudagraph_num_of_warmups: int = 0
    cudagraph_capture_sizes: Optional[List[int]] = None

    # not configurable, computed after init
    compile_sizes: List[int] = PrivateAttr
    capture_sizes: List[int] = PrivateAttr

    def model_post_init(self, __context: Any) -> None:
        for k, v in self.inductor_passes.items():
            if not isinstance(v, str):
                assert callable(v), (
                    f"pass {k} should be a function or a qualified name")
                self.inductor_passes[k] = v
                continue

            # resolve function from qualified name
            names = v.split(".")
            module = ".".join(names[:-1])
            func_name = names[-1]
            func = __import__(module).__dict__[func_name]
            self.inductor_compile_config[k] = func

        from vllm.compilation.backends import fix_functionalization
        from vllm.utils import combine_fx_passes
        if "post_grad_custom_post_pass" in self.inductor_compile_config:
            self.inductor_compile_config[
                "post_grad_custom_post_pass"] = combine_fx_passes(
                    fix_functionalization,
                    self.inductor_compile_config["post_grad_custom_post_pass"],
                )
        else:
            self.inductor_compile_config[
                "post_grad_custom_post_pass"] = fix_functionalization

    def init_during_runtime(self):
        """To complete the initialization of config,
        we need to know the compile context, which is only available
        during the first run of the model.
        """
        context = get_compile_context()
        context = copy.deepcopy(context) if context is not None else []
        sizes_to_specialize: List[int] = context
        if self.cudagraph_capture_sizes is None:
            self.capture_sizes = sizes_to_specialize
        else:
            self.capture_sizes = self.cudagraph_capture_sizes
            logger.info(("cudagraph sizes specified by model runner"
                         " %s is overridden by config %s"),
                        sizes_to_specialize, self.cudagraph_capture_sizes)
        if self.inductor_specialize_for_cudagraph_no_more_than is not None:
            assert self.inductor_compile_sizes is None, (
                "inductor_compile_sizes should be None when "
                "inductor_specialize_for_cudagraph_no_more_than is not None")
            self.compile_sizes = [
                x for x in self.capture_sizes
                if x <= self.inductor_specialize_for_cudagraph_no_more_than
            ]
        else:
            assert self.inductor_compile_sizes is not None, (
                "inductor_compile_sizes should not be None when "
                "inductor_specialize_for_cudagraph_no_more_than is None")
            self.compile_sizes = self.inductor_compile_sizes

    @staticmethod
    def select_and_init_config() -> "CompilationConfig":
        """The order of selecting config is:
        1. Use the config specified in environment variable.
        2. Use the config specified in plugins.
        3. Use the default config.
        """
        config_path = envs.VLLM_TORCH_COMPILE_CONFIG
        if config_path is not None:
            with open(config_path) as json_file:
                config = CompilationConfig.model_validate_json(
                    json_file.read())
        else:
            from vllm.plugins import get_compilation_config
            predefined_config = get_compilation_config()
            config = predefined_config if predefined_config is not None else (
                CompilationConfig())

        config.init_during_runtime()
        return config
