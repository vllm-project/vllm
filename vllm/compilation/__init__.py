import copy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

import vllm.envs as envs

from .compile_context import get_compile_context


class CompilationConfig(BaseModel):
    """
    Configuration for compilation.
    It has two parts:
    - CudaGraph capture:
        - use_cudagraph: whether to use cudagraph.
            - False: cudagraph inside compilation is not used.
            - True: cudagraph inside compilation is used. It requires
                that all input buffers have fixed addresses.
        - cudagraph_warmup_times: warmup times for cudagraph.
        NOTE: `capture_sizes` is always inferred from
        compilation context.
    - Inductor compilation:
        - use_inductor: whether to use inductor compilation.
            - False: inductor compilation is not used. graph runs in eager.
            - True: inductor compilation is used. one graph for symbolic shape
                is compiled. In addition, compile for different sizes specified
                in inductor_compile_sizes, using configurations
                in inductor_compile_config.
        - inductor_compile_sizes: sizes to compile for inductor.
        - inductor_specialize_as_cudagraph: if True, `inductor_compile_sizes`
            will be set to `capture_sizes`.
        - inductor_compile_config: additional configurations for inductor.
            - None: use default configurations.
        - inductor_passes: additional passes for inductor. It is a dictionary
            from pass name to pass function qualified name. We use function
            name because the config uses json format.
    """
    use_inductor: bool = True
    inductor_specialize_as_cudagraph: bool = False
    inductor_compile_sizes: Optional[List[int]] = Field(default_factory=dict)
    inductor_compile_config: Dict = Field(default_factory=dict)
    inductor_passes: Dict[str, str] = Field(default_factory=dict)

    use_cudagraph: bool = False
    cudagraph_warmup_times: int = 0

    # not configurable, computed after init
    capture_sizes: List[int] = PrivateAttr
    compile_sizes: List[int] = PrivateAttr

    def model_post_init(self, __context: Any) -> None:
        context = get_compile_context()
        context = copy.deepcopy(context) if context is not None else []
        sizes_to_specialize: List[int] = context
        self.capture_sizes = sizes_to_specialize

        if self.inductor_specialize_as_cudagraph:
            assert self.inductor_compile_sizes is None, (
                "inductor_compile_sizes should be None when "
                "inductor_specialize_as_cudagraph is True")
            self.compile_sizes = self.capture_sizes
        else:
            assert self.inductor_compile_sizes is not None, (
                "inductor_compile_sizes should not be None when "
                "inductor_specialize_as_cudagraph is False")
            self.compile_sizes = self.inductor_compile_sizes

        for k, v in self.inductor_passes.items():
            # resolve function from qualified name
            names = v.split(".")
            module = ".".join(names[:-1])
            func_name = names[-1]
            func = __import__(module).__dict__[func_name]
            self.inductor_compile_config[k] = func

        if "post_grad_custom_post_pass" in self.inductor_compile_config:
            from vllm.compilation.backends import fix_functionalization
            from vllm.utils import combine_fx_passes
            self.inductor_compile_config[
                "post_grad_custom_post_pass"] = combine_fx_passes(
                    fix_functionalization,
                    self.inductor_compile_config["post_grad_custom_post_pass"],
                )

    @staticmethod
    def default_config() -> "CompilationConfig":
        config_path = envs.VLLM_TORCH_COMPILE_CONFIG
        if config_path is not None:
            with open(config_path) as json_file:
                config = CompilationConfig.model_validate_json(
                    json_file.read())
        else:
            config = CompilationConfig()

        return config
