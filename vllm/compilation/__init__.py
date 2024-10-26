import copy
import json
from typing import Dict, List

from pydantic import BaseModel, Field, PrivateAttr

import vllm.envs as envs

from .compile_context import get_compile_context


class CompilationConfig(BaseModel):
    """
    Configuration for compilation.
    It has two parts:
    - CudaGraph capture:
        - cudagraph_warmup_times: warmup times for cudagraph.
        NOTE: `cudagraph_capture_sizes` is always inferred from
        compilation context.
    - Inductor compilation:
        - use_inductor: whether to use inductor compilation.
            - False: inductor compilation is not used. graph runs in eager.
            - True: inductor compilation is used. one graph for symbolic shape
                is compiled. In addition, compile for different sizes specified
                in inductor_compile_sizes, using configurations
                in inductor_compile_config.
        - inductor_specialize: how to specialize inductor compilation.
            - "none": no specialization.
            - "cudagraph": specialize using cudagraph capture sizes.
        - inductor_compile_config: additional configurations for inductor.
            - None: use default configurations.
        - inductor_passes: additional passes for inductor. It is a dictionary
            from pass name to pass function qualified name. We use function
            name because the config uses json format.
    """
    use_inductor: bool = True
    inductor_specialize: str = "none"  # "none", "cudagraph"
    inductor_compile_config: Dict = Field(default_factory=dict)
    inductor_passes: Dict[str, str] = Field(default_factory=dict)
    cudagraph_warmup_times: int = 0

    # not configurable, computed after init
    cudagraph_capture_sizes: List[int] = PrivateAttr
    inductor_compile_sizes: List[int] = PrivateAttr

    def model_post_init(self):
        context = get_compile_context()
        context = copy.deepcopy(context) if context is not None else []
        sizes_to_specialize: List[int] = context
        self.cudagraph_capture_sizes = sizes_to_specialize

        if self.inductor_specialize == "cudagraph":
            self.inductor_compile_sizes = self.cudagraph_capture_sizes
        elif self.inductor_specialize == "none":
            self.inductor_compile_sizes = []
        else:
            raise ValueError(
                f"Unknown inductor_specialize: {self.inductor_specialize}")

    @staticmethod
    def default_config() -> "CompilationConfig":
        config_path = envs.VLLM_TORCH_COMPILE_CONFIG
        if config_path is not None:
            with open(config_path) as json_file:
                data = json.load(json_file)
                config = CompilationConfig.model_validate_json(data)
        else:
            config = CompilationConfig()

        return config
