from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CompilationConfig(BaseModel):
    """
    Configuration for compilation.
    It has two parts:
    - Inductor compilation:
        - use_inductor: whether to use inductor compilation.
            - False: inductor compilation is not used. graph runs in eager.
            - True: inductor compilation is used. one graph for symbolic shape
                is compiled. In addition, compile for different sizes specified
                in inductor_compile_sizes, using configurations
                in inductor_compile_config.
        - inductor_compile_sizes: sizes for inductor to compile.
            - None: infer from compilation context.
        - inductor_compile_config: additional configurations for inductor.
            - None: use default configurations.
        - inductor_passes: additional passes for inductor. It is a dictionary
            from pass name to pass function qualified name. We use function
            name because the config uses json format.
    - CudaGraph capture:
        - cudagraph_capture_sizes: sizes for cudagraph to capture.
            - None: infer from compilation context.
        - cudagraph_warmup_times: warmup times for cudagraph.
    """
    use_inductor: bool = False
    inductor_compile_sizes: Optional[List[int]] = None
    inductor_compile_config: Optional[Dict] = None
    inductor_passes: Dict[str, str] = Field(default_factory=dict)
    cudagraph_capture_sizes: Optional[List[int]] = None
    cudagraph_warmup_times: int = 0
