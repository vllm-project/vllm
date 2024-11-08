"""
Generator function types.

Defines necessary information about each function type to generate.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from utils import get_script_dir


class GeneratorType(ABC):
    SCRIPT_DIR = get_script_dir()

    @staticmethod
    def description() -> str:
        raise NotImplementedError

    @abstractmethod
    def fn_defn_jinja_filepath(self) -> Path:
        # Function definition jinja - the entrypoint to the function to
        # generate.
        # Refer to csrc/quantization/cutlass_w8a8/scaled_mm_c3x.jinja for
        # an example.
        raise NotImplementedError

    @abstractmethod
    def fn_decl_jinja_filepath(self) -> Path:
        # Function decl jinja - the c++ function declaration of the function
        # to generate.
        # Refer to csrc/quantization/cutlass_w8a8/scaled_mm_c3x_fnprototype.jinja #noqa
        # for an example.

        raise NotImplementedError

    @abstractmethod
    def ops_def(self, fn_name: str) -> str:
        # torch binding ops.def template.
        raise NotImplementedError

    @abstractmethod
    def ops_impl(self, fn_name: str) -> str:
        # torch binding ops.impl template.
        raise NotImplementedError

    @staticmethod
    def from_str(s: str) -> "GeneratorType":
        if ScaledMMGenerator.description() == s:
            return ScaledMMGenerator()
        if SimpleGemmGenerator.description() == s:
            return SimpleGemmGenerator()
        if ScaledMMStreamKGenerator.description() == s:
            return ScaledMMStreamKGenerator()
        raise ValueError("Unknown generator type string {s}")


class ScaledMMGenerator(GeneratorType):

    def __init__(self):
        super().__init__()

    @staticmethod
    def description():
        return "scaled_mm"

    def fn_defn_jinja_filepath(self):
        return GeneratorType.SCRIPT_DIR / "scaled_mm_c3x.jinja"

    def fn_decl_jinja_filepath(self):
        return GeneratorType.SCRIPT_DIR / "scaled_mm_c3x_fnprototype.jinja"

    def ops_def(self, fn_name: str) -> str:
        return f'ops.def("{fn_name}(Tensor! out, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales) -> ()");'  #noqa

    def ops_impl(self, fn_name: str) -> str:
        return f'ops.impl("{fn_name}", torch::kCUDA, &{fn_name});'


class SimpleGemmGenerator(GeneratorType):

    def __init__(self):
        super().__init__()

    @staticmethod
    def description():
        return "simple_gemm"

    def fn_defn_jinja_filepath(self):
        return GeneratorType.SCRIPT_DIR / "simple_gemm_c3x.jinja"

    def fn_decl_jinja_filepath(self):
        return GeneratorType.SCRIPT_DIR / "simple_gemm_c3x_fnprototype.jinja"

    def ops_def(self, fn_name: str) -> str:
        return f'ops.def("{fn_name}(Tensor! out, Tensor a, Tensor b) -> ()");'

    def ops_impl(self, fn_name: str) -> str:
        # The {} should be filled in by the caller using the function name.
        return f'ops.impl("{fn_name}", torch::kCUDA, &{fn_name});'


class ScaledMMStreamKGenerator(GeneratorType):

    def __init__(self):
        super().__init__()

    @staticmethod
    def description():
        return "scaled_mm_streamk"

    def fn_defn_jinja_filepath(self):
        return GeneratorType.SCRIPT_DIR / "scaled_mm_c3x_streamk.jinja"

    def fn_decl_jinja_filepath(self):
        return GeneratorType.SCRIPT_DIR / "scaled_mm_c3x_streamk_fnprototype.jinja"

    def ops_def(self, fn_name: str) -> str:
        return f'ops.def("{fn_name}(Tensor! out, Tensor a, Tensor b, str reduction_mode, str decomposition_mode, Tensor a_scales, Tensor b_scales) -> ()");'  #noqa

    def ops_impl(self, fn_name: str) -> str:
        return f'ops.impl("{fn_name}", torch::kCUDA, &{fn_name});'

GeneratorTypes: List[GeneratorType] = [ScaledMMGenerator, SimpleGemmGenerator, ScaledMMStreamKGenerator]
