import functools
import logging
import torch
import torch.utils.cpp_extension

from .utils import get_function_schema

from pathlib import Path
from typing import List, Callable


def compose2(f: Callable, g: Callable) -> Callable:
    """
    Compose two functions.
    """
    return lambda *a, **kw: g(f(*a, **kw))


def compose(*fs: List[Callable]) -> Callable:
    """
    Compose a list of functions.
    """
    return functools.reduce(compose2, fs)


def is_optional_arg(n: torch.fx.Node, arg_num: int) -> bool:
    s = get_function_schema(n)
    if not s or arg_num >= len(s.arguments):
        return False
    return isinstance(s.arguments[arg_num].real_type, torch._C.OptionalType)


def build_extension(lib_name: str,
                    sources: List[str],
                    opt: str = '-O2',
                    verbose: bool = False):
    """
    Given a list of cpp and cuda source files, build and load a pytorch extension
    module with the given name.  Loaded ops will appear in the torch.ops.{lib_name}
    namespace.
    """
    vllm_root = Path(__file__).parent.parent.parent
    torch.utils.cpp_extension.load(
        name=lib_name,
        sources=sources,
        extra_cflags=[
            opt, f'-DLIBRARY_NAME={lib_name}', f'-I{vllm_root}/csrc'
        ],
        # TODO: Note: this is a total hack to get naive C++ fused ops working.
        # make an argument and push into fused_op_generator
        extra_ldflags=[
            f'{vllm_root}/vllm/_C.abi3.so'
        ],
        verbose=verbose,
        is_python_module=False,
    )
