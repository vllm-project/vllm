import dataclasses
from functools import cache
from itertools import product
from typing import Callable, List, Type

import torch
import tqdm
from test_utils import autogen_scaled_mm_fp8_gemm_test


@dataclasses.dataclass
class TestArgs:
    m: int
    n: int
    k: int
    out_dtype: Type[torch.dtype] = dataclasses.field(default=torch.bfloat16)
    device: str = dataclasses.field(default='cuda')

    @staticmethod
    @cache
    def default_test_args() -> List["TestArgs"]:
        Ms = [1, 16, 32, 64, 128, 256, 512, 222, 100, 33]
        Ns = [2048, 4096, 8192, 16384, 24576, 256, 1024]
        Ks = [128, 496, 1024]
        out_dtypes = [torch.bfloat16]

        args = []
        for m, n, k, out_dtype in product(Ms, Ns, Ks, out_dtypes):
            args.append(TestArgs(m, n, k, out_dtype))
        return args


@cache
def get_autogen_functions():
    import importlib
    from importlib.util import find_spec

    # import vllm nm_cutlass modules so torch._C can find it
    m_idx = 0
    m_name = f'vllm._nm_cutlass_{m_idx}_C'
    while find_spec(m_name):
        print(f"attempting import {m_name}")
        importlib.import_module(m_name)
        m_idx += 1
        m_name = f'vllm._nm_cutlass_{m_idx}_C'

    dispatch_names = torch._C._dispatch_get_all_op_names()
    autogen_dispatch_names = [x for x in dispatch_names if 'autogen' in x]
    assert all([x.startswith('_nm_cutlass') for x in autogen_dispatch_names])
    autogen_dispatch_modules_names = [(getattr(torch.ops,
                                               x.split('::')[0]),
                                       x.split('::')[1])
                                      for x in autogen_dispatch_names]
    name_fn = [(name, getattr(m, name))
               for m, name in autogen_dispatch_modules_names]
    print(f"#autogen functions found {len(name_fn)}")
    return name_fn


@cache
def test_kernel_function(name: str,
                         fn: Callable,
                         verbose: bool = False) -> bool:
    test_args: List[TestArgs] = TestArgs.default_test_args()
    for x in test_args:
        success = autogen_scaled_mm_fp8_gemm_test(
            fn,
            m=x.m,
            n=x.n,
            k=x.k,
            per_token_act_quant=False,
            per_out_channel_weight_quant=False,
            out_dtype=x.out_dtype,
            device=x.device)
        if not success:
            # Early exit
            if verbose:
                print(f"Test Fail : {name} failed for MNK : {x.m} {x.n} {x.k}")
            return False
    return True


@cache
def test_kernel(kernel_name: str) -> bool:
    name_fn = get_autogen_functions()
    name_fn = list(filter(lambda x: x[0] == kernel_name, name_fn))
    assert len(name_fn) == 1
    fn = name_fn[0][1]
    return test_kernel_function(kernel_name, fn)


def main(args):
    name_fn = get_autogen_functions()
    print(f"#{len(name_fn)} autogen functions found.")
    if args.pattern:
        name_fn = list(filter(lambda x: args.pattern in x[0], name_fn))
    print(f"${len(name_fn)} autogen functions match the pattern.")

    good_functions = []
    # Test each kernel one after another for correctness
    for name, fn in tqdm.tqdm(name_fn):
        test_kernel_function(name, fn, verbose=True)
        good_functions.append((name, fn))

    print(f"#{len(good_functions)} good functions found.")
    print(f"good functions \n{good_functions}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='''
            Test autogen cutlass kernels
            ''')
    parser.add_argument(
        '--pattern',
        default=None,
        help='Checks for this pattern in the autogen kernel name')
    args = parser.parse_args()
    main(args)
