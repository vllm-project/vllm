import torch

from vllm.utils import is_xpu

# Reference default values of atol and rtol are from
# https://github.com/pytorch/pytorch/blob/6d96beb6bec24d73ee3f080bac54d2104068f675/test/test_transformers.py#L67
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
default_rtol = {
    torch.float16: 1e-3,
    torch.bfloat16: 1.6e-2,
    torch.float: 1.3e-6
}

ipex_xpu_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
ipex_xpu_rtol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}


def get_default_atol(output) -> float:
    return default_atol[output.dtype] if not is_xpu() else ipex_xpu_atol[
        output.dtype]


def get_default_rtol(output) -> float:
    return default_rtol[output.dtype] if not is_xpu() else ipex_xpu_rtol[
        output.dtype]
