import torch
from typing import Optional
from vllm.utils import direct_register_custom_op
from vllm.platforms import current_platform


def rocm_aiter_tuned_gemm_impl(    
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor]=None,
    out_dtype: Optional[torch.dtype]=None,
    scale_a: Optional[torch.Tensor]=None,
    scale_b: Optional[torch.Tensor]=None) -> torch.Tensor:

    # This AITER function can be used for
    # - per-tensor activations + per-tensor weights
    #   e.g. vllm/model_executor/layers/linear.py
    # - per-token-activations + per-channel-weights
    #   e.g. vllm/model_executor/layers/quantization/utils/w8a8_utils.py
    from aiter.tuned_gemm import tgemm as aiter_tgemm

    print("AITER TUNED GEMM")

    return aiter_tgemm.mm(input,
                            weight.t(),
                            otype=out_dtype,
                            scale_a=scale_a,
                            scale_b=scale_b,
                            bias=bias)

def rocm_aiter_tuned_gemm_fake(    
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor]=None,
    out_dtype: Optional[torch.dtype]=None,
    scale_a: Optional[torch.Tensor]=None,
    scale_b: Optional[torch.Tensor]=None) -> torch.Tensor:

    m, _ = input.shape
    n = weight.shape[1]
    return torch.empty((m, n), dtype=out_dtype, device=input.device)

if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_tuned_gemm",
        op_func=rocm_aiter_tuned_gemm_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_tuned_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )

class aiter_ops:

    @staticmethod
    def rocm_aiter_tuned_gemm(    
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor]=None,
        out_dtype: Optional[torch.dtype]=None,
        scale_a: Optional[torch.Tensor]=None,
        scale_b: Optional[torch.Tensor]=None) -> torch.Tensor:

        return torch.ops.vllm.rocm_aiter_tuned_gemm(
            input,
            weight,
            bias=bias,
            out_dtype=out_dtype,
            scale_a=scale_a,
            scale_b=scale_b,
        )
