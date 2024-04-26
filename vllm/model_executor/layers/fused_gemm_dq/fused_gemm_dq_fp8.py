import cutlass
from cutlass import Tensor as FakeTensor
import cutlass.epilogue

import torch
from typing import Optional, Tuple, Dict


def setup_dequant_epilogue(
    plan: cutlass.op.Gemm,
    dq: torch.Tensor,
    scale_a: Optional[torch.Tensor],
    scale_b: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> Tuple[cutlass.op.Gemm, Dict]:
    assert bias is None

    if all([scale_a is None, scale_b is None]):
        return plan, None
    assert scale_b is not None

    def epilog_with_scale_b(accum, scale_b):
        D = scale_b * accum
        return D

    def epilog_with_both_scales(accum, scale_a, scale_b):
        D = scale_a * (scale_b * accum)
        return D

    visitor_args = {"scale_a": scale_a, "scale_b": scale_b, "D": dq}
    epilogue_tensors = {
        "accum": FakeTensor(
            element=torch.float32,
            shape=dq.shape,
            layout_tag=cutlass.LayoutType.RowMajor,
        ),
        "D": dq,
        "scale_b": scale_b,
    }
    epilog_fn = epilog_with_scale_b

    if scale_a is not None:
        epilogue_tensors["scale_a"] = scale_a
        visitor_args["scale_a"] = scale_a
        epilog_fn = epilog_with_both_scales

    plan.epilogue_visitor = cutlass.epilogue.trace(epilog_fn, epilogue_tensors)
    return plan, visitor_args


def fused_gemm_dq_fp8(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    out_dtype: torch.dtype,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dq = torch.empty((x_q.shape[0], w_q.shape[1]), dtype=out_dtype, device="cuda")
    C = torch.zeros((x_q.shape[0], w_q.shape[1]), dtype=out_dtype, device="cuda")

    plan = cutlass.op.Gemm(
        element_A=x_q.dtype,
        element_B=w_q.dtype,
        element_C=dq.dtype,
        element_D=dq.dtype,
        layout_A=cutlass.LayoutType.RowMajor,
        layout_B=cutlass.LayoutType.ColumnMajor,
        layout_C=cutlass.LayoutType.RowMajor,
        element_accumulator=torch.float32,
        kernel_cc=90,
    )

    plan, visitor_args = setup_dequant_epilogue(plan, dq, scale_a, scale_b, bias)

    plan.run(
        x_q,
        w_q,
        C,
        dq,
        alpha=1,
        beta=0,
        visitor_args=visitor_args,
        print_module=False,
    )

    return dq
