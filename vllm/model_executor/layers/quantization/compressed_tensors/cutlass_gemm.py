import cutlass
from cutlass import Tensor as FakeTensor
import cutlass.epilogue

import torch
from typing import Optional, Tuple, Dict

from vllm.logger import init_logger

logger = init_logger("cutlass_gemm")

def setup_dequant_epilogue(plan : cutlass.op.Gemm,
                           dq: torch.Tensor,
                           static_scales: Optional[torch.Tensor],
                           activation_scales: Optional[torch.Tensor]) \
                              -> Tuple[cutlass.op.Gemm, Dict]:

    if all([static_scales is None, activation_scales is None]):
        return plan, None
    assert static_scales is not None

    def epilog_with_scales_and_act_scales(accum, scales, act_scales):
        D = accum * scales * act_scales
        return D

    def epilog_with_scales(accum, scales):
        D = accum * scales
        return D

    epilog_tensors = {'scales': static_scales, 'D': dq}
    epilogue_trace_tensors = {
        "accum":
        FakeTensor(element=torch.int32,
                   shape=dq.shape,
                   layout_tag=cutlass.LayoutType.RowMajor),
        'scales':
        static_scales,
        'D':
        dq,
    }
    epilog_fn = epilog_with_scales

    if activation_scales is not None:
        epilog_tensors['act_scales'] = activation_scales
        epilogue_trace_tensors['act_scales'] = activation_scales
        epilog_fn = epilog_with_scales_and_act_scales

    plan.epilogue_visitor = cutlass.epilogue.trace(epilog_fn,
                                                   epilogue_trace_tensors)
    return plan, epilog_tensors


def cutlass_gemm_dq(
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        dtype: torch.dtype,
        static_scales: torch.Tensor,
        activation_scales: Optional[torch.Tensor] = None) -> torch.Tensor:

    dq = torch.empty((x_q.shape[0], w_q.shape[0]), dtype=dtype, device="cuda")

    log_str = (f"cutlass_gemm_dq: \n"
               f" - x_q {x_q.shape} {x_q.dtype} \n"
               f" - w_q {w_q.shape} {w_q.dtype} \n"
               f" - o_dq {dq.shape} {dq.dtype} \n")
    logger.debug(log_str)

    plan = cutlass.op.Gemm(
        element_A=x_q.dtype,
        element_B=w_q.dtype,
        element_C=dq.dtype,
        element_D=dq.dtype,
        layout_A=cutlass.LayoutType.RowMajor,
        layout_B=cutlass.LayoutType.ColumnMajor,
        layout_C=cutlass.LayoutType.RowMajor,
        element_accumulator=torch.int32)

    plan, visitor_args = setup_dequant_epilogue(plan, dq, static_scales,
                                                activation_scales)

    plan.run(x_q,
             w_q.t(),
             dq,
             dq,
             alpha=1,
             beta=0,
             visitor_args=visitor_args,
             print_module=False)

    dq = dq.view(*x_q.shape[:-1], -1)
    return dq
