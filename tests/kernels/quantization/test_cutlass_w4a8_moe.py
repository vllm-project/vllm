"""
Test the grouped gemm kernel itself and the entire moe layer
janky kernel tests for now
"""
import random
import torch

from dataclasses import dataclass
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows,
    quantize_weights,
)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return tensor.clamp(min=finfo.min, max=finfo.max).to(dtype=torch.float8_e4m3fn)


def cutlass_quantize(
    atype: torch.dtype,
    w: torch.Tensor,
    wtype: ScalarType,
    stype: torch.dtype | None,
    group_size: int | None,
    zero_points: bool = False,
):
    """
    dont do the encoding/reordering for weights and packing for scales
    cause that is done after all the things are combined.
    this is called for each expert then combined at the end 
    """
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w, wtype, group_size=group_size, zero_points=zero_points
    )

    # since scales are cast to fp8, we need to compute w_ref this way
    w_ref = (
        (w_q).to(torch.float32)
        * w_s.to(atype).to(torch.float32).repeat_interleave(group_size, dim=0)
    ).to(atype)

    # bit mask prevents sign extending int4 when packing
    w_q = pack_rows(w_q & 0x0F, wtype.size_bits, *w_q.shape)
    # w_q = w_q.t().contiguous().t()  # convert to col major
    w_q = w_q.t().contiguous() # (N, K)
    # print(f'{w_q.shape=}, {w_q.stride()=}')

    # reorder and pack can be done on everything at the end
    # w_q_packed = ops.cutlass_encode_and_reorder_int4b(w_q)
    # w_s_packed = ops.cutlass_pack_scale_fp8(w_s.to(atype))

    return w_ref, w_q, w_s.to(atype), w_zp


def cutlass_preprocess(
    w_q_experts: list[torch.Tensor],
    w_s_experts: list[torch.Tensor]
):
    """
    reorder/encode the expert weights
    pack the scale weights
    return the packed layout/strides to pass to grouped gemm
    """
    w_s_packed = ops.cutlass_pack_scale_fp8(torch.stack(w_s_experts))
    w_q_packed, packed_layout = ops.cutlass_encode_and_reorder_int4b_grouped(torch.stack(w_q_experts)) # expects dim 3
    return w_q_packed, w_s_packed, packed_layout


GROUP_SIZE = 128
NUM_EXPERTS = [8, 64]
Ks = [512, 256] # need divisible by gorup size
Ns = [2048, 1024]
ALIGNMENT = 16 # torch scaled mm alignment for M, needed for reference check

# faster test, qwen config
# Ks = [2048]
# Ns = [768]
# NUM_EXPERTS = [128]

if __name__ == '__main__':
    current_platform.seed_everything(42)
    for num_experts in NUM_EXPERTS:
        for K in Ks:
            for N in Ns:
                # generate random number of tokens per expert
                Ms = [ALIGNMENT * random.randint(1, 64) for _ in range(num_experts)]
                # set some random indices to 0
                # for zindex in [6, 7, 8, 17, 83, 94, 114]:
                #     Ms[zindex] = 0
                M_full = sum(Ms)
                scale_k = K // GROUP_SIZE

                # activations
                a = to_fp8(torch.randn((M_full, K), device='cuda'))
                a_ref = a.to(torch.float32)
                a_strides = torch.full((num_experts,), K, dtype=torch.int64).cuda()

                # output
                out = torch.empty((M_full, N), dtype=torch.bfloat16, device='cuda')
                c_strides = torch.full((num_experts,), N, dtype=torch.int64).cuda()

                # channel/token scales
                per_tok_scales = torch.randn((M_full, 1), dtype=torch.float32).cuda()
                per_chan_scales = torch.randn((num_experts, N, 1), dtype=torch.float32).cuda()

                # expert weights and scales
                wtype = scalar_types.int4
                atype = stype = torch.float8_e4m3fn
                w_refs, w_qs, w_ss = [], [], []
                for _ in range(num_experts):
                    b = to_fp8(torch.randn((K, N), device="cuda"))
                    w_ref, w_q, w_s, _ = cutlass_quantize(atype, b.to(torch.float16), wtype, stype, GROUP_SIZE, False)
                    w_refs.append(w_ref)
                    w_qs.append(w_q)
                    w_ss.append(w_s)
                
                w_q_packed, w_s_packed, packed_layout = cutlass_preprocess(w_qs, w_ss)
                problem_sizes = torch.tensor([[N, M, K] for M in Ms], dtype=torch.int32).cuda()
                expert_offsets = torch.cat([
                    torch.tensor([0], dtype=torch.int64),
                    torch.cumsum(torch.tensor(Ms, dtype=torch.int64), dim=0)[:-1]
                ]).cuda()

                # doesnt matter for now, we are constructing layout at runtime
                b_strides = packed_layout
                # since stride is like `(_1,2048,0)` it takes 2xint64 = 16 bytes 
                # we neeed to have each entry be like (2048, 0) in int64 to make the bytes align
                group_scale_strides = torch.zeros((num_experts, 2), dtype=torch.int64).cuda()
                group_scale_strides[:, 0] = N # set first column to N
                ops.cutlass_w4a8_moe_mm(
                    out,
                    a,
                    w_q_packed,
                    per_tok_scales,
                    per_chan_scales,
                    w_s_packed,
                    GROUP_SIZE,
                    expert_offsets,
                    problem_sizes,
                    a_strides,
                    b_strides,
                    c_strides,
                    group_scale_strides
                )
                torch.cuda.synchronize()
                # check result
                ends = torch.cumsum(torch.tensor(Ms), 0).tolist()
                starts = expert_offsets.cpu().tolist()
                for i in range(num_experts):
                    start, end = starts[i], ends[i]
                    if start == end: continue # no tokens for this expert
                    out_ref = torch._scaled_mm(
                        a_ref[start:end].to(torch.float8_e4m3fn),
                        w_refs[i].to(torch.float8_e4m3fn).t().contiguous().t(),
                        per_tok_scales[start:end], # (M, 1)
                        per_chan_scales[i].reshape(1, -1), # (1, N)
                        out_dtype=torch.bfloat16,
                        use_fast_accum=True
                    )
                    # rtol=1e-2, atol=1e-2 is closest to cutlass epsilon 1e-2, non zero floor 1e-4
                    torch.testing.assert_close(out[start:end], out_ref, rtol=1e-2, atol=1e-2)
    print('tests passed!')