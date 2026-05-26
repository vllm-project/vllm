from dataclasses import dataclass
from triton_kernels.numerics import InFlexData, OutFlexData
import torch
import triton
from .swiglu_details._swiglu import _swiglu, _swiglu_fn
from triton_kernels import target_info


@dataclass(frozen=True)
class FlexCtx:
    out_data: OutFlexData = OutFlexData()
    inp_data: InFlexData = InFlexData()
    saturate_inf: bool = False


@dataclass(frozen=True)
class PrecisionConfig:
    limit: float
    flex_ctx: FlexCtx = FlexCtx()


swiglu_fn = _swiglu_fn


class SwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, alpha, precision_config, routing_data):
        N = a.shape[-1]
        M = a.numel() // N
        assert a.stride()[-1] == 1
        assert a.shape[-1] % 2 == 0
        out = torch.empty(size=(M, N // 2), dtype=a.dtype, device=a.device)
        flex_ctx = precision_config.flex_ctx
        # optimization hyperparameters
        BLOCK_M, BLOCK_N = 32 // a.itemsize, 128
        num_warps = 4
        kwargs = {'maxnreg': 64} if not target_info.is_hip() else {}
        # launch semi-persistent kernel
        N_BLOCKS = triton.cdiv(N // 2, BLOCK_N)
        num_sms = target_info.num_sms()
        if routing_data is not None:
            waves_per_sm = 32 if target_info.is_hip() else 128
            num_pid = num_sms * (waves_per_sm // num_warps)
            M_BLOCKS = max(1, triton.cdiv(num_pid, N_BLOCKS))
            grid = (min(M_BLOCKS * N_BLOCKS, 4 * num_sms), )
        else:
            M_BLOCKS = triton.cdiv(M, BLOCK_M)
            if M_BLOCKS * N_BLOCKS >= 8 * num_sms:
                grid = (8 * num_sms, )
            else:
                grid = (min(M_BLOCKS * N_BLOCKS, 4 * num_sms), )
        n_tokens = None
        if routing_data is not None:
            n_tokens = routing_data.expt_data.token_offs_raw[routing_data.n_expts_tot]
        _swiglu[grid](
            flex_ctx.out_data.reinterpret(out),
            flex_ctx.out_data.expected_scale,
            flex_ctx.out_data.actual_scale,
            flex_ctx.out_data.checksum_scale,
            flex_ctx.inp_data.reinterpret(a),
            flex_ctx.inp_data.scale,
            alpha,
            M,
            N // 2,
            a.shape[-1],
            1,
            out.shape[-1],
            1,
            precision_config.limit,
            n_tokens,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EVEN_N=(N // 2) % BLOCK_N == 0,
            M_BLOCKS=M_BLOCKS,
            N_BLOCKS=N_BLOCKS,
            flexpoint_saturate_inf=flex_ctx.saturate_inf,
            num_warps=num_warps,
            **kwargs,
        )
        out = out.view(a.shape[:-1] + out.shape[-1:])
        return out


def swiglu(a, alpha, precision_config, routing_data=None):
    return SwiGLU.apply(a, alpha, precision_config, routing_data)


def swiglu_torch(a, alpha, precision_config):
    limit = precision_config.limit
    a_gelu = a[..., ::2]
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    out = out_gelu * (a_linear + 1)
    return out
