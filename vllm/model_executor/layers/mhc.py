import torch
import torch.nn.functional as F
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_aiter_mhc = None


def _mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    outer_shape = residual.shape[:-2]

    x_flat = residual.reshape(-1, hc_mult * hidden_size).float()
    num_tokens = x_flat.shape[0]

    rsqrt_val = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat, fn) * rsqrt_val

    pre_logits = mixes[:, :hc_mult]
    post_logits = mixes[:, hc_mult:hc_mult * 2]
    comb_logits = mixes[:, hc_mult * 2:]

    pre_mix = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    post_mix = torch.sigmoid(post_logits * hc_scale[1] + hc_base[hc_mult:hc_mult * 2]) * hc_post_mult_value
    comb = comb_logits * hc_scale[2] + hc_base[hc_mult * 2:]
    comb = comb.view(num_tokens, hc_mult, hc_mult)

    comb = comb.softmax(-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)

    res_view = residual.reshape(num_tokens, hc_mult, hidden_size).float()
    layer_input = (pre_mix.unsqueeze(-1) * res_view).sum(dim=-2).to(residual.dtype)

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _aiter_mhc is not None:
        return _aiter_mhc.mhc_pre(
            residual, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat,
        )
    return _mhc_pre_torch(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, n_splits,
    )


def _mhc_pre_fake(
    residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1,
):
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    post_mix = torch.empty(*outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(*outer_shape, hc_mult, hc_mult, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(*outer_shape, hidden_size, dtype=torch.bfloat16, device=residual.device)
    return post_mix, comb_mix, layer_input


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if _aiter_mhc is not None:
        out = torch.empty_like(residual)
        _aiter_mhc.mhc_post(out, x, residual, post_layer_mix, comb_res_mix)
        return out
    out = torch.einsum("...ij,...jh->...ih", comb_res_mix, residual.float())
    out = out + post_layer_mix * x.unsqueeze(-2).float()
    return out.to(residual.dtype)


def _mhc_post_fake(x, residual, post_layer_mix, comb_res_mix):
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="mhc_pre",
    op_func=mhc_pre,
    mutates_args=[],
    fake_impl=_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="mhc_post",
    op_func=mhc_post,
    mutates_args=[],
    fake_impl=_mhc_post_fake,
)
