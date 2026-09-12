# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    mhc_fused_post_pre_split_config,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op


def _torch_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
) -> None:
    assert out.shape[0] == 1
    assert sqrsum.shape[0] == 1
    x_float = x.float()
    out[0].copy_(x_float @ fn.t())
    sqrsum[0].copy_(x_float.square().sum(dim=-1))


def _hc_prenorm_gemm_outputs(
    x: torch.Tensor,
    fn: torch.Tensor,
    *,
    hidden_size: int,
    hc_mult: int,
    use_tilelang_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )

    use_deep_gemm = is_deep_gemm_supported() or not use_tilelang_fallback
    num_tokens = x.shape[0]
    n_splits = (
        compute_num_split(64, x.shape[1], cdiv(num_tokens, 64)) if use_deep_gemm else 1
    )
    out = torch.empty(
        n_splits,
        num_tokens,
        fn.shape[0],
        dtype=torch.float32,
        device=x.device,
    )
    sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=x.device,
    )
    if use_deep_gemm:
        tf32_hc_prenorm_gemm(x, fn, out, sqrsum, n_splits)
    else:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            _HC_PRENORM_GEMM_TILELANG_KERNEL,
        )

        _HC_PRENORM_GEMM_TILELANG_KERNEL(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
        )
    return out, sqrsum


def mhc_pre_delayed_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run mHC pre with a carried pre-mix and return the next pre-mix.

    Args:
        residual: BF16 residual streams of shape (tokens, hc_mult, hidden_size).
        fn: FP32 projection of shape (hc_mult * (hc_mult + 2), input_size).
        hc_scale: FP32 scales of shape (3,).
        hc_base: FP32 bias of shape (hc_mult * (hc_mult + 2),).
        rms_eps: RMS normalization epsilon.
        hc_pre_eps: Pre-mix epsilon.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult_value: Post-mix multiplier.
        sinkhorn_repeat: Number of Sinkhorn iterations.
        pre_mix: FP32 coefficients from the previous sublayer, or None to
            select residual stream zero at model entry.
        x: Optional BF16 projection input of shape (tokens, input_size), for
            the first layer's broadcast embedding and summed projection.
        norm_weight: Optional BF16 RMSNorm weight for the collapsed input.
        norm_eps: RMSNorm epsilon for the collapsed input.

    Returns:
        Post and residual coefficients, optionally normalized BF16 layer input,
        and the next FP32 pre-mix, with shapes (tokens, hc_mult, 1),
        (tokens, hc_mult, hc_mult), (tokens, hidden_size), and (tokens, hc_mult).
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_PRENORM_GEMM_TILELANG_KERNEL,
        mhc_pre_big_fuse_tilelang,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )

    assert residual.ndim == 3 and residual.dtype == torch.bfloat16
    assert residual.is_contiguous()
    num_tokens, hc_mult, hidden_size = residual.shape
    if x is None:
        x = residual.view(num_tokens, hc_mult * hidden_size)
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.shape[0] == num_tokens
    input_size = x.shape[1]
    mix_size = hc_mult * (hc_mult + 2)
    assert fn.shape == (mix_size, input_size) and fn.dtype == torch.float32
    assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
    assert hc_base.shape == (mix_size,) and hc_base.dtype == torch.float32
    if pre_mix is not None:
        assert pre_mix.shape == (num_tokens, hc_mult)
        assert pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()

    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    outputs = (
        post.unsqueeze(-1),
        comb.view(num_tokens, hc_mult, hc_mult),
        layer_input,
        next_pre_mix,
    )
    if num_tokens == 0:
        return outputs

    use_deep_gemm = is_deep_gemm_supported()
    n_splits = (
        compute_mhc_pre_num_splits(input_size, num_tokens) if use_deep_gemm else 1
    )
    mixes = torch.empty(
        n_splits, num_tokens, mix_size, dtype=torch.float32, device=residual.device
    )
    sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    if use_deep_gemm:
        tf32_hc_prenorm_gemm(x, fn, mixes, sqrsum, n_splits)
    else:
        _HC_PRENORM_GEMM_TILELANG_KERNEL(
            x,
            fn,
            mixes,
            sqrsum,
            input_size,
            1,
        )
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
        MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix if pre_mix is not None else post,
            next_pre_mix,
            layer_input,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_eps=norm_eps,
            hc_mult=hc_mult,
            use_pre_mix_in=pre_mix is not None,
            save_pre_mix=True,
            rms_numel=input_size,
        )
        return outputs
    mhc_pre_big_fuse_tilelang(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        pre_mix if pre_mix is not None else post,
        next_pre_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=input_size,
    )
    return outputs


def _mhc_pre_delayed_tilelang_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = residual.shape
    return (
        torch.empty(
            num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        ),
        torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=residual.device),
    )


def mhc_fused_post_pre_delayed_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Run one mHC post block followed by the next delayed mHC pre block.

    Within the fused kernel's token range the post mapping is folded into the
    pre-norm GEMM, so the updated residual streams feed the projection from
    registers instead of a second pass over global memory. Above it this runs
    the same post kernel and split-k GEMM as the unfused pair.

    Args:
        x: BF16 sublayer output of shape (tokens, hidden_size).
        residual: BF16 residual streams of shape (tokens, hc_mult, hidden_size).
        post_layer_mix: FP32 post coefficients of shape (tokens, hc_mult, 1).
        comb_res_mix: FP32 residual coefficients, (tokens, hc_mult, hc_mult).
        fn: FP32 projection of shape (hc_mult * (hc_mult + 2), input_size).
        hc_scale: FP32 scales of shape (3,).
        hc_base: FP32 bias of shape (hc_mult * (hc_mult + 2),).
        rms_eps: RMS normalization epsilon.
        hc_pre_eps: Pre-mix epsilon.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult_value: Post-mix multiplier.
        sinkhorn_repeat: Number of Sinkhorn iterations.
        pre_mix: FP32 coefficients from the previous sublayer, or None to
            select residual stream zero.
        norm_weight: Optional BF16 RMSNorm weight for the collapsed input.
        norm_eps: RMSNorm epsilon for the collapsed input.
        capture_aux: Also return the mean over the post-mapped streams, which
            draft models consume as the target's hidden state. It is folded
            into the collapse, which already reads those streams.

    Returns:
        The post-mapped residual streams, the post and residual coefficients,
        the optionally normalized BF16 layer input, the next FP32 pre-mix, and
        the BF16 stream mean (empty unless capture_aux), with shapes
        (tokens, hc_mult, hidden_size), (tokens, hc_mult, 1),
        (tokens, hc_mult, hc_mult), (tokens, hidden_size), (tokens, hc_mult),
        and (tokens, hidden_size).
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_PRENORM_GEMM_TILELANG_KERNEL,
        _MHC_FUSED_TILELANG_KERNEL,
        _MHC_POST_TILELANG_KERNEL,
        mhc_pre_big_fuse_tilelang,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )

    assert residual.ndim == 3 and residual.dtype == torch.bfloat16
    assert residual.is_contiguous()
    num_tokens, hc_mult, hidden_size = residual.shape
    input_size = hc_mult * hidden_size
    mix_size = hc_mult * (hc_mult + 2)
    assert x.shape == (num_tokens, hidden_size) and x.dtype == torch.bfloat16
    assert x.is_contiguous()
    assert post_layer_mix.shape[:2] == (num_tokens, hc_mult)
    assert post_layer_mix.dtype == torch.float32 and post_layer_mix.is_contiguous()
    assert comb_res_mix.shape == (num_tokens, hc_mult, hc_mult)
    assert comb_res_mix.dtype == torch.float32 and comb_res_mix.is_contiguous()
    assert fn.shape == (mix_size, input_size) and fn.dtype == torch.float32
    assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
    assert hc_base.shape == (mix_size,) and hc_base.dtype == torch.float32
    if pre_mix is not None:
        assert pre_mix.shape == (num_tokens, hc_mult)
        assert pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()

    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    aux = torch.empty(
        num_tokens if capture_aux else 0,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )
    if num_tokens == 0:
        return (
            torch.empty_like(residual),
            post.unsqueeze(-1),
            comb.view(num_tokens, hc_mult, hc_mult),
            layer_input,
            next_pre_mix,
            aux,
        )

    fused_config = mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult)
    if fused_config is not None:
        mixes, sqrsum, residual_cur = _MHC_FUSED_TILELANG_KERNEL(
            comb_res_mix,
            residual,
            post_layer_mix.view(num_tokens, hc_mult),
            x,
            fn.view(mix_size, hc_mult, hidden_size),
            hc_mult,
            hidden_size,
            mix_size,
        )
    else:
        residual_cur = torch.empty_like(residual)
        _MHC_POST_TILELANG_KERNEL(
            comb_res_mix,
            residual,
            post_layer_mix.view(num_tokens, hc_mult),
            x,
            residual_cur,
            hc_mult,
            hidden_size,
        )
        # The delayed epilogue is compiled per bucketed split count, so the
        # projection has to use the same bucket rather than the raw estimate.
        use_deep_gemm = is_deep_gemm_supported()
        n_splits = (
            compute_mhc_pre_num_splits(input_size, num_tokens) if use_deep_gemm else 1
        )
        mixes = torch.empty(
            n_splits, num_tokens, mix_size, dtype=torch.float32, device=residual.device
        )
        sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )
        residual_cur_2d = residual_cur.view(num_tokens, input_size)
        if use_deep_gemm:
            tf32_hc_prenorm_gemm(residual_cur_2d, fn, mixes, sqrsum, n_splits)
        else:
            _HC_PRENORM_GEMM_TILELANG_KERNEL(
                residual_cur_2d,
                fn,
                mixes,
                sqrsum,
                input_size,
                1,
            )

    outputs = (
        residual_cur,
        post.unsqueeze(-1),
        comb.view(num_tokens, hc_mult, hc_mult),
        layer_input,
        next_pre_mix,
        aux,
    )
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
        MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix if pre_mix is not None else post,
            next_pre_mix,
            aux if capture_aux else layer_input,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_eps=norm_eps,
            hc_mult=hc_mult,
            use_pre_mix_in=pre_mix is not None,
            save_pre_mix=True,
            rms_numel=input_size,
            write_aux=capture_aux,
        )
        return outputs
    mhc_pre_big_fuse_tilelang(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post,
        comb,
        layer_input,
        pre_mix if pre_mix is not None else post,
        next_pre_mix,
        aux if capture_aux else layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        mixes.shape[0],
        hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=input_size,
        write_aux=capture_aux,
    )
    return outputs


def _mhc_fused_post_pre_delayed_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    num_tokens, hc_mult, hidden_size = residual.shape
    return (
        torch.empty_like(residual),
        torch.empty(
            num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        ),
        torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=residual.device),
        torch.empty(
            num_tokens if capture_aux else 0,
            hidden_size,
            dtype=torch.bfloat16,
            device=residual.device,
        ),
    )


def mhc_pre_tilelang(
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
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: retained for the shared MHC operator API; the active GEMM
            backend selects its split factor internally.
        norm_weight: optional RMSNorm weight, shape (hidden_size,), dtype
            torch.bfloat16. When provided, RMSNorm is fused into the
            layer_input write path of the big_fuse kernel.
        norm_eps: epsilon for the fused RMSNorm; only consulted when
            norm_weight is given.

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
    )

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul, gemm_out_sqrsum = _hc_prenorm_gemm_outputs(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
    )
    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_tilelang_fake(
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
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_pre_broadcast_tilelang(
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
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fn_broadcast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-layer mHC pre for a residual broadcast from ``(T, H)``."""
    # n_splits is retained for the shared API; split selection is internal.
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
    )

    assert norm_weight is not None, "broadcast mHC pre currently requires fused RMSNorm"
    assert residual.dtype == torch.bfloat16
    assert residual.dim() == 2
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hidden_size = residual.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)
    assert fn_broadcast is not None
    assert fn_broadcast.dtype == torch.float32
    assert fn_broadcast.shape == (hc_mult3, hidden_size)

    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    residual_flat = residual
    num_tokens = residual.shape[0]

    residual_out = torch.empty(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul, gemm_out_sqrsum = _hc_prenorm_gemm_outputs(
        residual_flat,
        fn_broadcast,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_tilelang_fallback=False,
    )
    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        residual_out=residual_out,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        layer_input,
    )


def mhc_post_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_POST_TILELANG_KERNEL,
    )

    out = torch.empty_like(residual)
    _MHC_POST_TILELANG_KERNEL(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def mhc_fused_post_pre_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one MHC post block followed by the next MHC pre block.

    When ``norm_weight`` is provided, the layer_input_cur output is the
    RMSNorm'd activation (fused into the kernel); otherwise it is the
    raw pre-norm activation as before.

    ``n_splits`` and ``tile_n`` are retained for the shared MHC operator API.
    The TileLang path selects both values internally from the runtime shape.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_FUSED_TILELANG_KERNEL,
        _MHC_POST_TILELANG_KERNEL,
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
    )

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    fused_config = mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult)

    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if fused_config is not None:
        gemm_out_mul, gemm_out_sqrsum, residual_cur = _MHC_FUSED_TILELANG_KERNEL(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            hc_mult,
            hidden_size,
            hc_mult3,
        )
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    else:
        residual_cur = torch.empty_like(residual_flat)
        _MHC_POST_TILELANG_KERNEL(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            residual.shape[-2],
            residual.shape[-1],
        )
        gemm_out_mul, gemm_out_sqrsum = _hc_prenorm_gemm_outputs(
            residual_cur.view(num_tokens, hc_mult * hidden_size),
            fn,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
        )
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def _mhc_fused_post_pre_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _mhc_post_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def hc_head_fused_kernel_tilelang(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Apply the fused hc_head kernel and return the (T, H) bf16 result."""
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )
    if num_tokens == 0:
        return out
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_HEAD_FUSED_TILELANG_KERNEL,
    )

    _HC_HEAD_FUSED_TILELANG_KERNEL(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out


def _hc_head_fused_kernel_tilelang_fake(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = hs_flat.shape
    return torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )


direct_register_custom_op(
    op_name="mhc_pre_delayed_tilelang",
    op_func=mhc_pre_delayed_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_delayed_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_fused_post_pre_delayed_tilelang",
    op_func=mhc_fused_post_pre_delayed_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_delayed_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_pre_tilelang",
    op_func=mhc_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_post_tilelang",
    op_func=mhc_post_tilelang,
    mutates_args=[],
    fake_impl=_mhc_post_tilelang_fake,
)

direct_register_custom_op(
    op_name="mhc_fused_post_pre_tilelang",
    op_func=mhc_fused_post_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_tilelang_fake,
)

direct_register_custom_op(
    op_name="hc_head_fused_kernel_tilelang",
    op_func=hc_head_fused_kernel_tilelang,
    mutates_args=[],
    fake_impl=_hc_head_fused_kernel_tilelang_fake,
)
