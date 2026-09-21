# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch DSV4.1 mHC operations and overlap coefficient generation."""

from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_delayed_tilelang,
    mhc_post_tilelang,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

from .mega_mhc import is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

# GB200 TP4/FlashInfer improves through 16 tokens; larger screens tie or regress.
MHC_OVERLAP_MAX_TOKENS = 16


def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:
    """Check kernel requirements and safety of sharing the coefficient stream."""
    config = vllm_config.model_config.hf_config
    # DeepGEMM's prenorm kernel requires K % 64 == 0, N % 8 == 0, and N <= 32.
    mix_size = config.hc_mult * (config.hc_mult + 2)
    return (
        current_platform.is_device_capability_family(100)
        and is_deep_gemm_supported()
        and config.hidden_size % 64 == 0
        and 0 < mix_size <= 32
        and mix_size % 8 == 0
        and not vllm_config.parallel_config.use_ubatching
    )


def supports_mhc_all_reduce(vllm_config: VllmConfig) -> bool:
    parallel = vllm_config.parallel_config
    config = vllm_config.model_config.hf_config
    if (
        parallel.tensor_parallel_size != 4
        or parallel.enable_expert_parallel
        or config.hidden_size != 5120
        or config.hc_mult != 4
    ):
        return False
    comm = cast("CudaCommunicator", get_tp_group().device_communicator).ca_comm
    return comm is not None and bool(comm.mnnvl_lamport_ag_multicast_ptr)


def mhc_pre_delayed_overlap(
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
    *,
    stream: torch.cuda.Stream,
    layer_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the input on the caller stream and coefficients on another stream.

    Returns post mix, residual mix, normalized input, and next pre-mix. Only the
    input is ready on the caller stream; join stream before using coefficients.
    The caller must retain the weights and coefficient outputs until the join.
    """
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    n, hc, hidden = residual.shape
    assert residual.is_contiguous() and residual.dtype == torch.bfloat16
    assert norm_weight is not None
    if x is None:
        x = residual.view(n, hc * hidden)
    assert x.is_contiguous()
    post = torch.empty((n, hc), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n, hc * hc), device=residual.device, dtype=torch.float32)
    next_pre = torch.empty_like(post)
    compute_input = layer_input is None
    if layer_input is None:
        layer_input = torch.empty(
            (n, hidden), device=residual.device, dtype=torch.bfloat16
        )
    outputs = post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre
    if n == 0:
        return outputs
    splits = compute_mhc_pre_num_splits(x.shape[1], n)
    mix = torch.empty(
        (splits, n, hc * (hc + 2)), device=residual.device, dtype=torch.float32
    )
    sqr = torch.empty((splits, n), device=residual.device, dtype=torch.float32)
    epilogue = partial(
        MHC_PRE_NORM_KERNEL,
        mix,
        sqr,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        norm_weight,
        pre_mix if pre_mix is not None else post,
        next_pre,
        layer_input,  # Unused aux output in split modes.
        hidden_size=hidden,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        hc_mult=hc,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=x.shape[1],
    )
    main = torch.cuda.current_stream()
    # Prioritize input readiness for small batches before releasing statistics.
    if compute_input and n <= 8:
        epilogue(split_mode="input")
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        tf32_hc_prenorm_gemm(x, fn, mix, sqr, splits)
        epilogue(split_mode="stats")
    for tensor in (x, mix, sqr):
        tensor.record_stream(stream)
    if compute_input and n > 8:
        epilogue(split_mode="input")
    return outputs


def mhc_shifted_post_pre(
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
    *,
    stream: torch.cuda.Stream | None = None,
    reduce_results: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Dispatch shifted post/pre to overlap, Mega-mHC, or fused TileLang.

    When stream is supplied, join it before consuming the returned coefficients.
    """
    layer_input = None
    if reduce_results:
        tp = get_tp_group()
        if stream is not None and 0 < x.shape[0] <= MHC_OVERLAP_MAX_TOKENS:
            comm = cast("CudaCommunicator", tp.device_communicator).ca_comm
            assert comm is not None and comm.mnnvl_lamport_epochs is not None
            output = torch.empty_like(residual)
            layer_input = torch.empty_like(x)
            torch.ops._C_custom_ar.all_reduce_mhc(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                norm_weight,
                output,
                layer_input,
                comm.mnnvl_lamport_ag_local_ptr,
                comm.mnnvl_lamport_ag_multicast_ptr,
                comm.mnnvl_lamport_epochs[0],
                comm.rank,
                comm.mnnvl_buffer_size,
                norm_eps,
            )
            residual = output
        else:
            x = tp.all_reduce(x)
    if stream is not None:
        if layer_input is None:
            residual = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        aux = residual.mean(dim=1) if capture_aux else x.new_empty(0, x.shape[1])
        pre_outputs = mhc_pre_delayed_overlap(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            stream=stream,
            layer_input=layer_input,
        )
        return residual, *pre_outputs, aux

    if (
        pre_mix is not None
        and norm_weight is not None
        and not capture_aux
        and x.shape[0] <= 1 << 20
        and is_mega_mhc_supported(x.shape[1], residual.shape[1])
    ):
        outputs = mhc_shifted_post_pre_deep_gemm(
            x,
            residual,
            pre_mix,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_post_mult_value,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )
        return *outputs, x.new_empty(0, x.shape[1])

    return mhc_fused_post_pre_delayed_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix=pre_mix,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        capture_aux=capture_aux,
    )
