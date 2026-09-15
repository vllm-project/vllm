# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 GEMM with kernel-internal tensor-parallel reduce-scatter."""

from functools import cache

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor, make_ptr, nullptr

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.models.deepseek_v41.nvidia.ops.mxfp8_gemm_rs_kernel import Sm100Mxfp8GemmRS
from vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs_ar import GemmRsAr


@cache
def _compile(rank: int, world_size: int, n: int, k: int, num_sms: int):
    m, local_m, num_flags = cute.sym_int(), cute.sym_int(), cute.sym_int()
    a = make_fake_tensor(cutlass.Float8E4M3FN, (m, k), (k, 1), assumed_align=16)
    b = make_fake_tensor(cutlass.Float8E4M3FN, (n, k), (k, 1), assumed_align=16)
    partial = make_fake_tensor(cutlass.BFloat16, (m, n), (n, 1), assumed_align=32)
    output = make_fake_tensor(cutlass.BFloat16, (local_m, n), (n, 1), assumed_align=32)
    flags = make_fake_tensor(cutlass.Int32, (num_flags,), (1,), assumed_align=16)
    scale = nullptr(cutlass.Float8E8M0FNU, cute.AddressSpace.gmem, assumed_align=16)
    partial_mc = nullptr(cutlass.BFloat16, cute.AddressSpace.gmem, assumed_align=32)
    flags_mc = nullptr(cutlass.Int32, cute.AddressSpace.gmem, assumed_align=16)
    peers = nullptr(cutlass.Int64, cute.AddressSpace.gmem, assumed_align=8)
    alpha = make_fake_tensor(cutlass.Float32, (1,), (1,), assumed_align=4)
    return cute.compile(
        Sm100Mxfp8GemmRS(rank, world_size).run,
        a,
        b,
        partial,
        scale,
        scale,
        partial_mc,
        output,
        flags,
        flags_mc,
        peers,
        alpha,
        num_sms,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


class Mxfp8GemmRS:
    """Reuse K3's symmetric workspace; execute the MXFP8 fused kernel.

    The BF16 K3 GEMM is never invoked. Only its allocation and rendezvous
    machinery is shared; BF16 here is the projection's output dtype.
    """

    def __init__(self, max_tokens: int, n: int, k: int):
        assert max_tokens >= 128 and n % 128 == 0 and k % 128 == 0
        self.workspace = GemmRsAr(max_M=max_tokens, N=n)
        assert self.workspace.world_size == 4, "MXFP8 GEMM-RS currently supports TP4"
        self.n = n
        self.k = k
        self.alpha = torch.ones(1, dtype=torch.float32, device=self.workspace.device)
        self.compiled = _compile(
            self.workspace.rank,
            self.workspace.world_size,
            n,
            k,
            self.workspace.num_sms,
        )

    def quantized(self, a, b, a_scale, b_scale):
        ws = self.workspace
        m = a.shape[0]
        assert 128 <= m <= ws.max_M
        assert a.shape == (m, self.k) and b.shape == (self.n, self.k)
        assert a.is_contiguous() and b.is_contiguous()
        assert a.dtype == b.dtype == torch.float8_e4m3fn
        assert a.device == b.device == a_scale.device == b_scale.device == ws.device
        assert a_scale.is_contiguous() and b_scale.is_contiguous()
        assert a_scale.element_size() == b_scale.element_size() == 1
        assert a_scale.numel() == ((m + 127) // 128 * 128) * (self.k // 32)
        assert b_scale.numel() == self.n * (self.k // 32)
        output = torch.empty(
            ((m + ws.world_size - 1) // ws.world_size, self.n),
            dtype=torch.bfloat16,
            device=ws.device,
        )
        sf_a = make_ptr(
            cutlass.Float8E8M0FNU, a_scale.data_ptr(), cute.AddressSpace.gmem, 16
        )
        sf_b = make_ptr(
            cutlass.Float8E8M0FNU, b_scale.data_ptr(), cute.AddressSpace.gmem, 16
        )
        self.compiled(
            a,
            b,
            ws.partial[:m],
            sf_a,
            sf_b,
            ws.partial_mc_ptr,
            output,
            ws.flags,
            ws.flags_mc_ptr,
            ws.peer_flag_ptr,
            self.alpha,
        )
        return output

    def __call__(self, x, weight, weight_scale):
        assert x.ndim == 2 and x.shape[1] == self.k
        assert x.dtype == torch.bfloat16 and x.is_contiguous()
        a, a_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        return self.quantized(a, weight, a_scale, weight_scale)


class WoBGemmRS:
    """Bind a projection to one model-wide, sequentially reused workspace."""

    def __init__(self, runner, linear, transpose_weight):
        self.runner = runner
        self.linear = linear
        self.transpose_weight = transpose_weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter

        if x.shape[0] < 1024:
            return sp_reduce_scatter(self.linear(x))
        weight = self.linear.weight
        if self.transpose_weight:
            weight = weight.t()
        return self.runner(x, weight, self.linear.weight_scale)


def enable_wo_b_gemm_rs(layers, vllm_config) -> None:
    from vllm import envs
    from vllm.logger import init_logger
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.platforms import current_platform

    logger = init_logger(__name__)
    pc = vllm_config.parallel_config
    if (
        pc.tensor_parallel_size != 4
        or pc.use_ubatching
        or vllm_config.model_config.dtype != torch.bfloat16
        or vllm_config.lora_config is not None
        or envs.VLLM_BATCH_INVARIANT
        or not current_platform.is_device_capability_family(100)
    ):
        logger.warning_once(
            "DSV4.1 native GEMM-RS requires TP4, SM100, BF16 activations, "
            "and no ubatching, LoRA, or batch invariance."
        )
        return
    supported = (FlashInferCutlassMxfp8LinearKernel, FlashInferCutedslMxfp8LinearKernel)
    eligible = []
    for layer in layers:
        if not getattr(layer, "use_sequence_parallel", False):
            continue
        linear = layer.attn.wo_b
        kernel = getattr(linear.quant_method, "kernel", None)
        n, k = linear.output_size_per_partition, linear.input_size_per_partition
        if (
            isinstance(kernel, supported)
            and n % 128 == 0
            and k % 128 == 0
            and linear.bias is None
            and not linear.reduce_results
        ):
            eligible.append(
                (
                    layer.attn,
                    n,
                    k,
                    isinstance(kernel, FlashInferCutedslMxfp8LinearKernel),
                )
            )
    if not eligible:
        logger.warning_once(
            "DSV4.1 native GEMM-RS found no compatible SP MXFP8 wo_b projections."
        )
        return
    _, n, k, _ = eligible[0]
    assert all((item[1], item[2]) == (n, k) for item in eligible)
    runner = Mxfp8GemmRS(
        max(128, vllm_config.scheduler_config.max_num_batched_tokens), n, k
    )
    for attn, _, _, transpose in eligible:
        attn.wo_b_reduce_scatter = WoBGemmRS(runner, attn.wo_b, transpose)
    logger.info(
        "Enabled native MXFP8 GEMM-RS on %d DSV4.1 wo_b projections; "
        "one shared workspace, TP4, minimum 1024 tokens.",
        len(eligible),
    )
