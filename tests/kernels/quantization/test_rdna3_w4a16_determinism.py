#!/usr/bin/env python3
# SPDX-LicenseFileCopyrightText: Copyright contributors to the vLLM project
"""Bit-repeatability regression tests for the RDNA3 W4A16 GEMM.

The split-K epilogues of ``gptq_gemm_rdna3`` (scalar path) and the WMMA
path previously accumulated per-split low-precision partials directly into
the output via CAS atomics. With multiple writers per output element the
accumulation order — and therefore the rounded result — varied per call:
repeated fixed-input calls produced a different output almost every time on
gfx1100 once more than a few split blocks contended.

Both paths now store FP32 split partials and reduce them in a fixed order
with a single final low-precision rounding, which is deterministic by
construction. These tests assert exact repeatability through the public op
so they fail on the legacy atomic epilogue and pass on the deterministic
one. Both activation dtypes (bf16, fp16) are covered.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (  # noqa: E402
    RDNA3W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (  # noqa: E402
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

device = "cuda"
WEIGHT_TYPE = scalar_types.uint4b8

gfx1100_only = pytest.mark.skipif(
    not (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna3")
    ),
    reason="requires gfx1100 with the _rocm_C.gptq_gemm_rdna3 op built in",
)

REPEATS = 20
GROUP = 128
DTYPES = [torch.bfloat16, torch.float16]


def _build(K, N, seed, dtype):
    torch.manual_seed(seed)
    q_int4_kn = torch.randint(0, 16, (K, N), dtype=torch.int32)
    scales_gn = (torch.randn(K // GROUP, N) * 0.01 + 0.02).to(dtype)
    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE, packed_dim=0)
    no_loader = lambda *a, **k: None  # noqa: E731

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(data=qweight, weight_loader=no_loader, input_dim=0,
                            output_dim=1, packed_dim=0, packed_factor=8))
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(data=scales_gn, weight_loader=no_loader,
                                 input_dim=0, output_dim=1))
    return layer.to(device)


def _run(M, K, N, seed, dtype):
    from vllm.config import VllmConfig, set_current_vllm_config
    _cm = set_current_vllm_config(VllmConfig())
    _cm.__enter__()
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29741")
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    if not torch.distributed.is_initialized():
        init_distributed_environment(backend="cpu:gloo,cuda:hccl", world_size=1,
                                     rank=0, local_rank=0,
                                     distributed_init_method="env://")
        initialize_model_parallel(tensor_model_parallel_size=1)
    layer = _build(K, N, seed, dtype)
    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE, act_type=dtype,
        group_size=GROUP, zero_points=False, has_g_idx=False)
    kernel = RDNA3W4A16LinearKernel(cfg, w_q_param_name="qweight",
                                    w_s_param_name="scales",
                                    w_zp_param_name=None, w_gidx_param_name=None)
    kernel.process_weights_after_loading(layer)
    w_q, w_s, w_zp, w_g_idx = kernel._get_weight_params(layer)
    torch.manual_seed(seed + 1)
    x = torch.randn(M, K, device=device, dtype=dtype)
    return [
        torch.ops._rocm_C.gptq_gemm_rdna3(x, w_q, w_zp, w_s, w_g_idx, False)
        for _ in range(REPEATS)
    ]


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_splitk_bit_repeatable(dtype):
    """Scalar path, split-K active (Z = 1024/256 = 4 concurrent writers)."""
    outs = _run(M=1, K=1024, N=4096, seed=1234, dtype=dtype)
    for o in outs[1:]:
        assert torch.equal(o, outs[0]), (
            "RDNA3 W4A16 scalar split-K produced differing outputs for "
            "identical inputs")


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_small_m_bit_repeatable(dtype):
    """Scalar M>1 tiles share the same deterministic epilogue."""
    outs = _run(M=8, K=1024, N=4096, seed=1235, dtype=dtype)
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_splitk_bit_repeatable(dtype):
    """WMMA path (bf16 M >= 16, fp16 M >= 64), split-K active.

    K=6656 gives K_SPLIT=4 under the upstream heuristic; the deterministic
    epilogue must be bit-repeatable regardless of the split count.
    """
    M = 16 if dtype == torch.bfloat16 else 64
    outs = _run(M=M, K=6656, N=4096, seed=1236, dtype=dtype)
    for o in outs[1:]:
        assert torch.equal(o, outs[0]), (
            "RDNA3 W4A16 WMMA split-K produced differing outputs for "
            "identical inputs")


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_large_m_bit_repeatable(dtype):
    """Large-M WMMA tiles (128x64 kernels) with the deterministic epilogue."""
    outs = _run(M=128, K=6656, N=4096, seed=1237, dtype=dtype)
    for o in outs[1:]:
        assert torch.equal(o, outs[0])
