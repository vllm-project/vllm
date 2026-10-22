# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dory opt-in FP4 tensor-core MLP prefill (fp4_prefill_min_tokens).

Steps below the token threshold must run the W4A16 Marlin path unchanged; steps
at or above it run the NVFP4 x NVFP4 CUTLASS GEMM, which must stay close to
Marlin for unit-norm inputs (the only inputs Dory's MLP linears see).
"""

import pytest
import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
)
from vllm.model_executor.models import dory

pytestmark = pytest.mark.skipif(
    not (cutlass_fp4_supported() and is_fp4_marlin_supported()),
    reason="needs FP4 tensor cores (CUTLASS NVFP4 GEMM) and FP4 Marlin",
)

MIN_TOKENS = 128


class _MarlinW4A16:
    def apply(self, layer, x, bias=None):
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=None,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
        )


def _make_layer(name: str, n: int, k: int) -> nn.Module:
    """A W4A16 NVFP4 linear as ModelOptLinearMethod leaves it with
    _retain_weight_for_gather: Marlin layout + the serialized FP4 copies."""
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
    w_gs_inv = (448.0 * 6.0 / w.abs().max().float()).reshape(1)
    wq, w_sf = ops.scaled_fp4_quant(w, w_gs_inv, is_sf_swizzled_layout=False)
    w_sf = w_sf.view(torch.float8_e4m3fn).reshape(n, k // 16)

    layer = nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.weight = nn.Parameter(wq.clone(), requires_grad=False)
    layer.weight_scale = nn.Parameter(w_sf.clone(), requires_grad=False)
    layer.weight_global_scale = nn.Parameter(1.0 / w_gs_inv, requires_grad=False)
    layer.register_buffer("_nvfp4_weight_for_gather", wq.clone())
    layer.register_buffer("_nvfp4_weight_scale_for_gather", w_sf.clone())
    layer.register_buffer(
        "_nvfp4_weight_global_scale_for_gather", (1.0 / w_gs_inv).clone()
    )
    prepare_fp4_layer_for_marlin(layer)
    layer.quant_method = _MarlinW4A16()
    layer._dory_fp4_min_tokens = MIN_TOKENS
    dory._FP4_LINEARS[name] = layer
    return layer


@pytest.mark.parametrize("n,k", [(2 * 7168, 2560), (2560, 7168)])
@pytest.mark.parametrize("m", [1, 8, 127, 128, 512, 1536])
def test_fp4_linear_vs_marlin(m, n, k):
    torch.manual_seed(0)
    name = f"test.fp4.{n}x{k}"
    layer = _make_layer(name, n, k)
    x = dory.justnorm(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
    out = torch.ops.vllm.dory_fp4_linear(x, name)
    ref = layer.quant_method.apply(layer, x)
    assert out.shape == ref.shape == (m, n)
    if m < MIN_TOKENS:
        # below the threshold: the Marlin path itself, bit for bit
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
        return
    out, ref = out.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(out, ref, dim=-1)
    rel = (out - ref).norm() / ref.norm()
    # FP4 activation rounding: ~6% relative error per GEMM, never a wrong scale
    assert cos.min() > 0.99, cos.min()
    assert rel < 0.1, rel
