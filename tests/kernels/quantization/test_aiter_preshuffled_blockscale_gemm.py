# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AiterPreshuffledFp8BlockScaledMMKernel end to end on the GPU: it shuffles
the weight, quantizes the activation with a column-major scale and runs the
AITER bpreshuffle GEMM. The output must match a native torch reference and
the plain AITER blockscale kernel on the same weight."""

import importlib.util

import pytest
import torch

from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
    AiterPreshuffledFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    scaled_dequantize,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = [
    pytest.mark.skipif(
        not (_on_gfx950() and importlib.util.find_spec("aiter")),
        reason="Requires gfx950 + aiter",
    ),
    pytest.mark.usefixtures("default_vllm_config"),
]

BLOCK = 128
# Mean relative error of the bf16 output; the same bound as the CK rows of
# test_w8a8_block_fp8_aiter_matmul. CK split-K configs reduce partial sums
# into the bf16 output with atomics, which costs about one bf16 rounding.
REL_TOL = 5e-3


@pytest.fixture(autouse=True)
def aiter_linear_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    # can_implement rules out shapes the plain Triton GEMM has tuned, so the
    # parent always resolves to CK `gemm_a8w8_blockscale` here.
    monkeypatch.setattr(rocm_aiter_ops, "is_triton_gemm_w8a8_tuned", lambda n, k: False)
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _config(n: int, k: int) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(n, k),
    )


def _fp8_weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()
    fp8_info = torch.finfo(fp8_dtype)
    w = ((torch.rand(n, k, device="cuda") - 0.5) * 2 * fp8_info.max).clamp(
        min=fp8_info.min, max=fp8_info.max
    )
    ws = torch.rand(cdiv(n, BLOCK), cdiv(k, BLOCK), device="cuda") * 1e-2
    return w.to(fp8_dtype), ws


def _layer(w: torch.Tensor, ws: torch.Tensor, **tags: bool) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(ws.clone(), requires_grad=False)
    for name, value in tags.items():
        setattr(layer, name, value)
    return layer


def _reference(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    x_q, x_s = native_per_token_group_quant_fp8(x, BLOCK)
    return native_w8a8_block_matmul(x_q, w, x_s, ws, [BLOCK, BLOCK], x.dtype)


def _rel_diff(out: torch.Tensor, ref: torch.Tensor) -> float:
    out, ref = out.float(), ref.float()
    return (torch.mean(torch.abs(out - ref)) / torch.mean(torch.abs(ref))).item()


# GLM-5.x and DeepSeek-V4 linear shapes; 576 is a multiple of 16, not of 128.
@pytest.mark.parametrize(
    ("n", "k"), [(2048, 4096), (7168, 2048), (576, 7168), (16384, 1536)]
)
@pytest.mark.parametrize("m", [1, 33, 256, 4096])
def test_preshuffled_gemm_matches_reference(m: int, n: int, k: int) -> None:
    torch.manual_seed(0)
    w, ws = _fp8_weight(n, k)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

    kernel = AiterPreshuffledFp8BlockScaledMMKernel(_config(n, k))
    layer = _layer(w, ws)
    kernel.process_weights_after_loading(layer)

    assert kernel.preshuffled
    assert layer.aiter_bpreshuffled
    assert not torch.equal(layer.weight.view(torch.uint8), w.view(torch.uint8))

    out = kernel.apply_weights(layer, x)

    plain_kernel = AiterFp8BlockScaledMMKernel(_config(n, k))
    plain_layer = _layer(w, ws)
    plain_kernel.process_weights_after_loading(plain_layer)
    plain_out = plain_kernel.apply_weights(plain_layer, x)

    assert out.shape == (m, n)
    assert _rel_diff(out, _reference(x, w, ws)) < REL_TOL
    assert _rel_diff(out, plain_out) < REL_TOL


@pytest.mark.parametrize("m", [1, 256])
def test_bmm_layer_keeps_plain_layout(m: int) -> None:
    n, k = 2048, 4096
    torch.manual_seed(0)
    w, ws = _fp8_weight(n, k)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

    kernel = AiterPreshuffledFp8BlockScaledMMKernel(_config(n, k))
    layer = _layer(w, ws, is_bmm=True)
    kernel.process_weights_after_loading(layer)

    assert not kernel.preshuffled
    assert kernel.quant_fp8 is kernel._plain_quant_fp8
    assert not getattr(layer, "aiter_bpreshuffled", False)
    assert torch.equal(layer.weight.view(torch.uint8), w.view(torch.uint8))

    out = kernel.apply_weights(layer, x)

    assert _rel_diff(out, _reference(x, w, ws)) < REL_TOL


def test_dequant_reads_preshuffled_weight() -> None:
    # MLA reads kv_b_proj back through get_and_maybe_dequant_weights; for a
    # preshuffled weight it runs the layer on an identity matrix.
    n, k = 2048, 4096
    torch.manual_seed(0)
    w, ws = _fp8_weight(n, k)

    kernel = AiterPreshuffledFp8BlockScaledMMKernel(_config(n, k))
    layer = _layer(w, ws)
    kernel.process_weights_after_loading(layer)
    assert kernel.preshuffled

    method = object.__new__(Fp8LinearMethod)
    method.use_marlin = False
    method.use_deep_gemm = False
    method.block_quant = True
    method.weight_block_size = [BLOCK, BLOCK]
    method.fp8_linear = kernel
    layer.quant_method = method
    layer.weight_block_size = [BLOCK, BLOCK]
    layer.input_size_per_partition = k

    out = get_and_maybe_dequant_weights(layer, out_dtype=torch.bfloat16)
    ref = scaled_dequantize(w, ws, group_shape=[BLOCK, BLOCK], out_dtype=torch.bfloat16)

    assert out.shape == (n, k)
    assert _rel_diff(out, ref) < REL_TOL
