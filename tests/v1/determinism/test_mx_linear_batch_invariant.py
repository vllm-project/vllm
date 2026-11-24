# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MX linear kernels used under batch invariance must not depend on the
number of rows."""

import pytest
import torch
from utils import requires_mx, skip_if_not_cuda

import vllm.envs as envs
import vllm.model_executor.determinism.batch_invariant as bi
from vllm.model_executor.kernels.linear import (
    EmulationMxfp8LinearKernel,
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    HummingMxfp8LinearKernel,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.utils.flashinfer import has_flashinfer_batch_invariant_quantizers
from vllm.utils.import_utils import has_humming
from vllm.utils.torch_utils import set_random_seed

SEED = 0

# E8M0 exponent half-width. With 4-bit operands and a narrow spread the fp32
# partial sums are exact, so a reordering of them would be invisible.
SCALE_SPREAD = 15

# (N, K, use_asm_gemm)
MXFP4_CASES = [
    (4096, 2048, False),
    (2048, 6144, False),
    (4608, 7168, False),
    (8192, 2048, True),
    (8192, 1024, True),
    (8192, 14336, True),
]

MXFP8_CASES = [(4096, 2048), (2048, 6144), (1024, 768), (1536, 1024)]

# The untuned AITER MXFP4 configs switch at M = 9/33/65/129/257/513.
MXFP4_TOKEN_COUNTS = [
    1,
    2,
    8,
    9,
    16,
    32,
    33,
    64,
    65,
    128,
    129,
    256,
    257,
    512,
    513,
    1024,
    2048,
]

MXFP8_TOKEN_COUNTS = [1, 32, 64, 65, 128, 129, 256, 257, 512, 1024, 1025, 2048]

# Row 0 lands in the first tile of every decomposition, so it can stay
# invariant when the rest of the output does not.
CHECK_ROWS = [0, 1, 2, 3, 7, 15, 31]


def _probe(out: torch.Tensor) -> torch.Tensor:
    """The leading rows of an output, kept so the sweep does not hold every GEMM."""
    return out[: CHECK_ROWS[-1] + 1].clone()


def _variant_rows(probes: dict[int, torch.Tensor]) -> list[str]:
    """Rows whose contents changed with the number of rows in the launch."""
    failures = []
    for row in CHECK_ROWS:
        counts = [n for n in sorted(probes) if n > row]
        if not counts:
            continue
        reference = probes[counts[0]][row]
        variant = [n for n in counts if not torch.equal(probes[n][row], reference)]
        if variant:
            failures.append(f"row {row} changed at row counts {variant}")
    return failures


def _mxfp4_weights(n: int, k: int, use_asm_gemm: bool):
    """Weights laid out as AiterMxfp4LinearKernel.process_weights_after_loading."""
    weight = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.randint(
        127 - SCALE_SPREAD,
        128 + SCALE_SPREAD,
        (n, k // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    if not use_asm_gemm:
        return weight, weight_scale.T.contiguous()

    from aiter.ops.shuffle import shuffle_weight

    sm, sn = weight_scale.shape
    weight_scale = (
        weight_scale.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        .permute(0, 3, 5, 2, 4, 1, 6)
        .contiguous()
        .view(sm, sn)
    )
    return shuffle_weight(weight, layout=(16, 16)), weight_scale


def _mxfp8_weights(n: int, k: int):
    """E4M3 weights with E8M0 block scales, as the MXFP8 layer stores them."""
    weight = (torch.randn(n, k, device="cuda") / 8).to(torch.float8_e4m3fn)
    weight_scale = torch.randint(
        127 - SCALE_SPREAD,
        128 + SCALE_SPREAD,
        (n, k // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    return weight, weight_scale


@requires_mx
@pytest.mark.parametrize("n,k,use_asm_gemm", MXFP4_CASES)
def test_mxfp4_linear_is_batch_invariant(n: int, k: int, use_asm_gemm: bool):
    pytest.importorskip("aiter")

    from vllm._aiter_ops import rocm_aiter_ops

    set_random_seed(SEED)
    weight, weight_scale = _mxfp4_weights(n, k, use_asm_gemm)
    x = torch.randn(max(MXFP4_TOKEN_COUNTS), k, device="cuda", dtype=torch.bfloat16)

    probes = {
        num_tokens: _probe(
            rocm_aiter_ops.mxfp4_gemm_dynamic_quant(
                x[:num_tokens], weight, weight_scale, use_asm_gemm, torch.bfloat16
            )
        )
        for num_tokens in MXFP4_TOKEN_COUNTS
    }

    failures = _variant_rows(probes)
    assert not failures, (
        f"MXFP4 linear depends on the row count "
        f"(N={n}, K={k}, asm={use_asm_gemm}):\n  " + "\n  ".join(failures)
    )


@requires_mx
@pytest.mark.parametrize("n,k", MXFP8_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_linear_is_batch_invariant(n: int, k: int, dtype: torch.dtype):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_linear,
    )

    set_random_seed(SEED)
    weight, weight_scale = _mxfp8_weights(n, k)
    x = torch.randn(max(MXFP8_TOKEN_COUNTS), k, device="cuda", dtype=dtype)

    probes = {
        num_tokens: _probe(
            _mxfp8_dot_scaled_linear(x[:num_tokens], weight, weight_scale)
        )
        for num_tokens in MXFP8_TOKEN_COUNTS
    }

    failures = _variant_rows(probes)
    assert not failures, (
        f"MXFP8 linear depends on the row count (N={n}, K={k}, {dtype}):\n  "
        + "\n  ".join(failures)
    )


requires_flashinfer_batch_invariant = pytest.mark.skipif(
    not has_flashinfer_batch_invariant_quantizers(),
    reason="FlashInfer's batch-invariant MXFP8 quantizer and GEMM need FlashInfer "
    ">= 0.7.4 on SM100 / SM103 / SM107",
)

_FLASHINFER_MXFP8_KERNELS = {
    "flashinfer_cutedsl": FlashInferCutedslMxfp8LinearKernel,
    "flashinfer_cutlass": FlashInferCutlassMxfp8LinearKernel,
}
# Both sides of the FlashInfer dispatcher's 64-row boundary, both tile regimes
# and a tile-unaligned count; the full MXFP8_TOKEN_COUNTS compiles one kernel
# per configuration and would take minutes.
_FLASHINFER_MXFP8_TOKEN_COUNTS = [1, 32, 64, 65, 128, 257, 1024, 2048]
_MXFP8_QUANT_CONFIG = ModelOptMxFp8Config(
    is_checkpoint_mxfp8_serialized=True, kv_cache_quant_algo=None, exclude_modules=[]
)


def _mxfp8_column_linear(n: int, k: int, prefix: str) -> ColumnParallelLinear:
    """An MXFP8 ColumnParallelLinear with the weights of `_mxfp8_weights`,
    built under the current value of VLLM_BATCH_INVARIANT."""
    layer = ColumnParallelLinear(
        input_size=k,
        output_size=n,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=_MXFP8_QUANT_CONFIG,
        prefix=prefix,
        disable_tp=True,
    ).cuda()
    weight, weight_scale = _mxfp8_weights(n, k)
    layer.weight.data.copy_(weight)
    layer.weight_scale.data.copy_(weight_scale)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


@skip_if_not_cuda
@requires_flashinfer_batch_invariant
@pytest.mark.parametrize("n,k", MXFP8_CASES)
@pytest.mark.parametrize("linear_backend", sorted(_FLASHINFER_MXFP8_KERNELS))
def test_flashinfer_mxfp8_linear_is_batch_invariant(
    default_vllm_config, dist_init, n: int, k: int, linear_backend: str
):
    """Under VLLM_BATCH_INVARIANT (set by conftest) on SM100-family parts the
    layer selects the requested FlashInfer MXFP8 kernel, whose quantizer is the
    one a direct `flashinfer.mxfp8_quantize(..., batch_invariant=True)` call
    runs, and the leading rows of `apply` do not depend on the row count."""
    from flashinfer import mxfp8_quantize

    default_vllm_config.kernel_config.linear_backend = linear_backend
    set_random_seed(SEED)
    layer = _mxfp8_column_linear(n, k, "model.layers.0.self_attn.o_proj")
    kernel = getattr(layer.quant_method, "kernel", None)
    assert type(kernel) is _FLASHINFER_MXFP8_KERNELS[linear_backend]
    x = torch.randn(max(_FLASHINFER_MXFP8_TOKEN_COUNTS), k, device="cuda").to(
        torch.bfloat16
    )

    codes, sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True, batch_invariant=True)
    ref_codes, ref_sf = mxfp8_quantize(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
        backend="auto",
        batch_invariant=True,
    )
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(sf, ref_sf)

    probes = {
        num_tokens: _probe(layer.quant_method.apply(layer, x[:num_tokens]))
        for num_tokens in _FLASHINFER_MXFP8_TOKEN_COUNTS
    }
    failures = _variant_rows(probes)
    assert not failures, (
        f"FlashInfer MXFP8 linear depends on the row count "
        f"(N={n}, K={k}, {linear_backend}):\n  " + "\n  ".join(failures)
    )


@skip_if_not_cuda
@requires_flashinfer_batch_invariant
def test_flashinfer_mxfp8_selection_under_mode(default_vllm_config):
    """Under the mode a FlashInfer MXFP8 kernel is selected for a layer inside
    its shape rules; K % 32 != 0, N or K below 128 and activation rows over
    128 KiB keep the Emulation fallback."""
    kernel = init_mxfp8_linear_kernel(weight_shape=(4096, 2048))
    assert isinstance(kernel, tuple(_FLASHINFER_MXFP8_KERNELS.values())), type(kernel)
    for weight_shape in [(4096, 4112), (64, 4096), (4096, 64), (256, 65568)]:
        kernel = init_mxfp8_linear_kernel(weight_shape=weight_shape)
        assert type(kernel) is EmulationMxfp8LinearKernel, weight_shape


@skip_if_not_cuda
@requires_flashinfer_batch_invariant
def test_flashinfer_quantizers_flag_follows_the_mode(default_vllm_config, monkeypatch):
    """The resolved flag the layers read agrees with the kernel selection,
    which reads the mode live, when the mode changes within one process."""
    assert bi.flashinfer_quantizers_enabled()
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert not bi.flashinfer_quantizers_enabled()
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert bi.flashinfer_quantizers_enabled()
    kernel = init_mxfp8_linear_kernel(weight_shape=(4096, 2048))
    assert isinstance(kernel, tuple(_FLASHINFER_MXFP8_KERNELS.values()))


@skip_if_not_cuda
@requires_flashinfer_batch_invariant
def test_minimax_m3_fused_gemm1_off_under_batch_invariance(
    default_vllm_config, dist_init, monkeypatch
):
    """MiniMax-M3's fused GEMM1 switches by token count and quantizes the
    activation itself; with the FlashInfer MXFP8 kernels selected under the mode
    it must stay off, while the same layers take it with the mode off."""
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.models.minimax_m3.nvidia.ops.dense_gemm1_mxfp8 import can_fuse_gemm1

    act_fn = SiluAndMulWithClamp(swiglu_limit=7.0, alpha=1.702, beta=1.0)

    def layers():
        gate_up = _mxfp8_column_linear(2 * 768, 1024, "model.layers.0.mlp.gate_up")
        down = RowParallelLinear(
            input_size=768,
            output_size=1024,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_MXFP8_QUANT_CONFIG,
            prefix="model.layers.0.mlp.down_proj",
            disable_tp=True,
        ).cuda()
        return gate_up, down

    gate_up, down = layers()
    assert not can_fuse_gemm1(gate_up, down, act_fn)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    gate_up, down = layers()
    assert can_fuse_gemm1(gate_up, down, act_fn)


@skip_if_not_cuda
@pytest.mark.skipif(not has_humming(), reason="requires Humming")
@pytest.mark.parametrize("n,k", MXFP8_CASES)
def test_humming_mxfp8_linear_is_batch_invariant(
    default_vllm_config, dist_init, n: int, k: int
):
    """The layer captures Humming's compute config while loading, so
    VLLM_BATCH_INVARIANT (set by conftest) must be on before it is built."""
    default_vllm_config.kernel_config.linear_backend = "humming"
    set_random_seed(SEED)
    layer = ColumnParallelLinear(
        input_size=k,
        output_size=n,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        prefix="model.layers.0.self_attn.o_proj",
        disable_tp=True,
    ).cuda()
    assert type(layer.quant_method.kernel) is HummingMxfp8LinearKernel
    weight, weight_scale = _mxfp8_weights(n, k)
    layer.weight.data.copy_(weight)
    layer.weight_scale.data.copy_(weight_scale)
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(max(MXFP8_TOKEN_COUNTS), k, device="cuda", dtype=torch.bfloat16)

    probes = {
        num_tokens: _probe(layer.quant_method.apply(layer, x[:num_tokens]))
        for num_tokens in MXFP8_TOKEN_COUNTS
    }

    failures = _variant_rows(probes)
    assert not failures, (
        f"Humming MXFP8 linear depends on the row count (N={n}, K={k}):\n  "
        + "\n  ".join(failures)
    )
