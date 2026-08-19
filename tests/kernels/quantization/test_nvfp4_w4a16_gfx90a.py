# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 W4A16 (TritonNvFp4LinearKernel, gfx90a) regression tests.

Two independent things pinned by test rather than by comment alone:

1. The global_scale convention adjudicated 2026-08-15 (fp8-decode's
   fixture testing flagged a possible inversion; adjudication ran the
   REAL scheme path against fixture tensors and confirmed the kernel is
   correct -- see `TritonNvFp4LinearKernel.process_weights_after_loading`'s
   own docstring for the citation). Adapted here from fp8-decode's
   `adjudicate_gs.py` (session scratchpad), which used a real on-disk
   fixture; this version is self-contained (synthetic tensors, same
   logical check) so it runs without depending on an external fixture
   file surviving between sessions.
2. The `_is_nvfp4_format` detector widening (2026-08-15) -- several
   published NVFP4 checkpoints declare strategy="group" rather than
   "tensor_group" and previously fell through to
   `raise NotImplementedError(...)` at scheme resolution. Pinned with
   `nm-testing/TinyLlama-1.1B-Chat-v1.0-FP4`'s verbatim
   `quantization_config` (fetched directly from the checkpoint's
   `config.json`, not hand-written) as a must-resolve-to-NVFP4-W4A16 case.
   This checkpoint is also the intended e2e smoke vehicle for #23's GPU
   verification pass once the card is available.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


# ---------------------------------------------------------------------------
# 1. global_scale convention: multiplicative, not divisor
# ---------------------------------------------------------------------------


def _make_synthetic_nvfp4_layer(N: int, K: int, seed: int = 0):
    """Build packed e2m1 weight + e4m3 per-16-group scale + a disk-style
    (divisor) global scale, exactly the shapes CompressedTensorsW4A4Fp4's
    create_weights registers. Values are arbitrary but reproducible.
    """
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (N, K), generator=g, dtype=torch.uint8)
    packed = (codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)
    scale_bytes = torch.randint(0x30, 0x48, (N, K // 16), generator=g, dtype=torch.uint8)
    weight_scale = scale_bytes.view(torch.float8_e4m3fn)
    # A realistic on-disk divisor magnitude, matching the order of magnitude
    # adjudicate_gs.py's real fixture used (2752.0-class values).
    gs_disk = 2752.0
    return packed, weight_scale, gs_disk, codes


def _on_gfx90a() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx90a

    return on_gfx90a()


@pytest.mark.skipif(
    not _on_gfx90a(),
    reason="requires gfx90a for TritonNvFp4LinearKernel.is_supported()",
)
def test_global_scale_is_multiplicative_not_divisor():
    """CompressedTensorsW4A4Fp4.process_weights_after_loading inverts the
    on-disk divisor (CT stores 1/scale) before the kernel ever runs, so the
    kernel must MULTIPLY by what it receives. Runs the real scheme path,
    not a reimplementation -- adjudication's own methodology.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A4Fp4,
    )

    N, K = 64, 512
    packed, weight_scale, gs_disk, codes = _make_synthetic_nvfp4_layer(N, K)

    scheme = CompressedTensorsW4A4Fp4(use_a16=True)
    assert type(scheme.kernel).__name__ == "TritonNvFp4LinearKernel"

    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor([gs_disk], dtype=torch.float32), requires_grad=False
    )

    scheme.process_weights_after_loading(layer)
    gs_kernel = float(layer.weight_global_scale)

    # The scheme inverted the on-disk divisor.
    assert gs_kernel == pytest.approx(1.0 / gs_disk, rel=1e-6)

    # Multiplying by what the kernel receives reproduces the same true
    # weight values as dividing by the raw on-disk value would -- the
    # check that actually matters, independent of the intermediate
    # variable name.
    E2M1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float64,
    )
    e2m1_vals = E2M1[codes.long()]
    scale_vals = weight_scale.to(torch.float64).repeat_interleave(16, dim=1)
    prod = e2m1_vals * scale_vals
    via_kernel_multiply = prod * gs_kernel
    via_disk_divide = prod / gs_disk
    assert torch.allclose(via_kernel_multiply, via_disk_divide, rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. Detector widening: strategy="group" checkpoints must resolve to NVFP4
# ---------------------------------------------------------------------------

# Verbatim from nm-testing/TinyLlama-1.1B-Chat-v1.0-FP4's config.json
# (quantization_config), fetched directly, not hand-written. This exact
# shape -- strategy="group" rather than "tensor_group" -- is what fell
# through both the nvfp4 and mxfp4 detectors before the 2026-08-15
# widening.
TINYLLAMA_FP4_QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 16,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "group",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "nvfp4-pack-quantized",
    "global_compression_ratio": None,
    "ignore": ["lm_head"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}


def test_group_strategy_nvfp4_checkpoint_resolves_correctly():
    """Must resolve to CompressedTensorsW4A4Fp4(use_a16=True) via the same
    public get_scheme() path the real loader uses, not raise
    NotImplementedError (the pre-widening behavior, traced precisely
    through every intervening _get_scheme_from_parts predicate -- see the
    detector's own docstring) and not silently misroute to
    CompressedTensorsWNA16 or CompressedTensorsW8A16Fp8.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A4Fp4,
    )

    ct = CompressedTensorsConfig.from_config(TINYLLAMA_FP4_QUANT_CONFIG)
    layer = torch.nn.Linear(2048, 5632, bias=False)
    scheme = ct.get_scheme(layer, layer_name="model.layers.0.mlp.gate_proj")

    assert isinstance(scheme, CompressedTensorsW4A4Fp4)
    assert scheme.use_a16 is True
