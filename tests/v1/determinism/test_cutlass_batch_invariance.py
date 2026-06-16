# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from tests.utils import TestFP8Layer, requires_fp8
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

pytest.importorskip("torch.cuda")


@pytest.fixture(autouse=True)
def setup_cuda():
    if not current_platform.is_cuda():
        pytest.skip("CUTLASS FP8 kernels require CUDA.")
    torch.set_default_device("cuda")


@requires_fp8
@pytest.mark.parametrize("weight_shape", [(1024, 2048), (4608, 4096)])
@pytest.mark.parametrize("batch_size", [1, 16, 17, 32, 64, 65, 256, 257])
@torch.inference_mode()
def test_cutlass_fp8_batch_invariant_fixed_config(
    weight_shape: tuple[int, int],
    batch_size: int,
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    torch.manual_seed(0)
    layer = TestFP8Layer(
        weight_shape=weight_shape,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_quant_key=kFp8StaticTensorSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        force_kernel=CutlassFP8ScaledMMLinearKernel,
    )
    assert isinstance(layer.kernel, CutlassFP8ScaledMMLinearKernel)

    in_features = weight_shape[1]
    needle = torch.randn((1, in_features), device="cuda", dtype=torch.bfloat16)
    baseline = layer(needle)[0]

    filler = torch.randn(
        (max(batch_size - 1, 0), in_features), device="cuda", dtype=torch.bfloat16
    )

    front_batch = torch.cat([needle, filler], dim=0)
    back_batch = torch.cat([filler, needle], dim=0)

    front_output = layer(front_batch)[0]
    back_output = layer(back_batch)[-1]

    torch.testing.assert_close(front_output, baseline, rtol=0, atol=0)
    torch.testing.assert_close(back_output, baseline, rtol=0, atol=0)
