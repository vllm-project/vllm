# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming's MXFP4 path for FP8 and INT8 activations (fused E8M0 scales)
decodes a weight block exactly only if its E8M0 scale is at most 11 (FP8) or
3 (INT8) binades below the largest scale of the weight tensor. Layers with
nonzero blocks further down must use 16-bit activations instead."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.humming import linear
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
GROUP = 32


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """Pack (N, K) E2M1 codes into (N, K // 2) bytes, low nibble first."""
    codes = codes.to(torch.uint8)
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def _blocks(scale_codes: list[list[int]]):
    """Weights with every E2M1 value set to 0.5, one block per scale."""
    scale = torch.tensor(scale_codes, dtype=torch.uint8)
    return _pack(torch.ones(scale.shape[0], scale.shape[1] * GROUP)), scale


@pytest.mark.parametrize(
    ("scale_codes", "pad_n", "expected"),
    [
        ([[127, 116]], False, 0),  # 11 binades below the largest scale
        ([[127, 115, 0]], False, 2),  # 12 and 127 below
        ([[130, 119]], False, 0),  # the range follows the largest scale
        ([[127, 127], [115, 112]], False, 2),  # fused shards share one range
        ([[116, 116]], True, 0),  # Humming pads N with scale 2^0
        ([[115, 115]], True, 2),
    ],
)
def test_blocks_below_range(scale_codes, pad_n, expected):
    weight, scale = _blocks(scale_codes)
    assert linear._fused_e8m0_lossy_blocks(weight, scale, 11, pad_n=pad_n) == expected


def test_per_row_range():
    """With a per-channel secondary scale (INT8), each row has its own range."""
    weight, scale = _blocks([[127, 123], [100, 100]])
    assert linear._fused_e8m0_lossy_blocks(weight, scale, 3, per_row=True) == 1
    assert linear._fused_e8m0_lossy_blocks(weight, scale, 3) == 3


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not has_humming()
    or not current_platform.has_device_capability(89)
    or current_platform.has_device_capability(100),
    reason="Humming runs MXFP4 layers with FP8 activations on SM89 and SM90",
)
@pytest.mark.parametrize("case", ["in_range", "below_range", "n_padded", "zero_block"])
def test_humming_mxfp4_layer_decodes_checkpoint(
    case, default_vllm_config, dist_init, caplog_vllm
):
    """Read back the weights a Humming MXFP4 layer uses, with identity inputs."""
    from vllm.model_executor.kernels.linear import HummingMxFp4LinearKernel
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

    default_vllm_config.kernel_config.linear_backend = "humming"
    mxfp4 = {
        "num_bits": 4,
        "type": "float",
        "strategy": "group",
        "group_size": GROUP,
        "symmetric": True,
    }
    group = {
        "targets": ["Linear"],
        "weights": {**mxfp4, "dynamic": False},
        "input_activations": {**mxfp4, "dynamic": True},
    }
    quant_config = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "ignore": [],
            "config_groups": {"group_0": group},
        }
    )
    # Humming pads N to a multiple of 256.
    n, k = (128 if case == "n_padded" else 256), 512
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            k,
            n,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix="proj",
        )
    assert isinstance(layer.scheme.kernel, HummingMxFp4LinearKernel)

    codes = (torch.arange(n)[:, None] + torch.arange(k)) % 7 + 1
    codes[1::2] += 8  # negative values in odd rows
    scales = torch.full((n, k // GROUP), 127, dtype=torch.uint8)
    if case == "in_range":
        scales[0, 1] = 127 - 11
    elif case == "below_range":
        scales[0, 1:5] = torch.tensor([127 - 12, 127 - 14, 127 - 15, 127 - 16])
    elif case == "n_padded":
        scales.fill_(127 - 16)
    else:
        scales[0, 1] = 0
        codes[0, GROUP : 2 * GROUP] = 8  # -0
    for name, tensor in (("weight_packed", _pack(codes)), ("weight_scale", scales)):
        param = getattr(layer, name)
        param.weight_loader(param, tensor.cuda())
    layer.quant_method.process_weights_after_loading(layer)

    with torch.inference_mode():
        out = layer(torch.eye(k, dtype=torch.bfloat16, device="cuda"))[0]
    scale = torch.exp2(scales.float() - 127).repeat_interleave(GROUP, dim=1)
    torch.testing.assert_close(out.float().T.cpu(), E2M1[codes] * scale, rtol=0, atol=0)
    fallback = case in ("below_range", "n_padded")
    assert layer.scheme.kernel.layer_config.use_fused_e8m0_scale != fallback
    assert ("16-bit activations" in caplog_vllm.text) == fallback
