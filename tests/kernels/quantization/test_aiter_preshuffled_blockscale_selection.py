# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The plain AITER blockscale GEMM is preferred; the bpreshuffle one only takes
shapes whose plain table is less tuned than its own."""

import importlib.util

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterPreshuffledFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not (current_platform.is_rocm() and importlib.util.find_spec("aiter")),
        reason="Requires ROCm + aiter",
    ),
    pytest.mark.usefixtures("default_vllm_config"),
]


@pytest.mark.parametrize(
    ("plain", "shuffled", "picks_bpreshuffle"),
    [
        (0, 15, True),  # plain table empty: bpreshuffle covers the shape
        (6, 15, True),  # plain tuned at a few M only
        (15, 15, False),  # tie: keep the checkpoint layout
        (15, 6, False),
        (15, 0, False),  # bpreshuffle untuned
        (0, 0, False),
    ],
)
def test_plain_blockscale_gemm_has_priority(
    monkeypatch: pytest.MonkeyPatch, plain: int, shuffled: int, picks_bpreshuffle
) -> None:
    def tuned_m_count(n: int, k: int, preshuffled: bool) -> int:
        return shuffled if preshuffled else plain

    monkeypatch.setattr(
        rocm_aiter_ops, "blockscale_tuned_m_count", staticmethod(tuned_m_count)
    )
    monkeypatch.setattr(rocm_aiter_ops, "is_linear_fp8_enabled", lambda: True)
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(2048, 4096),
    )

    ok, reason = AiterPreshuffledFp8BlockScaledMMKernel.can_implement(config)

    assert ok is picks_bpreshuffle, reason
