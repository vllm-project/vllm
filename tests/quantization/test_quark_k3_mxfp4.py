# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.quark import quark_moe


@pytest.mark.parametrize(
    ("scheme", "opt_in", "aiter_supported", "expected"),
    [
        ("w_mxfp4_a_mxfp4", True, True, True),
        ("w_mxfp4_a_mxfp4", False, True, False),
        ("w_mxfp4_a_mxfp4", True, False, False),
        ("w_mxfp4", True, True, False),
        ("w_mxfp4_a_fp8", True, True, False),
    ],
)
def test_use_k3_situ_aiter_a8w4(monkeypatch, scheme, opt_in, aiter_supported, expected):
    monkeypatch.setattr(quark_moe.envs, "VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4", opt_in)
    monkeypatch.setattr(quark_moe, "_use_k3_situ_aiter", lambda _moe: aiter_supported)

    assert quark_moe._use_k3_situ_aiter_a8w4(MagicMock(), scheme) is expected


def test_k3_situ_a8w4_keeps_tp8_intermediate_size():
    method = object.__new__(quark_moe.QuarkOCP_MX_MoEMethod)
    method.is_k3_situ_aiter_a8w4 = True
    method.mxfp4_backend = Mxfp4MoeBackend.AITER_MXFP4_BF16

    with patch.object(
        quark_moe.QuarkMoEMethod,
        "maybe_roundup_sizes",
        return_value=(3584, 384),
    ):
        hidden_size, intermediate_size = method.maybe_roundup_sizes(
            hidden_size=3584,
            intermediate_size_per_partition=384,
            act_dtype=MagicMock(),
            moe_parallel_config=MagicMock(),
        )

    assert hidden_size == 3584
    assert intermediate_size == 384
