# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

This file contains a single canonical demonstration that
`UnquantizedMoEKernelOracle` methods delegate one-to-one to the
existing module-level functions in `oracle/unquantized.py`. Each method
on `UnquantizedMoEKernelOracle` follows the same `return module_fn(args)`
pattern, so verifying delegation for one method (`make_kernel`) gives
high confidence in the rest.
"""

from unittest import mock
from unittest.mock import patch

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.platforms import current_platform


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

    def test_make_kernel_delegates(self) -> None:
        quant_config = object()
        moe_config = object()
        experts_cls = TritonExperts
        sentinel_kernel = object()

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "make_unquantized_moe_kernel",
            return_value=sentinel_kernel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().make_kernel(
                quant_config,
                moe_config,
                UnquantizedMoeBackend.TRITON,
                experts_cls,
            )

        mocked.assert_called_once_with(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            experts_cls,
            None,  # routing_tables default
        )
        assert out is sentinel_kernel


# AITER's CK FP8 MoE GEMM (per-tensor, or per-channel with gelu_tanh) needs
# hidden % 128 == 0 and intermediate % 256 == 0 (% 128 for SwiGLU).


@pytest.mark.parametrize(
    ("weight_key_name", "activation", "hidden", "intermediate", "expected"),
    [
        # Per-tensor weights.
        ("kFp8StaticTensorSym", "SILU", 2048, 1536, (2048, 1536)),  # aligned
        ("kFp8StaticTensorSym", "SILU", 4096, 3072, (4096, 3072)),
        ("kFp8StaticTensorSym", "SILU", 2048, 1408, (2048, 1536)),  # DeepSeek-V2-Lite
        ("kFp8StaticTensorSym", "SILU", 2048, 384, (2048, 512)),  # Qwen3-30B-A3B TP2
        ("kFp8StaticTensorSym", "SILU", 2048, 192, (2048, 256)),
        ("kFp8StaticTensorSym", "GELU", 4096, 704, (4096, 768)),
        ("kFp8StaticTensorSym", "SWIGLUOAI", 2048, 1408, (2048, 1408)),  # needs 128
        ("kFp8StaticTensorSym", "SWIGLUOAI", 2880, 2880, (2944, 2944)),  # gpt-oss
        ("kFp8StaticTensorSym", "SWIGLUOAI_UNINTERLEAVE", 2880, 1440, (2944, 1536)),
        # Per-channel weights: only gelu_tanh (DiffusionGemma-26B-A4B).
        ("kFp8StaticChannelSym", "GELU_TANH", 2816, 704, (2816, 768)),  # TP1
        ("kFp8StaticChannelSym", "GELU_TANH", 2816, 352, (2816, 512)),  # TP2
        ("kFp8StaticChannelSym", "SILU", 2816, 704, (2816, 704)),  # not padded
    ],
)
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(
            "AITER",
            marks=pytest.mark.skipif(
                not is_aiter_found_and_supported(),
                reason="Only test on ROCm with AITER installed and supported",
            ),
        ),
    ],
)
def test_maybe_round_up_hidden_size_and_intermediate_size(
    backend_name, weight_key_name, activation, hidden, intermediate, expected
):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        maybe_round_up_hidden_size_and_intermediate_size,
    )
    from vllm.model_executor.layers.quantization.utils import quant_utils

    actual = maybe_round_up_hidden_size_and_intermediate_size(
        Fp8MoeBackend[backend_name],
        getattr(quant_utils, weight_key_name),
        MoEActivation[activation],
        hidden,
        intermediate,
    )
    assert actual == expected, f"Expected {expected}, got {actual}."


_requires_aiter = pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)


@pytest.mark.parametrize(
    (
        "backend_name",
        "weight_key_name",
        "activation_name",
        "hidden",
        "intermediate",
        "expect_round_up",
    ),
    [
        # Per-tensor weights, DeepSeek-V2-Lite shape.
        pytest.param(
            "AITER",
            "kFp8StaticTensorSym",
            "SILU",
            2048,
            1408,
            True,
            marks=_requires_aiter,
        ),
        ("TRITON", "kFp8StaticTensorSym", "SILU", 2048, 1408, False),
        pytest.param(
            "AITER",
            "kFp8StaticChannelSym",
            "SILU",
            2048,
            1408,
            False,
            marks=_requires_aiter,
        ),
        pytest.param(
            "AITER",
            "kFp8Static128BlockSym",
            "SILU",
            2048,
            1408,
            False,
            marks=_requires_aiter,
        ),
        # Per-channel gelu_tanh, DiffusionGemma-26B-A4B shape.
        pytest.param(
            "AITER",
            "kFp8StaticChannelSym",
            "GELU_TANH",
            2816,
            704,
            True,
            marks=_requires_aiter,
        ),
        pytest.param(
            "AITER",
            "kFp8Static128BlockSym",
            "GELU_TANH",
            2816,
            704,
            False,
            marks=_requires_aiter,
        ),
        ("TRITON", "kFp8StaticChannelSym", "GELU_TANH", 2816, 704, False),
    ],
)
def test_maybe_round_up_calls_fp8_moe_round_up_sizes_only_when_needed(
    backend_name,
    weight_key_name,
    activation_name,
    hidden,
    intermediate,
    expect_round_up,
):
    """Round up only for AITER with per-tensor weights, or per-channel weights
    with gelu_tanh."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        maybe_round_up_hidden_size_and_intermediate_size,
    )
    from vllm.model_executor.layers.quantization.utils import quant_utils

    with mock.patch(
        "vllm.model_executor.layers.fused_moe.oracle.fp8.fp8_moe_round_up_sizes"
    ) as round_up:
        maybe_round_up_hidden_size_and_intermediate_size(
            Fp8MoeBackend[backend_name],
            getattr(quant_utils, weight_key_name),
            MoEActivation[activation_name],
            hidden,
            intermediate,
        )

    if expect_round_up:
        round_up.assert_called_once_with(
            MoEActivation[activation_name], hidden, intermediate
        )
    else:
        round_up.assert_not_called()


@pytest.mark.parametrize(
    ("hidden", "inter", "hidden_pad", "inter_pad"),
    [
        (2880, 1408, 2944, 1536),  # both padded
        (2048, 1408, 2048, 1536),  # intermediate only
        (2880, 1536, 2944, 1536),  # hidden only
        (704, 640, 768, 768),  # both padded, small
        (2048, 1536, 2048, 1536),  # aligned, nothing to pad
    ],
)
def test_pad_fp8_moe_weights(hidden, inter, hidden_pad, inter_pad):
    """Loaded FP8 expert weights and per-channel scales are zero-padded to the
    kernel's sizes, with gate and up each starting at their padded offsets."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import pad_fp8_moe_weights

    moe_config = make_dummy_moe_config(
        num_experts=2, hidden_dim=hidden, intermediate_size=inter
    )

    # Gate is all ones and up all twos, so a misplaced half shows up.
    fp8_dtype = current_platform.fp8_dtype()
    w13 = torch.cat([torch.ones(2, inter, hidden), 2 * torch.ones(2, inter, hidden)], 1)
    w13 = w13.to(fp8_dtype)
    w2 = torch.ones(2, hidden, inter).to(fp8_dtype)
    w13_scale = torch.cat([3 * torch.ones(2, inter, 1), 4 * torch.ones(2, inter, 1)], 1)
    w2_scale = 5 * torch.ones(2, hidden, 1)

    w13, w2, w13_scale, w2_scale = pad_fp8_moe_weights(
        moe_config, w13, w2, w13_scale, w2_scale, hidden_pad, inter_pad
    )

    assert w13.shape == (2, 2 * inter_pad, hidden_pad)
    assert w2.shape == (2, hidden_pad, inter_pad)
    assert w13_scale.shape == (2, 2 * inter_pad, 1)
    assert w2_scale.shape == (2, hidden_pad, 1)

    # fp8 has no comparison ops on ROCm; 0, 1 and 2 convert to float exactly.
    w13, w2 = w13.float(), w2.float()

    # The loaded values are in place.
    assert (w13[:, :inter, :hidden] == 1).all(), "Expected w13 gate to be all ones."
    assert (w13[:, inter_pad : inter_pad + inter, :hidden] == 2).all(), (
        "Expected w13 up to be all twos."
    )
    assert (w2[:, :hidden, :inter] == 1).all(), "Expected w2 to be all ones."
    assert (w13_scale[:, :inter] == 3).all(), "Expected the gate scales."
    assert (w13_scale[:, inter_pad : inter_pad + inter] == 4).all(), (
        "Expected the up scales."
    )
    assert (w2_scale[:, :hidden] == 5).all(), "Expected the w2 scales."

    # The padding is zero.
    assert (w13[:, inter:inter_pad] == 0).all(), "Expected zero w13 gate padding."
    assert (w13[:, inter_pad + inter :] == 0).all(), "Expected zero w13 up padding."
    assert (w13[:, :, hidden:] == 0).all(), "Expected zero w13 hidden padding."
    assert (w2[:, hidden:] == 0).all(), "Expected zero w2 hidden padding."
    assert (w2[:, :, inter:] == 0).all(), "Expected zero w2 intermediate padding."
