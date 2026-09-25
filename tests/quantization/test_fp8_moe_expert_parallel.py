# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 MoE block_n / expert-parallel alignment helpers."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.engine.arg_utils import (
    EngineArgs,
    _fp8_weight_block_n_from_model_config,
    _moe_intermediate_size_from_model_config,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    fp8_moe_tp_requires_expert_parallel,
    validate_fp8_block_shape_moe,
)


def _flash_next_fp8_model_config(
    moe_intermediate_size: int = 640,
    is_moe: bool = True,
    weight_block_size: list[int] | None = None,
):
    quant_config = None
    if weight_block_size is not None:
        quant_config = {
            "quant_method": "fp8",
            "weight_block_size": weight_block_size,
        }
    return SimpleNamespace(
        is_moe=is_moe,
        hf_text_config=SimpleNamespace(moe_intermediate_size=moe_intermediate_size),
        hf_config=SimpleNamespace(quantization_config=quant_config),
        model_arch_config=SimpleNamespace(quantization_config=quant_config),
    )


@pytest.mark.parametrize(
    ("moe_n", "tp_size", "block_n", "expected"),
    [
        # Qwen3.8-Flash-Next-FP8: N=640, block_n=128. TP>1 cannot shard N.
        (640, 1, 128, False),
        (640, 2, 128, True),
        (640, 4, 128, True),
        (640, 8, 128, True),
        # Qwen3-30B-A3B-FP8-like: N=768. TP2 fits; TP4 does not.
        (768, 2, 128, False),
        (768, 4, 128, True),
        # Dense / invalid inputs should never force EP.
        (0, 4, 128, False),
        (640, 3, 128, False),
        (640, 4, 0, False),
    ],
)
def test_fp8_moe_tp_requires_expert_parallel(moe_n, tp_size, block_n, expected):
    assert (
        fp8_moe_tp_requires_expert_parallel(moe_n, tp_size, block_n) is expected
    )


def test_model_config_extractors():
    cfg = _flash_next_fp8_model_config(weight_block_size=[128, 128])
    assert _moe_intermediate_size_from_model_config(cfg) == 640
    assert _fp8_weight_block_n_from_model_config(cfg) == 128

    empty = _flash_next_fp8_model_config(weight_block_size=None)
    assert _fp8_weight_block_n_from_model_config(empty) is None


def test_auto_enable_ep_for_flash_next_fp8_tp4():
    cfg = _flash_next_fp8_model_config(weight_block_size=[128, 128])
    args = EngineArgs(tensor_parallel_size=4)
    assert args._maybe_auto_enable_fp8_moe_expert_parallel(cfg) is True

    args_already = EngineArgs(tensor_parallel_size=4, enable_expert_parallel=True)
    assert args_already._maybe_auto_enable_fp8_moe_expert_parallel(cfg) is True

    args_tp1 = EngineArgs(tensor_parallel_size=1)
    assert args_tp1._maybe_auto_enable_fp8_moe_expert_parallel(cfg) is False


def test_auto_enable_ep_uses_tp_times_dp_world_size():
    cfg = _flash_next_fp8_model_config(weight_block_size=[128, 128])
    # TP=1, DP=4 is a TP group of size 4 for MoE without EP.
    args = EngineArgs(tensor_parallel_size=1, data_parallel_size=4)
    assert args._maybe_auto_enable_fp8_moe_expert_parallel(cfg) is True


def test_auto_enable_skips_non_moe_and_unquantized():
    moe_unquant = _flash_next_fp8_model_config(weight_block_size=None)
    dense = _flash_next_fp8_model_config(is_moe=False, weight_block_size=[128, 128])
    args = EngineArgs(tensor_parallel_size=4)
    assert args._maybe_auto_enable_fp8_moe_expert_parallel(moe_unquant) is False
    assert args._maybe_auto_enable_fp8_moe_expert_parallel(dense) is False


def test_validate_fp8_block_shape_moe_mentions_expert_parallel():
    with (
        patch(
            "vllm.distributed.get_tensor_model_parallel_world_size",
            return_value=4,
        ),
        pytest.raises(ValueError, match="--enable-expert-parallel"),
    ):
        validate_fp8_block_shape_moe(160, [128, 128])
