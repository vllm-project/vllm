# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash on compute capability 12: sink forwarding and NVFP4 expert names."""

import pytest

from vllm.models.hy_v4_flash.common.model import _EXPERT_WEIGHT
from vllm.models.hy_v4_flash.nvidia.sm120 import (
    FlashSM120Impl,
    FlashSM120SparseBackend,
    FlashSM120SWABackend,
)


def test_sm120_flash_backends_apply_sinks_under_the_shared_name() -> None:
    # The shared name keeps the fp8_ds_mla KV layout selection.
    assert FlashSM120SparseBackend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert FlashSM120SparseBackend.supports_sink()
    assert FlashSM120SparseBackend.get_impl_cls() is FlashSM120Impl
    assert FlashSM120SWABackend.supports_sink()
    assert FlashSM120SWABackend.supports_sliding_window()


@pytest.mark.parametrize(
    ("name", "expert", "projection", "suffix"),
    [
        ("layers.1.mlp.experts.7.gate_proj.weight", "7", "gate_proj", ".weight"),
        (
            "layers.1.mlp.experts.7.up_proj.weight_scale_2",
            "7",
            "up_proj",
            ".weight_scale_2",
        ),
        (
            "layers.1.mlp.experts.7.down_proj.input_scale",
            "7",
            "down_proj",
            ".input_scale",
        ),
        (
            "layers.1.mlp.experts.gate_up_proj_scale_inv",
            None,
            "gate_up_proj",
            "_scale_inv",
        ),
    ],
)
def test_expert_weight_names_cover_modelopt_nvfp4(
    name: str, expert: str | None, projection: str, suffix: str
) -> None:
    match = _EXPERT_WEIGHT.fullmatch(name)
    assert match is not None
    assert match.groups()[1:] == (expert, projection, suffix)
