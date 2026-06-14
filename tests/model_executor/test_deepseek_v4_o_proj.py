# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


@pytest.mark.parametrize(
    ("capability", "expected_recipe", "expected_tma_aligned"),
    [
        ((9, 0), (1, 128, 128), False),
        ((10, 0), (1, 1, 128), True),
        ((12, 0), (1, 128, 128), False),
        ((12, 1), (1, 128, 128), False),
    ],
)
def test_deepseek_v4_o_proj_recipe_is_arch_specific(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected_recipe: tuple[int, int, int],
    expected_tma_aligned: bool,
):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(*capability),
    )

    assert compute_fp8_einsum_recipe() == (expected_recipe, expected_tma_aligned)
