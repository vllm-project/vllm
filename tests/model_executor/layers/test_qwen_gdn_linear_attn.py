# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn


def _make_config(requested: str, head_k_dim: int = 128) -> Any:
    return SimpleNamespace(
        additional_config={"gdn_prefill_backend": requested},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=head_k_dim)
        ),
    )


@pytest.mark.parametrize(
    "sm100,sm120,requested,cuda_runtime,head_k_dim,expected",
    [
        (False, True, "auto", 13, 128, "flashinfer"),
        (False, True, "flashinfer", 13, 128, "flashinfer"),
        (False, True, "cutedsl", 13, 128, "triton"),
        (True, False, "cutedsl", 13, 128, "cutedsl"),
        (False, True, "auto", 12, 128, "triton"),
        (False, True, "auto", 13, 64, "triton"),
    ],
)
def test_resolve_gdn_prefill_backend(
    sm100: bool,
    sm120: bool,
    requested: str,
    cuda_runtime: int,
    head_k_dim: int,
    expected: str,
) -> None:
    platform = MagicMock()
    platform.is_cuda.return_value = True
    platform.is_device_capability.return_value = False
    platform.is_device_capability_family.side_effect = {100: sm100, 120: sm120}.get
    platform.get_cuda_runtime_major.return_value = cuda_runtime

    with patch.object(qwen_gdn_linear_attn, "current_platform", platform):
        _, active_backend = qwen_gdn_linear_attn._resolve_gdn_prefill_backend(
            _make_config(requested, head_k_dim)
        )

    assert active_backend == expected
