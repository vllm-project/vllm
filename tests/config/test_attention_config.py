# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.attention import AttentionConfig


def test_kimi_k3_fp8_scale_paths_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        AttentionConfig(
            rocm_kimi_k3_fp8_prefill_scale_path="scales.safetensors",
            rocm_kimi_k3_fp8_prefill_scale_save_path="calibration",
        )


@pytest.mark.parametrize("margin", [float("nan"), float("inf"), 0.99])
def test_kimi_k3_fp8_scale_margin_is_validated(margin: float) -> None:
    with pytest.raises(ValueError, match="must be finite and >= 1"):
        AttentionConfig(rocm_kimi_k3_fp8_prefill_scale_margin=margin)


def test_kimi_k3_fp8_scale_defaults() -> None:
    config = AttentionConfig()
    assert config.rocm_kimi_k3_fp8_prefill_scale_path is None
    assert config.rocm_kimi_k3_fp8_prefill_scale_save_path is None
    assert config.rocm_kimi_k3_fp8_prefill_scale_margin == 1.1
