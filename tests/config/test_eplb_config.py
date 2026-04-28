# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.model import (
    ModelConfig,
    get_default_eplb_num_redundant_experts,
)
from vllm.config.parallel import EPLBConfig, ParallelConfig


@pytest.mark.parametrize(
    ("num_logical_experts", "ep_size", "expected"),
    [
        (8, 2, 0),
        (10, 3, 2),
        (15, 4, 1),
        (0, 4, 0),
    ],
)
def test_default_eplb_num_redundant_experts_formula(
    num_logical_experts: int,
    ep_size: int,
    expected: int,
) -> None:
    assert (
        get_default_eplb_num_redundant_experts(num_logical_experts, ep_size)
        == expected
    )


def test_eplb_config_default_is_unspecified() -> None:
    assert EPLBConfig().num_redundant_experts is None


def test_parallel_config_normalizes_unspecified_redundant_experts_without_eplb() -> None:
    parallel_config = ParallelConfig(eplb_config=EPLBConfig())
    assert parallel_config.eplb_config.num_redundant_experts == 0


def test_parallel_config_rejects_redundant_experts_without_eplb() -> None:
    with pytest.raises(ValueError, match="EPLB is not enabled"):
        ParallelConfig(eplb_config=EPLBConfig(num_redundant_experts=1))


def test_model_config_defaults_eplb_num_redundant_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.config.parallel.current_platform.is_cuda_alike",
        lambda: True,
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=3,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(),
    )
    model_config = SimpleNamespace(
        model_arch_config=SimpleNamespace(
            total_num_attention_heads=12,
            num_experts=10,
        ),
        registry=SimpleNamespace(
            is_pp_supported_model=lambda _architectures, _config: True
        ),
        architectures=[],
        use_mla=True,
        multimodal_config=None,
        _verify_with_expert_parallelism=lambda: None,
        get_num_experts=lambda: 10,
    )

    ModelConfig.verify_with_parallel_config(model_config, parallel_config)

    assert parallel_config.eplb_config.num_redundant_experts == 2


def test_model_config_preserves_explicit_eplb_num_redundant_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.config.parallel.current_platform.is_cuda_alike",
        lambda: True,
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=3,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(num_redundant_experts=0),
    )
    model_config = SimpleNamespace(
        model_arch_config=SimpleNamespace(
            total_num_attention_heads=12,
            num_experts=10,
        ),
        registry=SimpleNamespace(
            is_pp_supported_model=lambda _architectures, _config: True
        ),
        architectures=[],
        use_mla=True,
        multimodal_config=None,
        _verify_with_expert_parallelism=lambda: None,
        get_num_experts=lambda: 10,
    )

    ModelConfig.verify_with_parallel_config(model_config, parallel_config)

    assert parallel_config.eplb_config.num_redundant_experts == 0
