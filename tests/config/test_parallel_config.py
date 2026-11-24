# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.config.parallel as parallel_module
from vllm.config import ParallelConfig
from vllm.platforms.interface import Platform, PlatformEnum


class _DefaultOOTPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "cpu"
    device_type = "cpu"


class _SingleDPSequenceParallelPlatform(_DefaultOOTPlatform):
    @classmethod
    def use_sequence_parallel_moe_without_dp(cls) -> bool:
        return True


@pytest.mark.parametrize(
    (
        "platform_cls",
        "tensor_parallel_size",
        "data_parallel_size",
        "enable_expert_parallel",
        "all2all_backend",
        "expected",
    ),
    [
        (_DefaultOOTPlatform, 2, 1, True, "allgather_reducescatter", False),
        (_DefaultOOTPlatform, 2, 2, True, "allgather_reducescatter", True),
        (
            _SingleDPSequenceParallelPlatform,
            2,
            1,
            True,
            "allgather_reducescatter",
            True,
        ),
        (
            _SingleDPSequenceParallelPlatform,
            1,
            1,
            True,
            "allgather_reducescatter",
            False,
        ),
        (
            _SingleDPSequenceParallelPlatform,
            2,
            1,
            False,
            "allgather_reducescatter",
            False,
        ),
        (
            _SingleDPSequenceParallelPlatform,
            2,
            1,
            True,
            "flashinfer_nvlink_two_sided",
            False,
        ),
    ],
)
def test_use_sequence_parallel_moe_platform_dp1_hook(
    monkeypatch: pytest.MonkeyPatch,
    platform_cls: type[Platform],
    tensor_parallel_size: int,
    data_parallel_size: int,
    enable_expert_parallel: bool,
    all2all_backend: str,
    expected: bool,
) -> None:
    """Platforms whose MoE comm shards tokens across TP at DP=1 can opt into SP MoE."""
    monkeypatch.setattr(parallel_module, "current_platform", platform_cls())

    parallel_config = ParallelConfig(
        distributed_executor_backend="mp",
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        all2all_backend=all2all_backend,
    )

    assert parallel_config.use_sequence_parallel_moe is expected
    if data_parallel_size == 1:
        assert parallel_config.use_all2all is expected


def test_default_platform_disables_sequence_parallel_moe_without_dp() -> None:
    assert _DefaultOOTPlatform.use_sequence_parallel_moe_without_dp() is False
