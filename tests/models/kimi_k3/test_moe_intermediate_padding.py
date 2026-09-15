# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE intermediate padding of the Kimi K3 model.

The intermediate is padded so that every MoE shard is at least
``min_moe_intermediate_per_partition`` wide. The number of shards is not the
tensor-parallel world size: ``FusedMoEParallelConfig.make()`` shards the
experts over ``dp * pcp * tp`` devices, and with expert parallelism it does not
shard the intermediate at all, so padding it there would allocate the padding
in full on every rank.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import ParallelConfig
from vllm.model_executor.layers.fused_moe import config as moe_config
from vllm.models.kimi_k3.common.moe_padding import (
    effective_moe_tp_size,
    padded_moe_intermediate_size,
)
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.platforms import current_platform

# Kimi K3: moe_intermediate_size=3072, min_moe_intermediate_per_partition=256.
MOE_INTERMEDIATE_SIZE = 3072
MIN_PER_PARTITION = 256


@pytest.mark.parametrize(
    ("moe_tp_size", "expected"),
    [
        (1, 3072),  # not sharded
        (8, 3072),  # 3072 // 8 == 384, already >= 256
        (16, 4096),  # 3072 // 16 == 192 -> pad to 256 * 16
        (32, 8192),  # 3072 // 32 == 96 -> pad to 256 * 32
    ],
)
def test_pads_narrow_shards(moe_tp_size: int, expected: int) -> None:
    assert (
        padded_moe_intermediate_size(
            MOE_INTERMEDIATE_SIZE, moe_tp_size, MIN_PER_PARTITION
        )
        == expected
    )


@pytest.mark.parametrize(
    ("tp_size", "dp_size", "enable_expert_parallel", "expected"),
    [
        (16, 1, False, 16),  # plain tensor parallel
        (8, 4, False, 32),  # data parallel also shards the experts
        (16, 4, True, 1),  # expert parallel: whole experts per rank
        (1, 1, True, 1),  # single device: no sharding either way
    ],
)
def test_effective_moe_tp_size(
    tp_size: int, dp_size: int, enable_expert_parallel: bool, expected: int
) -> None:
    vllm_config = SimpleNamespace(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            enable_expert_parallel=enable_expert_parallel,
        )
    )
    assert effective_moe_tp_size(vllm_config) == expected


def test_expert_parallel_makes_moe_tp_size_one(monkeypatch) -> None:
    """The core invariant ``effective_moe_tp_size`` mirrors.

    ``FusedMoEParallelConfig.make()`` collapses the MoE tensor parallel size to
    1 under expert parallelism, so ``intermediate_size_per_partition`` is the
    whole (padded) intermediate on every rank.
    """
    monkeypatch.setattr(current_platform, "device_count", lambda: 2)
    monkeypatch.setattr(moe_config, "get_tensor_model_parallel_rank", lambda: 0)
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )

    moe_parallel_config = moe_config.FusedMoEParallelConfig.make(
        tp_size_=2,
        pcp_size_=1,
        dp_size_=1,
        sp_size_=1,
        vllm_parallel_config=parallel_config,
    )

    assert moe_parallel_config.tp_size == 1
    assert moe_parallel_config.ep_size == 2


class _Stub(nn.Module):
    """Stands in for a layer whose construction needs a real device."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.e_score_correction_bias = None
        self.moe_config = SimpleNamespace(intermediate_size_per_partition_unpadded=None)


def build_moe(monkeypatch, *, tp_size: int, enable_expert_parallel: bool):
    """Construct a ``KimiMoE`` on stubbed layers and report its padding."""
    for name in (
        "GateLinear",
        "ReplicatedLinear",
        "RMSNorm",
        "KimiRoutedOutputTransform",
        "KimiMLP",
        "FusedMoEFactory",
    ):
        monkeypatch.setattr(kimi_model, name, _Stub)
    monkeypatch.setattr(kimi_model, "aux_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", _Stub)
    monkeypatch.setattr(
        kimi_model, "get_tensor_model_parallel_world_size", lambda: tp_size
    )

    config = SimpleNamespace(
        hidden_size=16,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        min_moe_intermediate_per_partition=MIN_PER_PARTITION,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_renormalize=True,
        routed_expert_hidden_size=16,
        latent_moe_use_norm=False,
        routed_scaling_factor=1.0,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
        hidden_act="silu",
        activation_situ_beta=1.0,
        activation_situ_linear_beta=None,
        rms_norm_eps=1e-5,
    )
    return kimi_model.KimiMoE(
        config=config,
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(moe_backend="auto"),
            parallel_config=ParallelConfig(
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
            ),
        ),
        prefix="mlp",
    )


def test_expert_parallel_leaves_the_intermediate_unpadded(monkeypatch) -> None:
    """Under EP the experts are not intermediate-sharded, so 3072 is enough.

    Padding to ``256 * tp_size`` here would allocate 4096 (TP16) on every rank
    instead of 3072, in full, because the MoE tensor-parallel size is 1.
    """
    monkeypatch.setattr(current_platform, "device_count", lambda: 16)

    moe = build_moe(monkeypatch, tp_size=16, enable_expert_parallel=True)

    assert moe.moe_tp_size == 1
    assert moe.padded_moe_intermediate_size == MOE_INTERMEDIATE_SIZE


def test_tensor_parallel_still_pads_narrow_shards(monkeypatch) -> None:
    """Without EP the experts really are sharded by TP, so padding applies."""
    monkeypatch.setattr(current_platform, "device_count", lambda: 16)

    moe = build_moe(monkeypatch, tp_size=16, enable_expert_parallel=False)

    assert moe.moe_tp_size == 16
    assert moe.padded_moe_intermediate_size == MIN_PER_PARTITION * 16
