# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for ``FusedMoE.load_routed_expert_weights``.

These tests spawn real ``FusedMoE`` instances across multiple ranks using
``distributed_run`` (mirrors ``test_eplb_fused_moe_layer.py``) and verify that
EP-sharded ``[local_num_routed_experts, ...]`` tensors are bit-exact written
into the correct local slots — exercising the new sharded-load contract added
for train-infer weight sync.
"""

from __future__ import annotations

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from .eplb_utils import distributed_run, set_env_vars_and_device

# ---------------------------------------------------------------------------
# Fixtures: build a FusedMoE with EP enabled and a minimal expert_mapping
# ---------------------------------------------------------------------------


_EXPERT_MAPPING = [
    # (param_name, weight_name, expert_id_or_shard_idx, shard_id)
    # Use FusedMoE's flat attribute layout (w13_weight / w2_weight). Real
    # production mappings usually include an "experts." prefix because the
    # model wraps FusedMoE inside an ``experts`` submodule, but here we test
    # FusedMoE directly so the param names are flat.
    # For 3D fused w13 input, expert_id slot is the shard_idx (0=w1, 1=w3).
    ("w13_weight", "gate_up_proj", 0, "w1"),
    ("w13_weight", "gate_up_proj", 1, "w3"),
    ("w2_weight", "down_proj", 0, "w2"),
]


def _make_fused_moe(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    prefix: str = "sharded_test",
) -> FusedMoE:
    fml = FusedMoE(
        num_experts=num_experts,
        top_k=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        prefix=prefix,
        activation="silu",
        is_act_and_mul=True,
        params_dtype=torch.bfloat16,
        expert_mapping=_EXPERT_MAPPING,
    )
    # Zero the existing params so bit-compare below distinguishes written
    # from untouched experts.
    fml.w13_weight.data.zero_()
    fml.w2_weight.data.zero_()
    return fml


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------


def _contiguous_partition_worker(env, world_size: int):
    """ep_size=world_size, contiguous partition. Each rank holds
    ``global // ep_size`` experts in a contiguous block.

    Send a ``[local_routed_E, 2I, H]`` fused tensor for the local slice only
    and assert the right local slots match bit-for-bit.
    """
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )

        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        num_experts = 8
        hidden = 32
        inter = 16
        fml = _make_fused_moe(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=inter,
        ).to(device)

        # Sanity: EP must actually be on.
        assert fml.ep_size == world_size
        assert fml.local_num_experts == num_experts // world_size
        local_E = fml.local_num_experts
        assert fml._expert_map is not None
        local_to_global = (fml._expert_map != -1).nonzero().flatten().tolist()
        assert len(local_to_global) == local_E

        # Build a deterministic 3D fused gate_up_proj for only this rank's
        # local experts: shape [local_E, 2*intermediate_per_partition, hidden].
        # Each local expert i is filled with a globally unique pattern based on
        # its global id so we can verify it ended up in the right slot.
        intermediate_per_partition = fml.w13_weight.shape[1] // 2
        w13_source = torch.zeros(
            local_E,
            2 * intermediate_per_partition,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        w2_source = torch.zeros(
            local_E,
            hidden,
            intermediate_per_partition,
            dtype=torch.bfloat16,
            device=device,
        )
        for i, gid in enumerate(local_to_global):
            # Unique, non-zero pattern tied to global id.
            val = float(gid + 1)
            w13_source[i].fill_(val)
            w2_source[i].fill_(-val)

        # Dispatch through load_routed_expert_weights.
        list(
            fml.load_routed_expert_weights(
                [("gate_up_proj", w13_source)],
                {"gate_up_proj": local_to_global},
            )
        )
        list(
            fml.load_routed_expert_weights(
                [("down_proj", w2_source)],
                {"down_proj": local_to_global},
            )
        )

        # Verify: for each local expert i in this rank, the stored gate/up
        # halves should equal the source expert i values.
        for i, gid in enumerate(local_to_global):
            expected_val = float(gid + 1)
            # w13 stores [gate | up] concatenated along dim 1 after chunk(2) split
            # -- our sender packed a [2I, H] slice that was split into two halves
            # by load_routed_expert_weights's chunk(2, dim=1) + shard_id handling.
            # Each half of size [I, H] should be filled with `expected_val`.
            assert torch.all(fml.w13_weight.data[i] == expected_val), (
                f"rank={torch.distributed.get_rank()} gid={gid} local={i} w13 "
                f"mismatch, got unique values "
                f"{fml.w13_weight.data[i].unique().tolist()[:5]}"
            )
            assert torch.all(fml.w2_weight.data[i] == -expected_val), (
                f"rank={torch.distributed.get_rank()} gid={gid} local={i} w2 mismatch"
            )


def _shape_mismatch_worker(env, world_size: int):
    """Validator: sender-side (caller) mismatch must raise ValueError."""
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )

        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        fml = _make_fused_moe(num_experts=8, hidden_size=16, intermediate_size=16).to(
            device
        )
        local_E = fml.local_num_experts
        intermediate_per_partition = fml.w13_weight.shape[1] // 2

        # Off-by-one: pass local_E + 1 slices but local_E ids.
        bad_tensor = torch.zeros(
            local_E + 1,
            2 * intermediate_per_partition,
            16,
            dtype=torch.bfloat16,
            device=device,
        )
        local_to_global = (fml._expert_map != -1).nonzero().flatten().tolist()
        with pytest.raises(ValueError, match="shape mismatch"):
            list(
                fml.load_routed_expert_weights(
                    [("gate_up_proj", bad_tensor)],
                    {"gate_up_proj": local_to_global},
                )
            )

        # Missing key must also raise.
        good_tensor = torch.zeros(
            local_E,
            2 * intermediate_per_partition,
            16,
            dtype=torch.bfloat16,
            device=device,
        )
        with pytest.raises(ValueError, match="missing from expert_ids_map"):
            list(
                fml.load_routed_expert_weights(
                    [("gate_up_proj", good_tensor)],
                    {},
                )
            )


def _regression_full_load_unchanged_worker(env, world_size: int):
    """Regression: ``load_weights`` with a full ``[global_num_experts, ...]``
    tensor must still behave exactly as before."""
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )

        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        num_experts = 8
        hidden = 16
        inter = 16
        fml = _make_fused_moe(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=inter,
        ).to(device)
        local_to_global = (fml._expert_map != -1).nonzero().flatten().tolist()
        intermediate_per_partition = fml.w13_weight.shape[1] // 2

        # Build a full [global_E, 2I, H] tensor, unique value per global id.
        w13_full = torch.zeros(
            num_experts,
            2 * intermediate_per_partition,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        w2_full = torch.zeros(
            num_experts,
            hidden,
            intermediate_per_partition,
            dtype=torch.bfloat16,
            device=device,
        )
        for g in range(num_experts):
            w13_full[g].fill_(float(g + 100))
            w2_full[g].fill_(-float(g + 100))

        list(
            fml.load_weights(
                [
                    ("gate_up_proj", w13_full),
                    ("down_proj", w2_full),
                ]
            )
        )

        # Our local slots should mirror the global slice.
        for i, gid in enumerate(local_to_global):
            expected = float(gid + 100)
            assert torch.all(fml.w13_weight.data[i] == expected)
            assert torch.all(fml.w2_weight.data[i] == -expected)


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need >=2 GPUs to exercise EP-sharded routed load.",
)
def test_contiguous_partition():
    distributed_run(_contiguous_partition_worker, 2)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need >=2 GPUs.",
)
def test_shape_mismatch_raises():
    distributed_run(_shape_mismatch_worker, 2)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need >=2 GPUs.",
)
def test_full_load_regression():
    distributed_run(_regression_full_load_unchanged_worker, 2)
