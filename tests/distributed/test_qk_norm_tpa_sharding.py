# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for resolver-native qk-norm sharding under TPA.

Exercises ``_resolve_qk_norm_sharding`` — the derivation that lets
``MiniMaxText01RMSNormTP`` self-configure its weight shard + variance-reduce
group from the per-layer parallel resolver instead of having the model thread
``weight_shard_world_size`` / ``weight_shard_rank`` / ``reduce_group`` in by
hand. No torch.distributed is needed: the resolver state is module-level.

The key correctness property is that the resolver path reproduces, bit for bit,
the manual computation MiniMax-M2 used before this refactor, across all four
(q_norm / k_norm) x (TPA on / off) combinations.
"""

import pytest

from vllm.distributed.layer_parallel_config import (
    clear_layer_parallel_resolver,
    init_layer_parallel_resolver,
)
from vllm.model_executor.custom_op import op_registry
from vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp import (
    MiniMaxText01RMSNormTP,
    _resolve_qk_norm_sharding,
)

Q_PREFIX = "model.layers.0.self_attn.q_norm"
K_PREFIX = "model.layers.0.self_attn.k_norm"
# A non-attention prefix (e.g. a Mamba/linear-attention output norm) must never
# pick up the attention-TP policy, even when TPA is active.
LINEAR_ATTN_PREFIX = "model.layers.0.linear_attn.norm"


def test_class_still_registered_as_custom_op():
    """Guard: the module-level helper must not sit between the
    ``@CustomOp.register`` decorator and the class (which would leave the class
    unregistered and crash model init with a missing ``name`` attribute)."""
    assert MiniMaxText01RMSNormTP.name == "minimax_text01_rmsnorm_tp"
    assert op_registry["minimax_text01_rmsnorm_tp"] is MiniMaxText01RMSNormTP


@pytest.fixture(autouse=True)
def _reset_resolver():
    clear_layer_parallel_resolver()
    yield
    clear_layer_parallel_resolver()


# ---------------------------------------------------------------- #
# Fallback: no resolver / TPA disabled -> global TP world
# ---------------------------------------------------------------- #


def test_no_resolver_falls_back_to_global_tp():
    # Resolver uninitialized: any prefix resolves to the passed global TP world.
    assert _resolve_qk_norm_sharding(Q_PREFIX, 1, 8, 3) == (8, 3, False)


def test_tpa_inactive_is_global_tp():
    # attn_tp == full_tp -> resolver returns no policy -> global TP, no attn group.
    init_layer_parallel_resolver(
        full_tp_size=8, full_tp_rank=3, attn_tp_size=8, attn_tp_rank=3
    )
    assert _resolve_qk_norm_sharding(Q_PREFIX, 1, 8, 3) == (8, 3, False)


def test_tpa_inactive_replicated_k_norm_shards_by_kv_heads():
    # No TPA, but KV heads (2) < full TP (8) -> replicas = 4. The k_norm weight
    # still shards by kv-head count (8 // 4 = 2), rank folds by the replica factor.
    init_layer_parallel_resolver(
        full_tp_size=8, full_tp_rank=5, attn_tp_size=8, attn_tp_rank=5
    )
    world, rank, tpa = _resolve_qk_norm_sharding(K_PREFIX, 4, 8, 5)
    assert (world, rank, tpa) == (2, 1, False)  # 8//4=2, 5//4=1


# ---------------------------------------------------------------- #
# TPA active: shard/reduce over the attention-TP subgroup
# ---------------------------------------------------------------- #


def test_tpa_active_q_norm():
    # full_tp=16, attn_tp=4. q_norm has no KV replication (replicas=1).
    # Under a policy the resolver supplies attn_tp_rank (1), overriding the
    # passed global rank (9).
    init_layer_parallel_resolver(
        full_tp_size=16, full_tp_rank=9, attn_tp_size=4, attn_tp_rank=1
    )
    assert _resolve_qk_norm_sharding(Q_PREFIX, 1, 16, 9) == (4, 1, True)


def test_tpa_active_k_norm_partitioned():
    # attn_tp=4, kv_heads=4 -> replicas=1: k_norm partitions like q_norm.
    init_layer_parallel_resolver(
        full_tp_size=16, full_tp_rank=9, attn_tp_size=4, attn_tp_rank=2
    )
    assert _resolve_qk_norm_sharding(K_PREFIX, 1, 16, 9) == (4, 2, True)


def test_tpa_active_k_norm_replicated():
    # attn_tp=4, kv_heads=2 -> replicas=2: k_norm shards by kv-head count (2),
    # rank folds (3 // 2 = 1).
    init_layer_parallel_resolver(
        full_tp_size=16, full_tp_rank=9, attn_tp_size=4, attn_tp_rank=3
    )
    assert _resolve_qk_norm_sharding(K_PREFIX, 2, 16, 9) == (2, 1, True)


def test_non_attention_prefix_ignored_under_tpa():
    # TPA active, but a non-attention prefix must resolve to the global TP world
    # and must NOT signal the attention-TP reduce group.
    init_layer_parallel_resolver(
        full_tp_size=16, full_tp_rank=9, attn_tp_size=4, attn_tp_rank=1
    )
    assert _resolve_qk_norm_sharding(LINEAR_ATTN_PREFIX, 1, 16, 9) == (16, 9, False)


# ---------------------------------------------------------------- #
# Parity with the pre-refactor manual computation (the merge-safety net)
# ---------------------------------------------------------------- #


@pytest.mark.parametrize("full_tp,attn_tp", [(16, 4), (8, 2), (8, 4), (4, 1)])
def test_matches_legacy_manual_computation(full_tp, attn_tp):
    """Resolver path == MiniMax-M2's old explicit weight_shard_* math."""
    for attn_tp_rank in range(attn_tp):
        init_layer_parallel_resolver(
            full_tp_size=full_tp,
            full_tp_rank=attn_tp_rank,  # exact full rank is irrelevant to the math
            attn_tp_size=attn_tp,
            attn_tp_rank=attn_tp_rank,
        )

        # q_norm — legacy: weight_shard_world_size=attn_tp, rank=attn_tp_rank
        assert _resolve_qk_norm_sharding(Q_PREFIX, 1, full_tp, attn_tp_rank)[:2] == (
            attn_tp,
            attn_tp_rank,
        )

        # replicated k_norm with 2 KV heads — legacy: replicas = attn_tp // 2,
        # (weight_shard_world_size=2, weight_shard_rank=attn_tp_rank // replicas)
        if attn_tp >= 2:
            replicas = attn_tp // 2
            legacy_world = 2
            legacy_rank = attn_tp_rank // replicas
            assert _resolve_qk_norm_sharding(K_PREFIX, replicas, full_tp, attn_tp_rank)[
                :2
            ] == (legacy_world, legacy_rank)
        clear_layer_parallel_resolver()
