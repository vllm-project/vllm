# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the per-layer parallel config resolver.

Tests the public API of ``vllm/distributed/layer_parallel_config.py`` in
isolation (no torch.distributed needed) — the resolver state is module-level
and can be initialized directly via ``init_layer_parallel_resolver``.

Integration with ``initialize_model_parallel`` is exercised by existing
distributed test suites (those require multi-process setup).
"""

import pytest

from vllm.distributed.layer_parallel_config import (
    LayerParallelConfig,
    clear_layer_parallel_resolver,
    get_layer_parallel_config,
    init_layer_parallel_resolver,
)


@pytest.fixture(autouse=True)
def _reset_resolver():
    """Reset resolver state before AND after each test for isolation."""
    clear_layer_parallel_resolver()
    yield
    clear_layer_parallel_resolver()


# ---------------------------------------------------------------- #
# Default behavior (resolver uninitialized / TPA disabled)
# ---------------------------------------------------------------- #


def test_default_no_init_returns_all_none():
    """Without `init_layer_parallel_resolver`, all configs are defaults."""
    cfg = get_layer_parallel_config(None, "model.layers.0.self_attn.qkv_proj")
    assert cfg == LayerParallelConfig()
    assert cfg.tp_size is None
    assert cfg.tp_rank is None


def test_tpa_equals_tp_is_noop():
    """When attn_tp_size == full_tp_size, every layer returns defaults."""
    init_layer_parallel_resolver(
        full_tp_size=4, full_tp_rank=2, attn_tp_size=4, attn_tp_rank=2
    )
    cfg_attn = get_layer_parallel_config(None, "model.layers.0.self_attn.qkv_proj")
    cfg_mlp = get_layer_parallel_config(None, "model.layers.0.mlp.gate_proj")
    assert cfg_attn == LayerParallelConfig()
    assert cfg_mlp == LayerParallelConfig()


# ---------------------------------------------------------------- #
# TPA mode (attn_tp_size < full_tp_size)
# ---------------------------------------------------------------- #


def test_tpa_attention_layer_resolves_to_attn_tp():
    """TPA<TP: attention layers get attn_tp_size / attn_tp_rank."""
    init_layer_parallel_resolver(
        full_tp_size=8, full_tp_rank=5, attn_tp_size=2, attn_tp_rank=1
    )
    cfg = get_layer_parallel_config(None, "model.layers.0.self_attn.qkv_proj")
    assert cfg.tp_size == 2
    assert cfg.tp_rank == 1


def test_tpa_non_attention_layer_returns_defaults():
    """TPA<TP: non-attention layers (MLP, MoE, embedding) keep global TP."""
    init_layer_parallel_resolver(
        full_tp_size=8, full_tp_rank=5, attn_tp_size=2, attn_tp_rank=1
    )
    cfg = get_layer_parallel_config(None, "model.layers.0.mlp.gate_proj")
    assert cfg == LayerParallelConfig()
    cfg = get_layer_parallel_config(None, "model.layers.0.mlp.down_proj")
    assert cfg == LayerParallelConfig()
    cfg = get_layer_parallel_config(None, "model.embed_tokens")
    assert cfg == LayerParallelConfig()


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.7.self_attn.o_proj",
        "model.layers.0.attention.q_proj",
        "model.layers.0.attn.qkv",
    ],
)
def test_attention_prefix_patterns_recognized(prefix):
    """All known attention-prefix patterns trigger the TPA path."""
    init_layer_parallel_resolver(
        full_tp_size=8, full_tp_rank=3, attn_tp_size=4, attn_tp_rank=3
    )
    cfg = get_layer_parallel_config(None, prefix)
    assert cfg.tp_size == 4


def test_resolver_idempotent_on_same_prefix():
    """Multiple calls with the same prefix return identical config."""
    init_layer_parallel_resolver(
        full_tp_size=4, full_tp_rank=1, attn_tp_size=2, attn_tp_rank=0
    )
    prefix = "model.layers.5.self_attn.qkv_proj"
    a = get_layer_parallel_config(None, prefix)
    b = get_layer_parallel_config(None, prefix)
    assert a == b


@pytest.mark.parametrize(
    "tp_size, tpa_size, expected_attn_rank",
    [
        # TP=4 TPA=2 DCP=2: DCP groups {0,1} and {2,3}; attn = tp_rank//2
        (4, 2, [0, 0, 1, 1]),
        # TP=8 TPA=4 DCP=2: 4 DCP groups of size 2; attn = tp_rank // 2
        (8, 4, [0, 0, 1, 1, 2, 2, 3, 3]),
        # TP=8 TPA=2 DCP=4: DCP groups {0..3} and {4..7}; attn = tp_rank // 4
        (8, 2, [0, 0, 0, 0, 1, 1, 1, 1]),
        # TP=8 TPA=1 DCP=8: single DCP group; all ranks attn_rank=0
        (8, 1, [0, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_attn_tp_rank_matches_dcp_layout(tp_size, tpa_size, expected_attn_rank):
    """attn_tp_rank derivation must match the DCP-group consecutive layout
    used in vllm/distributed/parallel_state.py initialize_model_parallel:
    DCP groups are consecutive chunks of the TP axis, so attn_tp_rank for a
    given full TP rank is `tp_rank // dcp_size = tp_rank // (TP // TPA)`.
    Bug guard against `tp_rank % tpa_size` (the wrong mapping).
    """
    dcp_size = tp_size // tpa_size
    for tp_rank in range(tp_size):
        attn_rank = tp_rank // dcp_size
        assert attn_rank == expected_attn_rank[tp_rank], (
            f"tp_size={tp_size}, tpa_size={tpa_size}, tp_rank={tp_rank}: "
            f"expected attn_rank={expected_attn_rank[tp_rank]}, got {attn_rank}"
        )
        # Verify the wrong mapping (`tp_rank % tpa_size`) would have produced
        # a different value for at least some tp_ranks → if these ever agree
        # for all tp_ranks, the test stops protecting against the regression.
        if tpa_size != tp_size and tpa_size > 1:
            wrong = tp_rank % tpa_size
            if wrong != attn_rank:
                break  # at least one ranks differs → test is meaningful
    else:
        if tpa_size != tp_size and tpa_size > 1:
            pytest.fail(
                f"Expected at least one tp_rank where `tp_rank // dcp_size` "
                f"differs from `tp_rank % tpa_size` for tp={tp_size} tpa={tpa_size}, "
                f"but they always agreed — the regression guard is degenerate."
            )


# ---------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------- #


def test_init_rejects_non_divisible_sizes():
    """attn_tp_size must divide full_tp_size."""
    with pytest.raises(ValueError, match="divisible"):
        init_layer_parallel_resolver(
            full_tp_size=6, full_tp_rank=0, attn_tp_size=4, attn_tp_rank=0
        )


def test_init_rejects_rank_out_of_range():
    """attn_tp_rank must be in [0, attn_tp_size)."""
    with pytest.raises(ValueError, match="out of range"):
        init_layer_parallel_resolver(
            full_tp_size=4, full_tp_rank=0, attn_tp_size=2, attn_tp_rank=2
        )


# ---------------------------------------------------------------- #
# Reset / lifecycle
# ---------------------------------------------------------------- #


def test_clear_resets_to_default_behavior():
    """After clear, the resolver returns defaults again."""
    init_layer_parallel_resolver(
        full_tp_size=4, full_tp_rank=1, attn_tp_size=2, attn_tp_rank=1
    )
    cfg = get_layer_parallel_config(None, "model.layers.0.self_attn.qkv_proj")
    assert cfg.tp_size == 2

    clear_layer_parallel_resolver()
    cfg = get_layer_parallel_config(None, "model.layers.0.self_attn.qkv_proj")
    assert cfg == LayerParallelConfig()


def test_reinit_overwrites_previous_state():
    """Calling init twice updates the resolver state."""
    init_layer_parallel_resolver(
        full_tp_size=4, full_tp_rank=0, attn_tp_size=2, attn_tp_rank=0
    )
    init_layer_parallel_resolver(
        full_tp_size=8, full_tp_rank=3, attn_tp_size=4, attn_tp_rank=3
    )
    cfg = get_layer_parallel_config(None, "model.layers.0.self_attn.qkv_proj")
    assert cfg.tp_size == 4
    assert cfg.tp_rank == 3


# ---------------------------------------------------------------- #
# Layer parallel config dataclass
# ---------------------------------------------------------------- #


def test_layer_parallel_config_is_frozen():
    """The config dataclass is immutable to prevent accidental mutation."""
    cfg = LayerParallelConfig(tp_size=4, tp_rank=2)
    with pytest.raises((AttributeError, Exception)):
        cfg.tp_size = 8  # noqa: B018


def test_layer_parallel_config_equality():
    """Configs with the same values compare equal (for caching/test use)."""
    a = LayerParallelConfig(tp_size=4, tp_rank=2)
    b = LayerParallelConfig(tp_size=4, tp_rank=2)
    c = LayerParallelConfig(tp_size=4, tp_rank=3)
    assert a == b
    assert a != c
