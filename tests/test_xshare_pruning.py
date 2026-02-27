# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE batch-aware expert pruning (XShare)."""

import pytest
import torch

from vllm.config.moe import MoEConfig, MoEPruningConfig
from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    _fused_prune_experts,
    _get_moe_pruning_config,
    _prune_experts,
    set_moe_config,
)


def _cfg(**kwargs):
    defaults = dict(
        enable=True,
        expert_budget=None,
        budget_alpha=0.5,
        top_per_token=0,
        group_budget=0,
        group_size=0,
        min_batch=0,
        max_batch=0,
    )
    defaults.update(kwargs)
    return MoEConfig(pruning=MoEPruningConfig(**defaults))


@pytest.fixture(autouse=True)
def _reset_config():
    set_moe_config(None)
    yield
    set_moe_config(None)


MASK_VAL_F32 = torch.finfo(torch.float32).min


def _unmasked(result):
    return (result > torch.finfo(result.dtype).min).sum(dim=1)


def _unmasked_cols(result):
    return (result > torch.finfo(result.dtype).min).any(dim=0).sum().item()


# ── Disabled / no-op ──


@pytest.mark.parametrize(
    "config,label",
    [
        (None, "no config"),
        (MoEConfig(pruning=MoEPruningConfig(enable=False)), "disabled"),
        (
            MoEConfig(
                pruning=MoEPruningConfig(
                    enable=True,
                    expert_budget=None,
                    budget_alpha=0.0,
                    top_per_token=0,
                    group_budget=0,
                )
            ),
            "no knobs",
        ),
    ],
)
def test_pruning_disabled(config, label):
    set_moe_config(config)
    gating = torch.randn(8, 128)
    assert torch.equal(_prune_experts(gating), gating), label


# ── Expert budget (greedy total) ──


def test_greedy_total_masks_experts():
    set_moe_config(_cfg(expert_budget=4))
    result = _prune_experts(torch.randn(4, 8))
    assert (_unmasked(result) == 4).all()


def test_greedy_total_preserves_top():
    set_moe_config(_cfg(expert_budget=2))
    gating = torch.zeros(4, 8)
    gating[:, 0], gating[:, 1] = 10.0, 8.0
    gating[:, 2] = 1.0
    result = _prune_experts(gating)
    assert (result[:, 0] == 10.0).all()
    assert (result[:, 1] == 8.0).all()
    assert (result[:, 2] == MASK_VAL_F32).all()


@pytest.mark.parametrize("budget", [8, 200])
def test_greedy_total_ge_num_experts(budget):
    set_moe_config(_cfg(expert_budget=budget))
    gating = torch.randn(4, 8)
    assert torch.equal(_prune_experts(gating), gating)


def test_batch_size_one():
    set_moe_config(_cfg(expert_budget=4))
    assert _unmasked(_prune_experts(torch.randn(1, 8))).item() == 4


# ── Top per token ──


def test_top_per_token_preserves_best():
    set_moe_config(_cfg(top_per_token=1))
    gating = torch.tensor(
        [
            [1.0, 5.0, 3.0, 2.0],
            [4.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 6.0, 1.0],
        ]
    )
    result = _prune_experts(gating)
    assert result[0, 1] == 5.0 and result[1, 0] == 4.0 and result[2, 2] == 6.0


def test_top_per_token_with_greedy_total():
    set_moe_config(_cfg(expert_budget=2, top_per_token=1))
    gating = torch.tensor(
        [
            [1.0, 5.0, 3.0, 2.0, 0.1, 0.1, 0.1, 0.1],
            [4.0, 1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    result = _prune_experts(gating)
    assert result[0, 1] == 5.0 and result[1, 0] == 4.0
    assert (result[:, 4:] == MASK_VAL_F32).all()


# ── Per-group (GPR) ──


def test_per_group_routes_correctly():
    set_moe_config(_cfg(group_budget=2, group_size=4))
    gating = torch.zeros(8, 8)
    gating[0:4, 0], gating[0:4, 1] = 5.0, 4.0
    gating[4:8, 6], gating[4:8, 7] = 5.0, 4.0
    result = _prune_experts(gating)
    assert (result[0:4, 0] == 5.0).all() and (result[4:8, 6] == 5.0).all()


def test_remainder_tokens_not_fully_masked():
    set_moe_config(_cfg(group_budget=2, group_size=4, budget_alpha=0.0))
    result = _prune_experts(torch.randn(6, 8))
    assert (_unmasked(result)[4:] > 0).all()


def test_batch_smaller_than_group_returns_original():
    set_moe_config(_cfg(group_budget=2, group_size=4, budget_alpha=0.0))
    gating = torch.randn(1, 8)
    assert torch.equal(_prune_experts(gating), gating)


# ── Batch guards ──


@pytest.mark.parametrize(
    "M,cfg_kwargs",
    [
        (4, dict(expert_budget=4, min_batch=8)),
        (8, dict(expert_budget=4, max_batch=4)),
    ],
)
def test_batch_guard_skips(M, cfg_kwargs):
    set_moe_config(_cfg(**cfg_kwargs))
    gating = torch.randn(M, 16)
    assert torch.equal(_prune_experts(gating), gating)


# ── Topk guarantee ──


def test_topk_guarantee():
    set_moe_config(_cfg(expert_budget=2))
    result = _prune_experts(torch.randn(8, 16), topk=6)
    assert (_unmasked(result) >= 6).all()


def test_topk_zero_no_guarantee():
    set_moe_config(_cfg(expert_budget=2))
    result = _prune_experts(torch.randn(4, 8), topk=0)
    assert (_unmasked(result) == 2).all()


def test_topk_selective_not_all():
    set_moe_config(_cfg(expert_budget=4))
    result = _prune_experts(torch.randn(8, 16), topk=6)
    u = _unmasked(result)
    assert (u >= 6).all() and (u <= 10).all()


# ── Budget alpha ──


def test_budget_alpha():
    set_moe_config(_cfg(budget_alpha=0.25))  # E=16 -> 4
    result = _prune_experts(torch.randn(8, 16), topk=2)
    assert _unmasked_cols(result) <= 4


def test_expert_budget_overrides_alpha():
    set_moe_config(_cfg(expert_budget=8, budget_alpha=0.1))
    result = _prune_experts(torch.randn(8, 16), topk=2)
    cols = _unmasked_cols(result)
    assert 2 < cols <= 8


# ── Non-contiguous input ──


def test_non_contiguous():
    set_moe_config(_cfg(expert_budget=4))
    gating = torch.randn(16, 8)[::2]
    assert not gating.is_contiguous()
    assert (_unmasked(_prune_experts(gating)) == 4).all()


# ── Route-level behavior ──


def test_pruning_reduces_active_experts():
    set_moe_config(_cfg(expert_budget=4))
    torch.manual_seed(42)
    gating = torch.randn(8, 16)
    pruned = _prune_experts(gating, topk=2)
    pruned_experts = set(torch.topk(pruned, 2, dim=-1).indices.flatten().tolist())
    assert len(pruned_experts) <= 4


# ── CUDA tests ──


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_basic():
    set_moe_config(_cfg(expert_budget=4))
    result = _prune_experts(torch.randn(8, 16, device="cuda"), topk=6)
    assert (_unmasked(result) >= 6).all() and result.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_capture():
    set_moe_config(_cfg(expert_budget=4))
    M, E, topk = 8, 16, 6
    _get_moe_pruning_config.cache_clear()
    _get_moe_pruning_config()
    gating = torch.randn(M, E, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _prune_experts(gating, topk=topk)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        result = _prune_experts(gating, topk=topk)
    gating.copy_(torch.randn(M, E, device="cuda"))
    g.replay()
    assert (_unmasked(result) >= topk).all()


# ── Fused path ──


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("budget,topk", [(4, 2), (None, 4)])
def test_fused_matches_python(budget, topk):
    cfg = _cfg(expert_budget=budget, budget_alpha=0.25 if budget is None else 0.5)
    set_moe_config(cfg)
    gating = torch.randn(16, 32, device="cuda")
    python_result = _prune_experts(gating.clone(), topk=topk)
    fused_result = _fused_prune_experts(gating.clone(), topk=topk)
    assert torch.allclose(python_result, fused_result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        dict(expert_budget=4, top_per_token=1),
        dict(expert_budget=4, group_budget=2, group_size=4),
    ],
)
def test_fused_fallback(cfg_kwargs):
    set_moe_config(_cfg(**cfg_kwargs))
    result = _fused_prune_experts(torch.randn(8, 16, device="cuda"), topk=2)
    assert (_unmasked(result) >= 2).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_disabled():
    gating = torch.randn(8, 16, device="cuda")
    assert torch.equal(_fused_prune_experts(gating), gating)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_cuda_graph():
    set_moe_config(_cfg(expert_budget=4))
    M, E, topk = 8, 16, 2
    _get_moe_pruning_config.cache_clear()
    _get_moe_pruning_config()
    gating = torch.randn(M, E, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _fused_prune_experts(gating, topk=topk)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        result = _fused_prune_experts(gating, topk=topk)
    gating.copy_(torch.randn(M, E, device="cuda"))
    g.replay()
    assert (_unmasked(result) >= topk).all()
    assert _unmasked_cols(result) <= 4


# --- Integration tests ---


def test_create_moe_config_from_dict():
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs.__new__(EngineArgs)
    args.moe_config = {"pruning": {"enable": True, "expert_budget": 24}}
    cfg = args._create_moe_config()
    assert cfg is not None
    assert cfg.pruning.enable is True
    assert cfg.pruning.expert_budget == 24


def test_create_moe_config_none():
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs.__new__(EngineArgs)
    args.moe_config = None
    assert args._create_moe_config() is None


def test_create_moe_config_empty_pruning():
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs.__new__(EngineArgs)
    args.moe_config = {}
    cfg = args._create_moe_config()
    assert cfg is not None
    assert cfg.pruning is None


def test_create_moe_config_invalid_type():
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs.__new__(EngineArgs)
    args.moe_config = [1, 2, 3]
    with pytest.raises((AttributeError, TypeError)):
        args._create_moe_config()


def test_context_manager_restores_config():
    from vllm.config.moe import MoEConfig, MoEPruningConfig

    cfg_with = MoEConfig(pruning=MoEPruningConfig(enable=True, expert_budget=4))
    set_moe_config(cfg_with)
    _get_moe_pruning_config.cache_clear()
    assert _get_moe_pruning_config() is not None

    set_moe_config(None)
    _get_moe_pruning_config.cache_clear()
    assert _get_moe_pruning_config() is None

    gating = torch.randn(8, 16)
    assert torch.equal(_prune_experts(gating), gating)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "expert_budget,topk,E",
    [
        (4, 2, 16),
        (8, 4, 32),
        (2, 6, 16),
        (24, 4, 128),
    ],
)
def test_fused_matches_python_extended(expert_budget, topk, E):
    set_moe_config(_cfg(expert_budget=expert_budget))
    torch.manual_seed(42)
    gating = torch.randn(16, E, device="cuda")
    fused = _fused_prune_experts(gating, topk=topk)
    python = _prune_experts(gating, topk=topk)
    assert torch.allclose(fused, python)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_with_ties():
    set_moe_config(_cfg(expert_budget=4))
    gating = torch.ones(8, 16, device="cuda", dtype=torch.bfloat16)
    gating[:, :4] = 10.0
    result = _fused_prune_experts(gating, topk=2)
    mask_val = torch.finfo(gating.dtype).min
    assert (result[:, :4] > mask_val).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_dtype(dtype):
    set_moe_config(_cfg(expert_budget=4))
    gating = torch.randn(8, 16, device="cuda", dtype=dtype)
    result = _fused_prune_experts(gating, topk=2)
    assert _unmasked_cols(result) <= 4
