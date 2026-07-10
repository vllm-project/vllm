# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for ExampleHiddenStatesConnector KV-cache-group logic."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def _full(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )


def _hidden(block_size: int) -> HiddenStateCacheSpec:
    return HiddenStateCacheSpec(
        block_size=block_size, num_kv_heads=6, head_size=2048, dtype=torch.bfloat16
    )


def _config(*specs):
    """Minimal stand-in exposing only ``kv_cache_groups`` (all the helpers read)."""
    return SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"layer.{i}"], kv_cache_spec=spec)
            for i, spec in enumerate(specs)
        ]
    )


# ---- _find_cache_kv_group_id ------------------------------------------------


def test_find_group_id_none_config_returns_zero():
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(None) == 0


def test_find_group_id_single_non_hidden_group_returns_zero():
    # Uniform (dense) model: one group, no HiddenStateCacheSpec -> group 0.
    cfg = _config(_full(16))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 0


def test_find_group_id_locates_hidden_group_when_not_first():
    # Hybrid layout: the hidden-states group is not group 0.
    cfg = _config(_full(528), _hidden(22), _full(528))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 1


def test_find_group_id_locates_hidden_group_last():
    cfg = _config(_full(528), _full(528), _hidden(22))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 2


def test_find_group_id_raises_when_no_hidden_group_and_multiple_groups():
    cfg = _config(_full(16), _full(16))
    with pytest.raises(ValueError, match="Could not uniquely identify"):
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg)


def test_find_group_id_raises_when_multiple_hidden_groups():
    cfg = _config(_hidden(22), _hidden(22))
    with pytest.raises(ValueError, match="Could not uniquely identify"):
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg)


# ---- _get_cache_block_size --------------------------------------------------


def test_get_block_size_reads_hidden_group_spec_not_global():
    # Hidden group keeps block size 22; the global is bumped to 528 for hybrids.
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=528))
    cfg = _config(_full(528), _hidden(22))
    block_size = ExampleHiddenStatesConnector._get_cache_block_size(
        vllm_config, cfg, cache_kv_group_id=1
    )
    assert block_size == 22


def test_get_block_size_falls_back_to_cache_config_when_no_kv_cache_config():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    block_size = ExampleHiddenStatesConnector._get_cache_block_size(
        vllm_config, None, cache_kv_group_id=0
    )
    assert block_size == 16


# ---- MLA-verifier absorption ------------------------------------------------


def test_find_group_id_errors_clearly_when_absorbed_by_mla_swa_verifier():
    # HiddenStateCacheSpec subclasses MLAAttentionSpec, so an MLA + sliding-
    # window MLA verifier absorbs it into the MLA group instead of isolating it.
    dt = torch.bfloat16
    spec = {
        "layers.0.mla": MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=576, dtype=dt
        ),
        "layers.1.swa": SlidingWindowMLASpec(
            block_size=64, num_kv_heads=1, head_size=576, dtype=dt, sliding_window=512
        ),
        "cache_only_layers.61": _hidden(64),
    }
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )
    groups = get_kv_cache_groups(vllm_config, spec)
    assert not any(isinstance(g.kv_cache_spec, HiddenStateCacheSpec) for g in groups)
    cfg = SimpleNamespace(kv_cache_groups=groups)
    with pytest.raises(ValueError, match="MLA verifiers are unsupported"):
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg)
