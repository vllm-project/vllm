# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the install-time MLA-cache contract (guards.check_cache_contract) and
the LiveConfig defaults."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.deepseek_v32.amd.mono import guards as G
from vllm.models.deepseek_v32.amd.mono.live import LiveConfig


def model(kvs):
    return NS(
        model=NS(layers={L: NS(self_attn=NS(kv_cache=kv)) for L, kv in kvs.items()})
    )


def test_cache_contract():
    cfg = NS(layers=[3, 4, 5])
    vc = NS(cache_config=NS(block_size=16))
    good = {L: torch.zeros(4, 16, 576, dtype=torch.bfloat16) for L in cfg.layers}
    assert G.check_cache_contract(model(good), cfg, vc).startswith("ok")
    bad_caches = {
        "dtype": torch.zeros(4, 16, 576, dtype=torch.float16),
        "not contiguous": torch.zeros(4, 576, 16, dtype=torch.bfloat16).transpose(1, 2),
        "row width": torch.zeros(4, 16, 512, dtype=torch.bfloat16),
        "share one MLA cache": good[3],
    }
    for frag, kv in bad_caches.items():
        bad = dict(good)
        bad[5] = kv
        with pytest.raises(RuntimeError, match=frag):
            G.check_cache_contract(model(bad), cfg, vc)
    # the block size is checked against the metadata at the first step, not here
    other_block = NS(cache_config=NS(block_size=64))
    assert G.check_cache_contract(model(good), cfg, other_block).startswith("ok")
    # size-only stand-ins (test_guards) are skipped
    fake = {L: NS(numel=lambda: 1) for L in cfg.layers}
    assert G.check_cache_contract(model(fake), cfg, vc).startswith("skipped")


def test_live_config_defaults():
    """The validated optimizations are on; the debug / safety extras are off."""
    c = LiveConfig(ckpt="x")
    assert c.early_cache_checks and c.indexer_trim and c.poll_early_out
    assert c.attention_weight == "fp8_block128" and c.indexer_mode == "attn"
    assert c.fused_indexer is None  # AUTO: on when supported
    assert c.fused_index_rowpar_cache and c.fused_index_batched_score
    assert c.fused_select_radix11 and c.fused_index_proj_spread and c.index_q_fp8
    assert c.cache_hoist and c.split_keys64
    assert not c.device_nonfinite and not c.failstop_nonfinite
    assert c.step_sync and c.check_every == 1 and c.enabled
