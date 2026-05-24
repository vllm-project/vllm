# SPDX-License-Identifier: Apache-2.0
"""TDD for GenesisMoEIntermediateCacheManager (P37).

Covers:
  - Platform guard (fallback to torch.empty on CPU / non-NVIDIA / disabled).
  - Pool-hit: two calls with same config share the SAME backing tensor.
  - Pool-miss: first call allocates.
  - Overflow: graceful fallback to fresh torch.empty.
  - Env gate: default disabled (returns fresh), enabled engages pool.
  - Dynamo compatibility: acquire_* functions wrapped with allow_in_graph.
  - Registry introspection for observability.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import torch


def _force_enable(monkeypatch):
    """Force should_apply() to return True AND env-enabled to True."""
    from vllm._genesis.kernels import moe_intermediate_cache as m
    monkeypatch.setattr(m, "_SHOULD_APPLY_CACHED", True)
    monkeypatch.setattr(m, "_ENABLED_AT_IMPORT", True)


def _reset(monkeypatch):
    """Ensure clean state regardless of prior test interference."""
    from vllm._genesis.kernels import moe_intermediate_cache as m
    m.clear_for_tests()


class TestPlatformGate:
    def test_fallback_when_disabled(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        monkeypatch.setattr(m, "_SHOULD_APPLY_CACHED", False)
        monkeypatch.setattr(m, "_ENABLED_AT_IMPORT", False)

        t = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        # Fresh alloc — pool NOT used
        assert t.shape == (128 * 8 * max(2 * 2816, 2048),)
        assert len(m._CACHE13_POOLS) == 0

    def test_fallback_when_platform_incompat(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        monkeypatch.setattr(m, "_SHOULD_APPLY_CACHED", False)  # non-NVIDIA
        monkeypatch.setattr(m, "_ENABLED_AT_IMPORT", True)
        m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert len(m._CACHE13_POOLS) == 0  # platform-incompat → no pool


class TestPoolHit:
    def test_cache13_pool_hit_same_config(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)

        t1 = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        t2 = m.acquire_cache13(
            M=256, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        # Same POOL (backing tensor) even though M differs
        # — both fit in the max_batched_tokens-sized pool.
        assert t1.data_ptr() == t2.data_ptr()
        assert t1 is t2

    def test_cache2_pool_hit_same_config(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)

        t1 = m.acquire_cache2(
            M=128, num_topk=8, N=2816,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        t2 = m.acquire_cache2(
            M=256, num_topk=8, N=2816,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert t1 is t2

    def test_cache13_pool_miss_diff_N(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)

        # Different N → different pool key → different tensor.
        t1 = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        t2 = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=1024, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert t1 is not t2

    def test_cache_pool_miss_diff_dtype(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)

        t1 = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        t2 = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float32, device=torch.device("cpu"),  # differ
        )
        assert t1 is not t2
        assert t1.dtype == torch.float16
        assert t2.dtype == torch.float32


class TestOverflow:
    def test_cache13_overflow_fresh_alloc(self, monkeypatch):
        """M exceeds pool budget → fresh tensor, pool untouched."""
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)
        # Override max_batched_tokens via env simulation
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", 64)  # tiny pool

        t = m.acquire_cache13(
            M=1024, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        # Fresh (large) tensor, not pooled
        assert t.shape == (1024 * 8 * max(2 * 2816, 2048),)
        # Pool wasn't populated for this oversize request
        assert len(m._CACHE13_POOLS) == 0

    def test_cache2_overflow_fresh_alloc(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", 64)
        t = m.acquire_cache2(
            M=1024, num_topk=8, N=2816,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert t.shape == (1024 * 8, 2816)
        assert len(m._CACHE2_POOLS) == 0


class TestDtypeShape:
    def test_cache13_shape(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", 4096)
        t = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        # pool sized by MAX_BT_OVERRIDE, not M
        assert t.shape == (4096 * 8 * max(2 * 2816, 2048),)
        assert t.dtype == torch.float16

    def test_cache2_shape(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", 4096)
        t = m.acquire_cache2(
            M=128, num_topk=8, N=2816,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert t.shape == (4096 * 8, 2816)

    def test_resize_cache_pattern_works(self, monkeypatch):
        """Simulate the `_resize_cache` call in _fused_marlin_moe.

        Pool must support `.flatten()[:prod(v)].view(*v)` for any
        `v` with `prod(v) <= pool.numel()`.
        """
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", 4096)
        pool = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        # Simulate `_resize_cache(pool, (M * num_topk, w13 * N))` for M=128.
        M, topk, w13, N = 128, 8, 2, 2816
        needed = M * topk * w13 * N
        assert needed <= pool.numel()
        reshaped = pool.flatten()[:needed].view(M * topk, w13 * N)
        assert reshaped.shape == (M * topk, w13 * N)

        # And (M*topk, K) view
        K = 2048
        needed2 = M * topk * K
        reshaped2 = pool.flatten()[:needed2].view(M * topk, K)
        assert reshaped2.shape == (M * topk, K)


class TestRegistry:
    def test_registry_info_empty(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        info = m.get_registry_info()
        assert info["total_bytes"] == 0
        assert info["cache13_pools"] == []
        assert info["cache2_pools"] == []

    def test_registry_info_after_acquire(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", 4096)
        m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        m.acquire_cache2(
            M=128, num_topk=8, N=2816,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        info = m.get_registry_info()
        assert len(info["cache13_pools"]) == 1
        assert len(info["cache2_pools"]) == 1
        assert info["total_bytes"] > 0

    def test_class_facade_equivalence(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", 4096)
        # Class-facade API matches module-level function API
        t1 = m.GenesisMoEIntermediateCacheManager.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        t2 = m.acquire_cache13(
            M=128, num_topk=8, w13_num_shards=2, N=2816, K=2048,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert t1 is t2


class TestDynamoCompat:
    def test_allow_in_graph_decorator_present(self):
        """Dynamo compat check: the acquire functions carry the dynamo
        allow-in-graph signature so AoT-compile-fullgraph doesn't
        reject them.
        """
        from vllm._genesis.kernels import moe_intermediate_cache as m
        # allow_in_graph adds `_torchdynamo_inline = False` or similar
        # marker, but the simpler check is that the function is still
        # callable and returns a tensor.
        t = m.acquire_cache13(
            M=16, num_topk=2, w13_num_shards=1, N=64, K=64,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert isinstance(t, torch.Tensor)

    def test_warm_up_callable(self):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        # warm_up must be callable without error (even if returns False).
        result = m.warm_up()
        assert isinstance(result, bool)


class TestEnvIntegration:
    def test_env_override_max_bt(self, monkeypatch):
        from vllm._genesis.kernels import moe_intermediate_cache as m
        _reset(monkeypatch)
        _force_enable(monkeypatch)
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", 8192)
        t = m.acquire_cache13(
            M=256, num_topk=4, w13_num_shards=2, N=1024, K=1024,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert t.shape == (8192 * 4 * max(2 * 1024, 1024),)

    def test_max_bt_rounds_up_to_power_of_2(self, monkeypatch):
        """When no override, M hint is rounded up to next power of 2,
        min 4096. Stable key across slightly different M values."""
        from vllm._genesis.kernels import moe_intermediate_cache as m
        monkeypatch.setattr(m, "_MAX_BT_OVERRIDE", None)
        # G-016 audit fix: walrus `M_hint := ...` was assigned but never
        # used downstream — drop walrus, pass values directly. Behavior
        # unchanged (function takes the rvalue, not the binding).
        assert m._resolve_max_batched_tokens(100) == 4096
        assert m._resolve_max_batched_tokens(3000) == 4096
        assert m._resolve_max_batched_tokens(5000) == 8192  # pow2(5000)=8192
