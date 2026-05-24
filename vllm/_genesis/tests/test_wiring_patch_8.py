# SPDX-License-Identifier: Apache-2.0
"""TDD for P8 — KV hybrid reporting wiring (critical 3.76× KV-cache fix).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest


_KV_BASELINE = """# SPDX-License-Identifier: Apache-2.0
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request


def _report_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    \"\"\"Log resolved KV cache configuration.\"\"\"
    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]
    )

    # Log the KV cache size and maximum concurrency.
    num_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )
    print("done")
"""


_SCHED_BASELINE = """# SPDX-License-Identifier: Apache-2.0
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface


class Scheduler:
    def __init__(self, vllm_config, kv_cache_config):
        self.vllm_config = vllm_config
        if True:
            min_block_size = min(
                [
                    group.kv_cache_spec.block_size
                    for group in kv_cache_config.kv_cache_groups
                ]
            )
            num_groups = len(kv_cache_config.kv_cache_groups)
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // num_groups
            ) * min_block_size
"""


@pytest.fixture
def fake_p8_files(tmp_path, monkeypatch):
    kv_path = tmp_path / "kv_cache_utils.py"
    kv_path.write_text(_KV_BASELINE)

    sched_path = tmp_path / "scheduler.py"
    sched_path.write_text(_SCHED_BASELINE)

    def resolve(rel):
        if "kv_cache_utils" in rel:
            return str(kv_path)
        if "scheduler" in rel:
            return str(sched_path)
        return None

    from vllm._genesis.wiring.legacy import patch_8_kv_hybrid_reporting as p8
    monkeypatch.setattr(p8, "resolve_vllm_file", resolve)
    monkeypatch.setattr(p8, "vllm_install_root", lambda: "/fake")

    # Stub the Issue #5 post-apply import probe — in unit-test env we
    # don't have a real vllm installed, so simulate "helper imports
    # cleanly" by giving importlib.import_module a stand-in module that
    # exposes the helper. Tests for the negative case override this.
    import importlib, types
    stub_kv = types.ModuleType("vllm.v1.core.kv_cache_utils")
    stub_kv.token_capacity_kv_cache_groups = lambda *a, **kw: None
    real_import = importlib.import_module
    def fake_import(name, *args, **kwargs):
        if name == "vllm.v1.core.kv_cache_utils":
            return stub_kv
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", fake_import)

    return str(kv_path), str(sched_path)


class TestPatch8Wiring:
    def test_apply_both_files(self, fake_p8_files):
        from vllm._genesis.wiring.legacy import patch_8_kv_hybrid_reporting as p8
        kv_path, sched_path = fake_p8_files

        status, reason = p8.apply()
        assert status == "applied", f"{status}: {reason}"

        kv_content = open(kv_path).read()
        sched_content = open(sched_path).read()

        # kv_cache_utils.py: imports added, helper injected, callsite rewritten
        assert "AttentionSpec,  # [Genesis P8]" in kv_content
        assert "MambaSpec,  # [Genesis P8]" in kv_content
        assert "def token_capacity_kv_cache_groups(" in kv_content
        assert "capacity_groups = token_capacity_kv_cache_groups(" in kv_content

        # scheduler.py: import added, callsite rewritten
        assert ("from vllm.v1.core.kv_cache_utils import token_capacity_kv_cache_groups"
                in sched_content)
        assert "capacity_groups = token_capacity_kv_cache_groups(" in sched_content
        # Old inline logic gone from scheduler
        assert "num_groups = len(kv_cache_config.kv_cache_groups)" not in sched_content

    def test_idempotent(self, fake_p8_files):
        from vllm._genesis.wiring.legacy import patch_8_kv_hybrid_reporting as p8
        kv_path, sched_path = fake_p8_files

        from vllm._genesis.wiring.legacy.patch_8_kv_hybrid_reporting import (
            GENESIS_P8_MARKER_KV, GENESIS_P8_MARKER_SCHED,
        )
        s1, _ = p8.apply()
        s2, _ = p8.apply()
        assert s1 == "applied" and s2 == "applied"

        with open(kv_path) as _f:
            kv_content = _f.read()
        # Each marker inserted exactly once (marker version follows
        # GENESIS_P8_MARKER_KV constant, not a hardcoded literal).
        assert kv_content.count(GENESIS_P8_MARKER_KV) == 1

        with open(sched_path) as _f:
            sched_content = _f.read()
        assert sched_content.count(GENESIS_P8_MARKER_SCHED) == 1

    def test_skip_when_upstream_merged_kv(self, fake_p8_files):
        """If helper function already exists in kv_cache_utils (upstream landed),
        KV sub-patch skips — but scheduler may still need the import."""
        from vllm._genesis.wiring.legacy import patch_8_kv_hybrid_reporting as p8
        kv_path, _sched_path = fake_p8_files

        # Prepend helper to kv_cache_utils to simulate upstream merge
        content = open(kv_path).read()
        open(kv_path, "w").write(
            "def token_capacity_kv_cache_groups(): pass\n" + content
        )

        status, reason = p8.apply()
        # Scheduler can still apply → overall "applied"
        assert status == "applied"
        assert "kv_cache_utils=skipped" in reason

    def test_missing_files_skip(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_8_kv_hybrid_reporting as p8
        monkeypatch.setattr(p8, "resolve_vllm_file", lambda rel: None)
        monkeypatch.setattr(p8, "vllm_install_root", lambda: "/fake")
        status, reason = p8.apply()
        assert status == "skipped"
        assert "not found" in reason.lower() or "kv_cache_utils.py" in reason.lower()


class TestP8Semantics:
    """Sanity-check the math intent of P8 with mock group counts."""

    def test_filter_logic_excludes_mamba_when_cache_mode_not_all(self):
        """On mamba_cache_mode=='align' (default for Qwen3.6): mamba excluded."""
        # Pure-python sim of the helper
        class AttnSpec:
            block_size = 2768
        class MambaSpec:
            block_size = 2768
        groups = [
            type("G", (), {"kv_cache_spec": AttnSpec()})(),
            type("G", (), {"kv_cache_spec": MambaSpec()})(),
        ]

        def is_attn(g):
            return type(g.kv_cache_spec).__name__.startswith("Attn")
        def is_mamba(g):
            return type(g.kv_cache_spec).__name__.startswith("Mamba")

        mamba_scales = False  # align-mode, not 'all'
        filtered = [g for g in groups if is_attn(g) or (is_mamba(g) and mamba_scales)]
        assert len(filtered) == 1  # only attn
        # → num_tokens is 2× what upstream computes (divisor 1 vs 2)

    def test_filter_includes_mamba_when_all(self):
        class AttnSpec:
            block_size = 2768
        class MambaSpec:
            block_size = 2768
        groups = [
            type("G", (), {"kv_cache_spec": AttnSpec()})(),
            type("G", (), {"kv_cache_spec": MambaSpec()})(),
        ]

        def is_attn(g):
            return type(g.kv_cache_spec).__name__.startswith("Attn")
        def is_mamba(g):
            return type(g.kv_cache_spec).__name__.startswith("Mamba")

        mamba_scales = True  # mamba_cache_mode=='all'
        filtered = [g for g in groups if is_attn(g) or (is_mamba(g) and mamba_scales)]
        assert len(filtered) == 2

    def test_fallback_when_filter_empty(self):
        """If no attn groups and mamba not scaled, return all (original behavior)."""
        class MambaSpec:
            block_size = 2768
        groups = [
            type("G", (), {"kv_cache_spec": MambaSpec()})(),
        ]

        def is_attn(g):
            return type(g.kv_cache_spec).__name__.startswith("Attn")
        def is_mamba(g):
            return type(g.kv_cache_spec).__name__.startswith("Mamba")

        mamba_scales = False
        filtered = [g for g in groups if is_attn(g) or (is_mamba(g) and mamba_scales)]
        if not filtered:
            filtered = list(groups)
        assert len(filtered) == 1

    def test_capacity_ratio_on_qwen36_topology(self):
        """Qwen3.6-A3B: 1 attn group + 1 mamba group → 2× under-report without P8.

        Prod measured (integration round 6 comparison):
          prod v5.14.1: 1,104,432 tokens (with P8)
          v7.0 round 6: 293,408 tokens (without P8 equivalent)
          Ratio 3.76× — more than the simple 2× model-topology reason, so there
          are compounding effects (block_size differences between groups, etc).
        """
        # Simulate upstream-style divisor
        num_blocks = 100
        min_block_size = 2768

        num_groups_with_mamba = 2  # upstream counts both
        num_groups_attn_only = 1   # P8 counts only attn

        upstream_tokens = (num_blocks // num_groups_with_mamba) * min_block_size
        p8_tokens = (num_blocks // num_groups_attn_only) * min_block_size

        assert p8_tokens == 2 * upstream_tokens
