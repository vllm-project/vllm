# SPDX-License-Identifier: Apache-2.0
"""TDD for P5 wiring.

Active algorithm: v1 (LCM-pad-max). The v2 pad-smaller-to-max attempt
worked at the Python spec layer but broke TurboQuant attention kernel
reshape (integration round 4, 2026-04-24) because vLLM stores the
allocated tensor at `page_size_padded` bytes but the kernel views it
using the natural `block_size × num_kv_heads × slot_size_aligned` shape.

Therefore v1 is kept as the production algorithm. The v2 text (as _V2_FN)
and _V1_FN migration support remain in the module as reference and for
any future experiments that also modify the TQ kernel reshape path.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest


_BASELINE_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""KV-Cache Utilities."""

import copy
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, replace

from vllm import envs


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec):
    return not kv_cache_spec
'''


@pytest.fixture
def fake_kv_utils_file(tmp_path, monkeypatch):
    path = tmp_path / "kv_cache_utils.py"
    path.write_text(_BASELINE_SOURCE)

    from vllm._genesis import guards
    monkeypatch.setattr(guards, "resolve_vllm_file",
                        lambda rel: str(path) if "kv_cache_utils" in rel else None)
    monkeypatch.setattr(guards, "vllm_install_root", lambda: str(tmp_path))
    monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
    return str(path)


def _patch_module_guards(monkeypatch, fake_path):
    from vllm._genesis.wiring.legacy import patch_5_page_size as p5
    monkeypatch.setattr(p5, "resolve_vllm_file",
                        lambda rel: fake_path if "kv_cache_utils" in rel else None)
    monkeypatch.setattr(p5, "vllm_install_root", lambda: "/fake")
    monkeypatch.setattr(p5, "is_nvidia_cuda", lambda: True)


class TestPatch5V1Active:
    def test_apply_writes_v1_lcm_algorithm(self, fake_kv_utils_file, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_5_page_size
        _patch_module_guards(monkeypatch, fake_kv_utils_file)

        status, reason = patch_5_page_size.apply()
        assert status == "applied", f"{status}: {reason}"

        content = open(fake_kv_utils_file).read()

        # v7.0 marker (v1 active)
        assert "Genesis P5 page_size unification v7.0" in content
        # v1 uses math.lcm for LCM padding
        assert "import math" in content
        assert "smaller_lcm" in content
        # Old NotImplementedError gone
        assert "Cannot unify by adjusting block_size." not in content

    def test_idempotent(self, fake_kv_utils_file, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_5_page_size
        _patch_module_guards(monkeypatch, fake_kv_utils_file)

        s1, _ = patch_5_page_size.apply()
        s2, _ = patch_5_page_size.apply()
        assert s1 == "applied"
        assert s2 == "applied"

        content = open(fake_kv_utils_file).read()
        # Marker exactly once — no duplicate prepending
        assert content.count("Genesis P5 page_size unification v7.0") == 1
        # import math exactly once
        assert content.count("import math") == 1

    def test_skip_on_non_nvidia(self, fake_kv_utils_file, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_5_page_size
        monkeypatch.setattr(patch_5_page_size, "is_nvidia_cuda", lambda: False)

        status, reason = patch_5_page_size.apply()
        assert status == "skipped"
        assert "non-NVIDIA" in reason

    def test_skip_when_upstream_pr37429_landed(self, fake_kv_utils_file, monkeypatch):
        """If upstream PR #37429 (per-group block pools) merges, our patch
        becomes obsolete and should auto-skip."""
        from vllm._genesis.wiring.legacy import patch_5_page_size
        _patch_module_guards(monkeypatch, fake_kv_utils_file)

        content = open(fake_kv_utils_file).read()
        open(fake_kv_utils_file, "w").write(
            "# def _has_mixed_mamba_attention(): pass\n"
            "# mamba_num_blocks tracking\n"
            + content
        )

        status, reason = patch_5_page_size.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()


class TestPatch5MathAnalysis:
    """Document the overhead tradeoff in this test suite as evidence."""

    def test_v1_lcm_overhead_on_real_topology(self):
        """Observed 2026-04-24 integration: max=1073152, smaller=813248."""
        import math

        max_page = 1_073_152
        smaller = 813_248

        # GCD: 1073152 = 2^13 × 131 = 8192×131; 813248 = 2^6 × 131 × 97.
        # GCD = 2^6 × 131 = 8384.
        g = math.gcd(max_page, smaller)
        assert g == 8384

        # v1 with single smaller: smaller_lcm = smaller itself.
        smaller_lcm = smaller
        target = ((max_page + smaller_lcm - 1) // smaller_lcm) * smaller_lcm
        assert target == 1_626_496
        overhead = (target - max_page) / max_page
        assert 0.515 < overhead < 0.516  # 51.56%

    def test_v2_would_save_memory_but_breaks_tq_kernel(self):
        """Document why v2 is not active: saves ~34% memory but breaks
        TurboQuant attention kernel reshape (RuntimeError on integration
        round 4). Future work: align with real_page_size_bytes in
        vLLM's kernel-layer allocation."""
        max_page = 1_073_152
        num_attn = 10
        num_mamba = 30

        # v1: all 40 layers padded to 1,626,496
        v1_target = 1_626_496
        v1_bytes_per_block = (num_attn + num_mamba) * v1_target

        # v2 (theoretical): attn untouched, mamba padded to max
        v2_bytes_per_block = num_attn * max_page + num_mamba * max_page

        savings = v1_bytes_per_block - v2_bytes_per_block
        savings_pct = savings / v1_bytes_per_block * 100
        assert 33 < savings_pct < 35  # ~34%

        # But v2 is BLOCKED until we fix the kernel reshape
        # (integration round 4 crashed; v2 stays in file as reference).
