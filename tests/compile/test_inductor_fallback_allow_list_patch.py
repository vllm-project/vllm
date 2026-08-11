# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Inductor FALLBACK_ALLOW_LIST patch in env_override.py.

The patch wraps ``torch._inductor.lowering.FALLBACK_ALLOW_LIST`` in a thin
proxy that auto-allows any custom op in the ``vllm::`` or ``vllm_aiter::``
namespaces. This routes those ops through Inductor's fast-path
``make_fallback(target, warn=False, override_decomp=True)`` and avoids the
expensive ``error.operator_str(target, args, kwargs)`` formatting that
recursively stringifies every input ``TensorBox``.

The slow path is what made ``torch.compile`` effectively hang on Kimi-K2.6
TP=8 (deep MoE/TP IR provenance trees). These tests cover both the proxy's
semantics in isolation and the membership-check fast-path that Inductor's
``GraphLowering.call_function`` actually performs, so we can validate the
optimization without needing a full GPU compile.
"""

import time

import pytest

from vllm.env_override import (
    _patch_inductor_fallback_allow_list,
    _VllmFallbackAllowList,
)


class TestVllmFallbackAllowListProxy:
    """Unit tests for the membership-proxy semantics."""

    def test_vllm_namespace_auto_allowed(self):
        proxy = _VllmFallbackAllowList(set())
        assert "vllm::all_reduce" in proxy
        assert "vllm::fused_add_rms_norm" in proxy
        assert "vllm::all_reduce.default" in proxy

    def test_vllm_aiter_namespace_auto_allowed(self):
        proxy = _VllmFallbackAllowList(set())
        assert "vllm_aiter::fused_add_rms_norm" in proxy
        assert "vllm_aiter::rocm_aiter_fused_moe" in proxy

    def test_unknown_namespace_falls_through(self):
        proxy = _VllmFallbackAllowList({"torchvision::roi_align"})
        assert "torchvision::roi_align" in proxy
        assert "made_up_ns::nonexistent_op" not in proxy

    def test_non_string_falls_through_to_inner(self):
        sentinel = object()
        inner = {sentinel}
        proxy = _VllmFallbackAllowList(inner)
        assert sentinel in proxy
        assert object() not in proxy

    def test_prefix_only_match_not_substring(self):
        proxy = _VllmFallbackAllowList(set())
        assert "not_vllm::something" not in proxy
        assert "  vllm::space_prefixed" not in proxy

    def test_standard_entries_preserved(self):
        base = {"torchvision::roi_align", "aten::index_add"}
        proxy = _VllmFallbackAllowList(base)
        assert "torchvision::roi_align" in proxy
        assert "aten::index_add" in proxy
        assert "aten::__not_present__" not in proxy

    def test_add_and_discard_delegate_to_inner(self):
        inner: set[str] = set()
        proxy = _VllmFallbackAllowList(inner)
        proxy.add("custom::op")
        assert "custom::op" in inner
        proxy.discard("custom::op")
        assert "custom::op" not in inner

    def test_iter_len_repr(self):
        base = {"torchvision::roi_align", "aten::index_add"}
        proxy = _VllmFallbackAllowList(base)
        assert set(iter(proxy)) == base
        assert len(proxy) == len(base)
        assert "torchvision::roi_align" in repr(proxy)

    def test_getattr_delegates_to_inner(self):
        class _Inner:
            sentinel = "i_am_inner"

            def some_method(self):
                return 42

        inner = _Inner()
        proxy = _VllmFallbackAllowList(inner)
        assert proxy.sentinel == "i_am_inner"
        assert proxy.some_method() == 42

    def test_sentinel_attribute(self):
        proxy = _VllmFallbackAllowList(set())
        assert proxy._vllm_patched is True


class TestPatchApplication:
    """Integration tests verifying the patch reaches ``torch._inductor``."""

    def test_patch_applied_to_lowering(self):
        import torch._inductor.lowering as _lowering

        assert getattr(_lowering.FALLBACK_ALLOW_LIST, "_vllm_patched", False), (
            "env_override._patch_inductor_fallback_allow_list did not run"
        )

    def test_graph_module_local_binding_rebound(self):
        # ``torch/_inductor/graph.py`` does:
        #   from torch._inductor.lowering import FALLBACK_ALLOW_LIST
        # so the patch has to overwrite the graph module's local binding too,
        # otherwise the fast-path check in GraphLowering.call_function still
        # sees the original (unwrapped) OrderedSet.
        import torch._inductor.graph as _graph
        import torch._inductor.lowering as _lowering

        if not hasattr(_graph, "FALLBACK_ALLOW_LIST"):
            pytest.skip(
                "torch._inductor.graph no longer imports FALLBACK_ALLOW_LIST "
                "as a module-level symbol; nothing to rebind."
            )

        assert _graph.FALLBACK_ALLOW_LIST is _lowering.FALLBACK_ALLOW_LIST

    def test_patch_is_idempotent(self):
        import torch._inductor.lowering as _lowering

        first = _lowering.FALLBACK_ALLOW_LIST
        _patch_inductor_fallback_allow_list()
        _patch_inductor_fallback_allow_list()
        assert _lowering.FALLBACK_ALLOW_LIST is first

    def test_real_vllm_ops_in_real_allow_list(self):
        # End-to-end membership check using the live (already-patched) object.
        import torch._inductor.lowering as _lowering

        allow_list = _lowering.FALLBACK_ALLOW_LIST
        assert "vllm::all_reduce" in allow_list
        assert "vllm::fused_add_rms_norm" in allow_list
        assert "vllm_aiter::fused_add_rms_norm" in allow_list


class TestInductorFallbackFastPath:
    """Emulates ``GraphLowering.call_function``'s FALLBACK_ALLOW_LIST check.

    The relevant snippet in ``torch/_inductor/graph.py`` is roughly::

        base_name = target.name()
        if base_name not in FALLBACK_ALLOW_LIST:
            log.info(
                "Creating implicit fallback for:\\n%s",
                error.operator_str(target, args, kwargs),
            )
        out = make_fallback(target, ...)

    On a deep MoE/TP graph (Kimi-K2.6 at TP=4/8) ``operator_str`` recurses
    through every input ``TensorBox.__str__`` and ends up taking many minutes
    of CPU per encountered op. The patch ensures the membership test
    short-circuits for ``vllm::*``/``vllm_aiter::*`` ops so the slow path is
    never entered. These tests pin that behaviour without needing a real
    GPU compile.
    """

    def _simulate_graph_lowering(self, target_names: list[str]):
        """Returns the set of target names that would have hit the slow
        operator_str() path under the patched FALLBACK_ALLOW_LIST.
        """
        import torch._inductor.lowering as _lowering

        allow_list = _lowering.FALLBACK_ALLOW_LIST
        slow_path_hits: list[str] = []
        for name in target_names:
            if name not in allow_list:
                slow_path_hits.append(name)
        return slow_path_hits

    def test_vllm_ops_skip_slow_path(self):
        slow = self._simulate_graph_lowering(
            [
                "vllm::all_reduce",
                "vllm::fused_add_rms_norm",
                "vllm_aiter::rocm_aiter_fused_moe",
                "vllm_aiter::asm_moe",
            ]
        )
        assert slow == [], (
            "Patched FALLBACK_ALLOW_LIST must short-circuit for all "
            f"vllm::*/vllm_aiter::* ops; got slow-path hits: {slow}"
        )

    def test_non_vllm_ops_still_hit_slow_path(self):
        # Without the patch this is also what would happen; with the patch
        # the behaviour for non-vllm namespaces must be unchanged.
        slow = self._simulate_graph_lowering(
            ["my_user_ns::custom_op", "fancy_ns::something_else"]
        )
        assert "my_user_ns::custom_op" in slow
        assert "fancy_ns::something_else" in slow

    def test_kimi_k2_6_style_op_stream(self):
        """Emulates one decoder layer's worth of fallback hits.

        Kimi-K2.6 at TP=4 lowers a stream of ``vllm::all_reduce`` +
        ``vllm_aiter::fused_add_rms_norm`` calls (one per residual block)
        plus a handful of fused-MoE ops. Pre-patch every one of these would
        invoke ``operator_str`` and stringify a hundreds-deep IR provenance
        tree; post-patch they must all short-circuit.
        """
        n_layers = 64  # Kimi-K2.6 has ~64 decoder layers per replica
        op_stream: list[str] = []
        for _ in range(n_layers):
            op_stream.extend(
                [
                    "vllm::all_reduce",
                    "vllm_aiter::fused_add_rms_norm",
                    "vllm_aiter::rocm_aiter_fused_moe",
                ]
            )

        start = time.perf_counter()
        slow = self._simulate_graph_lowering(op_stream)
        elapsed_s = time.perf_counter() - start

        assert slow == [], (
            f"Expected all {len(op_stream)} vllm/vllm_aiter ops to take "
            f"the fast path; got {len(slow)} slow-path hits."
        )
        # ``__contains__`` is O(1) per call, so a Kimi-sized stream should
        # complete in well under a second even on a slow runner. The
        # pre-patch slow path took many minutes per op on Kimi-K2.6 TP=8.
        assert elapsed_s < 1.0, (
            f"FALLBACK_ALLOW_LIST membership check is unexpectedly slow: "
            f"{elapsed_s:.3f}s for {len(op_stream)} ops"
        )

    def test_inner_set_membership_still_works_for_standard_ops(self):
        """The patch must not break Inductor's existing fallback decisions
        for non-vllm ops such as ``torchvision::roi_align``."""
        import torch._inductor.lowering as _lowering

        allow_list = _lowering.FALLBACK_ALLOW_LIST
        # ``torchvision::roi_align`` has been a member of the upstream
        # FALLBACK_ALLOW_LIST since the original Inductor implementation.
        # If the proxy ever broke pass-through, this would regress.
        if "torchvision::roi_align" not in allow_list:
            pytest.skip(
                "Upstream FALLBACK_ALLOW_LIST no longer ships "
                "torchvision::roi_align; nothing to verify."
            )
