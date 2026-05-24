# SPDX-License-Identifier: Apache-2.0
"""Tests for PN17 — FA2 softmax_lse runtime clamp.

PN17 closes Genesis Issue #11 mechanism A: FA2's
`flash_attn_varlen_func` allocates softmax_lse buffer sized by the
caller-passed `max_seqlen_k`, not by actual `seqused_k`. vLLM sets
`attn_metadata.max_seq_len = max_model_len` during cudagraph capture
for shape stability; that value leaks into runtime decode/prefill
and over-allocates 50-100 MiB at long context.

PN17 inserts a runtime-only clamp:
- eager mode: `max_seqlen_k = int(seqused_k.max().item())`
- cudagraph capture: keep upstream `max_seqlen_k = max_model_len`

The patch is opt-in via `GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1`.

These tests pin:
- Module imports cleanly
- Anchor + replacement strings preserve cudagraph guard
- Replacement preserves the upstream variable assignments and only
  modifies `max_seqlen_k`
- apply() respects the env-flag gate (default skip)
- PN17 is registered in PATCH_REGISTRY + apply_all + PATCHES.md
"""
from __future__ import annotations



class TestPN17ModuleStructure:
    def test_module_importable(self):
        from vllm._genesis.wiring.kv_cache import patch_N17_fa2_softmax_lse_clamp  # noqa: F401

    def test_anchor_old_block_present(self):
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            PN17_OLD,
        )
        # Anchor must include the four-line non-cascade preamble that
        # uniquely identifies the FA2 varlen call site.
        assert "if not attn_metadata.use_cascade:" in PN17_OLD
        assert "cu_seqlens_q = attn_metadata.query_start_loc" in PN17_OLD
        assert "seqused_k = attn_metadata.seq_lens" in PN17_OLD
        assert "max_seqlen_q = attn_metadata.max_query_len" in PN17_OLD
        assert "max_seqlen_k = attn_metadata.max_seq_len" in PN17_OLD

    def test_replacement_preserves_unmodified_lines(self):
        """The replacement must keep the four assignments (cu_seqlens_q,
        seqused_k, max_seqlen_q) unchanged — only the `max_seqlen_k`
        line is replaced. If we accidentally drop one, the call site
        loses a required local."""
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            PN17_NEW,
        )
        for line in (
            "cu_seqlens_q = attn_metadata.query_start_loc",
            "seqused_k = attn_metadata.seq_lens",
            "max_seqlen_q = attn_metadata.max_query_len",
        ):
            assert line in PN17_NEW, (
                f"PN17 replacement must preserve `{line}` — call site "
                "needs it"
            )


class TestPN17CudagraphGuard:
    """The clamp MUST fall back to upstream behavior during cudagraph
    capture — otherwise we change the captured shape and break replay.
    """

    def test_replacement_includes_capture_check(self):
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            PN17_NEW,
        )
        assert "is_current_stream_capturing" in PN17_NEW, (
            "PN17 must guard the clamp behind is_current_stream_capturing() "
            "or it will alter cudagraph-captured tensor shapes"
        )

    def test_replacement_capture_path_uses_upstream_value(self):
        """During capture, the replacement assigns `max_seqlen_k =
        attn_metadata.max_seq_len` — exactly upstream behavior."""
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            PN17_NEW,
        )
        # The capture-path branch must still use max_seq_len so cudagraph
        # captures get the same shape upstream would produce.
        assert "max_seqlen_k = attn_metadata.max_seq_len" in PN17_NEW

    def test_replacement_runtime_path_uses_seqused_k_max(self):
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            PN17_NEW,
        )
        # Runtime path: clamp to actual chunk max
        assert "seqused_k.max().item()" in PN17_NEW, (
            "Runtime path must clamp to int(seqused_k.max().item())"
        )

    def test_replacement_has_defensive_upper_bound(self):
        """The clamped value must never exceed max_seq_len — defensive
        guard against metadata corruption."""
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            PN17_NEW,
        )
        # Must contain a check that the clamped value doesn't exceed
        # the upstream max_seq_len bound.
        assert "max_seqlen_k > attn_metadata.max_seq_len" in PN17_NEW


class TestPN17EnvGate:
    def test_apply_skipped_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PN17_FA2_LSE_CLAMP", raising=False)
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            apply,
        )
        status, reason = apply()
        assert status == "skipped"
        assert "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP" in reason

    def test_apply_skipped_explains_issue_11(self, monkeypatch):
        """The skip reason must reference Issue #11 + the diagnosis
        credit so future operators discover the institutional context.
        """
        monkeypatch.delenv("GENESIS_ENABLE_PN17_FA2_LSE_CLAMP", raising=False)
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            apply,
        )
        _, reason = apply()
        assert "Issue #11" in reason
        assert "noonghunna" in reason

    def test_env_flag_truthy_values_recognized(self, monkeypatch):
        """`1`, `true`, `yes`, `on` (case-insensitive) all enable PN17.
        Anything else is treated as OFF."""
        from vllm._genesis.wiring.kv_cache.patch_N17_fa2_softmax_lse_clamp import (
            _is_enabled,
        )
        for val in ("1", "true", "TRUE", "Yes", "on", "ON"):
            monkeypatch.setenv("GENESIS_ENABLE_PN17_FA2_LSE_CLAMP", val)
            assert _is_enabled() is True, f"truthy value {val!r} ignored"
        for val in ("", "0", "false", "off", "no", "garbage"):
            monkeypatch.setenv("GENESIS_ENABLE_PN17_FA2_LSE_CLAMP", val)
            assert _is_enabled() is False, f"falsy value {val!r} enabled patch"


class TestPN17DispatcherIntegration:
    def test_pn17_in_patch_registry(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert "PN17" in PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN17"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP"
        assert meta["default_on"] is False
        # Issue #11 cross-reference required
        assert "#11" in meta["title"] or "Issue #11" in meta["credit"]

    def test_pn17_category_is_memory_savings(self):
        """PN17 frees softmax_lse over-allocation — falls under
        memory_savings, not perf_hotfix."""
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert PATCH_REGISTRY["PN17"]["category"] == "memory_savings"

    def test_pn17_in_apply_all(self):
        """`@register_patch` for PN17 must be hooked into apply_all so
        the dry-run patcher walks it."""
        from vllm._genesis.patches import apply_all
        assert hasattr(apply_all, "apply_patch_N17_fa2_softmax_lse_clamp"), (
            "PN17 must be registered in apply_all.py"
        )

    def test_pn17_in_patches_md(self):
        """PATCHES.md must list PN17 — the existing
        test_patches_md_sync test enforces this generically; this is
        the explicit per-patch pin."""
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[3]
        patches_md = (repo_root / "docs" / "PATCHES.md").read_text()
        assert "PN17" in patches_md
        assert "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP" in patches_md
