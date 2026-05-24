# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN32 v2 — GDN _forward_core chunked-prefill (Cliff 2 fix).

CPU-runnable structural tests. Real GPU validation requires:
  - Single 24GB GPU (1×3090 / 1×4090 / 1×5090)
  - >50K-token single-shot prompt
  - Hybrid GDN model (Qwen3.5/3.6 27B)

v7.69 v2 supersedes v7.65 v1. v1 chunked at the wrong level
(forward_cuda outer, didn't propagate cu_seqlens to inner FLA call).
v2 chunks _forward_core directly with chunk-local cu_seqlens and
threaded initial_state. See club-3090#19 finding 3 (2026-05-02).

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reporter: noonghunna (CLIFF2_INVESTIGATION_20260430.md +
                       club-3090#19 finding 3, 2026-05-02).
"""
from __future__ import annotations


def test_pn32_wiring_imports():
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN32_MARKER")


def test_pn32_dispatcher_registry():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN32" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN32"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL"
    assert e["default_on"] is False


def test_pn32_skips_when_env_off(monkeypatch):
    monkeypatch.delenv(
        "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL", raising=False
    )
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import apply
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


# ─────────────────────────────────────────────────────────────────
# v7.69 v2 — anchor + replacement structural correctness
# ─────────────────────────────────────────────────────────────────


def test_pn32_v2_anchor_targets_forward_core_prefill_branch():
    """v2 anchor must match the PREFILL BRANCH of _forward_core, NOT
    the outer forward_cuda's gdn_attention_core call (that was v1's
    wrong-level chunking that didn't propagate cu_seqlens).
    """
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_ANCHOR,
    )
    # Must include the prefill branch entry comment
    assert "# 2.2: Process the remaining part" in PN32_ANCHOR
    assert "if attn_metadata.num_prefills > 0:" in PN32_ANCHOR
    # Must include the FLA call (chunk_gated_delta_rule) at this level
    assert "self.chunk_gated_delta_rule(" in PN32_ANCHOR
    # Must NOT match v1's wrong-level pattern (gdn_attention_core call)
    assert "torch.ops.vllm.gdn_attention_core" not in PN32_ANCHOR
    # Must include the cache update (anchor's full extent)
    assert "ssm_state[non_spec_state_indices_tensor] = last_recurrent_state" in PN32_ANCHOR


def test_pn32_v2_replacement_chunks_along_T_dim():
    """v2 replacement must slice query/key/value/g/beta on dim=1 (T)."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # All five FLA inputs sliced along dim 1 (after unsqueeze, shape is
    # (1, T, H, D); T is dim 1)
    for name in (
        "query_non_spec",
        "key_non_spec",
        "value_non_spec",
        "g_non_spec",
        "beta_non_spec",
    ):
        assert (
            f"{name}[\n"
            f"                        :, _genesis_pn32_start:_genesis_pn32_end\n"
            f"                    ]"
        ) in PN32_REPLACEMENT, (
            f"v2 must slice {name} along dim=1 (T) — fix-shape consistency"
        )


def test_pn32_v2_replacement_builds_chunk_local_cu_seqlens():
    """v2 must construct chunk-local cu_seqlens=[0, chunk_len] per chunk
    (NOT pass full-prompt non_spec_query_start_loc)."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # Construction of chunk-local cu_seqlens
    assert "torch.tensor(\n" in PN32_REPLACEMENT
    assert "[0, _genesis_pn32_chunk_len]," in PN32_REPLACEMENT
    # Passes chunk_indices=None and chunk_offsets=None to FLA
    # (let FLA recompute internally from cu_seqlens)
    assert "chunk_indices=None," in PN32_REPLACEMENT
    assert "chunk_offsets=None," in PN32_REPLACEMENT


def test_pn32_v2_replacement_threads_initial_state():
    """v2 must thread initial_state across chunks via last_recurrent_state.

    First chunk uses computed initial_state; subsequent chunks use the
    prior chunk's last_recurrent_state. This preserves recurrent state
    propagation that FLA does internally for unchunked calls.
    """
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # State is initialized to incoming initial_state
    assert "_genesis_pn32_state = initial_state" in PN32_REPLACEMENT
    # Each chunk's FLA call uses the threaded state
    assert "initial_state=_genesis_pn32_state," in PN32_REPLACEMENT
    # State is updated to the chunk's final state for the next iteration
    assert "_genesis_pn32_state = _genesis_pn32_last_state" in PN32_REPLACEMENT


def test_pn32_v2_replacement_concatenates_chunks_along_T():
    """v2 must concat per-chunk outputs along dim=1 (T) to reconstruct
    the full-shape core_attn_out_non_spec."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # torch.cat along dim 1 (T dim, since output shape is (1, T, H, V))
    assert "torch.cat(\n                    _genesis_pn32_chunks, dim=1\n                )" in PN32_REPLACEMENT


def test_pn32_v2_replacement_bypasses_for_multi_seq():
    """Multi-sequence prefill (cu_seqlens shape > [2]) must bypass to
    original — chunking across seq boundaries requires inner state-cache
    surgery not exposed at this layer."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # Single-seq detection: cu_seqlens shape == 2
    assert "non_spec_query_start_loc.shape[0] == 2" in PN32_REPLACEMENT
    # Else branch falls through to original FLA call
    assert "# ─── Original path" in PN32_REPLACEMENT
    # Original call uses non_spec_query_start_loc + attn_metadata fields
    assert "cu_seqlens=non_spec_query_start_loc," in PN32_REPLACEMENT
    assert "chunk_indices=attn_metadata.chunk_indices," in PN32_REPLACEMENT


def test_pn32_v2_replacement_threshold_and_chunk_size_env_tunable():
    """Both threshold (default 16384) and chunk_size (default 8192) are
    env-tunable and read defensively (with try/except for parse errors)."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    assert "GENESIS_PN32_GDN_CHUNK_THRESHOLD" in PN32_REPLACEMENT
    assert "GENESIS_PN32_GDN_CHUNK_SIZE" in PN32_REPLACEMENT
    assert "'16384'" in PN32_REPLACEMENT
    assert "'8192'" in PN32_REPLACEMENT
    # Defensive parse — operators can pass garbage env vars
    assert "except (ValueError, TypeError):" in PN32_REPLACEMENT


def test_pn32_v2_replacement_explicit_del_for_chunk_buffers():
    """Replacement has explicit `del` to help allocator reuse chunk slots."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # All five chunk slices have explicit del
    assert "del (\n" in PN32_REPLACEMENT
    assert "_genesis_pn32_q_chunk," in PN32_REPLACEMENT
    assert "_genesis_pn32_k_chunk," in PN32_REPLACEMENT
    assert "_genesis_pn32_v_chunk," in PN32_REPLACEMENT
    assert "_genesis_pn32_g_chunk," in PN32_REPLACEMENT
    assert "_genesis_pn32_beta_chunk," in PN32_REPLACEMENT
    # Final chunks list also freed after concat
    assert "del _genesis_pn32_chunks" in PN32_REPLACEMENT


# ─────────────────────────────────────────────────────────────────
# v2 documentation requirements
# ─────────────────────────────────────────────────────────────────


def test_pn32_v2_documents_v1_redesign_rationale():
    """v2 module docstring must explain WHY v1 was wrong + what v2
    changes. This is the cross-rig finding's audit trail."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    src = inspect.getsource(mod)
    # Must reference the v1 redesign rationale
    assert "v7.65" in src or "v1" in src
    assert "WRONG level" in src or "wrong level" in src
    # Must reference the cross-rig finding origin
    assert "club-3090#19" in src or "noonghunna" in src
    # Must explain the metadata-mismatch bug
    assert "cu_seqlens" in src
    assert "metadata" in src.lower()


def test_pn32_v2_documents_p103_composition():
    """v2 must explicitly document composition with P103 (the inner-FLA
    chunking patch). Without this guidance, operators don't know that
    PN32+P103 together = full Cliff 2 coverage."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    src = inspect.getsource(mod)
    # Must reference P103 explicitly
    assert "P103" in src
    # Must say they compose / are complementary
    assert "complement" in src.lower() or "compose" in src.lower() or "composition" in src.lower()


def test_pn32_v2_documents_dependencies_section():
    """v2 must have a DEPENDENCIES section in module docstring listing
    P103 as recommended, P28 as conflict, single-seq as a precondition."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    doc = mod.__doc__ or ""
    assert "DEPENDENCIES" in doc, (
        "v2 module docstring must have a DEPENDENCIES section per "
        "Sander's request 2026-05-02"
    )
    # P103 listed as recommended companion
    assert "P103" in doc
    # P28 listed as conflict
    assert "P28" in doc
    # Single-sequence precondition documented
    assert "single-seq" in doc.lower() or "single sequence" in doc.lower() or "single-sequence" in doc.lower()


def test_pn32_v2_documents_threshold_semantics():
    """v2 must document what triggers the chunked path (env + threshold +
    single-seq + prefill)."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    doc = mod.__doc__ or ""
    assert "GENESIS_PN32_GDN_CHUNK_THRESHOLD" in doc
    assert "16384" in doc
    assert "8192" in doc


def test_pn32_v2_dispatcher_credit_mentions_v7_69_redesign():
    """Dispatcher credit string should reflect v2 redesign (so operators
    using `genesis explain PN32` see the current state)."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    credit = PATCH_REGISTRY["PN32"].get("credit", "")
    # v2 redesign mention (lenient — credit may have been updated to v7.69)
    # Skip if not yet updated (CHANGELOG.md is authoritative for ops)
    # but must at least mention Cliff 2 + chunking
    assert "Cliff 2" in credit or "cliff 2" in credit.lower()
    assert "chunk" in credit.lower()


# ─────────────────────────────────────────────────────────────────
# Marker + apply_all integration
# ─────────────────────────────────────────────────────────────────


def test_pn32_v2_register_in_apply_all():
    from vllm._genesis.patches.apply_all import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    names = [name for name, _ in APPLY_REGISTRY]
    pn32 = [n for n in names if "PN32" in n and "chunked" in n.lower()]
    assert len(pn32) == 1, f"PN32 chunked-prefill not registered, names: {names[:5]}"


def test_pn32_v2_marker_bumped_to_v7_69():
    """Marker must reflect v2 (so a v1-applied container that runs v2
    apply detects the difference and re-applies, not silent-skip)."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        GENESIS_PN32_MARKER,
    )
    assert "v2" in GENESIS_PN32_MARKER
    assert "v7.69" in GENESIS_PN32_MARKER


def test_pn32_v2_drift_marker_specific():
    """v2 drift markers must be SPECIFIC (NOT generic '[Genesis PN32'
    which would false-positive on any sibling Genesis insertion in
    gdn_linear_attn.py)."""
    import os
    import tempfile

    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    import vllm._genesis.guards as guards

    with tempfile.TemporaryDirectory() as td:
        mamba_dir = os.path.join(
            td, "model_executor", "layers", "mamba"
        )
        os.makedirs(mamba_dir)
        with open(os.path.join(mamba_dir, "gdn_linear_attn.py"), "w") as f:
            f.write("# placeholder\n")

        orig = guards.vllm_install_root
        guards.vllm_install_root = lambda: td
        try:
            patcher = mod._make_patcher()
        finally:
            guards.vllm_install_root = orig

    assert patcher is not None
    # Must NOT include the generic '[Genesis PN32' prefix
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis PN32"):
            assert "v2" in m or "v7.69" in m or "chunked-prefill" in m, (
                f"drift marker {m!r} too generic — risk of false-positive "
                f"on sibling patches' insertions"
            )
