# SPDX-License-Identifier: Apache-2.0
"""TDD for PN52 — vllm#41411 backport (multi-file).

Validates each of the 7 sub-patches' anchors + idempotency + env-flag +
registry consistency.
"""
from __future__ import annotations

import pytest


def _wiring():
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N52_prompt_logprobs_eviction as M,
    )
    return M


def test_anchors_target_correct_buggy_code():
    """All 7 anchors must contain the bug signatures from vllm pre-#41411."""
    M = _wiring()
    assert "computed_prefill < prompt_lens - 1" in M.PROMPT_LOGPROB_OLD
    # Replacement must drop the `-1` from the actual code line.
    code_lines_new = [
        ln for ln in M.PROMPT_LOGPROB_NEW.splitlines()
        if ln.lstrip().startswith("includes_prompt")
    ]
    assert len(code_lines_new) == 1
    assert "computed_prefill < prompt_lens" in code_lines_new[0]
    assert "prompt_lens - 1" not in code_lines_new[0]

    assert "lora_request: LoRARequest | None = None" in M.INPUT_BATCH_FIELD_OLD
    assert "in_progress_prompt_logprobs_cpu" in M.INPUT_BATCH_FIELD_NEW

    assert "self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}" in M.INPUT_BATCH_INIT_OLD
    assert "in_progress_prompt_logprobs_cpu: dict" not in M.INPUT_BATCH_INIT_NEW

    assert "self.in_progress_prompt_logprobs_cpu.pop(req_id, None)" in M.INPUT_BATCH_REMOVE_OLD
    assert "self.in_progress_prompt_logprobs_cpu.pop" not in M.INPUT_BATCH_REMOVE_NEW

    assert "in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu" in M.RUNNER_DICT_ALIAS_OLD
    assert "in_progress_dict =" not in M.RUNNER_DICT_ALIAS_NEW

    assert "logprobs_tensors = in_progress_dict.get(req_id)" in M.RUNNER_GET_OLD
    assert "logprobs_tensors = request.in_progress_prompt_logprobs_cpu" in M.RUNNER_GET_NEW

    assert "del in_progress_dict[req_id]" in M.RUNNER_DEL_OLD
    assert "in_progress_prompt_logprobs_cpu = None" in M.RUNNER_DEL_NEW


def test_replacements_carry_pn52_marker():
    M = _wiring()
    for n, new in [
        ("PROMPT_LOGPROB_NEW", M.PROMPT_LOGPROB_NEW),
        ("INPUT_BATCH_FIELD_NEW", M.INPUT_BATCH_FIELD_NEW),
        ("INPUT_BATCH_INIT_NEW", M.INPUT_BATCH_INIT_NEW),
        ("INPUT_BATCH_REMOVE_NEW", M.INPUT_BATCH_REMOVE_NEW),
        ("RUNNER_DICT_ALIAS_NEW", M.RUNNER_DICT_ALIAS_NEW),
        ("RUNNER_GET_NEW", M.RUNNER_GET_NEW),
        ("RUNNER_DEL_NEW", M.RUNNER_DEL_NEW),
    ]:
        assert "PN52" in new, f"{n} missing PN52 marker"
        assert "vllm#41411" in new, f"{n} missing upstream PR ref"


def test_idempotent_on_synthetic_each_file(tmp_path):
    """Each file's anchors apply once then no-op on second pass."""
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()

    cases = [
        ("prompt_logprob_synthetic", M.PROMPT_LOGPROB_OLD, M.PROMPT_LOGPROB_NEW, "pn52_drop_minus1"),
        ("input_batch_field_synthetic", M.INPUT_BATCH_FIELD_OLD, M.INPUT_BATCH_FIELD_NEW, "pn52_field"),
        ("input_batch_init_synthetic", M.INPUT_BATCH_INIT_OLD, M.INPUT_BATCH_INIT_NEW, "pn52_init"),
        ("input_batch_remove_synthetic", M.INPUT_BATCH_REMOVE_OLD, M.INPUT_BATCH_REMOVE_NEW, "pn52_remove"),
        ("runner_dict_synthetic", M.RUNNER_DICT_ALIAS_OLD, M.RUNNER_DICT_ALIAS_NEW, "pn52_dict_alias"),
        ("runner_get_synthetic", M.RUNNER_GET_OLD, M.RUNNER_GET_NEW, "pn52_get"),
        ("runner_del_synthetic", M.RUNNER_DEL_OLD, M.RUNNER_DEL_NEW, "pn52_del"),
    ]
    for fname, old, new, sub_name in cases:
        target = tmp_path / fname
        # wrap in some surrounding text so anchor is interior of file
        target.write_text("# header\n" + old + "\n# footer\n")
        patcher = TextPatcher(
            patch_name=fname,
            target_file=str(target),
            marker=M.GENESIS_PN52_MARKER,
            sub_patches=[TextPatch(name=sub_name, anchor=old, replacement=new, required=True)],
        )
        r1, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED, f"{fname} apply 1 failed"
        body1 = target.read_text()
        assert "PN52" in body1, f"{fname} marker missing"
        r2, _ = patcher.apply()
        assert r2 == TextPatchResult.IDEMPOTENT, f"{fname} not idempotent"
        assert target.read_text() == body1, f"{fname} second apply mutated body"


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION", raising=False)
    decision, reason = should_apply("PN52")
    assert decision is False
    assert "opt-in" in reason.lower() or "off" in reason.lower()


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION", "1")
    decision, _ = should_apply("PN52")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN52" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN52"]
    assert meta["env_flag"] == "GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION"
    assert meta["default_on"] is False
    assert meta["upstream_pr"] == 41411
    assert "prompt_logprobs" in meta["title"].lower()


def test_apply_all_registers_pn52():
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_N52_prompt_logprobs_eviction")
