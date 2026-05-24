# SPDX-License-Identifier: Apache-2.0
"""TDD for structured boot summary — v7.70 operator readability feature.

Covers:
  - System info header rendering (Genesis version, vllm pin, GPU, model)
  - Per-category APPLY/SKIP counters with totals
  - APPLIED table grouping by category + upstream PR cite
  - SKIPPED table grouping by reason class (upstream_merged / env_disabled
    / model_incompat / conflict / other)
  - FAILED highlighted section
  - Multi-worker dedup (same patch_id reported twice → counted once)
"""
from __future__ import annotations

import pytest


@pytest.fixture
def reset_decisions():
    """Clear the dispatcher decision log before/after each test."""
    from vllm._genesis import dispatcher
    saved = list(dispatcher._DECISIONS)
    dispatcher._DECISIONS.clear()
    yield
    dispatcher._DECISIONS.clear()
    dispatcher._DECISIONS.extend(saved)


def test_summary_empty_returns_helpful_message(reset_decisions):
    from vllm._genesis.dispatcher import dump_structured_boot_summary
    out = dump_structured_boot_summary()
    assert "no Genesis decisions recorded" in out


def test_summary_header_includes_genesis_and_vllm_versions(reset_decisions):
    from vllm._genesis.dispatcher import (
        dump_structured_boot_summary, log_decision,
    )
    log_decision("PN59", True, "opt-in env (config: neutral)")
    out = dump_structured_boot_summary()
    assert "Genesis:" in out
    assert "vLLM:" in out
    # Genesis version should NOT be doubled "vv..."
    assert "vv" not in out


def test_summary_counters_match_decisions(reset_decisions):
    from vllm._genesis.dispatcher import (
        dump_structured_boot_summary, log_decision,
    )
    log_decision("PN59", True, "opt-in env")
    log_decision("PN51", True, "opt-in env")
    log_decision("PN58", False, "PN58 SKIPPED — P62 mutually exclusive")
    out = dump_structured_boot_summary()
    assert "3 total" in out
    assert "2 APPLY" in out
    assert "1 SKIP" in out


def test_summary_category_breakdown(reset_decisions):
    from vllm._genesis.dispatcher import (
        dump_structured_boot_summary, log_decision,
    )
    # PN59 → hybrid; PN51 → perf_hotfix; PN58 → structured_output
    log_decision("PN59", True, "opt-in env")
    log_decision("PN51", True, "opt-in env")
    log_decision("PN58", False, "P62 mutually exclusive — conflict")
    out = dump_structured_boot_summary()
    assert "By category:" in out
    assert "hybrid" in out
    assert "perf_hotfix" in out


def test_summary_groups_applied_by_category(reset_decisions):
    from vllm._genesis.dispatcher import (
        dump_structured_boot_summary, log_decision,
    )
    log_decision("PN59", True, "opt-in env")
    log_decision("PN51", True, "opt-in env")
    out = dump_structured_boot_summary()
    assert "APPLIED (2)" in out
    assert "PN59" in out
    assert "PN51" in out


def test_summary_skip_reason_classes(reset_decisions):
    from vllm._genesis.dispatcher import (
        dump_structured_boot_summary, log_decision,
    )
    log_decision(
        "PN14",
        False,
        "upstream_merged — marker safe_page_idx=tl.where(kv_mask, page_idx, 0) present",
    )
    log_decision("PN50", False, "opt-in only — set GENESIS_ENABLE_PN50=1")
    log_decision(
        "PN58",
        False,
        "PN58 SKIPPED — P62 (vllm#36138 broader) is active. Mutually exclusive",
    )
    out = dump_structured_boot_summary()
    # Pretty labels (v7.70+)
    assert "Upstream merged" in out
    assert "Opt-in" in out
    assert "Conflict" in out
    assert "PN14" in out
    assert "PN50" in out
    assert "PN58" in out


def test_summary_dedup_multiple_workers(reset_decisions):
    """Same patch_id logged twice (TP=2 worker boots) → counted once."""
    from vllm._genesis.dispatcher import (
        dump_structured_boot_summary, log_decision,
    )
    log_decision("PN59", True, "opt-in env (config: neutral)")
    log_decision("PN59", True, "opt-in env (config: neutral)")
    log_decision("P107", True, "opt-in env")
    out = dump_structured_boot_summary()
    # Should report 2 unique patches, not 3 raw decisions
    assert "2 total" in out
    assert "2 APPLY" in out


def test_summary_failed_section_highlighted(reset_decisions):
    from vllm._genesis.dispatcher import (
        dump_structured_boot_summary, log_decision,
    )
    log_decision("PN59", True, "opt-in env")
    log_decision("PFAKE", False, "failed: anchor not found in v1/core/sched/scheduler.py")
    out = dump_structured_boot_summary()
    assert "FAILED" in out


def test_log_structured_boot_summary_emits_info_block(reset_decisions, caplog):
    import logging
    from vllm._genesis.dispatcher import log_decision, log_structured_boot_summary
    log_decision("PN59", True, "opt-in env")
    with caplog.at_level(logging.INFO, logger="genesis.dispatcher"):
        log_structured_boot_summary()
    assert any("structured boot summary" in r.message for r in caplog.records)


def test_summary_caps_skip_class_with_overflow_marker(reset_decisions):
    """Long opt-in lists should be capped + show '… and N more' marker.

    Cap was bumped from 8 → 12 in v7.70 for better operator visibility.
    """
    from vllm._genesis.dispatcher import (
        dump_structured_boot_summary, log_decision,
    )
    # Generate 16 env-disabled decisions to exceed 12 cap
    for pid in ["PN50", "PN54", "PN58", "PN29", "PN30", "PN31", "PN32",
                "PN34", "PN38", "P83", "P85", "P75", "P77", "P86", "P79b",
                "P79c"]:
        log_decision(pid, False, "opt-in only — set GENESIS_ENABLE_*=1")
    out = dump_structured_boot_summary()
    # Should cap the env_disabled list to 12 entries + show overflow marker
    assert "and 4 more" in out
