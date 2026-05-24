# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.migrate — pin-bump migration runbook
generator.

`genesis migrate-vllm` analyzes a target upstream-vllm checkout and
produces a per-patch migration plan:

  - Anchors that still match → patch ports cleanly, no action
  - Anchors that drifted    → operator must re-derive against new source
  - Upstream PRs now merged → patch self-retires (auto, no action)
  - Files moved/renamed     → wiring needs file-path update
  - JartX-class refactors   → multiple patches in the family need re-review

The tool is read-only against the upstream checkout. Output is JSON +
human-readable runbook (markdown).

Test strategy:
  - Use synthetic vllm tree fixtures (small files mirroring the patch
    target shape) — no need for a real 1GB+ upstream clone in CI
  - Verify each migration verdict (clean / anchor-drift / upstream-merged
    / file-moved) against synthetic fixtures
"""
from __future__ import annotations


import pytest


@pytest.fixture
def synthetic_vllm_tree(tmp_path):
    """Build a synthetic vllm/ directory mimicking the file layout for
    text-patch targets we care about (PN14, PN13)."""
    vllm = tmp_path / "vllm"
    (vllm / "v1" / "attention" / "ops").mkdir(parents=True)
    (vllm / "compilation").mkdir(parents=True)

    # PN14 target — pristine version (anchor still matches)
    (vllm / "v1" / "attention" / "ops" / "triton_turboquant_decode.py").write_text(
        '''
@triton.jit
def _tq_decode_stage1(...):
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = ...
'''
    )

    # PN13 target — upstream MERGED variant (var-arg lambda already there)
    (vllm / "compilation" / "cuda_graph.py").write_text(
        '''
class CUDAGraphWrapper:
    def __call__(self):
        with stack:
            stack.enter_context(
                patch("gc.collect", lambda *args, **kwargs: None)
            )
            stack.enter_context(
                patch(
                    "torch.accelerator.empty_cache",
                    lambda *args, **kwargs: None,
                )
            )
'''
    )

    return tmp_path


class TestAnchorChecker:
    def test_anchor_present_returns_clean(self, synthetic_vllm_tree):
        from vllm._genesis.compat.migrate import check_patch_against_upstream
        verdict = check_patch_against_upstream(
            "PN14", upstream_root=synthetic_vllm_tree,
        )
        assert verdict["status"] in ("clean", "would_apply", "anchor_present")
        assert verdict["patch_id"] == "PN14"

    def test_anchor_drift_detected(self, synthetic_vllm_tree):
        # Mutate the PN14 target file to break the anchor
        target = (synthetic_vllm_tree / "vllm" / "v1" / "attention" / "ops"
                  / "triton_turboquant_decode.py")
        target.write_text(target.read_text().replace(
            "page_idx = kv_offs // BLOCK_SIZE",
            "computed_idx = kv_offs // BLOCK_SIZE",
        ))
        from vllm._genesis.compat.migrate import check_patch_against_upstream
        verdict = check_patch_against_upstream(
            "PN14", upstream_root=synthetic_vllm_tree,
        )
        assert verdict["status"] == "anchor_drift"
        assert "anchor" in verdict["message"].lower()

    def test_upstream_merged_marker_present(self, synthetic_vllm_tree):
        """PN13 fixture has var-arg lambdas already in upstream → patch
        will self-retire on merge. The migrate tool should detect this."""
        from vllm._genesis.compat.migrate import check_patch_against_upstream
        verdict = check_patch_against_upstream(
            "PN13", upstream_root=synthetic_vllm_tree,
        )
        assert verdict["status"] == "upstream_merged"
        assert verdict["patch_id"] == "PN13"

    def test_target_file_missing(self, synthetic_vllm_tree):
        # Remove the target file
        target = (synthetic_vllm_tree / "vllm" / "v1" / "attention" / "ops"
                  / "triton_turboquant_decode.py")
        target.unlink()
        from vllm._genesis.compat.migrate import check_patch_against_upstream
        verdict = check_patch_against_upstream(
            "PN14", upstream_root=synthetic_vllm_tree,
        )
        assert verdict["status"] in ("file_missing", "anchor_drift")

    def test_unknown_patch_id(self, synthetic_vllm_tree):
        from vllm._genesis.compat.migrate import check_patch_against_upstream
        verdict = check_patch_against_upstream(
            "PN_NONEXISTENT", upstream_root=synthetic_vllm_tree,
        )
        assert verdict["status"] in ("unknown_patch", "error")


class TestRunbookGenerator:
    def test_generate_runbook_returns_dict(self, synthetic_vllm_tree):
        from vllm._genesis.compat.migrate import generate_runbook
        runbook = generate_runbook(
            upstream_root=synthetic_vllm_tree,
            patch_ids=["PN14", "PN13"],
        )
        assert isinstance(runbook, dict)
        assert "summary" in runbook
        assert "patches" in runbook
        assert len(runbook["patches"]) == 2

    def test_runbook_buckets_by_status(self, synthetic_vllm_tree):
        from vllm._genesis.compat.migrate import generate_runbook
        runbook = generate_runbook(
            upstream_root=synthetic_vllm_tree,
            patch_ids=["PN14", "PN13"],
        )
        assert "by_status" in runbook["summary"]
        statuses = runbook["summary"]["by_status"]
        # Both PN14 (clean) and PN13 (merged) should be in their buckets
        assert sum(statuses.values()) == 2

    def test_runbook_includes_action_items(self, synthetic_vllm_tree):
        """Runbook must produce actionable guidance for each patch."""
        from vllm._genesis.compat.migrate import generate_runbook
        runbook = generate_runbook(
            upstream_root=synthetic_vllm_tree,
            patch_ids=["PN14", "PN13"],
        )
        for p in runbook["patches"]:
            assert "action" in p, f"patch {p.get('patch_id')!r} missing action"


class TestMarkdownFormatter:
    def test_format_runbook_produces_lines(self, synthetic_vllm_tree):
        from vllm._genesis.compat.migrate import (
            format_runbook_md, generate_runbook,
        )
        runbook = generate_runbook(
            upstream_root=synthetic_vllm_tree,
            patch_ids=["PN14", "PN13"],
        )
        md = format_runbook_md(runbook)
        assert isinstance(md, str)
        assert "Genesis migration runbook" in md
        assert "PN14" in md
        assert "PN13" in md

    def test_format_includes_action_section(self, synthetic_vllm_tree):
        from vllm._genesis.compat.migrate import (
            format_runbook_md, generate_runbook,
        )
        runbook = generate_runbook(
            upstream_root=synthetic_vllm_tree,
            patch_ids=["PN14"],
        )
        md = format_runbook_md(runbook)
        # Action section should appear in the output
        assert "Action" in md or "action" in md


class TestCLI:
    def test_cli_returns_int(self, synthetic_vllm_tree):
        from vllm._genesis.compat.migrate import main
        rc = main([str(synthetic_vllm_tree), "--patches", "PN14"])
        assert isinstance(rc, int)

    def test_cli_unknown_path_returns_nonzero(self):
        from vllm._genesis.compat.migrate import main
        rc = main(["/nonexistent/path"])
        assert rc != 0

    def test_cli_json_output(self, synthetic_vllm_tree, capsys):
        from vllm._genesis.compat.migrate import main
        import json
        main([str(synthetic_vllm_tree), "--patches", "PN14", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "patches" in parsed
