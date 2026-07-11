# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.prefix_cache_analysis import (
    PromptRecord,
    _reusable_tokens_from_chains,
    analyze,
    load_plain_prompt_jsonl,
)
from vllm.v1.core.kv_cache_utils import BlockHash

MODEL_NAME = "facebook/opt-125m"


def test_load_plain_prompt_jsonl(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(
        '{"id": "r1", "prompt": "hello world"}\n\n{"prompt": "no id given"}\n'
    )
    records = load_plain_prompt_jsonl(path)
    # Default id is the 0-indexed *physical line number*, including blank
    # lines that get skipped -- so it stays anchored to the source file
    # (line 2 is blank, line 3 -> index 2) rather than silently renumbering.
    assert [r.request_id for r in records] == ["r1", "2"]
    assert records[1].text == "no id given"


def test_load_plain_prompt_jsonl_missing_field(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text('{"not_prompt": "oops"}\n')
    with pytest.raises(ValueError, match="missing required 'prompt' field"):
        load_plain_prompt_jsonl(path)


def test_analyze_groups_shared_prefix_and_splits_on_divergence():
    # Long enough shared lead-in that it survives BPE tokenization as
    # multiple full blocks at a small block size, then diverges.
    shared = "The quick brown fox jumps over the lazy dog. " * 4
    record_a = PromptRecord("a", shared + "Ending for request A.")
    record_b = PromptRecord("b", shared + "A completely different ending B.")
    record_c = PromptRecord("c", "Unrelated short prompt.")

    report = analyze(
        [record_a, record_b, record_c],
        model=MODEL_NAME,
        block_size=4,
    )

    assert report.num_requests == 3
    assert report.total_prompt_tokens > 0
    # a and b share a real prefix -> at least one shared group, and it must
    # not include the unrelated c.
    assert report.top_prefix_groups, "expected at least one shared-prefix group"
    top = report.top_prefix_groups[0]
    assert set(top.request_ids) == {"a", "b"}
    assert report.estimated_reusable_full_block_tokens > 0


def test_analyze_report_json_roundtrip():
    record_a = PromptRecord("a", "identical prompt text for both requests")
    record_b = PromptRecord("b", "identical prompt text for both requests")

    report = analyze([record_a, record_b], model=MODEL_NAME, block_size=4)
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["num_requests"] == 2
    assert 0.0 <= payload["cacheability_ratio"] <= 1.0
    assert payload["block_size"] == 4


def test_analyze_no_full_blocks_reports_zero_cacheability():
    record = PromptRecord("a", "hi")
    report = analyze([record], model=MODEL_NAME, block_size=16)
    assert report.total_full_block_tokens == 0
    assert report.cacheability_ratio == 0.0
    assert report.top_prefix_groups == []


def test_reusable_tokens_does_not_double_count_overlapping_groups():
    """Regression test: a and b share a 5-block prefix; a, b, and c all
    share a shorter 2-block prefix within that. These are two distinct,
    legitimately-reported groups ({a,b}@depth5 and {a,b,c}@depth2), but
    they overlap on blocks 0-1 for a and b. Reusable tokens must count
    that overlap once, not once per group it appears in.

    Ground truth via a manual walk of the trie (block_size=16):
      block 0 (h0): shared by a,b,c -> saved 2 * 16 = 32
      block 1 (h1): shared by a,b,c -> saved 2 * 16 = 32
      block 2 (h2): shared by a,b   -> saved 1 * 16 = 16
      block 3 (h3): shared by a,b   -> saved 1 * 16 = 16
      block 4 (h4): shared by a,b   -> saved 1 * 16 = 16
      total = 112

    The old (buggy) formula summed len(prefix) * block_size * (n - 1) per
    reported group instead: 5*16*(2-1) [a,b @ depth5] + 2*16*(3-1)
    [a,b,c @ depth2] = 80 + 64 = 144, double-counting blocks 0-1.
    """
    chains = {
        "a": [BlockHash(h) for h in (b"h0", b"h1", b"h2", b"h3", b"h4", b"hA_only")],
        "b": [BlockHash(h) for h in (b"h0", b"h1", b"h2", b"h3", b"h4", b"hB_only")],
        "c": [BlockHash(h) for h in (b"h0", b"h1", b"hC_diverge")],
    }
    assert _reusable_tokens_from_chains(chains, block_size=16) == 112
