# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The token->request mapping cache must not survive a query-layout rewrite.

CommonAttentionMetadata.token_to_req_indices() caches its result on the
object. The drafter's multi-step loop rewrites the SAME object in place from
the ragged first-pass layout (one row per token) to one row per request; a
stale cache hit hands draft rows > 0 an EARLIER request's identity in mixed
batches. The SWA window then anchors at that request's seq_len -- one slot
past its last written token -- and the draft attention reads unwritten fp8,
poisoning the entire hidden row with NaN. Pure-decode batches alias the two
layouts (mapping is the identity in both), which kept this invisible in
every pure-decode test and probe."""

import ast
import inspect

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata


def _ragged_cad() -> CommonAttentionMetadata:
    # request 0 carries 3 tokens (a prefill chunk or accepted spec tokens),
    # requests 1-2 carry 1 each: the smallest mixed batch.
    qsl = torch.tensor([0, 3, 4, 5], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.tensor([100, 50, 60], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=100,
        block_table_tensor=torch.zeros((3, 4), dtype=torch.int32),
        slot_mapping=torch.zeros(5, dtype=torch.int64),
        causal=True,
    )


def test_layout_rewrite_with_invalidation_yields_identity_mapping():
    cad = _ragged_cad()
    buf = torch.zeros(16, dtype=torch.int32)
    ragged = cad.token_to_req_indices(buf)
    assert ragged.tolist() == [0, 0, 0, 1, 2]

    # The drafter loop-entry rewrite, including the cache invalidation.
    batch_size = 3
    cad.num_actual_tokens = batch_size
    cad.max_query_len = 1
    cad.query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
    cad.query_start_loc_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
    cad._token_to_req_indices_cache = None
    cad._num_computed_tokens_cache = None

    per_row = cad.token_to_req_indices(torch.zeros(16, dtype=torch.int32))
    assert per_row.tolist() == [0, 1, 2], (
        "row r must map to request r after the per-row rewrite; a stale "
        "ragged mapping gives rows > 0 an earlier request's identity"
    )


def test_proposer_loop_entry_invalidates_the_mapping_cache():
    """AST pin: the drafter's in-place layout rewrite must clear both caches."""
    from vllm.v1.spec_decode import llm_base_proposer

    src = inspect.getsource(llm_base_proposer)
    tree = ast.parse(src)
    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr in (
                        "_token_to_req_indices_cache",
                        "_num_computed_tokens_cache",
                    )
                    and isinstance(t.value, ast.Attribute)
                    and t.value.attr == "common_attn_metadata"
                    == False  # value is Name for local var; handle both
                ):
                    pass
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr in (
                    "_token_to_req_indices_cache",
                    "_num_computed_tokens_cache",
                ):
                    hits.add(t.attr)
    assert hits == {
        "_token_to_req_indices_cache",
        "_num_computed_tokens_cache",
    }, (
        f"proposer no longer invalidates the layout-derived caches ({hits}); "
        "the ragged->per-row rewrite will serve stale request identities"
    )


def test_extend_all_queries_does_not_carry_the_cache():
    from vllm.v1.spec_decode.utils import extend_all_queries_by_N

    cad = _ragged_cad()
    cad.token_to_req_indices(torch.zeros(16, dtype=torch.int32))
    assert cad._token_to_req_indices_cache is not None
    new_cad = extend_all_queries_by_N(
        cad,
        N=1,
        arange=torch.arange(8, dtype=torch.int32),
        new_slot_mapping=torch.zeros(8, dtype=torch.int64),
    )
    assert new_cad._token_to_req_indices_cache is None, (
        "replace() carried the old layout's token->request cache into the "
        "extended layout"
    )
