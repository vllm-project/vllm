# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level tests for the Model Runner V2 suffix decoding speculator."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

from vllm.v1.worker.gpu.spec_decode.suffix.speculator import (  # noqa: E402
    SuffixSpeculator,
)

DEVICE = torch.device("cuda")


def _local_draft(hist, k, depth):
    """Continuation of the longest, rightmost earlier occurrence of a suffix
    of ``hist`` within ``hist`` (empty if no token matches)."""
    best_len, best_pos = 0, 0
    for i in range(1, len(hist)):
        m = 0
        while m < depth and i - 1 - m >= 0 and hist[i - 1 - m] == hist[-1 - m]:
            m += 1
        if m > 0 and (m, i) >= (best_len, best_pos):
            best_len, best_pos = m, i
    return [int(t) for t in hist[best_pos : best_pos + k]] if best_len else []


def _follows_in_corpus(docs, hist, draft, n=2):
    """Draft continues some occurrence of hist's last n tokens in a doc."""
    tail = [int(t) for t in hist[-n:]]
    for doc in docs:
        for i in range(n, len(doc)):
            if doc[i - n : i] == tail and doc[i : i + len(draft)] == draft:
                return True
    return False


def _make(k, depth, max_reqs, max_len, corpus_tokens, adaptive=False):
    spec = SimpleNamespace(
        num_speculative_tokens=k,
        suffix_decoding_max_tree_depth=depth,
        suffix_decoding_corpus_tokens=corpus_tokens,
        enable_adaptive_verification=adaptive,
        suffix_decoding_max_cached_requests=10000,
    )
    cfg = SimpleNamespace(
        speculative_config=spec,
        scheduler_config=SimpleNamespace(max_num_seqs=max_reqs),
        model_config=SimpleNamespace(max_model_len=max_len),
    )

    def buf(*shape):
        return SimpleNamespace(
            gpu=torch.zeros(*shape, dtype=torch.int32, device=DEVICE)
        )

    req_states = SimpleNamespace(
        all_token_ids=buf(max_reqs, max_len),
        total_len=buf(max_reqs),
        prompt_len=SimpleNamespace(np=np.zeros(max_reqs, dtype=np.int32)),
        req_id_to_index={},
    )
    return SuffixSpeculator(cfg, DEVICE, req_states), req_states


def _set(req_states, slot, prompt, output):
    toks = [int(t) for t in prompt] + [int(t) for t in output]
    req_states.all_token_ids.gpu[slot, : len(toks)] = torch.tensor(toks)
    req_states.total_len.gpu[slot] = len(toks)
    req_states.prompt_len.np[slot] = len(prompt)


def _finish(spec, req_states, slot_by_id):
    req_states.req_id_to_index = dict(slot_by_id)
    spec.on_requests_finished(list(slot_by_id))


def _propose(spec, req_states, slots, num_sampled=None, num_rejected=None):
    n = len(slots)
    max_reqs = req_states.total_len.gpu.shape[0]
    last_sampled = torch.zeros(max_reqs, 1, dtype=torch.int64, device=DEVICE)
    for s in slots:
        last = int(req_states.total_len.gpu[s]) - 1
        last_sampled[s, 0] = req_states.all_token_ids.gpu[s, last]
    batch = SimpleNamespace(
        num_reqs=n,
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
    )
    if num_sampled is None:
        num_sampled = [1] * n
    if num_rejected is None:
        num_rejected = [0] * n
    drafts = spec.propose(
        batch,
        None,
        None,
        None,
        None,
        torch.tensor(num_sampled, dtype=torch.int32, device=DEVICE),
        torch.tensor(num_rejected, dtype=torch.int32, device=DEVICE),
        last_sampled,
        None,
        None,
        None,
    )
    return drafts.cpu().numpy(), spec.num_valid[:n].cpu().numpy()


@pytest.mark.parametrize("vocab", [4, 16, 1000])
def test_local_matches_reference(vocab):
    k, depth, max_reqs, max_len = 6, 8, 8, 300
    rng = np.random.default_rng(vocab)
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=0)
    for _ in range(4):
        hists = []
        for s in range(max_reqs):
            plen, olen = int(rng.integers(1, 60)), int(rng.integers(1, 200))
            toks = rng.integers(0, vocab, plen + olen)
            _set(rs, s, toks[:plen], toks[plen:])
            hists.append(toks)
        drafts, num_valid = _propose(spec, rs, list(range(max_reqs)))
        for s, hist in enumerate(hists):
            ref = _local_draft(hist, k, depth)
            assert num_valid[s] == len(ref)
            assert drafts[s, : len(ref)].tolist() == ref
            # Filler slots carry the last sampled token.
            assert (drafts[s, len(ref) :] == hist[-1]).all()


@pytest.mark.parametrize("vocab", [16, 1000])
def test_corpus_drafts_are_real_continuations(vocab):
    """A draft is the exact local draft, or follows the request's trailing
    n-gram somewhere in a finished response, and is at least as long a
    match as the local one."""
    k, depth, max_reqs, max_len = 6, 8, 8, 300
    rng = np.random.default_rng(vocab)
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1 << 15)
    docs: list[list[int]] = []
    num_from_corpus = 0
    for rnd in range(6):
        hists = []
        for s in range(max_reqs):
            plen, olen = int(rng.integers(1, 60)), int(rng.integers(1, 200))
            toks = rng.integers(0, vocab, plen + olen)
            _set(rs, s, toks[:plen], toks[plen:])
            hists.append(toks)
        drafts, num_valid = _propose(spec, rs, list(range(max_reqs)))
        for s, hist in enumerate(hists):
            got = drafts[s, : num_valid[s]].tolist()
            if got == _local_draft(hist, k, depth):
                continue
            num_from_corpus += 1
            assert _follows_in_corpus(docs, hist, got), (rnd, s)
        ids = {f"r{rnd}-{s}": s for s in range(max_reqs)}
        _finish(spec, rs, ids)
        for rid in sorted(ids):
            plen = int(rs.prompt_len.np[ids[rid]])
            docs.append(hists[ids[rid]][max(plen - 3, 0) :].tolist())
    if vocab == 16:
        # Small vocabularies make cross-request matches common.
        assert num_from_corpus > 0


def test_corpus_indexes_response_start():
    """The first response token is drafted from the shared prompt tail."""
    k, depth, max_reqs, max_len = 3, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    header = [90, 91, 92]
    _set(rs, 0, [1, 2] + header, [7, 8, 9])
    _finish(spec, rs, {"done": 0})
    # A different prompt with the same template tail, nothing generated yet
    # beyond the prefill token.
    _set(rs, 1, [60, 61] + header, [7])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 2
    assert drafts[0].tolist() == [8, 9, 7]


def test_corpus_draft_from_finished_response():
    k, depth, max_reqs, max_len = 5, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [1, 2, 3], [7, 8, 9, 10, 11, 12, 13, 14])
    _finish(spec, rs, {"done": 0})
    # A new request whose output starts like the finished response.
    _set(rs, 1, [50, 51], [7, 8, 9])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 5
    assert drafts[0].tolist() == [10, 11, 12, 13, 14]


def test_longer_corpus_match_wins_over_local():
    k, depth, max_reqs, max_len = 3, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [], [1, 2, 3, 4, 70, 71, 72])
    _finish(spec, rs, {"done": 0})
    # Locally only "4" recurs (continuing with 99); the corpus matches
    # "1 2 3 4".
    _set(rs, 1, [4, 99, 98], [1, 2, 3, 4])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 3
    assert drafts[0].tolist() == [70, 71, 72]


def test_draft_stops_at_document_end():
    k, depth, max_reqs, max_len = 6, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [], [5, 6, 7, 8])
    _finish(spec, rs, {"a": 0})
    _set(rs, 0, [], [40, 41, 42])
    _finish(spec, rs, {"b": 0})
    _set(rs, 1, [], [5, 6])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 2
    assert drafts[0].tolist() == [7, 8, 6, 6, 6, 6]


def test_newest_document_wins_ties():
    k, depth, max_reqs, max_len = 2, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    for i, cont in enumerate(([30, 31], [40, 41], [50, 51])):
        _set(rs, 0, [], [5, 6] + cont)
        _finish(spec, rs, {f"d{i}": 0})
    _set(rs, 1, [], [5, 6])
    drafts, _ = _propose(spec, rs, [1])
    assert drafts[0].tolist() == [50, 51]


def test_most_frequent_continuation_wins():
    k, depth, max_reqs, max_len = 2, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    # Two older responses continue "5 6" with "30 31", the newest with "40 41".
    for i, cont in enumerate(([30, 31], [30, 31], [40, 41])):
        _set(rs, 0, [], [5, 6] + cont)
        _finish(spec, rs, {f"d{i}": 0})
    _set(rs, 1, [], [5, 6])
    drafts, _ = _propose(spec, rs, [1])
    assert drafts[0].tolist() == [30, 31]


def test_vote_prefers_longer_supported_match():
    """A longer match is used when its continuation is as well supported."""
    k, depth, max_reqs, max_len = 2, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [], [9, 9, 5, 6, 30, 31])
    _finish(spec, rs, {"a": 0})
    _set(rs, 0, [], [1, 2, 3, 4, 5, 6, 70, 71])
    _finish(spec, rs, {"b": 0})
    _set(rs, 1, [], [1, 2, 3, 4, 5, 6])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 2
    assert drafts[0].tolist() == [70, 71]


def test_no_draft_without_sampled_token():
    k, depth, max_reqs, max_len = 3, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=0)
    _set(rs, 0, [], [1, 2, 3, 1, 2])
    # A partial prefill samples no token and must not draft.
    drafts, num_valid = _propose(spec, rs, [0], num_sampled=[0])
    assert num_valid[0] == 0
    drafts, num_valid = _propose(spec, rs, [0])
    assert drafts[0].tolist() == [3, 1, 2]


def test_corpus_wraps_without_straddling():
    k, depth, max_reqs, max_len = 4, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=50)
    for rnd in range(5):
        _set(rs, 0, [100], list(range(rnd * 10, rnd * 10 + 20)))
        _finish(spec, rs, {f"r{rnd}": 0})
    corpus = spec.corpus.tolist()
    # Each document (1 prompt token + 20 response tokens + separator) takes
    # 22 slots: two fit, and every third wraps to offset 0, so documents 4
    # and 3 survive at offsets 0 and 22.
    assert int(spec.corpus_head) == 22
    assert int(spec.doc_seq) == 5
    assert corpus[:22] == [100] + list(range(40, 60)) + [-1]
    assert corpus[22:44] == [100] + list(range(30, 50)) + [-1]
    # Stale index entries for overwritten documents are never drafted.
    _set(rs, 1, [], [12, 13])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 0


def test_ingestion_is_deterministic():
    """The same finished set, reported in different orders, builds identical
    corpus and index state (what keeps TP ranks consistent)."""
    k, depth, max_reqs, max_len = 4, 8, 8, 256
    rng = np.random.default_rng(0)
    toks = [rng.integers(0, 50, 200) for _ in range(max_reqs)]
    states = []
    for order in (list(range(max_reqs)), list(reversed(range(max_reqs)))):
        spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1 << 14)
        for s in range(max_reqs):
            _set(rs, s, toks[s][:20], toks[s][20:])
        ids = [f"r{s}" for s in order]
        rs.req_id_to_index = {rid: int(rid[1:]) for rid in ids}
        spec.on_requests_finished(ids)
        states.append((spec.corpus.clone(), spec.table.clone()))
    assert torch.equal(states[0][0], states[1][0])
    assert torch.equal(states[0][1], states[1][1])


def test_adaptive_verification_confidences():
    k, depth, max_reqs, max_len = 4, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=0, adaptive=True)
    _set(rs, 0, [], [1, 2, 3, 1, 2])
    drafts, num_valid = _propose(spec, rs, [0])
    assert num_valid[0] == 3
    conf = spec.draft_token_confidence_probs[0].tolist()
    # No observations yet: smoothed 1/2 for real drafts, 0 for the filler.
    assert conf == pytest.approx([0.5, 0.5, 0.5, 0.0])

    # The target accepted 2 of the 3 real tokens and rejected the third.
    _set(rs, 0, [], [1, 2, 3, 1, 2, 3, 1, 9])
    _propose(spec, rs, [0], num_sampled=[3], num_rejected=[1])
    stats = spec.accept_stats.sum(dim=0).tolist()  # [K, (accepted, observed)]
    assert stats[:3] == [[1, 1], [1, 1], [0, 1]]
    assert stats[3] == [0, 0]
