# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score
from vllm.pooling_params import LateInteractionParams, PoolingParams
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    build_late_interaction_doc_params,
    build_late_interaction_query_params,
)
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner


def _make_pooling_params(
    late_interaction_params: LateInteractionParams,
) -> PoolingParams:
    return PoolingParams(
        task="token_embed",
        late_interaction_params=late_interaction_params,
    )


def _make_emb(rows, dim=32):
    """Create a deterministic embedding tensor with realistic dimensions."""
    gen = torch.Generator().manual_seed(rows * 1000 + dim)
    return torch.randn(rows, dim, dtype=torch.float32, generator=gen)


def test_postprocess_scores_and_releases_query_cache():
    runner = LateInteractionRunner()
    query_key = "query-0"
    query_emb = _make_emb(8)
    doc_emb = _make_emb(12)

    query_params = _make_pooling_params(
        build_late_interaction_query_params(query_key=query_key, query_uses=1)
    )
    query_output = runner.postprocess_pooler_output(
        raw_pooler_output=[query_emb],
        pooling_params=[query_params],
        req_ids=["query-req"],
        finished_mask=[True],
    )
    assert isinstance(query_output, list)
    assert query_output[0] is not None
    assert query_output[0].shape == torch.Size([])

    doc_params = _make_pooling_params(
        build_late_interaction_doc_params(query_key=query_key)
    )
    doc_output = runner.postprocess_pooler_output(
        raw_pooler_output=[doc_emb],
        pooling_params=[doc_params],
        req_ids=["doc-req"],
        finished_mask=[True],
    )
    assert isinstance(doc_output, list)
    assert doc_output[0] is not None
    assert torch.allclose(doc_output[0], compute_maxsim_score(query_emb, doc_emb))

    with pytest.raises(ValueError, match="query cache miss"):
        runner.postprocess_pooler_output(
            raw_pooler_output=[doc_emb],
            pooling_params=[doc_params],
            req_ids=["doc-req-2"],
            finished_mask=[True],
        )


def test_postprocess_scores_docs_in_batch():
    runner = LateInteractionRunner()
    query_key = "query-batch"
    query_emb = _make_emb(8)
    doc_emb_1 = _make_emb(10)
    doc_emb_2 = _make_emb(14)

    query_params = _make_pooling_params(
        build_late_interaction_query_params(query_key=query_key, query_uses=2)
    )
    runner.postprocess_pooler_output(
        raw_pooler_output=[query_emb],
        pooling_params=[query_params],
        req_ids=["query-req"],
        finished_mask=[True],
    )

    doc_params = _make_pooling_params(
        build_late_interaction_doc_params(query_key=query_key)
    )
    doc_output = runner.postprocess_pooler_output(
        raw_pooler_output=[doc_emb_1, doc_emb_2],
        pooling_params=[doc_params, doc_params],
        req_ids=["doc-req-1", "doc-req-2"],
        finished_mask=[True, True],
    )
    assert isinstance(doc_output, list)
    assert doc_output[0] is not None
    assert doc_output[1] is not None
    assert torch.allclose(doc_output[0], compute_maxsim_score(query_emb, doc_emb_1))
    assert torch.allclose(doc_output[1], compute_maxsim_score(query_emb, doc_emb_2))

    with pytest.raises(ValueError, match="query cache miss"):
        runner.postprocess_pooler_output(
            raw_pooler_output=[doc_emb_1],
            pooling_params=[doc_params],
            req_ids=["doc-req-3"],
            finished_mask=[True],
        )


def test_finished_request_releases_unscored_doc_use():
    runner = LateInteractionRunner()
    query_key = "query-cancel"
    query_emb = _make_emb(8)
    doc_emb = _make_emb(10)

    query_params = _make_pooling_params(
        build_late_interaction_query_params(query_key=query_key, query_uses=1)
    )
    runner.postprocess_pooler_output(
        raw_pooler_output=[query_emb],
        pooling_params=[query_params],
        req_ids=["query-req"],
        finished_mask=[True],
    )

    doc_params = _make_pooling_params(
        build_late_interaction_doc_params(query_key=query_key)
    )
    runner.register_request("doc-req", doc_params)
    runner.on_requests_finished({"doc-req"})

    with pytest.raises(ValueError, match="query cache miss"):
        runner.postprocess_pooler_output(
            raw_pooler_output=[doc_emb],
            pooling_params=[doc_params],
            req_ids=["doc-req-retry"],
            finished_mask=[True],
        )


def test_invalid_query_uses_raises():
    runner = LateInteractionRunner()
    bad_meta = LateInteractionParams(
        mode=LATE_INTERACTION_MODE_CACHE_QUERY,
        query_key="query-bad",
    )
    bad_meta.query_uses = "bad-int"  # type: ignore[assignment]
    bad_query_params = _make_pooling_params(bad_meta)

    with pytest.raises(ValueError, match="must be an integer value"):
        runner.postprocess_pooler_output(
            raw_pooler_output=[_make_emb(8)],
            pooling_params=[bad_query_params],
            req_ids=["query-req"],
            finished_mask=[True],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_score_zerocopy_pairs_matches_reference():
    """N:N scoring (multiple distinct queries) must match a per-pair fp64
    MaxSim oracle and the shared-query kernel path (PR #40337 review:
    single-launch pairwise kernel replaces per-unique-query launches).
    """
    torch.manual_seed(0)
    device = "cuda"
    d = 128

    # Ragged docs scattered in one projected batch tensor.
    doc_lengths = [180, 37, 512, 1, 300, 64, 1030, 256]
    doc_offsets = []
    total = 0
    for ld in doc_lengths:
        doc_offsets.append(total)
        total += ld
    batch = torch.randn(total, d, device=device, dtype=torch.float16)

    # Distinct query per pair except two pairs sharing one query (mixed N:N).
    q_lens = [32, 32, 1030, 16, 96, 5, 512, 32]
    queries = [torch.randn(lq, d, device=device, dtype=torch.float16) for lq in q_lens]
    queries[1] = queries[0]  # shared query across pairs 0 and 1

    scores = LateInteractionRunner._score_zerocopy(
        queries, batch, doc_offsets, doc_lengths
    )

    for i, (q, off, ld) in enumerate(zip(queries, doc_offsets, doc_lengths)):
        doc = batch[off : off + ld]
        ref = (q.double() @ doc.double().T).max(dim=1).values.sum().float()
        assert torch.isfinite(scores[i]), f"pair {i} score not finite"
        torch.testing.assert_close(
            scores[i].to(torch.float32), ref, atol=5e-2, rtol=1e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_score_zerocopy_single_query_path_unchanged():
    """1:N (one distinct query) must still route through the shared-query
    kernel and agree with the fp64 oracle."""
    torch.manual_seed(1)
    device = "cuda"
    d = 128
    doc_lengths = [64, 200, 7, 128]
    doc_offsets = [0, 64, 264, 271]
    batch = torch.randn(sum(doc_lengths), d, device=device, dtype=torch.float16)
    q = torch.randn(32, d, device=device, dtype=torch.float16)
    queries = [q, q, q, q]

    scores = LateInteractionRunner._score_zerocopy(
        queries, batch, doc_offsets, doc_lengths
    )
    for i, (off, ld) in enumerate(zip(doc_offsets, doc_lengths)):
        doc = batch[off : off + ld]
        ref = (q.double() @ doc.double().T).max(dim=1).values.sum().float()
        torch.testing.assert_close(
            scores[i].to(torch.float32), ref, atol=5e-2, rtol=1e-3
        )


# ---------------------------------------------------------------------------
# MRv2 PoolingRunner integration (review request: exercise pool() itself,
# not just the kernel dispatch).
# ---------------------------------------------------------------------------
def _make_mrv2_runner(hidden_dim: int, embed_dim: int = 128):
    """PoolingRunner with a minimal late-interaction pooler around the real
    AllPool + TokenEmbeddingPoolerHead. Bypasses __init__ (which needs a
    loaded model + full VllmConfig) but every field pool() touches is real.
    """
    from types import SimpleNamespace

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.pooler.tokwise.heads import (
        TokenEmbeddingPoolerHead,
    )
    from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
    from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner

    with set_current_vllm_config(VllmConfig()):
        pooling = AllPool()
    projector = torch.nn.Linear(
        hidden_dim, embed_dim, bias=False, dtype=torch.float32, device="cuda"
    )
    projector.weight.requires_grad_(False)

    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    head = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32, projector=projector, activation=_normalize
    )

    class _Pooler:
        def __init__(self):
            self.pooling = pooling
            self.head = head

        def __call__(self, hidden_states, pooling_metadata):
            pooled = self.pooling(hidden_states, pooling_metadata)
            return [
                self.head.forward_chunk(pooled[i], p)
                for i, p in enumerate(pooling_metadata.pooling_params)
            ]

    runner = object.__new__(PoolingRunner)
    runner.model = SimpleNamespace(pooler=_Pooler())
    runner.model_config = SimpleNamespace(pooler_config=None)
    runner.max_num_reqs = 128
    runner.supported_tasks = frozenset({"token_embed"})
    runner.pooling_params = {}
    runner.pooling_states = {}
    runner.prompt_token_ids = {}
    runner.late_interaction_runner = LateInteractionRunner()
    runner._flash_late_interaction_enabled = True
    return runner


def _make_mrv2_step(lens: list[int], hidden_dim: int, seed: int):
    """(input_batch, req_states, hidden_states) shims for one pool() step."""
    from types import SimpleNamespace

    import numpy as np

    gen = torch.Generator(device="cuda").manual_seed(seed)
    lens_np = np.asarray(lens, dtype=np.int32)
    total = int(lens_np.sum())
    hidden_states = torch.randn(
        total, hidden_dim, generator=gen, device="cuda", dtype=torch.bfloat16
    )
    query_start_loc = torch.zeros(len(lens) + 1, device="cuda", dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(
        lens_np, device="cuda", dtype=torch.int32
    ).cumsum(0)
    input_batch = SimpleNamespace(
        num_tokens=total,
        num_reqs=len(lens),
        req_ids=[f"r{seed}_{i}" for i in range(len(lens))],
        idx_mapping_np=np.arange(len(lens)),
        num_scheduled_tokens=lens_np,
        seq_lens_cpu_upper_bound=torch.tensor(lens_np, dtype=torch.int32),
        query_start_loc=query_start_loc,
    )
    req_states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=lens_np.astype(np.int64))
    )
    return input_batch, req_states, hidden_states


def _register_step(runner, input_batch, params_list):
    from vllm.v1.pool.metadata import PoolingStates

    runner.pooling_params = dict(enumerate(params_list))
    runner.pooling_states = {i: PoolingStates() for i in range(len(params_list))}
    for req_id, p in zip(input_batch.req_ids, params_list):
        runner.late_interaction_runner.register_request(req_id, p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_mrv2_pooling_runner_zerocopy_scores_match_reference():
    """Full pool() flow on the MRv2 runner: query-cache step, then a doc
    step that must take the zero-copy path (project_batch called once) and
    produce scores matching an fp64 MaxSim oracle on the projected
    embeddings."""
    hidden_dim = 64
    runner = _make_mrv2_runner(hidden_dim)
    head = runner.model.pooler.head

    calls = {"project_batch": 0}
    orig_project_batch = head.project_batch

    def _counting_project_batch(hidden_states):
        calls["project_batch"] += 1
        return orig_project_batch(hidden_states)

    head.project_batch = _counting_project_batch

    # Step 1: cache one query (normal pooler path, nothing to score).
    q_params = [
        _make_pooling_params(build_late_interaction_query_params("q0", 4))
    ]
    ib, rs, hs = _make_mrv2_step([32], hidden_dim, seed=7)
    _register_step(runner, ib, q_params)
    runner.pool(hs, ib, rs)
    assert calls["project_batch"] == 0
    query_emb = runner.late_interaction_runner._query_cache["q0"]

    # Step 2: score three docs against the cached query via zero-copy.
    doc_lens = [50, 3, 77]
    d_params = [
        _make_pooling_params(build_late_interaction_doc_params("q0"))
        for _ in doc_lens
    ]
    ib, rs, hs = _make_mrv2_step(doc_lens, hidden_dim, seed=8)
    _register_step(runner, ib, d_params)
    out, finished = runner.pool(hs, ib, rs)

    assert calls["project_batch"] == 1, "doc step must take the zero-copy path"
    assert finished == [True] * len(doc_lens)

    # Reference: fp64 MaxSim over the same projected embeddings.
    projected = orig_project_batch(hs)
    start = 0
    for i, ld in enumerate(doc_lens):
        doc = projected[start : start + ld]
        start += ld
        ref = (
            (query_emb.double() @ doc.double().T).max(dim=1).values.sum().float()
        )
        torch.testing.assert_close(
            out[i].to(torch.float32), ref, atol=5e-2, rtol=1e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_mrv2_pooling_runner_no_projection_without_docs_in_batch():
    """A step whose batch has no SCORE_DOC request must not pay the
    full-batch projection, even while doc requests are pending globally
    (review: has_pending_docs alone was too broad a gate)."""
    hidden_dim = 64
    runner = _make_mrv2_runner(hidden_dim)
    head = runner.model.pooler.head

    calls = {"project_batch": 0}
    orig_project_batch = head.project_batch

    def _counting_project_batch(hidden_states):
        calls["project_batch"] += 1
        return orig_project_batch(hidden_states)

    head.project_batch = _counting_project_batch

    # Make a doc request pending globally (registered, not in this batch).
    runner.late_interaction_runner.register_request(
        "other_doc", _make_pooling_params(build_late_interaction_doc_params("qx"))
    )
    assert runner.late_interaction_runner.has_pending_docs

    # This step carries only a query request.
    q_params = [
        _make_pooling_params(build_late_interaction_query_params("q1", 1))
    ]
    ib, rs, hs = _make_mrv2_step([16], hidden_dim, seed=9)
    _register_step(runner, ib, q_params)
    runner.pool(hs, ib, rs)

    assert calls["project_batch"] == 0, (
        "query-only batch must not trigger full-batch projection"
    )
