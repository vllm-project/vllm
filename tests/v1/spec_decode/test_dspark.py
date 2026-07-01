# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.platforms import current_platform
from vllm.models.deepseek_v4.nvidia.dspark import (
    _apply_rope_gptj_last,
    _rmsnorm_no_weight,
)
from vllm.models.deepseek_v4.nvidia.dspark_triton import (
    dspark_context_kv_store,
    dspark_inv_rope_bf16_layout,
    dspark_markov_greedy_argmax,
    dspark_qkv_postprocess,
)
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dspark import DSparkProposer
from vllm.v1.spec_decode.dspark_sampling import (
    sample_dspark_markov_block,
    sample_dspark_markov_block_fused,
)

DEVICE_TYPE = current_platform.device_type


def _sampling_metadata(all_greedy: bool, batch_size: int) -> SamplingMetadata:
    temperature = None
    if not all_greedy:
        temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE)
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([], device=DEVICE_TYPE),
        presence_penalties=torch.tensor([], device=DEVICE_TYPE),
        repetition_penalties=torch.tensor([], device=DEVICE_TYPE),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=[],
    )


def test_dspark_markov_sampling_chains_sampled_tokens():
    base_logits = torch.zeros(1, 3, 6, device=DEVICE_TYPE)
    first_prev_token_ids = torch.tensor([1], device=DEVICE_TYPE)

    def apply_markov_bias(
        logits: torch.Tensor, prev_token_ids: torch.Tensor, step_idx: int
    ) -> torch.Tensor:
        del step_idx
        biased = logits.clone()
        rows = torch.arange(logits.shape[0], device=logits.device)
        biased[rows, prev_token_ids + 1] = 10
        return biased

    tokens, draft_probs = sample_dspark_markov_block(
        base_logits,
        first_prev_token_ids,
        apply_markov_bias,
        _sampling_metadata(all_greedy=True, batch_size=1),
        return_probs=False,
    )

    assert draft_probs is None
    assert tokens.tolist() == [[2, 3, 4]]


def test_dspark_markov_sampling_returns_corrected_draft_probs():
    base_logits = torch.zeros(1, 2, 5, device=DEVICE_TYPE)
    first_prev_token_ids = torch.tensor([0], device=DEVICE_TYPE)

    def apply_markov_bias(
        logits: torch.Tensor, prev_token_ids: torch.Tensor, step_idx: int
    ) -> torch.Tensor:
        del step_idx
        biased = logits.clone()
        biased[:, 1] = 30.0
        rows = torch.arange(logits.shape[0], device=logits.device)
        biased[rows, prev_token_ids] = -30.0
        return biased

    tokens, draft_probs = sample_dspark_markov_block(
        base_logits,
        first_prev_token_ids,
        apply_markov_bias,
        _sampling_metadata(all_greedy=False, batch_size=1),
        return_probs=True,
    )

    assert tokens.shape == (1, 2)
    assert draft_probs is not None
    expected_step0 = apply_markov_bias(base_logits[:, 0, :], first_prev_token_ids, 0)
    assert torch.allclose(draft_probs[:, 0, :], expected_step0.softmax(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_fused_markov_sampler_matches_reference_probs():
    torch.manual_seed(0)
    batch, block, vocab, rank = 2, 5, 4096, 64
    base_logits = torch.randn(batch, block, vocab, device=DEVICE_TYPE) * 3.0
    first_prev = torch.tensor([7, 100], device=DEVICE_TYPE)
    w1 = torch.randn(vocab, rank, device=DEVICE_TYPE) * 0.05
    w2 = torch.randn(vocab, rank, device=DEVICE_TYPE) * 0.05

    def apply_markov_bias(logits, prev_token_ids, step_idx):
        del step_idx
        return logits + w1[prev_token_ids.long()] @ w2.t()

    tokens, draft_probs = sample_dspark_markov_block_fused(
        base_logits,
        first_prev,
        apply_markov_bias,
        _sampling_metadata(all_greedy=False, batch_size=batch),
    )

    assert tokens.shape == (batch, block)
    assert draft_probs is not None and draft_probs.shape == (batch, block, vocab)
    # Reconstruct the sequential step logits from the sampled tokens and verify
    # every step's draft_probs is exactly the softmax of what was fed, and each
    # sampled token has positive probability under its own draft distribution.
    rows = torch.arange(batch, device=DEVICE_TYPE)
    prev = first_prev.long()
    for step in range(block):
        step_logits = apply_markov_bias(base_logits[:, step, :], prev, step)
        assert torch.allclose(
            draft_probs[:, step, :], step_logits.softmax(dim=-1), atol=1e-5
        )
        assert (draft_probs[rows, step, tokens[:, step]] > 0).all()
        prev = tokens[:, step]
    assert torch.allclose(
        draft_probs.sum(dim=-1),
        torch.ones(batch, block, device=DEVICE_TYPE),
        atol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_fused_markov_sampler_applies_top_k_top_p():
    import dataclasses

    torch.manual_seed(0)
    batch, block, vocab, rank = 2, 4, 4096, 32
    base_logits = torch.randn(batch, block, vocab, device=DEVICE_TYPE) * 2.0
    first_prev = torch.tensor([7, 100], device=DEVICE_TYPE)
    w1 = torch.randn(vocab, rank, device=DEVICE_TYPE) * 0.05
    w2 = torch.randn(vocab, rank, device=DEVICE_TYPE) * 0.05
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32, device=DEVICE_TYPE)
    top_k = torch.tensor([1, vocab], dtype=torch.int32, device=DEVICE_TYPE)
    top_p = torch.tensor([1.0, 0.75], dtype=torch.float32, device=DEVICE_TYPE)
    metadata = dataclasses.replace(
        _sampling_metadata(all_greedy=False, batch_size=batch),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    def apply_markov_bias(logits, prev_token_ids, step_idx):
        del step_idx
        return logits + w1[prev_token_ids.long()] @ w2.t()

    tokens, draft_probs = sample_dspark_markov_block_fused(
        base_logits,
        first_prev,
        apply_markov_bias,
        metadata,
    )

    assert tokens.shape == (batch, block)
    assert draft_probs is not None and draft_probs.shape == (batch, block, vocab)
    rows = torch.arange(batch, device=DEVICE_TYPE)
    prev = first_prev.long()
    for step in range(block):
        step_logits = apply_markov_bias(base_logits[:, step, :], prev, step)
        expected_logits = step_logits.to(torch.float32).clone()
        expected_logits.div_(temperature.view(-1, 1))
        expected_logits = apply_top_k_top_p(expected_logits, top_k, top_p)
        expected_probs = expected_logits.softmax(dim=-1, dtype=torch.float32)
        assert torch.allclose(draft_probs[:, step, :], expected_probs, atol=1e-5)
        assert (draft_probs[rows, step, tokens[:, step]] > 0).all()
        prev = tokens[:, step]

    assert torch.allclose(
        draft_probs.sum(dim=-1),
        torch.ones(batch, block, device=DEVICE_TYPE),
        atol=1e-4,
    )


def test_dspark_fused_markov_seed_is_cross_rank_deterministic():
    # The fused sampler draws its Gumbel noise from an in-kernel Philox seed
    # counter. Under TP>1 the draft sampler runs redundantly on every rank and
    # all ranks MUST draw identical draft tokens, which requires the seed
    # sequence to be identical across independently-initialized processes. Guard
    # the property that makes that true: a fixed start + lockstep increment, so
    # the same call index always yields the same seed (never per-process
    # entropy).
    import importlib

    from vllm.v1.spec_decode import dspark_sampling as ds

    # Module init must be a fixed constant (identical on every rank), not
    # per-process entropy: reloading twice must land on the same start value.
    importlib.reload(ds)
    start_a = ds._FUSED_MARKOV_SEED_COUNTER
    importlib.reload(ds)
    start_b = ds._FUSED_MARKOV_SEED_COUNTER
    assert start_a == start_b, "fused Markov seed must start identical across ranks"

    # From a shared start, the seed sequence is deterministic and distinct.
    ds._FUSED_MARKOV_SEED_COUNTER = 123
    seq_a = [ds._next_fused_markov_seed() for _ in range(10)]
    ds._FUSED_MARKOV_SEED_COUNTER = 123
    seq_b = [ds._next_fused_markov_seed() for _ in range(10)]
    assert seq_a == seq_b, "fused Markov seed must advance deterministically"
    assert len(set(seq_a)) == len(seq_a), "seeds within a run must be distinct"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_fused_markov_sampler_falls_back_when_seeded():
    import dataclasses

    batch, block, vocab = 1, 3, 512
    base_logits = torch.randn(batch, block, vocab, device=DEVICE_TYPE)
    first_prev = torch.tensor([2], device=DEVICE_TYPE)

    def apply_markov_bias(logits, prev_token_ids, step_idx):
        del prev_token_ids, step_idx
        return logits.clone()

    def seeded_meta(seed):
        gen = torch.Generator(device=DEVICE_TYPE)
        gen.manual_seed(seed)
        return dataclasses.replace(
            _sampling_metadata(all_greedy=False, batch_size=batch),
            all_random=False,
            generators={0: gen},
        )

    # With a per-request generator, the fused path must fall back to the exact
    # reference sampler, so a fixed seed reproduces the reference bit-for-bit.
    fused_tokens, fused_probs = sample_dspark_markov_block_fused(
        base_logits, first_prev, apply_markov_bias, seeded_meta(1234)
    )
    ref_tokens, ref_probs = sample_dspark_markov_block(
        base_logits,
        first_prev,
        apply_markov_bias,
        seeded_meta(1234),
        return_probs=True,
    )
    assert torch.equal(fused_tokens, ref_tokens)
    assert torch.allclose(fused_probs, ref_probs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_fused_markov_sampler_falls_back_for_fp64_gumbel():
    batch, block, vocab = 1, 3, 512
    base_logits = torch.randn(batch, block, vocab, device=DEVICE_TYPE)
    first_prev = torch.tensor([2], device=DEVICE_TYPE)
    metadata = _sampling_metadata(all_greedy=False, batch_size=batch)

    def apply_markov_bias(logits, prev_token_ids, step_idx):
        del prev_token_ids, step_idx
        return logits.clone()

    torch.manual_seed(1234)
    fused_tokens, fused_probs = sample_dspark_markov_block_fused(
        base_logits,
        first_prev,
        apply_markov_bias,
        metadata,
        use_fp64_gumbel=True,
    )
    torch.manual_seed(1234)
    ref_tokens, ref_probs = sample_dspark_markov_block(
        base_logits,
        first_prev,
        apply_markov_bias,
        metadata,
        return_probs=True,
        use_fp64_gumbel=True,
    )
    assert torch.equal(fused_tokens, ref_tokens)
    assert torch.allclose(fused_probs, ref_probs)


def test_dspark_speculative_config_predicates_and_hash_include_sampler():
    cfg = object.__new__(SpeculativeConfig)
    cfg.method = "dspark"
    cfg.draft_model_config = None
    for name in (
        "dspark_materialized_attention",
        "dspark_triton_attention",
        "dspark_triton_qkv_postprocess",
        "dspark_triton_context_kv_store",
        "dspark_markov_inplace_add",
        "dspark_fused_markov_sampler",
        "dspark_forward_cudagraph",
        "dspark_forward_cudagraph_allow_tp",
        "dspark_fused_o_proj_quant",
        "dspark_fused_shared_experts_quant",
    ):
        setattr(cfg, name, False)

    assert cfg.use_dspark()
    assert not cfg.use_eagle()
    hash_without_fused_sampler = cfg.compute_hash()
    cfg.dspark_fused_markov_sampler = True
    assert cfg.compute_hash() != hash_without_fused_sampler


def test_dspark_fast_path_defaults_enabled():
    for name in (
        "dspark_materialized_attention",
        "dspark_triton_attention",
        "dspark_triton_qkv_postprocess",
        "dspark_triton_context_kv_store",
        "dspark_markov_inplace_add",
        "dspark_fused_markov_sampler",
        "dspark_fused_o_proj_quant",
        "dspark_fused_shared_experts_quant",
    ):
        assert getattr(SpeculativeConfig, name) is True


def test_dspark_shares_target_embedding_and_lm_head(monkeypatch):
    class DummyPPGroup:
        world_size = 1

    class TargetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(8, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    class DraftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)

    from vllm.v1.spec_decode import dspark as dspark_proposer

    monkeypatch.setattr(dspark_proposer, "get_pp_group", lambda: DummyPPGroup())
    target = TargetModel()
    draft = DraftModel()
    old_embed = draft.embed_tokens
    old_head = draft.head

    proposer = object.__new__(DSparkProposer)
    proposer.model = draft

    proposer._maybe_share_embeddings(target)
    proposer._maybe_share_lm_head(target)

    assert proposer.model.embed_tokens is target.model.embed_tokens
    assert proposer.model.head is target.lm_head
    assert proposer.model.embed_tokens is not old_embed
    assert proposer.model.head is not old_head


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_triton_qkv_postprocess_matches_reference():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    q = torch.randn(4, 3, 512, dtype=dtype, device=device).contiguous()
    kv = torch.randn(4, 512, dtype=dtype, device=device).contiguous()
    positions = torch.tensor([0, 3, 7, 11], dtype=torch.int64, device=device)
    angles = torch.randn(16, 32, dtype=torch.float32, device=device)
    cos_sin_cache = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    eps = 1e-6

    q_ref = _rmsnorm_no_weight(q, eps).to(dtype)
    q_ref = _apply_rope_gptj_last(q_ref, positions, cos_sin_cache)
    kv_ref = _apply_rope_gptj_last(kv, positions, cos_sin_cache)

    q_out, kv_out = dspark_qkv_postprocess(q, kv, positions, cos_sin_cache, eps)

    assert torch.allclose(q_out.float(), q_ref.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(kv_out.float(), kv_ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_triton_inv_rope_bf16_layout_matches_reference():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    tokens = 5
    n_groups = 2
    heads_per_group = 3
    head_dim = 512
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    o = torch.randn(
        tokens,
        n_groups * heads_per_group,
        head_dim,
        dtype=dtype,
        device=device,
    ).contiguous()
    positions = torch.tensor([0, 3, 7, 11, 15], dtype=torch.int64, device=device)
    angles = torch.randn(32, rope_dim // 2, dtype=torch.float32, device=device)
    cos_sin_cache = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    out = dspark_inv_rope_bf16_layout(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )

    cs = cos_sin_cache.index_select(0, positions).to(torch.float32)
    cos = cs[:, : rope_dim // 2].unsqueeze(1)
    sin = cs[:, rope_dim // 2 :].unsqueeze(1)
    ref = o.clone()
    rope = ref[..., nope_dim:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], rope_dim // 2, 2)
    even = rope[..., 0]
    odd = rope[..., 1]
    inv_even = even * cos + odd * sin
    inv_odd = odd * cos - even * sin
    ref[..., nope_dim:] = torch.stack((inv_even, inv_odd), dim=-1).reshape(
        shape
    ).to(dtype)
    ref = ref.view(tokens, n_groups, heads_per_group, head_dim)
    ref = ref.reshape(tokens, n_groups, heads_per_group * head_dim)
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_triton_context_kv_store_matches_reference():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    kv = torch.randn(7, 512, dtype=dtype, device=device)
    cache = torch.randn(2, 8, 512, dtype=dtype, device=device)
    cache_ref = cache.clone()
    positions = torch.tensor([8, 9, 10, 16, 17, 18, 19], device=device)
    query_start_loc = torch.tensor([0, 3, 7], device=device)
    num_rejected_tokens = torch.tensor([1, 0], device=device)
    weight = torch.randn(512, dtype=dtype, device=device)
    angles = torch.randn(32, 32, dtype=torch.float32, device=device)
    cos_sin_cache = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    eps = 1e-6

    kv_float = kv.float()
    rrms = torch.rsqrt(kv_float.square().mean(-1, keepdim=True) + eps)
    kv_norm = (kv_float * rrms * weight.float()).to(dtype)
    kv_ref = _apply_rope_gptj_last(kv_norm, positions, cos_sin_cache)

    cache_ref[0, positions[0:2] % 8] = kv_ref[0:2]
    cache_ref[1, positions[3:7] % 8] = kv_ref[3:7]

    dspark_context_kv_store(
        kv,
        cache,
        positions,
        query_start_loc,
        2,
        num_rejected_tokens,
        weight,
        cos_sin_cache,
        eps,
    )

    assert torch.allclose(cache.float(), cache_ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_triton_markov_greedy_argmax_matches_reference():
    device = torch.device("cuda")
    torch.manual_seed(0)
    batch_size = 2
    vocab_size = 137
    rank = 16
    dtype = torch.bfloat16

    base_logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)
    w1 = torch.randn(vocab_size, rank, dtype=dtype, device=device)
    w2 = torch.randn(vocab_size, rank, dtype=dtype, device=device)
    prev_token_ids = torch.tensor([3, 41], dtype=torch.int64, device=device)
    num_blocks = (vocab_size + 31) // 32
    block_vals = torch.empty(batch_size, num_blocks, dtype=torch.float32, device=device)
    block_ids = torch.empty(batch_size, num_blocks, dtype=torch.int64, device=device)
    out = torch.empty(batch_size, dtype=torch.int64, device=device)

    dspark_markov_greedy_argmax(
        base_logits,
        prev_token_ids,
        w1,
        w2,
        block_vals,
        block_ids,
        out,
        block_v=32,
    )

    reference = (
        base_logits
        + w1[prev_token_ids].float() @ w2.float().T
    ).argmax(dim=-1)
    assert out.tolist() == reference.tolist()
