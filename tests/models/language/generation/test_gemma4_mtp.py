# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gemma4_mtp import (
    Gemma4MTP,
    Gemma4MTPMaskedEmbedder,
)


@pytest.mark.parametrize("H", [256, 1024])
@pytest.mark.parametrize("T", [1, 4, 16])
def test_gemma4_mtp_masked_embedder_select_and_score_and_top_tokens(H: int, T: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton kernel tests")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    vocab_size = 2048
    num_centroids = 4
    top_k = 2

    torch.manual_seed(42)
    embedder = Gemma4MTPMaskedEmbedder(
        hidden_size=H,
        vocab_size=vocab_size,
        num_centroids=num_centroids,
        centroid_intermediate_top_k=top_k,
    ).to(device=device, dtype=dtype)
    embedder.token_ordering.copy_(torch.randperm(vocab_size, device=device))
    embedder.centroids.weight.data.normal_(0.0, 0.02)

    lm_head_weight = torch.randn(vocab_size, H, device=device, dtype=dtype)
    h = torch.randn(T, H, device=device, dtype=dtype)

    logits, selected = embedder._select_and_score(h, lm_head_weight)
    S = embedder.num_selected

    ref_scores = torch.einsum(
        "td,tsd->ts",
        h,
        lm_head_weight[selected.reshape(-1)].view(T, S, H),
    )
    torch.testing.assert_close(logits, ref_scores, atol=1e-2, rtol=1e-2)

    top_tokens = embedder.get_top_tokens(h, lm_head_weight)
    expected_tokens = selected[
        torch.arange(T, device=device), ref_scores.argmax(dim=-1)
    ]
    torch.testing.assert_close(top_tokens, expected_tokens)

    # Full forward pass verification
    full_logits = embedder(h, lm_head_weight)
    assert full_logits.shape == (T, vocab_size)
    assert torch.equal(full_logits.argmax(dim=-1), top_tokens)

    # Empty tensor handling (T=0)
    h_empty = h[:0]
    empty_logits, empty_selected = embedder._select_and_score(h_empty, lm_head_weight)
    assert empty_logits.shape == (0, S)
    assert empty_selected.shape == (0, S)
    assert embedder.get_top_tokens(h_empty, lm_head_weight).shape == (0,)


@pytest.mark.usefixtures("dist_init")
def test_gemma4_mtp_get_top_tokens_masked_and_unmasked():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for test")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    H = 256
    T = 4
    vocab_size = 1024

    torch.manual_seed(42)
    h = torch.randn(T, H, device=device, dtype=dtype)
    lm_head_weight = torch.randn(vocab_size, H, device=device, dtype=dtype)

    mtp = Gemma4MTP.__new__(Gemma4MTP)
    torch.nn.Module.__init__(mtp)
    mtp._stable_full_lm_head_weight = lm_head_weight

    # 1. Masked embedding path (use_ordered_embeddings=True)
    embedder = Gemma4MTPMaskedEmbedder(
        hidden_size=H,
        vocab_size=vocab_size,
        num_centroids=4,
        centroid_intermediate_top_k=2,
    ).to(device=device, dtype=dtype)
    embedder.token_ordering.copy_(torch.arange(vocab_size, device=device))
    embedder.centroids.weight.data.normal_(0.0, 0.02)
    mtp.masked_embedding = embedder

    tokens_masked = mtp.get_top_tokens(h)
    expected_masked = embedder.get_top_tokens(h, lm_head_weight)
    assert torch.equal(tokens_masked, expected_masked)

    # 2. Unmasked embedding path (use_ordered_embeddings=False, soft_cap=30.0)
    mtp.masked_embedding = None
    mtp.logits_processor = LogitsProcessor(vocab_size, soft_cap=30.0)
    lm_head = ParallelLMHead(vocab_size, H)
    lm_head.to(device=device, dtype=dtype)
    lm_head.weight.data.copy_(lm_head_weight)  # type: ignore[operator]
    mtp.lm_head = lm_head
    mtp._suppress_idx = None

    tokens_unmasked = mtp.get_top_tokens(h)
    full_logits = mtp.compute_logits(h)
    assert full_logits is not None
    assert torch.equal(tokens_unmasked, full_logits.argmax(dim=-1))

    # Verify _suppress_idx masking parity in get_top_tokens vs compute_logits
    mtp._suppress_idx = tokens_unmasked.unique()
    tokens_suppressed = mtp.get_top_tokens(h)
    suppressed_logits = mtp.compute_logits(h)
    assert suppressed_logits is not None
    assert torch.equal(tokens_suppressed, suppressed_logits.argmax(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemma4_moe_active_expert_cap_exact_and_padded():
    from vllm.model_executor.models.gemma4 import (
        gemma4_fused_routing_kernel_triton,
        set_gemma4_moe_cap,
    )

    device = torch.device("cuda:0")
    E = 128
    K = 8
    chunk_len = 7
    cap_s = 14
    torch.manual_seed(123)
    per_expert_scale = torch.ones(E, device=device, dtype=torch.float32)
    # Test T = 16 (2 full 7-token request chunks + 2 CUDA-graph padding tokens)
    gating = torch.randn(16, E, device=device, dtype=torch.bfloat16)

    set_gemma4_moe_cap(0, 1)
    uncapped_weights, uncapped_ids = gemma4_fused_routing_kernel_triton(
        gating, K, per_expert_scale
    )

    try:
        set_gemma4_moe_cap(cap_s, chunk_len)
        capped_weights, capped_ids = gemma4_fused_routing_kernel_triton(
            gating, K, per_expert_scale
        )
    finally:
        set_gemma4_moe_cap(0, 1)

    for req_idx in range(2):
        base = req_idx * chunk_len
        # Token 0's K experts and weights are preserved identically
        assert torch.equal(capped_ids[base], uncapped_ids[base])
        torch.testing.assert_close(capped_weights[base], uncapped_weights[base])
        # Unique active experts across the 7-token chunk are bounded by cap_s=14
        unique_experts = torch.unique(capped_ids[base : base + chunk_len])
        assert unique_experts.numel() <= cap_s
        # Token 0's experts are a subset of the active expert set
        tok0_set = set(uncapped_ids[base].tolist())
        chunk_set = set(unique_experts.tolist())
        assert tok0_set.issubset(chunk_set)

    # Trailing CUDA-graph padding tokens (indices 14..15) are untouched
    assert torch.equal(capped_ids[14:], uncapped_ids[14:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemma4_cot_lenient_rejection_sampling():
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
        set_v2_cot_lenience,
    )

    device = torch.device("cuda:0")
    vocab_size = 512
    num_steps = 3
    num_logits = num_steps + 1  # 1 bonus/prev slot + 3 draft steps

    # Target argmax at all steps is token 200 (logit = 10.0).
    # Draft proposes [201, 101, 202] where:
    # - step 0 (token 201) has logit 9.0 (diff = 1.0 <= 1.8 -> accepted in CoT)
    # - step 1 (token 101 = `<channel|>`) matches target argmax 101 (exits CoT)
    # - step 2 (token 202) has logit 9.0 vs target 200 logit 10.0 (diff = 1.0,
    #   but `<channel|>` exited CoT at step 1 -> strictly rejected!)
    target_logits = torch.zeros(num_logits, vocab_size, device=device)
    target_logits[0, 200] = 10.0
    target_logits[0, 201] = 9.0
    target_logits[1, 101] = 10.0
    target_logits[2, 200] = 10.0
    target_logits[2, 202] = 9.0
    target_logits[3, 200] = 10.0

    draft_sampled = torch.tensor([100, 201, 101, 202], dtype=torch.int64, device=device)
    cu_num_logits = torch.tensor([0, num_logits], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int32, device=device)
    expanded_local_pos = torch.arange(num_logits, dtype=torch.int32, device=device)
    pos = torch.tensor([10, 11, 12, 13], dtype=torch.int64, device=device)
    temperature = torch.zeros(1, dtype=torch.float32, device=device)
    seed = torch.zeros(1, dtype=torch.int64, device=device)

    # 1. Strict lossless mode (margin = 0.0): step 0 (201 != 200) rejected immediately
    set_v2_cot_lenience(0.0)
    sampled_strict, num_strict = rejection_sample(
        target_logits=target_logits,
        draft_logits=None,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temperature,
        seed=seed,
        pos=pos,
        num_speculative_steps=num_steps,
    )
    assert num_strict.item() == 1
    assert sampled_strict[0, 0].item() == 200

    # 2. CoT-lenient mode (margin = 1.8): step 0 (201) accepted leniently,
    # step 1 (101 `<channel|>`) accepted and exits CoT, step 2 (202) rejected
    # strictly and replaced by target argmax 200.
    try:
        set_v2_cot_lenience(1.8)
        sampled_lenient, num_lenient = rejection_sample(
            target_logits=target_logits,
            draft_logits=None,
            draft_sampled=draft_sampled,
            cu_num_logits=cu_num_logits,
            idx_mapping=idx_mapping,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            temperature=temperature,
            seed=seed,
            pos=pos,
            num_speculative_steps=num_steps,
        )
        assert num_lenient.item() == 3
        assert sampled_lenient[0, :3].tolist() == [201, 101, 200]

        # Next round on the same request (continuing at pos = 10 + 3 = 13)
        # must remain outside CoT (in_cot == 0) even though pos = 13 < 96!
        pos_next = torch.tensor([13, 14, 15, 16], dtype=torch.int64, device=device)
        draft_next = torch.tensor(
            [200, 201, 200, 200], dtype=torch.int64, device=device
        )
        sampled_after_cot, num_after_cot = rejection_sample(
            target_logits=target_logits,
            draft_logits=None,
            draft_sampled=draft_next,
            cu_num_logits=cu_num_logits,
            idx_mapping=idx_mapping,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            temperature=temperature,
            seed=seed,
            pos=pos_next,
            num_speculative_steps=num_steps,
        )
        assert num_after_cot.item() == 1
        assert sampled_after_cot[0, 0].item() == 200
    finally:
        set_v2_cot_lenience(0.0)


def test_gemma4_proposer_use_local_argmax_reduction(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    def mock_init_greedy(self, *args, **kwargs):
        self._enable_probabilistic_draft_probs = False

    monkeypatch.setattr(SpecDecodeBaseProposer, "__init__", mock_init_greedy)
    proposer = Gemma4Proposer(MagicMock(), torch.device("cpu"))
    assert proposer.use_local_argmax_reduction is True

    def mock_init_prob(self, *args, **kwargs):
        self._enable_probabilistic_draft_probs = True

    monkeypatch.setattr(SpecDecodeBaseProposer, "__init__", mock_init_prob)
    proposer = Gemma4Proposer(MagicMock(), torch.device("cpu"))
    assert proposer.use_local_argmax_reduction is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prepare_q_only_decode_inputs():
    from unittest.mock import MagicMock

    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        prepare_q_only_decode_inputs,
    )

    device = torch.device("cuda")
    num_reqs = 3
    max_num_reqs = 4
    hidden_size = 256

    query_start_loc = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device=device)
    target_tokens = 6
    target_positions = torch.tensor(
        [10, 11, 20, 21, 22, 30], dtype=torch.int64, device=device
    )
    target_seq_lens = torch.tensor([12, 23, 31, 0], dtype=torch.int32, device=device)

    num_rejected = torch.tensor([1, 0, 0], dtype=torch.int32, device=device)
    expected_last_indices = torch.tensor([0, 4, 5], dtype=torch.int64, device=device)

    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int64, device=device)

    num_sampled = torch.tensor([1, 0, 2], dtype=torch.int32, device=device)
    last_sampled = torch.tensor([100, 200, 300, 400], dtype=torch.int64, device=device)
    next_prefill_tokens = torch.tensor(
        [1000, 2000, 3000, 4000], dtype=torch.int64, device=device
    )
    expected_draft_tokens = torch.tensor(
        [300, 1000, 200], dtype=torch.int32, device=device
    )

    target_hidden_states = torch.randn(
        target_tokens, hidden_size, device=device, dtype=torch.float16
    )

    draft_hidden_states = torch.empty(
        max_num_reqs, hidden_size, device=device, dtype=torch.float16
    )
    current_draft_step = torch.full((1,), 999, dtype=torch.int32, device=device)
    sample_src_positions = torch.empty(max_num_reqs, dtype=torch.int64, device=device)

    input_buffers = InputBuffers(max_num_reqs, max_num_reqs, device)

    input_batch = MagicMock()
    input_batch.num_reqs = num_reqs
    input_batch.query_start_loc = query_start_loc
    input_batch.positions = target_positions
    input_batch.seq_lens = target_seq_lens
    input_batch.idx_mapping = idx_mapping

    prepare_q_only_decode_inputs(
        draft_hidden_states,
        current_draft_step,
        input_buffers,
        sample_src_positions,
        target_hidden_states,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        max_num_reqs,
    )

    # 1. draft_input_ids
    assert torch.equal(input_buffers.input_ids[:num_reqs], expected_draft_tokens)
    # 2. draft_positions
    expected_positions = target_positions[expected_last_indices]
    assert torch.equal(input_buffers.positions[:num_reqs], expected_positions)
    # 3. sample_src_positions == draft_positions + 1
    assert torch.equal(sample_src_positions[:num_reqs], expected_positions + 1)
    # 4. draft_seq_lens[:num_reqs] == target_seq_lens[:num_reqs] - num_rejected
    expected_seq_lens = target_seq_lens[:num_reqs] - num_rejected
    assert torch.equal(input_buffers.seq_lens[:num_reqs], expected_seq_lens)
    # 5. draft_seq_lens[num_reqs:] == 0
    assert torch.equal(
        input_buffers.seq_lens[num_reqs:],
        torch.zeros(max_num_reqs - num_reqs, dtype=torch.int32, device=device),
    )
    # 6. draft_query_start_loc == [0, 1, 2, 3, 3]
    expected_q_start = torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32, device=device)
    assert torch.equal(input_buffers.query_start_loc, expected_q_start)
    # 7. draft_current_step == 0
    assert current_draft_step.item() == 0
    # 8. draft_hidden_states[:num_reqs] == target_hidden_states[last_token_indices]
    assert torch.equal(
        draft_hidden_states[:num_reqs],
        target_hidden_states[expected_last_indices],
    )
