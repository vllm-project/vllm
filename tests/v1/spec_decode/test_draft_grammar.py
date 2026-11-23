# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MRV2 worker-side grammar-aware speculative drafting."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import (
    DeviceConfig,
    ModelConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.math_utils import cdiv
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.draft_grammar import (
    DraftGrammarManager,
    apply_safe_draft_grammar_bitmask,
    find_single_allowed_token,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

TOKENIZER = "gpt2"
AGENT_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": ["get_weather"]},
            "arguments": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["Singapore", "Tokyo"]},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }
)


def _pack_allowed_tokens(allowed: list[int], vocab_size: int) -> np.ndarray:
    row = np.zeros(cdiv(vocab_size, 32), dtype=np.uint32)
    for tok in allowed:
        row[tok >> 5] |= np.uint32(1 << (tok & 31))
    return row.view(np.int32)


def _make_vllm_config(num_spec_tokens: int = 4) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        speculative_config=SpeculativeConfig(
            model="[ngram]",
            num_speculative_tokens=num_spec_tokens,
        ),
    )


def _wait_for_grammar(request: Request) -> None:
    structured_req = request.structured_output_request
    assert structured_req is not None
    deadline = time.time() + 10.0
    while not structured_req._check_grammar_completion():
        assert time.time() < deadline, "Grammar compilation timed out"
        time.sleep(0.01)


def _require_grammar(request: Request) -> StructuredOutputGrammar:
    structured_req = request.structured_output_request
    assert structured_req is not None
    grammar = structured_req.grammar
    assert isinstance(grammar, StructuredOutputGrammar)
    return grammar


def _make_request(
    manager: StructuredOutputManager,
    tokenizer,
    req_id: str,
    schema: str = AGENT_TOOL_SCHEMA,
    reasoning_ended: bool | None = True,
) -> Request:
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    assert sp.structured_outputs is not None
    sp.structured_outputs._backend = "xgrammar"
    sp.update_from_generation_config({}, tokenizer.eos_token_id)
    req = Request(
        request_id=req_id,
        prompt_token_ids=tokenizer.encode("Call the weather tool."),
        sampling_params=sp,
        pooling_params=None,
        reasoning_ended=reasoning_ended,
    )
    manager.grammar_init(req)
    _wait_for_grammar(req)
    return req


def test_find_single_allowed_token_handles_tail_padding():
    vocab_size = 70  # 3 int32 words; top 26 bits of word 2 are padding
    assert (
        find_single_allowed_token(_pack_allowed_tokens([42], vocab_size), vocab_size)
        == 42
    )
    assert (
        find_single_allowed_token(
            _pack_allowed_tokens([42, 43], vocab_size), vocab_size
        )
        is None
    )
    assert (
        find_single_allowed_token(_pack_allowed_tokens([], vocab_size), vocab_size)
        is None
    )
    # Padding bits set above vocab_size in the final word must be ignored.
    row = _pack_allowed_tokens([69], vocab_size).copy()
    row.view(np.uint32)[2] |= np.uint32(1 << 10)  # bit 74 >= vocab_size
    assert find_single_allowed_token(row, vocab_size) == 69


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_safe_draft_grammar_bitmask_kernel_and_eagle3_nan_guard(dtype):
    """Normal rows are masked in-place; cropped-vocab (EAGLE-3 d2t) rows with
    empty grammar intersection stay unmasked so softmax never produces NaN."""
    device = torch.device("cuda")
    vocab_size = 10000
    torch.manual_seed(0)
    logits = torch.randn((4, vocab_size), dtype=dtype, device=device)

    # Row 0: unconstrained request (not in req_indices) — must stay untouched.
    row0_orig = logits[0].clone()
    # Row 1: normal structured request allowing tokens [7, 42, 9999].
    # Row 2: EAGLE-3 cropped-vocab row whose finite support is [0..99],
    #        while grammar allows ONLY [5000, 5001] -> empty intersection!
    logits[2, 100:] = float("-inf")
    row2_orig = logits[2].clone()
    # Row 3: EAGLE-3 cropped-vocab row whose finite support is [0..99],
    #        grammar allows [50, 5000] -> non-empty intersection {50}!
    logits[3, 100:] = float("-inf")

    bm_np = np.stack(
        [
            _pack_allowed_tokens([7, 42, 9999], vocab_size),
            _pack_allowed_tokens([5000, 5001], vocab_size),
            _pack_allowed_tokens([50, 5000], vocab_size),
        ]
    )
    req_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    bm_gpu = torch.from_numpy(bm_np).to(device)
    has_valid_buf = torch.zeros(4, dtype=torch.int32, device=device)

    has_valid = apply_safe_draft_grammar_bitmask(
        logits, req_indices, bm_gpu, has_valid_buf
    )
    assert has_valid.cpu().tolist() == [1, 0, 1]

    # Row 0 untouched.
    assert torch.equal(logits[0], row0_orig)
    # Row 1 finite ONLY on {7, 42, 9999}.
    row1_finite = torch.isfinite(logits[1]).nonzero(as_tuple=True)[0].cpu().tolist()
    assert row1_finite == [7, 42, 9999]
    # Row 2 left unmasked (no all -inf, no NaN in softmax).
    assert torch.equal(logits[2], row2_orig)
    assert torch.isfinite(torch.softmax(logits[2].float(), dim=-1)).all()
    # Row 3 finite ONLY on {50}.
    row3_finite = torch.isfinite(logits[3]).nonzero(as_tuple=True)[0].cpu().tolist()
    assert row3_finite == [50]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_draft_grammar_manager_lifecycle_rollback_fastpath_and_preemption():
    """End-to-end CPU/GPU lifecycle: Scheduler init handoff, multi-step
    speculative masking & rollback, deterministic fast-path (including EAGLE-3
    draft_vocab guard), and preemption resume."""
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = _make_vllm_config(num_spec_tokens=4)
    vocab_size = vllm_config.model_config.get_vocab_size()

    sched_mgr = StructuredOutputManager(vllm_config)
    req = _make_request(sched_mgr, tokenizer, "req-1", reasoning_ended=True)
    # Second request sharing the exact same schema exercises compiled cache.
    req2 = _make_request(sched_mgr, tokenizer, "req-2", reasoning_ended=True)

    inits = sched_mgr.get_draft_grammar_inits(
        {"req-1": req, "req-2": req2}, ["req-1", "req-2"]
    )
    assert len(inits) == 2
    assert inits[0].cache_key == inits[1].cache_key
    # Second call must not re-emit already synced requests.
    assert (
        sched_mgr.get_draft_grammar_inits(
            {"req-1": req, "req-2": req2}, ["req-1", "req-2"]
        )
        == []
    )

    worker_mgr = DraftGrammarManager(
        vllm_config=vllm_config,
        max_num_reqs=8,
        vocab_size=vocab_size,
        device=device,
    )
    worker_mgr.register_inits(inits)
    assert len(worker_mgr._compiled_cache) == 1  # shared compiled grammar cache

    # Run a 4-step draft proposal for a mixed batch [unconstrained, req-1].
    req_ids = ["req-free", "req-1"]
    worker_mgr.begin_propose(req_ids, num_spec_steps=4)
    assert worker_mgr.has_active_requests

    drafted_seq: list[int] = []
    for _step in range(4):
        logits = torch.randn((2, vocab_size), dtype=torch.float32, device=device)
        worker_mgr.apply_draft_bitmask(logits)
        sampled = logits.argmax(dim=-1)
        drafted_seq.append(int(sampled[1].item()))
        worker_mgr.advance_after_draft_step(sampled)

    # Every drafted token must be valid under the scheduler's ground-truth grammar.
    grammar = _require_grammar(req)
    assert grammar.validate_tokens(drafted_seq) == drafted_seq

    # Rollback via end_propose must restore shadow matcher to initial state.
    worker_mgr.end_propose()
    assert not worker_mgr.has_active_requests

    # Simulate partial verification: target accepted first 2 drafts only.
    verified = drafted_seq[:2]
    sampled_np = np.full((2, 5), -1, dtype=np.int32)
    sampled_np[1, :2] = verified
    num_sampled_np = np.array([1, 2], dtype=np.int32)

    worker_mgr.sync_verified_tokens(req_ids, sampled_np, num_sampled_np)
    assert sched_mgr.accept_tokens(req, verified)
    assert req.structured_output_request.accepted_grammar_token_ids == verified

    # Verify bit-identical state between scheduler and worker shadow matcher.
    sched_bm = sched_mgr.grammar_bitmask({"req-1": req}, ["req-1"], {})
    assert sched_bm is not None
    worker_bm = torch.zeros_like(torch.from_numpy(sched_bm[:1]))
    worker_mgr._entries["req-1"].matcher.fill_next_token_bitmask(worker_bm, 0)
    assert np.array_equal(worker_bm[0].numpy(), sched_bm[0])

    # Simulate preemption + resume: worker drops req-1, scheduler resets flag.
    worker_mgr.remove_request("req-1")
    req.structured_output_request.draft_grammar_synced = False
    resume_inits = sched_mgr.get_draft_grammar_inits({"req-1": req}, ["req-1"])
    assert len(resume_inits) == 1
    assert resume_inits[0].accepted_token_ids == tuple(verified)
    worker_mgr.register_inits(resume_inits)
    resumed_bm = torch.zeros_like(worker_bm)
    worker_mgr._entries["req-1"].matcher.fill_next_token_bitmask(resumed_bm, 0)
    assert np.array_equal(resumed_bm[0].numpy(), sched_bm[0])

    # Deterministic fast-path test: 1-character literal "Z" has popcount == 1
    # (no shorter prefix token exists in GPT-2).
    single_tok_id = tokenizer.encode("Z")[0]
    choice_req = Request(
        request_id="req-choice",
        prompt_token_ids=tokenizer.encode("Letter:"),
        sampling_params=SamplingParams(
            structured_outputs=StructuredOutputsParams(choice=["Z"])
        ),
        pooling_params=None,
        reasoning_ended=True,
    )
    choice_req.sampling_params.structured_outputs._backend = "xgrammar"
    choice_req.sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)
    sched_mgr.grammar_init(choice_req)
    _wait_for_grammar(choice_req)
    choice_inits = sched_mgr.get_draft_grammar_inits(
        {"req-choice": choice_req}, ["req-choice"]
    )
    worker_mgr.register_inits(choice_inits)

    before_fast = worker_mgr.num_forced_fast_path_steps
    worker_mgr.begin_propose(["req-choice"], num_spec_steps=2)
    logits = torch.randn((1, vocab_size), dtype=torch.float32, device=device)
    worker_mgr.apply_draft_bitmask(logits)
    sampled = logits.argmax(dim=-1)
    assert int(sampled[0].item()) == single_tok_id
    worker_mgr.advance_after_draft_step(sampled)
    worker_mgr.end_propose()
    assert worker_mgr.num_forced_fast_path_steps == before_fast + 1

    # EAGLE-3 draft_vocab guard: if the sole allowed token is outside the
    # cropped draft vocab, fast-path must NOT fire (must sync/fallback safely).
    worker_mgr.draft_vocab_target_ids = frozenset({0, 1, 2})  # excludes "Z"
    before_fast = worker_mgr.num_forced_fast_path_steps
    before_sync = worker_mgr.num_synced_draft_steps
    worker_mgr.begin_propose(["req-choice"], num_spec_steps=2)
    logits = torch.full(
        (1, vocab_size), float("-inf"), dtype=torch.float32, device=device
    )
    logits[0, :3] = torch.tensor([0.1, 0.9, 0.2], device=device)
    worker_mgr.apply_draft_bitmask(logits)
    # Empty intersection -> logits untouched, argmax stays token 1 (no NaN).
    sampled = logits.argmax(dim=-1)
    assert int(sampled[0].item()) == 1
    worker_mgr.advance_after_draft_step(sampled)
    worker_mgr.end_propose()
    assert worker_mgr.num_forced_fast_path_steps == before_fast
    assert worker_mgr.num_synced_draft_steps == before_sync + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reasoning_end_transition_defers_init_until_structured_phase():
    """While an Agent request is still in <think> (reasoning_ended=False),
    scheduler must not emit DraftGrammarInit; once reasoning ends and initial
    structured tokens are accepted, DraftGrammarInit carries them so worker
    shadow state aligns bit-for-bit."""
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = _make_vllm_config(num_spec_tokens=4)
    vocab_size = vllm_config.model_config.get_vocab_size()

    sched_mgr = StructuredOutputManager(vllm_config)
    think_end_tok = tokenizer.encode("END")[0]
    open_brace = tokenizer.encode("{")[0]

    class _StubReasoner:
        reasoning_end_token_ids = ()

        def __init__(self, tokenizer=None, **kwargs):
            pass

        def is_reasoning_end(self, input_ids: list[int]) -> bool:
            return think_end_tok in input_ids

        def is_reasoning_end_streaming(
            self, input_ids: list[int], delta_ids: list[int]
        ) -> bool:
            return think_end_tok in delta_ids

    sched_mgr.reasoner_cls = _StubReasoner
    req = _make_request(sched_mgr, tokenizer, "req-think", reasoning_ended=False)

    # During reasoning phase: no DraftGrammarInit emitted.
    assert sched_mgr.get_draft_grammar_inits({"req-think": req}, ["req-think"]) == []

    # Simulate reasoning end: target emits [</think>, '{'].
    req.append_output_token_ids([think_end_tok, open_brace])
    assert sched_mgr.accept_tokens(req, [think_end_tok, open_brace])
    assert req.structured_output_request is not None
    assert req.structured_output_request.reasoning_ended is True

    inits = sched_mgr.get_draft_grammar_inits({"req-think": req}, ["req-think"])
    assert len(inits) == 1
    assert inits[0].accepted_token_ids == (open_brace,)

    worker_mgr = DraftGrammarManager(
        vllm_config=vllm_config,
        max_num_reqs=4,
        vocab_size=vocab_size,
        device=device,
    )
    worker_mgr.register_inits(inits)

    sched_bm = sched_mgr.grammar_bitmask({"req-think": req}, ["req-think"], {})
    assert sched_bm is not None
    worker_bm = torch.zeros_like(torch.from_numpy(sched_bm[:1]))
    worker_mgr._entries["req-think"].matcher.fill_next_token_bitmask(worker_bm, 0)
    assert np.array_equal(worker_bm[0].numpy(), sched_bm[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_draft_model_speculator_sample_draft_integration():
    """Verify DraftModelSpeculator.sample_draft() bypasses local argmax when
    draft grammar is active, masks logits BEFORE writing DraftLogitsCache, and
    keeps the fast local-argmax path when no grammar request is active."""
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = _make_vllm_config(num_spec_tokens=2)
    vocab_size = vllm_config.model_config.get_vocab_size()

    sched_mgr = StructuredOutputManager(vllm_config)
    req = _make_request(sched_mgr, tokenizer, "req-spec", reasoning_ended=True)
    inits = sched_mgr.get_draft_grammar_inits({"req-spec": req}, ["req-spec"])

    worker_mgr = DraftGrammarManager(
        vllm_config=vllm_config,
        max_num_reqs=4,
        vocab_size=vocab_size,
        device=device,
    )
    worker_mgr.register_inits(inits)

    class _DummySpeculator(DraftModelSpeculator):
        def __init__(self):
            self.draft_grammar_manager = worker_mgr
            self.use_local_argmax_reduction = True
            self.draft_watermarker = None
            self.use_fp64_gumbel = False
            self.acceptance_estimator = None
            self.top_tokens_calls = 0
            self.compute_logits_calls = 0

        def load_draft_model(self, target_model, target_attn_layer_names):
            raise NotImplementedError

        def propose(self, *args, **kwargs):
            raise NotImplementedError

        def capture(self, *args, **kwargs):
            raise NotImplementedError

        def init_cudagraph_manager(self, *args, **kwargs):
            raise NotImplementedError

        def get_draft_top_tokens(self, hidden_states, spec_step_idx):
            self.top_tokens_calls += 1
            return torch.zeros(
                hidden_states.shape[0], dtype=torch.int64, device=hidden_states.device
            )

        def compute_draft_logits(self, hidden_states, spec_step_idx):
            self.compute_logits_calls += 1
            # Put huge logit on an invalid token (0 = '!') so unmasked argmax
            # would pick 0, whereas masked logits must pick a valid JSON start.
            logits = torch.full(
                (hidden_states.shape[0], vocab_size),
                -1.0,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            logits[:, 0] = 20.0
            return logits

    spec = _DummySpeculator()
    hidden = torch.zeros((1, 16), dtype=torch.float32, device=device)
    idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    temp = torch.tensor([1.0], dtype=torch.float32, device=device)
    seeds = torch.tensor([42], dtype=torch.int64, device=device)
    src_pos = torch.tensor([0], dtype=torch.int64, device=device)
    draft_step = torch.tensor(0, dtype=torch.int64, device=device)

    # 1. No active grammar -> local argmax fast path IS taken.
    worker_mgr.begin_propose(["req-unconstrained"], num_spec_steps=2)
    out_free = spec.sample_draft(
        hidden, src_pos, idx_mapping, temp, seeds, draft_step, draft_logits=None
    )
    worker_mgr.end_propose()
    assert spec.top_tokens_calls == 1
    assert spec.compute_logits_calls == 0
    assert int(out_free[0].item()) == 0

    # 2. Active grammar (greedy) -> bypasses local argmax and masks token 0.
    worker_mgr.begin_propose(["req-spec"], num_spec_steps=2)
    out_greedy = spec.sample_draft(
        hidden, src_pos, idx_mapping, temp, seeds, draft_step, draft_logits=None
    )
    worker_mgr.end_propose()
    assert spec.top_tokens_calls == 1
    assert spec.compute_logits_calls == 1
    assert int(out_greedy[0].item()) != 0
    grammar = _require_grammar(req)
    assert grammar.validate_tokens([int(out_greedy[0].item())]) == [
        int(out_greedy[0].item())
    ]

    # 3. Active grammar (probabilistic) -> DraftLogitsCache stores masked -inf
    # on invalid token 0!
    cache = DraftLogitsCache(
        max_num_reqs=4,
        num_steps=2,
        vocab_size=vocab_size,
        dtype=torch.float32,
        fill=0.0,
        device=device,
    )
    worker_mgr.begin_propose(["req-spec"], num_spec_steps=2)
    out_prob = spec.sample_draft(
        hidden, src_pos, idx_mapping, temp, seeds, draft_step, draft_logits=cache
    )
    worker_mgr.end_propose()
    assert torch.isneginf(cache.values[0, 0, 0])
    assert grammar.validate_tokens([int(out_prob[0].item())]) == [
        int(out_prob[0].item())
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_probabilistic_rejection_sampling_lossless_with_dual_grammar_mask():
    """Verify MRV2 gumbel_sample + DraftLogitsCache + rejection_sample with
    dual grammar masking: zero NaNs, 100% grammar-valid outputs, huge acceptance
    gain over target-only masking, and empirical distribution matching p_M."""
    device = torch.device("cuda")
    vocab_size = 64
    num_trials = 512
    allowed = [3, 17, 41]  # e.g. valid JSON grammar tokens
    bm_row = _pack_allowed_tokens(allowed, vocab_size)

    # Target raw logits favor tokens 3 (~0.60), 17 (~0.30), 41 (~0.10) in allowed.
    target_raw = torch.full((vocab_size,), -2.0, dtype=torch.float32, device=device)
    target_raw[3] = 3.0
    target_raw[17] = 2.3
    target_raw[41] = 1.2
    target_masked = target_raw.clone()
    disallowed_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    disallowed_mask[allowed] = False
    target_masked[disallowed_mask] = float("-inf")
    expected_p = torch.softmax(target_masked, dim=-1)

    # Unconstrained drafter puts ~90% mass on INVALID tokens (e.g. "\n  "),
    # and matches target closely on the allowed tokens.
    draft_raw = torch.full((vocab_size,), -2.0, dtype=torch.float32, device=device)
    draft_raw[0] = 5.5  # invalid token that unconstrained drafter strongly prefers
    draft_raw[3] = 2.9
    draft_raw[17] = 2.35
    draft_raw[41] = 1.15

    def run_batch(dual_mask: bool) -> tuple[float, torch.Tensor]:
        idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device)
        temp = torch.ones(num_trials, dtype=torch.float32, device=device)
        seeds = torch.arange(100, 100 + num_trials, dtype=torch.int64, device=device)
        pos_draft = torch.zeros(num_trials, dtype=torch.int64, device=device)
        draft_step = torch.tensor(0, dtype=torch.int64, device=device)

        draft_logits_tensor = draft_raw.unsqueeze(0).expand(num_trials, -1).clone()
        if dual_mask:
            bm_gpu = torch.from_numpy(np.tile(bm_row, (num_trials, 1))).to(device)
            has_valid = torch.zeros(num_trials, dtype=torch.int32, device=device)
            apply_safe_draft_grammar_bitmask(
                draft_logits_tensor, idx_mapping, bm_gpu, has_valid
            )
            assert (has_valid == 1).all()

        cache = DraftLogitsCache(
            max_num_reqs=num_trials,
            num_steps=1,
            vocab_size=vocab_size,
            dtype=torch.float32,
            fill=0.0,
            device=device,
        )
        draft_sampled = gumbel_sample(
            draft_logits_tensor,
            idx_mapping,
            temp,
            seeds,
            pos_draft,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=cache.values,
            logits_cache_col=draft_step,
            use_fp64=False,
            logits_cache_row_mapping=cache.row_mapping,
        )

        # Target Verify: 2 logits per request (pos 0 verifies draft at input_ids[1],
        # pos 1 is bonus). In MRv2 rejection_sample, draft_sampled_flat[logit_idx + 1]
        # holds the draft token verified against target_logits[logit_idx].
        target_logits = (
            target_masked.unsqueeze(0).expand(num_trials * 2, -1).contiguous()
        )
        cu_num_logits = torch.arange(
            0, (num_trials + 1) * 2, 2, dtype=torch.int32, device=device
        )
        expanded_idx = idx_mapping.repeat_interleave(2)
        expanded_local_pos = torch.tensor(
            [0, 1] * num_trials, dtype=torch.int32, device=device
        )
        verify_pos = torch.zeros(num_trials * 2, dtype=torch.int64, device=device)

        draft_sampled_flat = torch.zeros(
            (num_trials * 2,), dtype=torch.int32, device=device
        )
        draft_sampled_flat[1::2] = draft_sampled.to(torch.int32)

        sampled, num_sampled = rejection_sample(
            target_logits,
            cache,
            draft_sampled_flat,
            cu_num_logits,
            verify_pos,
            idx_mapping,
            expanded_idx,
            expanded_local_pos,
            temp,
            seeds,
            num_speculative_steps=1,
            skip_masked_vocab_blocks=True,
        )
        # Position 0 output must ALWAYS be grammar-valid (never outside `allowed`).
        out_tok0 = sampled[:, 0].cpu().tolist()
        assert all(t in allowed for t in out_tok0)
        # Draft accepted iff num_sampled == 2 (accepted draft + bonus token).
        accept_rate = float((num_sampled == 2).float().mean().item())
        counts = (
            torch.bincount(sampled[:, 0], minlength=vocab_size).float() / num_trials
        )
        return accept_rate, counts

    naive_accept, _ = run_batch(dual_mask=False)
    dual_accept, dual_dist = run_batch(dual_mask=True)

    # Target-only masking collapses acceptance (< 0.25); dual masking jumps > 0.90.
    assert naive_accept < 0.25, (
        f"Expected collapsed naive acceptance, got {naive_accept}"
    )
    assert dual_accept > 0.90, (
        f"Expected high dual-masked acceptance, got {dual_accept}"
    )
    # Output marginal distribution matches p_M within Monte Carlo tolerance.
    max_tv = float((dual_dist.cpu() - expected_p.cpu()).abs().max().item())
    assert max_tv < 0.06, f"Distribution drift too large: max_tv={max_tv}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_eagle3_empty_intersection_rejection_sampling_recovers_rare_token_losslessly():
    """When EAGLE-3's cropped vocab (`d2t`) excludes all grammar-valid tokens,
    safe draft masking leaves draft logits unmasked (zero NaN) and target
    rejection sampling rejects 100% and recovers the rare grammar token."""
    device = torch.device("cuda")
    vocab_size = 64
    num_trials = 64
    rare_allowed = [58, 59]  # outside EAGLE-3 cropped draft vocab [0..31]
    bm_row = _pack_allowed_tokens(rare_allowed, vocab_size)

    # EAGLE-3 draft logits: finite ONLY on [0..31], -inf on [32..63].
    draft_logits = torch.randn(
        (num_trials, vocab_size), dtype=torch.float32, device=device
    )
    draft_logits[:, 32:] = float("-inf")
    idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device)
    bm_gpu = torch.from_numpy(np.tile(bm_row, (num_trials, 1))).to(device)
    has_valid = torch.zeros(num_trials, dtype=torch.int32, device=device)

    apply_safe_draft_grammar_bitmask(draft_logits, idx_mapping, bm_gpu, has_valid)
    assert (has_valid == 0).all()

    temp = torch.ones(num_trials, dtype=torch.float32, device=device)
    seeds = torch.arange(500, 500 + num_trials, dtype=torch.int64, device=device)
    pos_draft = torch.zeros(num_trials, dtype=torch.int64, device=device)
    cache = DraftLogitsCache(
        max_num_reqs=num_trials,
        num_steps=1,
        vocab_size=vocab_size,
        dtype=torch.float32,
        fill=0.0,
        device=device,
    )
    draft_sampled = gumbel_sample(
        draft_logits,
        idx_mapping,
        temp,
        seeds,
        pos_draft,
        apply_temperature=True,
        is_drafting=True,
        logits_cache=cache.values,
        logits_cache_col=torch.tensor(0, dtype=torch.int64, device=device),
        use_fp64=False,
        logits_cache_row_mapping=cache.row_mapping,
    )
    assert not torch.isnan(cache.values[:, 0, :32]).any()
    assert (draft_sampled < 32).all()

    # Target masked logits allow ONLY rare tokens [58, 59].
    target_row = torch.full(
        (vocab_size,), float("-inf"), dtype=torch.float32, device=device
    )
    target_row[58] = 2.0
    target_row[59] = 1.0
    target_logits = target_row.unsqueeze(0).expand(num_trials * 2, -1).contiguous()
    cu_num_logits = torch.arange(
        0, (num_trials + 1) * 2, 2, dtype=torch.int32, device=device
    )
    draft_sampled_flat = torch.zeros(
        (num_trials * 2,), dtype=torch.int32, device=device
    )
    draft_sampled_flat[1::2] = draft_sampled.to(torch.int32)

    sampled, num_sampled = rejection_sample(
        target_logits,
        cache,
        draft_sampled_flat,
        cu_num_logits,
        torch.zeros(num_trials * 2, dtype=torch.int64, device=device),
        idx_mapping,
        idx_mapping.repeat_interleave(2),
        torch.tensor([0, 1] * num_trials, dtype=torch.int32, device=device),
        temp,
        seeds,
        num_speculative_steps=1,
        skip_masked_vocab_blocks=True,
    )
    # Every draft outside [58, 59] must be rejected (num_sampled == 1) and
    # recovered strictly inside rare_allowed!
    assert (num_sampled == 1).all()
    recovered = sampled[:, 0].cpu().tolist()
    assert all(t in rare_allowed for t in recovered)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lark_grammar_uses_full_vocab_tokenizer_info():
    """Lark grammars compiled with full_vocab_compiler must set
    `use_full_vocab=True` so DraftGrammarManager deserializes with the matching
    full-vocab TokenizerInfo."""
    vllm_config = _make_vllm_config(num_spec_tokens=2)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    sched_mgr = StructuredOutputManager(vllm_config)

    lark_grammar = 'start: "hello" " " ("world" | "vllm")'
    sp = SamplingParams(
        structured_outputs=StructuredOutputsParams(grammar=lark_grammar)
    )
    assert sp.structured_outputs is not None
    sp.structured_outputs._backend = "xgrammar"
    sp.update_from_generation_config({}, tokenizer.eos_token_id)
    req = Request(
        request_id="req-lark",
        prompt_token_ids=tokenizer.encode("Say hello."),
        sampling_params=sp,
        pooling_params=None,
        reasoning_ended=True,
    )
    sched_mgr.grammar_init(req)
    _wait_for_grammar(req)

    inits = sched_mgr.get_draft_grammar_inits({"req-lark": req}, ["req-lark"])
    assert len(inits) == 1
    assert inits[0].use_full_vocab is True

    worker_mgr = DraftGrammarManager(
        vllm_config=vllm_config,
        max_num_reqs=4,
        vocab_size=vllm_config.model_config.get_vocab_size(),
        device=torch.device("cuda"),
    )
    worker_mgr.register_inits(inits)
    assert worker_mgr.has_requests_in_batch(["req-lark"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_open_string_state_bypasses_draft_masking_for_full_cudagraph():
    """Inside an open free-text JSON string value (~96%+ vocab allowed),
    apply_draft_bitmask must bypass masking and clear `has_active_requests` so
    remaining draft decode steps stay on FULL CUDA graphs."""
    vllm_config = _make_vllm_config(num_spec_tokens=4)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    sched_mgr = StructuredOutputManager(vllm_config)
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
    )
    req = _make_request(sched_mgr, tokenizer, "req-open", schema=schema)
    prefix_ids = tokenizer.encode('{"summary": "', add_special_tokens=False)
    assert sched_mgr.accept_tokens(req, prefix_ids)

    inits = sched_mgr.get_draft_grammar_inits({"req-open": req}, ["req-open"])
    assert len(inits) == 1

    vocab_size = vllm_config.model_config.get_vocab_size()
    worker_mgr = DraftGrammarManager(
        vllm_config=vllm_config,
        max_num_reqs=4,
        vocab_size=vocab_size,
        device=torch.device("cuda"),
    )
    worker_mgr.register_inits(inits)
    worker_mgr.begin_propose(["req-open"], num_spec_steps=4)
    assert worker_mgr.has_active_requests

    logits = torch.zeros((1, vocab_size), dtype=torch.float32, device="cuda")
    worker_mgr.apply_draft_bitmask(logits)
    # Logits left unmasked and remaining steps released back to FULL CG.
    assert not torch.isneginf(logits).any()
    assert not worker_mgr.has_active_requests
    worker_mgr.end_propose()
