# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionSampler projects logits tile-by-tile (#50699): sampling outputs
must not depend on the tile size, and peak memory must scale with the tile,
not the batch."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.utils.gpu_sync_debug as gsd
from vllm.model_executor.models import diffusion_gemma as dg
from vllm.sampling_params import SamplingParams
from vllm.utils.gpu_sync_debug import with_gpu_sync_check
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.states import SamplingStates

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda"


def _make_sampler(
    num_reqs, canvas_len, vocab, hidden, weight, top_k=0, params=None, swap_capacity=0
):
    states = dg.DiffusionGemmaRequestStates(
        max_num_reqs=num_reqs,
        canvas_length=canvas_len,
        vocab_size=vocab,
        max_denoising_steps=8,
        device=torch.device(DEVICE),
        hidden_size=hidden,
        stability_threshold=2,
        swap_capacity=swap_capacity,
    )
    base = SimpleNamespace(
        sampling_states=SamplingStates(num_reqs, vocab),
        logprob_token_ids_state=LogprobTokenIdsState(num_reqs, torch.device(DEVICE)),
        req_states=SimpleNamespace(
            draft_tokens=torch.zeros(
                num_reqs, canvas_len, dtype=torch.int64, device=DEVICE
            )
        ),
        logprobs_mode="raw_logprobs",
    )
    sampler = dg.DiffusionSampler(
        sampler=base,
        diffusion_config=SimpleNamespace(canvas_length=canvas_len),
        vocab_size=vocab,
        diffusion_states=states,
        compute_logits=lambda h: h @ weight.t(),
        confidence_threshold=0.5,
        t_min=0.0,
        t_max=0.0,
        entropy_bound=0.1,
        embed_weight=weight,
        normalizer=torch.tensor(1.0, device=DEVICE),
    )
    for i in range(num_reqs):
        sampler.add_request(i, params[i] if params else SamplingParams(top_k=top_k))
        states.is_encoder_phase[i] = False
    sampler.apply_staged_writes()
    return sampler, states


def _make_batch(num_reqs, valid_lens, slots=None):
    cu = np.zeros(num_reqs + 1, dtype=np.int64)
    cu[1:] = np.cumsum(valid_lens)
    qsl_np = cu.astype(np.int32)
    slots_np = np.asarray(slots or range(num_reqs), dtype=np.intp)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=int(max(valid_lens)),
        idx_mapping_np=slots_np,
        idx_mapping=torch.from_numpy(slots_np).to(DEVICE),
        cu_num_logits_np=cu,
        query_start_loc_np=qsl_np,
        query_start_loc=torch.from_numpy(qsl_np).to(DEVICE),
    )


@pytest.mark.parametrize("top_k", [0, 4])
def test_tiled_projection_matches_full(monkeypatch, top_k):
    """Deterministic outputs (t_min=t_max=0) are identical for group=1 tiling
    and a single full-batch pass, including a truncated (padded) canvas."""
    num_reqs, canvas_len, vocab, hidden = 4, 16, 1024, 32
    torch.manual_seed(0)
    weight = torch.randn(vocab, hidden, device=DEVICE)
    valid_lens = [canvas_len, canvas_len, canvas_len, canvas_len - 3]
    hs = torch.randn(sum(valid_lens), hidden, device=DEVICE)

    results = []
    for free in (2**60, 0):
        torch.manual_seed(0)
        sampler, states = _make_sampler(
            num_reqs, canvas_len, vocab, hidden, weight, top_k
        )
        monkeypatch.setattr(
            torch.accelerator, "get_memory_info", lambda free=free: (free, 2**60)
        )
        out = sampler(hs.clone(), _make_batch(num_reqs, valid_lens))
        results.append((out, states))

    (out_full, st_full), (out_tiled, st_tiled) = results
    torch.testing.assert_close(st_full.argmax_canvas, st_tiled.argmax_canvas)
    torch.testing.assert_close(st_full.is_encoder_phase, st_tiled.is_encoder_phase)
    torch.testing.assert_close(st_full.confident, st_tiled.confident)
    torch.testing.assert_close(st_full.step, st_tiled.step)
    torch.testing.assert_close(out_full.sampled_token_ids, out_tiled.sampled_token_ids)
    torch.testing.assert_close(out_full.num_sampled, out_tiled.num_sampled)


def test_tiling_bounds_peak_memory(monkeypatch):
    """Forced group=1 tiling must not materialize full-batch logits."""
    num_reqs, canvas_len, vocab, hidden = 8, 64, 32768, 64
    torch.manual_seed(0)
    weight = torch.randn(vocab, hidden, device=DEVICE)
    valid_lens = [canvas_len] * num_reqs
    hs = torch.randn(num_reqs * canvas_len, hidden, device=DEVICE)

    peaks = {}
    for name, free in (("full", 2**60), ("tiled", 0)):
        torch.manual_seed(0)
        sampler, _ = _make_sampler(num_reqs, canvas_len, vocab, hidden, weight)
        monkeypatch.setattr(
            torch.accelerator, "get_memory_info", lambda free=free: (free, 2**60)
        )
        sampler(hs.clone(), _make_batch(num_reqs, valid_lens))
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        base = torch.accelerator.memory_allocated()
        sampler(hs.clone(), _make_batch(num_reqs, valid_lens))
        torch.accelerator.synchronize()
        peaks[name] = torch.accelerator.max_memory_allocated() - base

    assert peaks["tiled"] < peaks["full"] / 2


def test_profile_run_resets_state():
    """profile_run exercises the decode pipeline for KV sizing, then leaves
    slot 0 freshly initialized."""
    num_reqs, canvas_len, vocab, hidden = 2, 16, 1024, 32
    torch.manual_seed(0)
    weight = torch.randn(vocab, hidden, device=DEVICE)
    sampler, states = _make_sampler(num_reqs, canvas_len, vocab, hidden, weight)
    sampler.profile_run(torch.zeros(num_reqs, hidden, device=DEVICE))
    assert bool(states.is_encoder_phase[0])
    assert int(states.step[0]) == 0


@pytest.fixture
def sync_check_error(monkeypatch):
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", False)
    gsd.enable_gpu_sync_check()


def test_logprobs_reach_the_emitting_step_without_a_host_sync(sync_check_error):
    """Logprobs stashed on the converging step are reported when the slot
    emits: a read at once (and again after its next read), a generation on
    its commit step. A request without logprobs gets no rows. The host never
    learns which slots converged, so no step syncs with the GPU."""
    num_reqs, canvas_len, vocab, hidden = 3, 16, 1024, 32
    set_random_seed(0)
    weight = torch.randn(vocab, hidden, device=DEVICE)
    one_step = {"diffusion_max_steps": 1}
    params = [
        SamplingParams(extra_args=one_step),
        SamplingParams(logprobs=2, extra_args=one_step),
        SamplingParams(logprobs=2, extra_args={**one_step, "diffusion_read_only": 1}),
    ]
    sampler, _ = _make_sampler(
        num_reqs, canvas_len, vocab, hidden, weight, params=params
    )
    hs = torch.randn(num_reqs * canvas_len, hidden, device=DEVICE)
    step = with_gpu_sync_check(sampler)

    emitted = []
    for _ in range(2):
        out = step(hs, _make_batch(num_reqs, [canvas_len] * num_reqs))
        lists = out.logprobs_tensors.tolists()
        assert len(lists.logprobs) == 2 * canvas_len
        for i, k in enumerate(out.num_sampled.tolist()):
            if k and params[i].logprobs:
                lp = lists.slice_request(i, k)
                tokens = out.sampled_token_ids[i, :k].tolist()
                assert lp.logprob_token_ids[:, 0].tolist() == tokens
                assert (lp.logprobs[:, :3] <= 0).all()
                emitted.append(i)

    assert emitted == [2, 1, 2]


def test_constrained_reads_report_their_allowed_ids_without_a_host_sync(
    sync_check_error,
):
    """Constrained read-only reads sharing one allowed set emit at
    convergence with every allowed id's logprob, without a host sync."""
    num_reqs, canvas_len, vocab, hidden = 2, 16, 1024, 32
    set_random_seed(0)
    weight = torch.randn(vocab, hidden, device=DEVICE)
    allowed = [3, 5, 700, 1000]
    extra = {"diffusion_max_steps": 1, "diffusion_read_only": 1}
    params = [
        SamplingParams(
            logprobs=0,
            logprob_token_ids=allowed,
            extra_args={**extra, "diffusion_constrained": 1},
        )
        for _ in range(num_reqs)
    ]
    sampler, _ = _make_sampler(
        num_reqs, canvas_len, vocab, hidden, weight, params=params
    )
    hs = torch.randn(num_reqs * canvas_len, hidden, device=DEVICE)

    out = with_gpu_sync_check(sampler)(hs, _make_batch(num_reqs, [canvas_len] * 2))

    lists = out.logprobs_tensors.tolists()
    for i in range(num_reqs):
        lp = lists.slice_request(i, canvas_len)
        tokens = out.sampled_token_ids[i, :canvas_len].tolist()
        assert set(tokens) <= set(allowed)
        assert lp.logprob_token_ids[:, 0].tolist() == tokens
        assert (lp.logprob_token_ids[:, 1:] == allowed).all()
        assert np.allclose(np.exp(lp.logprobs[:, 1:]).sum(axis=1), 1.0, atol=1e-4)


def test_full_vocab_logprobs_come_on_the_commit_step(sync_check_error):
    """logprobs=-1 under --max-logprobs -1 asks for every vocab id, next to a
    top-2 request in the same batch. Both get their rows on the step they
    emit, through syncs the checker is told about."""
    num_reqs, canvas_len, vocab, hidden = 2, 16, 1024, 32
    set_random_seed(0)
    weight = torch.randn(vocab, hidden, device=DEVICE)
    one_step = {"diffusion_max_steps": 1}
    params = [
        SamplingParams(logprobs=-1, extra_args=one_step),
        SamplingParams(logprobs=2, extra_args=one_step),
    ]
    sampler, _ = _make_sampler(
        num_reqs, canvas_len, vocab, hidden, weight, params=params
    )
    hs = torch.randn(num_reqs * canvas_len, hidden, device=DEVICE)
    step = with_gpu_sync_check(sampler)

    batch = _make_batch(num_reqs, [canvas_len] * num_reqs)
    # Both converge on the first step and emit on the commit step.
    assert step(hs, batch).logprobs_tensors is None
    out = step(hs, batch)

    lists = out.logprobs_tensors.tolists()
    for i, num_cols in enumerate((1 + vocab, 3)):
        lp = lists.slice_request(i, canvas_len)
        tokens = out.sampled_token_ids[i, :canvas_len].tolist()
        assert lp.logprob_token_ids[:, 0].tolist() == tokens
        assert np.isfinite(lp.logprobs[:, :num_cols]).all()
        assert np.isneginf(lp.logprobs[:, num_cols:]).all()


def _swap_sampler(swap_capacity=1):
    canvas_len, vocab, hidden = 16, 1024, 32
    set_random_seed(0)
    weight = torch.randn(vocab, hidden, device=DEVICE)
    sampler, states = _make_sampler(
        2, canvas_len, vocab, hidden, weight, swap_capacity=swap_capacity
    )
    return sampler, states, torch.randn(canvas_len, hidden, device=DEVICE)


def _admit(sampler, states, req_id, slot, params=None):
    """Add a request at `slot` and finish its prefill, as the runner does for
    a new request and for a preempted one that resumes."""
    states.add_request(slot, req_id)
    sampler.add_request(slot, params or SamplingParams())
    sampler.apply_staged_writes()
    last_prefill_step = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=0,
        idx_mapping_np=np.array([slot], dtype=np.intp),
        num_computed_prefill_tokens_np=np.zeros(1, dtype=np.int32),
        num_scheduled_tokens=np.ones(1, dtype=np.int32),
        prefill_len_np=np.ones(1, dtype=np.int32),
    )
    with_gpu_sync_check(sampler)(torch.empty(0, device=DEVICE), last_prefill_step)


def _preempt(states, req_id, slot):
    with_gpu_sync_check(states.swap_out)(req_id, slot)
    states.remove_request(slot)


def _denoise(sampler, hs, slot):
    return sampler(hs, _make_batch(1, [len(hs)], slots=[slot]))


def test_preempted_block_resumes_in_another_slot(sync_check_error):
    """A block preempted after the first of the two denoise steps it needs
    converges on its first step in the new slot. Keeping and resuming it
    never syncs with the GPU."""
    sampler, states, hs = _swap_sampler()
    _admit(sampler, states, "a", 0)
    _denoise(sampler, hs, 0)
    kept = states.canvas[0].clone()

    _preempt(states, "a", 0)
    _admit(sampler, states, "a", 1)
    assert torch.equal(states.canvas[1], kept)
    _denoise(sampler, hs, 1)
    assert bool(states.is_encoder_phase[1])


def test_converged_block_resumes_at_its_commit_with_logprobs():
    """A block preempted between converging and committing emits right after
    the request's prefill, with the logprobs it converged with."""
    params = SamplingParams(logprobs=2, extra_args={"diffusion_max_steps": 1})
    sampler, states, hs = _swap_sampler()
    _admit(sampler, states, "a", 0, params)
    _denoise(sampler, hs, 0)
    tokens = states.argmax_canvas[0].tolist()

    _preempt(states, "a", 0)
    _admit(sampler, states, "a", 1, params)
    out = _denoise(sampler, hs, 1)

    assert out.num_sampled.tolist() == [16]
    assert out.sampled_token_ids[0, :16].tolist() == tokens
    lp = out.logprobs_tensors.tolists().slice_request(0, 16)
    assert lp.logprob_token_ids[:, 0].tolist() == tokens


def test_kept_block_survives_a_preemption_during_its_prefill():
    sampler, states, hs = _swap_sampler()
    _admit(sampler, states, "a", 0)
    _denoise(sampler, hs, 0)
    kept = states.canvas[0].clone()

    _preempt(states, "a", 0)
    # Resumed, then preempted again before its prompt was prefilled.
    states.add_request(0, "a")
    _preempt(states, "a", 0)
    _admit(sampler, states, "a", 1)
    assert torch.equal(states.canvas[1], kept)


def test_oldest_kept_block_restarts_past_capacity():
    sampler, states, hs = _swap_sampler(swap_capacity=1)
    for req_id, slot in (("a", 0), ("b", 1)):
        _admit(sampler, states, req_id, slot)
        _denoise(sampler, hs, slot)

    _preempt(states, "a", 0)
    _preempt(states, "b", 1)
    _admit(sampler, states, "a", 0)
    _admit(sampler, states, "b", 1)
    assert states.step.tolist() == [0, 1]


def test_block_with_full_vocab_logprobs_restarts():
    """The full-vocabulary logprobs of a converged block are too large to
    keep, so the block is denoised again instead of committing without them."""
    wide = SamplingParams(logprobs=-1, extra_args={"diffusion_max_steps": 1})
    sampler, states, hs = _swap_sampler()
    _admit(sampler, states, "a", 0, wide)
    _denoise(sampler, hs, 0)
    assert 0 in states.wide_logprobs

    _preempt(states, "a", 0)
    _admit(sampler, states, "a", 1, wide)
    assert int(states.step[1]) == 0 and not states.wide_logprobs


def test_block_of_a_finished_request_is_not_resumed():
    """A request that ends while preempted frees its block: a later request
    with the same id starts from noise."""
    sampler, states, hs = _swap_sampler()
    _admit(sampler, states, "a", 0)
    _denoise(sampler, hs, 0)
    _preempt(states, "a", 0)

    states.swap_drop("a")
    _admit(sampler, states, "a", 1)
    assert int(states.step[1]) == 0
