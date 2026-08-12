# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionSampler projects logits tile-by-tile (#50699): sampling outputs
must not depend on the tile size, and peak memory must scale with the tile,
not the batch."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models import diffusion_gemma as dg
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.states import SamplingStates

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda"


def _make_sampler(num_reqs, canvas_len, vocab, hidden, weight, top_k=0):
    states = dg.DiffusionGemmaRequestStates(
        max_num_reqs=num_reqs,
        canvas_length=canvas_len,
        vocab_size=vocab,
        max_denoising_steps=8,
        device=torch.device(DEVICE),
        hidden_size=hidden,
        stability_threshold=2,
    )
    base = SimpleNamespace(
        sampling_states=SamplingStates(num_reqs, vocab),
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
        sampler.add_request(i, 4, SamplingParams(top_k=top_k))
        states.is_encoder_phase[i] = False
    sampler.apply_staged_writes()
    return sampler, states


def _make_batch(num_reqs, valid_lens):
    cu = np.zeros(num_reqs + 1, dtype=np.int64)
    cu[1:] = np.cumsum(valid_lens)
    qsl_np = cu.astype(np.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=int(max(valid_lens)),
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
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
            current_platform, "mem_get_info", lambda free=free: (free, 2**60)
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
            current_platform, "mem_get_info", lambda free=free: (free, 2**60)
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
