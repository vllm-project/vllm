# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.models import diffusion_gemma as dg
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _sampling_states(
    *,
    vocab_size: int,
    logprobs: int = dg.NO_LOGPROBS,
    seed_set: bool = False,
    top_k: int | None = None,
    top_p: float = 1.0,
    min_p: float = 0.0,
):
    top_k_value = vocab_size if top_k is None else top_k
    return SimpleNamespace(
        vocab_size=vocab_size,
        seeds_set=np.array([seed_set]),
        top_k=SimpleNamespace(np=np.array([top_k_value], dtype=np.int32)),
        top_p=SimpleNamespace(np=np.array([top_p], dtype=np.float32)),
        min_p=SimpleNamespace(np=np.array([min_p], dtype=np.float32)),
        max_num_logprobs=lambda slots: logprobs,
    )


def _local_sampler(
    *,
    vocab_size: int = 8,
    canvas_length: int = 3,
    tp_size: int = 2,
    tp_rank: int = 0,
    sc_vocab_start: int = 0,
    sc_vocab_end: int = 4,
    sampling_states=None,
):
    sampler = dg.DiffusionSampler.__new__(dg.DiffusionSampler)
    sampler.vocab_size = vocab_size
    sampler.canvas_length = canvas_length
    sampler.tp_size = tp_size
    sampler.tp_rank = tp_rank
    sampler.sc_vocab_start = sc_vocab_start
    sampler.sc_vocab_end = sc_vocab_end
    sampler.embed_weight = torch.empty(sc_vocab_end - sc_vocab_start, 2)
    sampler.sampling_states = sampling_states or _sampling_states(
        vocab_size=vocab_size
    )
    return sampler


def _input_batch(*, num_reqs: int = 1, num_draft_tokens: int = 3, num_logits=3):
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=num_draft_tokens,
        cu_num_logits_np=np.array([0, num_logits], dtype=np.int32),
        idx_mapping_np=np.array([0], dtype=np.int32),
    )


def test_dense_compatible_local_uniform_consumes_full_vocab_rng():
    device = torch.device("cpu")

    torch.manual_seed(1234)
    local = dg._dense_compatible_local_uniform((2, 3), 11, 4, 9, device)
    next_after_local = torch.randint(0, 11, (2, 3), device=device)

    torch.manual_seed(1234)
    dense = torch.rand((2, 3, 11), device=device, dtype=torch.float32)
    next_after_dense = torch.randint(0, 11, (2, 3), device=device)

    torch.testing.assert_close(local, dense[..., 4:9])
    assert torch.equal(next_after_local, next_after_dense)


def test_global_token_argmax_breaks_tp_ties_by_lowest_global_id(monkeypatch):
    local_values = torch.tensor([[5.0, 7.0, 1.0]])
    local_indices = torch.tensor([[4, 5, 0]])
    other_pair = torch.tensor([[[5.0, 2.0], [6.0, 1.0], [1.0, 7.0]]])

    def fake_all_gather(tensor, tp_size, tp_group_name):
        return torch.cat([tensor, other_pair], dim=-1)

    monkeypatch.setattr(dg, "_tp_all_gather_last_dim", fake_all_gather)

    tokens = dg._global_token_argmax(
        local_values, local_indices, vocab_size=8, tp_size=2, tp_group_name="tp"
    )

    assert torch.equal(tokens, torch.tensor([[2, 5, 0]]))


def test_local_sample_step_matches_dense_step_for_unsharded_vocab():
    vocab_size = 7
    canvas_length = 3
    hidden_size = 4
    stability_threshold = 2
    logits = torch.randn(canvas_length, vocab_size)
    embed_weight = torch.randn(vocab_size, hidden_size)
    normalizer = torch.tensor(1.5)

    def tensors():
        return {
            "canvas": torch.zeros(1, canvas_length, dtype=torch.int64),
            "argmax_canvas": torch.zeros(1, canvas_length, dtype=torch.int64),
            "step": torch.zeros(1, dtype=torch.int32),
            "is_encoder_phase": torch.zeros(1, dtype=torch.bool),
            "confident": torch.zeros(1, dtype=torch.bool),
            "sc_embeds": torch.zeros(1, canvas_length, hidden_size),
            "history": torch.zeros(
                1, stability_threshold, canvas_length, dtype=torch.int64
            ),
            "history_len": torch.zeros(1, dtype=torch.int32),
            "sampled": torch.zeros(1, canvas_length, dtype=torch.int32),
            "num_sampled": torch.zeros(1, dtype=torch.int32),
            "draft_tokens": torch.zeros(1, canvas_length, dtype=torch.int64),
        }

    common = {
        "decode_slots": torch.tensor([0], dtype=torch.int64),
        "decode_idx": torch.tensor([0], dtype=torch.int64),
        "all_slots": torch.tensor([0], dtype=torch.int64),
        "valid_canvas_len": torch.tensor([canvas_length], dtype=torch.int64),
        "embed_weight": embed_weight,
        "normalizer": normalizer,
        "max_denoising_steps": 8.0,
        "t_min": 0.2,
        "t_max": 1.0,
        "confidence_threshold": 100.0,
        "vocab_size": vocab_size,
        "CL": canvas_length,
        "ST": stability_threshold,
        "entropy_bound": 0.5,
        "tp_size": 1,
        "tp_group_name": "",
    }

    dense = tensors()
    torch.manual_seed(123)
    dense_sample_step = getattr(
        dg._compiled_sample_step, "__wrapped__", dg._compiled_sample_step
    )
    dense_sample_step(
        logits,
        common["decode_slots"],
        common["decode_idx"],
        common["all_slots"],
        common["valid_canvas_len"],
        dense["canvas"],
        dense["argmax_canvas"],
        dense["step"],
        dense["is_encoder_phase"],
        dense["confident"],
        dense["sc_embeds"],
        common["embed_weight"],
        common["normalizer"],
        dense["history"],
        dense["history_len"],
        dense["sampled"],
        dense["num_sampled"],
        dense["draft_tokens"],
        common["max_denoising_steps"],
        common["t_min"],
        common["t_max"],
        common["confidence_threshold"],
        common["vocab_size"],
        common["CL"],
        common["ST"],
        common["entropy_bound"],
        0,
        vocab_size,
        common["tp_size"],
        common["tp_group_name"],
    )
    next_after_dense = torch.randint(0, vocab_size, (2,))

    local = tensors()
    torch.manual_seed(123)
    dg._diffusion_gemma_local_sample_step(
        logits,
        common["decode_slots"],
        common["decode_idx"],
        common["all_slots"],
        common["valid_canvas_len"],
        local["canvas"],
        local["argmax_canvas"],
        local["step"],
        local["is_encoder_phase"],
        local["confident"],
        local["sc_embeds"],
        common["embed_weight"],
        common["normalizer"],
        local["history"],
        local["history_len"],
        local["sampled"],
        local["num_sampled"],
        local["draft_tokens"],
        common["max_denoising_steps"],
        common["t_min"],
        common["t_max"],
        common["confidence_threshold"],
        common["vocab_size"],
        common["CL"],
        common["ST"],
        common["entropy_bound"],
        0,
        vocab_size,
        common["tp_size"],
        common["tp_group_name"],
    )
    next_after_local = torch.randint(0, vocab_size, (2,))

    assert torch.equal(next_after_local, next_after_dense)
    for key in dense:
        torch.testing.assert_close(local[key], dense[key])


def test_local_logits_eligibility_fails_closed_for_full_vocab_consumers():
    sampler = _local_sampler()
    assert sampler.can_sample_local_logits(_input_batch())

    assert not _local_sampler(
        sampling_states=_sampling_states(vocab_size=8, logprobs=1)
    ).can_sample_local_logits(_input_batch())
    assert not _local_sampler(
        sampling_states=_sampling_states(vocab_size=8, top_k=4)
    ).can_sample_local_logits(_input_batch())
    assert not _local_sampler(
        sampling_states=_sampling_states(vocab_size=8, top_p=0.5)
    ).can_sample_local_logits(_input_batch())
    assert not _local_sampler(
        sampling_states=_sampling_states(vocab_size=8, min_p=0.1)
    ).can_sample_local_logits(_input_batch())
    assert not _local_sampler(
        sampling_states=_sampling_states(vocab_size=8, seed_set=True)
    ).can_sample_local_logits(_input_batch())


def test_local_logits_eligibility_fails_closed_for_geometry_and_routing():
    assert not _local_sampler(tp_size=1).can_sample_local_logits(_input_batch())
    assert not _local_sampler(vocab_size=9, sc_vocab_end=4).can_sample_local_logits(
        _input_batch()
    )
    assert not _local_sampler(sc_vocab_start=1, sc_vocab_end=5).can_sample_local_logits(
        _input_batch()
    )
    assert not _local_sampler().can_sample_local_logits(
        _input_batch(num_logits=2)
    )
    assert not _local_sampler().can_sample_local_logits(
        _input_batch(), grammar_output=object()
    )


def test_runner_uses_local_logits_only_after_sampler_accepts():
    hidden_states = torch.randn(3, 2)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=3,
        logits_indices=torch.tensor([0, 1, 2]),
    )
    output = SimpleNamespace(
        num_sampled=torch.tensor([0]), num_rejected=torch.tensor([0])
    )

    class FakeSampler:
        def __init__(self, enabled):
            self.enabled = enabled
            self.local_called = False
            self.dense_called = False

        def can_sample_local_logits(self, batch, grammar_output):
            return self.enabled

        def sample_local_logits(self, logits, batch):
            self.local_called = True
            assert logits.item() == 1
            return output

        def __call__(self, logits, batch):
            self.dense_called = True
            assert logits.item() == 2
            return output

    class FakeModel:
        def __init__(self):
            self.local_called = False
            self.dense_called = False

        def compute_diffusion_logits_local(self, states):
            self.local_called = True
            return torch.tensor(1)

        def compute_logits(self, states):
            self.dense_called = True
            return torch.tensor(2)

    runner = SimpleNamespace(
        sampler=FakeSampler(enabled=True),
        rejection_sampler=None,
        model=FakeModel(),
        batch_sharder=None,
    )
    sampler_output, _, _ = GPUModelRunner.sample(
        runner, hidden_states, input_batch, grammar_output=None
    )
    assert sampler_output is output
    assert runner.model.local_called
    assert runner.sampler.local_called
    assert not runner.model.dense_called

    runner = SimpleNamespace(
        sampler=FakeSampler(enabled=False),
        rejection_sampler=None,
        model=FakeModel(),
        batch_sharder=None,
    )
    sampler_output, _, _ = GPUModelRunner.sample(
        runner, hidden_states, input_batch, grammar_output=None
    )
    assert sampler_output is output
    assert runner.model.dense_called
    assert runner.sampler.dense_called
    assert not runner.model.local_called
