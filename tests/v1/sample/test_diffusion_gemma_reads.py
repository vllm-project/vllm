# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request DiffusionGemma state behind structured reads: seed canvases,
pinned positions, read-only slots and the per-slot step cap."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    _MASKED_LOGIT,
    DiffusionGemmaRequestStates,
    _concat_logprob_stashes,
    _denoise_temperature,
    _mask_rows_to_allowed,
    _store_self_conditioning,
)
from vllm.model_executor.models.diffusion_gemma_sampler import (
    sample_row_stats,
    sample_row_stats_reference,
    sample_step,
    sample_step_reference,
    softcap_logits,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.platforms import current_platform
from vllm.v1.outputs import LogprobsTensors

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="the sampler state lives on the GPU"
)

CL = 8
VOCAB = 64
MAX_REQS = 4
MAX_STEPS = 48


def _states(
    max_num_reqs: int = MAX_REQS, stability_threshold: int = 2
) -> DiffusionGemmaRequestStates:
    return DiffusionGemmaRequestStates(
        max_num_reqs=max_num_reqs,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=MAX_STEPS,
        device=torch.device("cuda"),
        hidden_size=4,
        stability_threshold=stability_threshold,
    )


def _slots(*idx: int) -> tuple[np.ndarray, torch.Tensor]:
    slots = np.array(idx, dtype=np.int64)
    return slots, torch.tensor(slots, device="cuda")


def test_seed_canvas_replaces_only_seeded_slots():
    states = _states()
    for slot in range(3):
        states.add_request(slot)
    seed = list(range(CL))
    states.set_seed_canvas(1, seed)
    slots, slots_gpu = _slots(0, 1, 2)
    states.init_canvas(slots_gpu)
    before = states.canvas[slots_gpu].clone()

    states.apply_seed_canvases(slots, slots_gpu)

    after = states.canvas[slots_gpu]
    assert after[1].tolist() == seed
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[2], before[2])


def test_apply_seed_canvases_leaves_unseeded_batches_alone():
    states = _states()
    states.add_request(0)
    slots, slots_gpu = _slots(0)
    states.init_canvas(slots_gpu)
    before = states.canvas[0].clone()

    states.apply_seed_canvases(slots, slots_gpu)

    assert torch.equal(states.canvas[0], before)


def test_add_request_clears_seed_and_read_only():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_read_only(0)
    assert states.seeded_slots == {0}
    assert states.read_only_slots == {0}

    states.add_request(0)

    assert not states.seeded_slots
    assert not states.read_only_slots
    assert not bool(states.has_seed[0])
    assert not bool(states.read_only[0])


def test_canvas_width_resets_with_the_slot():
    states = _states()
    states.add_request(0)
    states.canvas_width_np[0] = 4
    states.set_seed_canvas(0, [7, 7, 7, 7])
    assert states.seed_canvas[0, :4].tolist() == [7, 7, 7, 7]

    states.add_request(0)

    assert states.canvas_width_np[0] == CL


def test_remove_request_forgets_the_slot():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_read_only(0)

    states.remove_request(0)

    assert not states.seeded_slots
    assert not states.read_only_slots


def _step_kwargs(
    states: DiffusionGemmaRequestStates,
    decode_slots: torch.Tensor,
    sample: torch.Tensor,
    argmax: torch.Tensor,
    entropy: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
) -> dict:
    n, width = sample.shape
    device = states.device
    return dict(
        new_tokens=sample,
        argmax_tokens=argmax,
        token_entropy=entropy,
        random_tokens=torch.randint(0, VOCAB, (n, width), device=device),
        decode_slots=decode_slots,
        decode_idx=torch.arange(n, dtype=torch.int64, device=device),
        valid_canvas_len=torch.full((n,), width, dtype=torch.int64, device=device),
        canvas=states.canvas,
        argmax_canvas=states.argmax_canvas,
        history=states.accepted_canvas_history,
        pin_mask=states.pin_mask,
        seed_canvas=states.seed_canvas,
        step=states.step,
        is_encoder_phase=states.is_encoder_phase,
        confident=states.confident,
        history_len=states.accepted_canvas_history_len,
        max_steps=states.max_steps,
        read_only=states.read_only,
        sampled=sampled,
        num_sampled=num_sampled,
        confidence_threshold=0.1,
        entropy_bound=0.1,
    )


def _denoise_once(
    states: DiffusionGemmaRequestStates,
    slots: list[int],
    compute_sc: bool = True,
    width: int = CL,
    embed_weight: torch.Tensor | None = None,
    embed_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One denoise step over `slots` with flat logits, so nothing
    converges by stability or confidence and only the step cap can end it.
    `width` below CL runs the step on the first `width` columns, as the
    sampler does for a narrow tile."""
    n = len(slots)
    device = states.device
    decode_slots = torch.tensor(slots, dtype=torch.int64, device=device)
    sampled = torch.zeros(n, CL, dtype=torch.int32, device=device)
    num_sampled = torch.zeros(n, dtype=torch.int32, device=device)
    temp = _denoise_temperature(states.step, decode_slots, float(MAX_STEPS), 0.5, 1.0)
    argmax, sample, entropy, probs = sample_row_stats_reference(
        torch.zeros(n * width, VOCAB, device=device),
        temp,
        width,
        embed_dtype if compute_sc else None,
    )
    keep = sample_step(
        **_step_kwargs(
            states,
            decode_slots,
            sample.view(n, width),
            argmax.view(n, width),
            entropy.view(n, width),
            sampled,
            num_sampled,
        )
    )
    _store_self_conditioning(
        states.self_conditioning_embeds[:, :width],
        decode_slots,
        keep,
        probs.view(n, width, -1) if probs is not None else None,
        states.pin_mask[:, :width],
        (
            torch.zeros(VOCAB, 4, dtype=embed_dtype, device=device)
            if embed_weight is None
            else embed_weight
        ),
        torch.tensor(1.0, dtype=embed_dtype, device=device),
        sc_vocab_start=0,
        sc_vocab_end=VOCAB,
        tp_size=1,
        tp_group_name="",
    )
    return sampled[:, :width], num_sampled


def test_pinned_positions_hold_their_seed_through_a_step():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    seed = list(range(10, 10 + CL))
    states.set_seed_canvas(0, seed)
    states.canvas[0] = torch.tensor(seed, device="cuda")
    states.set_pins(0, [0, 1, 2, 3])

    # Flat logits accept nothing, so every free position is renoised.
    _denoise_once(states, [0], embed_weight=torch.ones(VOCAB, 4, device="cuda"))

    assert states.canvas[0, :4].tolist() == seed[:4]
    # The soft embed is zero at pinned positions and non-zero elsewhere.
    sc = states.self_conditioning_embeds[0]
    assert not sc[:4].any()
    assert sc[4:].any()


def test_add_request_clears_pins():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_pins(0, [2, 3])
    assert states.pin_mask[0].tolist() == [False, False, True, True] + [False] * 4

    states.add_request(0)

    assert not states.pin_mask[0].any()


def test_single_step_decode_does_not_sync():
    # The sampler runs on the engine's hot path, where vLLM forbids GPU->CPU
    # syncs. A single-step tile zeroes the soft embeds instead of storing them.
    states = _states()
    for slot in (0, 1):
        states.add_request(slot)
        states.is_encoder_phase[slot] = False
    device = states.device
    slots = torch.tensor([0, 1], device=device)
    entropy = torch.rand(2, CL, device=device)
    kwargs = _step_kwargs(
        states,
        slots,
        torch.randint(0, VOCAB, (2, CL), device=device),
        torch.randint(0, VOCAB, (2, CL), device=device),
        entropy,
        torch.zeros(2, CL, dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
    )
    embed_weight = torch.ones(VOCAB, 4, device=device)
    normalizer = torch.tensor(1.0, device=device)
    torch.accelerator.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        keep = sample_step(**kwargs)
        _store_self_conditioning(
            states.self_conditioning_embeds,
            slots,
            keep,
            None,
            states.pin_mask,
            embed_weight,
            normalizer,
            sc_vocab_start=0,
            sc_vocab_end=VOCAB,
            tp_size=1,
            tp_group_name="",
        )
    finally:
        torch.cuda.set_sync_debug_mode(0)


def test_single_step_tile_skips_self_conditioning():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.self_conditioning_embeds[0] = 1.0

    _denoise_once(states, [0], compute_sc=False)

    assert not states.self_conditioning_embeds[0].any()


@pytest.mark.parametrize("width", [CL, 3, 1])
@pytest.mark.parametrize("stability_threshold", [1, 3])
def test_sample_step_kernel_matches_reference(width, stability_threshold):
    torch.manual_seed(0)
    n, num_reqs = 6, 8
    states = _states(max_num_reqs=num_reqs, stability_threshold=stability_threshold)
    device = states.device
    states.canvas.random_(VOCAB)
    states.argmax_canvas.random_(VOCAB)
    states.seed_canvas.random_(VOCAB)
    states.accepted_canvas_history.random_(VOCAB)
    states.pin_mask.copy_(torch.rand(num_reqs, CL, device=device) < 0.3)
    states.accepted_canvas_history_len.fill_(stability_threshold)

    # One request per outcome: 0 converges, 1 is a read that converges and
    # emits, 2 is stable but uncertain, 3 is confident but unstable, 4
    # commits and 5 hits its step cap.
    decode_slots = torch.tensor([5, 0, 3, 7, 1, 6], device=device)
    argmax = torch.randint(0, VOCAB, (n, width), device=device)
    states.accepted_canvas_history[decode_slots[:3], :, :width] = argmax[:3, None]
    # The step writes ring row 0 (history_len % ST), so break a later row.
    states.accepted_canvas_history[decode_slots[3], -1, :width] = argmax[3] + 1
    states.read_only[decode_slots[1]] = True
    states.is_encoder_phase[decode_slots[4]] = True
    states.step[decode_slots[5]] = MAX_STEPS - 1
    # Confident rows, an uncertain one ending in a negative NaN (which sorts
    # last, as in torch.sort), and tied entropies.
    entropy = torch.rand(n, width, device=device) * 0.05
    entropy[2] = 0.15 + 0.01 * torch.rand(width, device=device)
    entropy[2, -1] = torch.tensor(float("nan")).neg()
    entropy[0] = 0.02
    kwargs = _step_kwargs(
        states,
        decode_slots,
        torch.randint(0, VOCAB, (n, width), device=device),
        argmax,
        entropy,
        torch.full((num_reqs, CL), -1, dtype=torch.int32, device=device),
        torch.full((num_reqs,), -1, dtype=torch.int32, device=device),
    )
    kwargs["decode_idx"] = torch.tensor([2, 0, 5, 1, 7, 3], device=device)
    kwargs["valid_canvas_len"] = torch.randint(1, width + 1, (n,), device=device)
    mutable = (
        "canvas",
        "argmax_canvas",
        "history",
        "step",
        "is_encoder_phase",
        "confident",
        "history_len",
        "sampled",
        "num_sampled",
    )
    ref_kwargs = {k: v.clone() if k in mutable else v for k, v in kwargs.items()}

    keep = sample_step(**kwargs)
    ref_keep = sample_step_reference(**ref_kwargs)

    assert torch.equal(keep, ref_keep)
    for name in mutable:
        assert torch.equal(kwargs[name], ref_kwargs[name]), name
    num_sampled = ref_kwargs["num_sampled"][kwargs["decode_idx"]]
    assert (num_sampled[[1, 4]] > 0).all()
    assert ref_keep.tolist() == [
        False,
        False,
        True,
        stability_threshold > 1,
        False,
        False,
    ]


def test_softcap_kernel_matches_reference():
    logits = torch.randn(5, 3 * 4096 + 7, dtype=torch.bfloat16, device="cuda") * 40

    out = softcap_logits(logits, 30.0)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.tanh(logits.float() / 30.0) * 30.0)


def test_jit_warmup_covers_every_serving_variant():
    """Startup warmup compiles every kernel variant serving reaches, so the
    JIT monitor sees no compile in the engine step."""
    registry = JitWarmupRegistry(
        SimpleNamespace(
            model_config=SimpleNamespace(
                try_get_generation_config=lambda: {"stability_threshold": 1},
                get_vocab_size=lambda: VOCAB,
                dtype=torch.bfloat16,
            ),
            diffusion_config=SimpleNamespace(canvas_length=CL),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=MAX_REQS),
        )
    )
    dispatchers = (softcap_logits, sample_row_stats, sample_step)
    for dispatcher in dispatchers:
        # Drop variants earlier tests compiled, which would hide a gap.
        dispatcher.kernel.device_caches.clear()
    with registry.activate():
        for dispatcher in dispatchers:
            dispatcher.register_warmup()
    registry.warmup()

    def num_variants() -> list[int]:
        return [
            sum(len(cache[0]) for cache in d.kernel.device_caches.values())
            for d in dispatchers
        ]

    warmed = num_variants()
    assert all(warmed)
    for cols in (VOCAB, 1, 7, 16, 48):
        softcap_logits(torch.randn(3, cols, device="cuda"), 30.0)
        for width in (4, 1, 3):
            for probs_dtype in (None, torch.bfloat16):
                sample_row_stats(
                    torch.randn(2 * width, cols, device="cuda"),
                    torch.ones(2, device="cuda"),
                    width,
                    123,
                    probs_dtype,
                )
    softcap_logits(torch.randn(3, VOCAB, dtype=torch.bfloat16, device="cuda"), 30.0)
    for width, slots in ((CL, [0]), (3, [0, 1, 2]), (1, [3, 1])):
        states = _states()
        for slot in slots:
            states.add_request(slot)
        _denoise_once(states, slots, width=width)

    assert num_variants() == warmed


def test_self_conditioning_stores_a_bf16_model_in_the_fp32_buffer():
    # The model's embeddings are bf16 while the buffer is fp32.
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False

    _denoise_once(
        states,
        [0],
        embed_weight=torch.ones(VOCAB, 4, dtype=torch.bfloat16, device="cuda"),
        embed_dtype=torch.bfloat16,
    )

    assert states.self_conditioning_embeds.dtype == torch.float32
    # uniform probs @ an all-ones embedding
    assert torch.allclose(
        states.self_conditioning_embeds[0], torch.ones(CL, 4, device="cuda")
    )


def test_narrow_tile_leaves_the_rest_of_the_canvas_alone():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.canvas[0] = 5
    states.argmax_canvas[0] = 5

    _denoise_once(states, [0], width=4)

    # Columns past the width are untouched; the step counted.
    assert states.canvas[0, 4:].tolist() == [5] * (CL - 4)
    assert states.argmax_canvas[0, 4:].tolist() == [5] * (CL - 4)
    assert states.step[0] == 1


def test_step_cap_is_per_slot():
    states = _states()
    for slot in (0, 1):
        states.add_request(slot)
        states.is_encoder_phase[slot] = False
    states.max_steps[0] = 1

    _denoise_once(states, [0, 1])

    # Slot 0 hit its cap and moves to commit. Slot 1 keeps denoising.
    assert states.is_encoder_phase[:2].tolist() == [True, False]
    assert states.step[:2].tolist() == [1, 1]


def _stash(rows: int, width: int, first_id: int) -> LogprobsTensors:
    ids = torch.arange(first_id, first_id + rows * width).reshape(rows, width)
    return LogprobsTensors(
        logprob_token_ids=ids,
        logprobs=-ids.float(),
        selected_token_ranks=torch.zeros(rows, dtype=torch.int64),
    )


def test_stashes_of_different_widths_join():
    # A read that asked for 10 label ids and a generation that asked for 2
    # logprobs converged in different steps, so their stashes differ in width.
    wide, narrow = _stash(2, 11, 100), _stash(3, 3, 0)

    out = _concat_logprob_stashes([narrow, wide], [0, 3])

    assert out.logprob_token_ids.shape == (5, 11)
    assert out.logprobs.shape == (5, 11)
    assert out.cu_num_generated_tokens == [0, 3]
    # The narrow rows keep their values and pad with id 0 at -inf.
    assert torch.equal(out.logprob_token_ids[:3, :3], narrow.logprob_token_ids)
    assert not out.logprob_token_ids[:3, 3:].any()
    assert torch.isneginf(out.logprobs[:3, 3:]).all()
    assert torch.equal(out.logprobs[3:], wide.logprobs)


@pytest.mark.parametrize("width", [4, CL])
@pytest.mark.parametrize("steps", [1, 2])
def test_read_emits_at_convergence_while_generation_waits_for_commit(width, steps):
    states = _states()
    for slot in (2, 0):
        states.add_request(slot)
        states.is_encoder_phase[slot] = False
        states.max_steps[slot] = steps
    states.set_read_only(2)

    for step in range(steps):
        sampled, counts = _denoise_once(states, [2, 0], width=width)
        assert counts.tolist() == ([width, 0] if step == steps - 1 else [0, 0])

    assert torch.equal(sampled[0], states.argmax_canvas[2, :width].int())
    assert not states.is_encoder_phase[2]
    assert states.is_encoder_phase[0]
    assert not states.self_conditioning_embeds[2].any()

    sampled, counts = _denoise_once(states, [0], width=width)
    assert counts.tolist() == [width]
    assert torch.equal(sampled[0], states.argmax_canvas[0, :width].int())


def test_batch_allowed_needs_one_shared_set():
    states = _states()
    states.constrained[0] = (1, 2, 3)
    states.constrained[1] = (4, 5)

    # Mixed sets, or a slot with no set, fall back to per-row masking.
    assert states.batch_allowed([0, 1]) is None
    assert states.batch_allowed([0, 2]) is None
    assert states.batch_allowed([]) is None

    shared = states.batch_allowed([0])
    assert shared is not None
    assert shared.tolist() == [1, 2, 3]
    assert shared.dtype == torch.int64
    assert states.batch_allowed([0, 0]) is shared
    assert states.allowed_tensor((1, 2, 3)) is shared


def test_mask_rows_to_allowed_masks_only_constrained_rows():
    logits = torch.randn(5, VOCAB, device="cuda")
    before = logits.clone()
    first = torch.tensor([3, 7], device="cuda")
    third = torch.tensor([0], device="cuda")

    out = _mask_rows_to_allowed(logits, [0, 2, 3], [2, 1, 2], [first, None, third])

    assert out is not logits
    assert torch.equal(logits, before)
    # Masked columns hold the finite sentinel, so entropy stays finite.
    kept = out > _MASKED_LOGIT
    assert torch.isfinite(out).all()
    assert kept[0:2].sum(dim=1).tolist() == [2, 2]
    assert kept[0:2][:, first].all()
    assert torch.equal(out[0:2][:, first], before[0:2][:, first])
    assert torch.equal(out[2], before[2])
    assert kept[3:5].sum(dim=1).tolist() == [1, 1]
    assert torch.equal(out[3:5][:, third], before[3:5][:, third])
    # The masked row's softmax is the distribution renormalized over the set.
    torch.testing.assert_close(
        out[0].softmax(dim=-1)[first], before[0, first].softmax(dim=-1)
    )


def test_mask_rows_to_allowed_is_a_no_op_without_constrained_rows():
    logits = torch.randn(3, VOCAB, device="cuda")
    assert _mask_rows_to_allowed(logits, [0, 1], [1, 2], [None, None]) is logits
