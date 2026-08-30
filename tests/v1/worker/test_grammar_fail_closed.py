# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance must stop at the last bitmask row that carries a real mask.

`grammar_bitmask` stops constraining rows once it meets a -1 placeholder in the
scheduled draft window, and the worker verifies the drafts it holds on the GPU
rather than the scheduler's padded copy. When the two disagree -- the hand-off
lost a step's drafts, see #54437 -- acceptance can walk into a row where every
token is allowed.

The grammar here admits only LEGAL. The scheduled window is back-filled one
draft deep, so rows 0 and 1 carry a real mask and rows 2.. are `_full_mask`,
while the drafts on the GPU would carry acceptance into row 2 and emit ILLEGAL.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

VOCAB = 128
K = 3  # drafts per step
NUM_LOGITS = K + 1  # + the bonus row
LEGAL = 7
ILLEGAL = 99
# Index of the first -1 in the scheduled window: rows 0..1 are masked.
NUM_ACCEPTABLE = 1


def _bitmask() -> np.ndarray:
    """Rows 0..NUM_ACCEPTABLE constrained to LEGAL, the rest fully permissive."""
    mask = np.zeros((NUM_LOGITS, cdiv(VOCAB, 32)), dtype=np.int32)
    for row in range(NUM_LOGITS):
        if row <= NUM_ACCEPTABLE:
            mask[row, LEGAL // 32] = 1 << (LEGAL % 32)
        else:
            mask[row, :] = -1  # every token allowed
    return mask


def _input_batch(device: torch.device) -> SimpleNamespace:
    cu_np = np.array([0, NUM_LOGITS], dtype=np.int32)
    # draft_sampled = input_ids[logits_indices]; entry i + 1 is draft i, so the
    # drafts the GPU holds are [LEGAL, LEGAL, ILLEGAL].
    input_ids = torch.tensor(
        [LEGAL, LEGAL, LEGAL, ILLEGAL], dtype=torch.int32, device=device
    )
    return SimpleNamespace(
        num_reqs=1,
        req_ids=["g"],
        num_draft_tokens=K,
        num_draft_tokens_per_req=np.array([K], dtype=np.int32),
        input_ids=input_ids,
        logits_indices=torch.arange(NUM_LOGITS, dtype=torch.int32, device=device),
        positions=torch.arange(NUM_LOGITS, dtype=torch.int64, device=device),
        cu_num_logits_np=cu_np,
        cu_num_logits=torch.from_numpy(cu_np).to(device),
        idx_mapping_np=np.array([0], dtype=np.int32),
        idx_mapping=torch.zeros(1, dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.zeros(NUM_LOGITS, dtype=torch.int32, device=device),
        expanded_local_pos=torch.arange(NUM_LOGITS, dtype=torch.int32, device=device),
        seq_lens=torch.tensor([64], dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([64], dtype=torch.int32),
    )


def _logits(device: torch.device) -> torch.Tensor:
    """Every row prefers its own draft, so nothing but the mask can reject one."""
    logits = torch.zeros(NUM_LOGITS, VOCAB, dtype=torch.float32, device=device)
    logits[0, LEGAL] = 10.0
    logits[1, LEGAL] = 10.0
    logits[2, ILLEGAL] = 10.0  # row 2 is permissive, so this wins there
    logits[3, LEGAL] = 10.0
    return logits


def _rejection_sampler(device: torch.device) -> RejectionSampler:
    sampler = object.__new__(RejectionSampler)
    sampler.num_speculative_steps = K
    sampler.enable_adaptive_verification = False
    sampler.use_block_verification = False
    sampler.synthetic_conditional_rates = None
    sampler.watermark_key = None
    zeros_f = torch.zeros(1, dtype=torch.float32, device=device)  # greedy
    sampler.sampler = SimpleNamespace(
        compute_nans=False,
        logprobs_mode="raw_logprobs",
        use_fp64_gumbel=False,
        apply_sampling_params=lambda logits, *args, **kwargs: logits,
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=zeros_f),
            seeds=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int64, device=device)),
            max_num_logprobs=lambda _: NO_LOGPROBS,
        ),
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(
                gpu=torch.zeros(1, dtype=torch.int32, device=device)
            )
        ),
    )
    return sampler


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_acceptance_stops_at_the_last_masked_row():
    device = torch.device("cuda")
    input_batch = _input_batch(device)
    logits = _logits(device)

    runner = object.__new__(GPUModelRunner)
    runner.device = device
    runner.vocab_size = VOCAB
    runner.batch_sharder = None
    runner.sampler = None
    runner.model = SimpleNamespace(compute_logits=lambda hidden: logits)
    runner.rejection_sampler = _rejection_sampler(device)
    runner.speculator = SimpleNamespace(draft_logits=None)
    runner.structured_outputs_worker = StructuredOutputsWorker(
        max_num_logits=NUM_LOGITS,
        vocab_size=VOCAB,
        device=device,
        mask_stride=NUM_LOGITS,
        num_bonus_tokens=1,
    )

    grammar_output = GrammarOutput(["g"], _bitmask())
    # The scheduler reports how many leading drafts the bitmask constrained.
    # Absent on an older scheduler, where the whole window is invalidated.
    grammar_output.num_acceptable_drafts = [NUM_ACCEPTABLE]

    hidden = torch.zeros(NUM_LOGITS, 8, dtype=torch.float32, device=device)
    sampler_output, num_sampled, _ = runner.sample(hidden, input_batch, grammar_output)

    accepted = int(num_sampled[0].item())
    emitted = sampler_output.sampled_token_ids[0, :accepted].tolist()
    assert ILLEGAL not in emitted, (
        f"emitted {emitted}: a draft verified against a permissive row was "
        f"accepted, so the request sampled with no grammar constraint"
    )
    # Draft 0 may be accepted; draft 1 must not be, because accepting it moves
    # sampling into row 2, which constrains nothing.
    assert accepted == NUM_ACCEPTABLE + 1
    assert emitted == [LEGAL] * accepted
