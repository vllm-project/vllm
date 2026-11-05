# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LogprobsProcessor.

These tests exercise the truncation invariant that the MRV2 sampler relies
on: when the sampler returns a row wider than a request's own
`num_logprobs + 1` (because another request in the batch needed a wider
row), the trailing positions are populated with sentinel values
(`token_id=0`, `logprob=-inf`). LogprobsProcessor must read only the first
`num_logprobs + 1` entries so those sentinels never reach the user.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from transformers import TokenizersBackend

from vllm.logprobs import Logprob, create_sample_logprobs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists, LogprobsRows, LogprobsTensors


def _make_processor(num_logprobs: int) -> LogprobsProcessor:
    return LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat_logprobs=False),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=num_logprobs,
        num_prompt_logprobs=None,
    )


def test_drops_trailing_sentinel_columns():
    """A request that asked for 3 custom token logprobs but ended up in a
    batch padded to width 5 must not surface the trailing -inf entries."""
    processor = _make_processor(num_logprobs=3)

    sampled = 42
    # Layout: [sampled, custom_1, custom_2, custom_3, SENTINEL, SENTINEL]
    # Use float32-exact values so cumulative_logprob compares cleanly.
    token_ids = np.array([[sampled, 100, 200, 300, 0, 0]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.0, -2.0, -3.0, -np.inf, -np.inf]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    assert len(processor.logprobs) == 1
    pos = processor.logprobs[0]
    # Exactly sampled + 3 requested tokens; trailing sentinels dropped.
    assert set(pos.keys()) == {sampled, 100, 200, 300}
    assert 0 not in pos
    assert all(np.isfinite(lp.logprob) for lp in pos.values())
    # cumulative_logprob comes from the sampled token's logprob only.
    assert processor.cumulative_logprob == -0.5


def test_accepts_exactly_sized_row():
    """When the row is exactly num_logprobs+1, no truncation needed."""
    processor = _make_processor(num_logprobs=2)

    token_ids = np.array([[7, 11, 13]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.5, -2.5]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    pos = processor.logprobs[0]
    assert set(pos.keys()) == {7, 11, 13}


def _byte_tokenizer() -> TokenizersBackend:
    vocab = {f"<0x{i:02X}>": i for i in range(256)}
    vocab["<unk>"] = 256
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.decoder = decoders.ByteFallback()
    return TokenizersBackend(tokenizer_object=tokenizer)


def _reference_sample_logprobs(
    tokenizer: TokenizersBackend,
    num_logprobs: int,
    steps: list[LogprobsLists],
) -> tuple[list[dict[int, Logprob]], float]:
    """Sample logprobs as LogprobsProcessor built them before LogprobsRows:
    per-position tolist(), fresh decodes, eager context, keyword Logprobs."""
    ref = LogprobsProcessor(
        tokenizer=tokenizer,
        logprobs=[],
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=num_logprobs,
        num_prompt_logprobs=None,
    )
    for token_ids_np, logprobs_np, ranks_np, _ in steps:
        for rank_np, position_np, ids_np in zip(ranks_np, logprobs_np, token_ids_np):
            rank, logprobs, token_ids = (
                rank_np.tolist(),
                position_np.tolist(),
                ids_np.tolist(),
            )
            context = ref._get_sampled_context_ids(ref.logprobs)
            decoded = [tokenizer.decode([tid]) for tid in token_ids]
            decoded = [
                ref._correct_decoded_token(tid, context) if text.endswith("�") else text
                for tid, text in zip(token_ids, decoded)
            ]
            ref.cumulative_logprob += logprobs[0]
            width = len(logprobs) if num_logprobs == -1 else num_logprobs
            ranks = itertools.chain((rank,), range(1, width + 1))
            ref.logprobs.append(
                {
                    tid: Logprob(logprob=lp, rank=r, decoded_token=text)
                    for tid, lp, r, text in zip(token_ids, logprobs, ranks, decoded)
                }
            )
    return ref.logprobs, ref.cumulative_logprob


@pytest.mark.parametrize("num_logprobs", [0, 1, 3, -1])
def test_logprob_rows_and_arrays_match_reference(num_logprobs: int):
    """Both wire forms, and a request switching between them (as when a wide
    request joins its batch), give the dicts the pinned implementation built,
    in the same insertion order: duplicate ids (sampled token in the top-k),
    rows wider than the request, multi-position steps (spec decode), U+FFFD
    corrected from context, and history dropped after each DELTA output."""
    rng = np.random.default_rng(0)
    tokenizer = _byte_tokenizer()
    width = 5
    # "é" and "中" as split UTF-8 bytes, so single-id decodes end in U+FFFD.
    utf8_ids = [0xC3, 0xA9, 0xE4, 0xB8, 0xAD]
    steps = []
    for step in range(60):
        num_positions = 1 + step % 3
        token_ids = rng.integers(32, 256, (num_positions, width)).astype(np.int64)
        token_ids[:, 0] = [utf8_ids[(step + p) % 5] for p in range(num_positions)]
        token_ids[0, 1] = token_ids[0, 0]
        logprobs = -np.abs(rng.standard_normal((num_positions, width)))
        logprobs = logprobs.astype(np.float32)
        logprobs[:, -1] = -np.inf
        ranks = rng.integers(1, 50, num_positions).astype(np.int64)
        steps.append(LogprobsLists(token_ids, logprobs, ranks))
    ref_logprobs, ref_cumulative = _reference_sample_logprobs(
        tokenizer, num_logprobs, steps
    )
    assert any(lp.decoded_token == "é" for d in ref_logprobs for lp in d.values())

    for form in ("arrays", "rows", "mixed"):
        processor = LogprobsProcessor(
            tokenizer=tokenizer,
            logprobs=create_sample_logprobs(flat_logprobs=False),
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=num_logprobs,
            num_prompt_logprobs=None,
        )
        returned = []
        for i, step in enumerate(steps):
            num_positions = len(step.sampled_token_ranks)
            output = EngineCoreOutput(
                request_id="req-0", new_token_ids=[0] * num_positions
            )
            if form == "rows" or (form == "mixed" and i % 2):
                output.new_logprob_rows = step.slice_request_rows(0, num_positions)
            else:
                output.new_logprobs = step
            processor.update_from_output(output)
            assert processor.logprobs is not None
            returned.extend(processor.logprobs[-num_positions:])
            processor.drop_returned_sample_logprobs()

        assert [list(d.items()) for d in returned] == [
            list(d.items()) for d in ref_logprobs
        ]
        assert processor.cumulative_logprob == ref_cumulative


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_drop_returned_sample_logprobs_keeps_context(flat_logprobs: bool):
    processor = _make_processor(num_logprobs=0)
    processor.logprobs = create_sample_logprobs(flat_logprobs)
    for token_id in range(20):
        processor._update_sample_logprob_rows(LogprobsRows([[token_id]], [[-0.5]], [1]))
    context = processor._get_sampled_context_ids(processor.logprobs)

    processor.drop_returned_sample_logprobs()

    assert processor._get_sampled_context_ids(processor.logprobs) == context
    assert len(processor.logprobs) == (20 if flat_logprobs else 4)


def test_prompt_token_id_logprobs_are_popped_once():
    """DELTA outputs carry fixed-ID prompt scores exactly once."""
    processor = _make_processor(num_logprobs=1)
    assert processor.pop_prompt_token_id_logprobs() is None

    processor.update_from_output(
        EngineCoreOutput(
            request_id="req-0",
            new_token_ids=[],
            prompt_token_id_logprobs=torch.tensor(
                [[-0.5, -1.5], [-2.5, -3.5]], dtype=torch.float32
            ),
        )
    )

    scores = processor.pop_prompt_token_id_logprobs()
    assert scores is not None
    assert scores.tolist() == [[-0.5, -1.5], [-2.5, -3.5]]
    assert processor.pop_prompt_token_id_logprobs() is None


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_prompt_logprob_start_pads_unscored_positions(flat_logprobs):
    """Positions up to prompt_logprob_start are None, so the rows the worker
    returns from the start onward stay aligned with prompt positions."""
    params = SamplingParams(
        prompt_logprobs=0, prompt_logprob_start=2, flat_logprobs=flat_logprobs
    )
    processor = LogprobsProcessor.from_new_request(
        None, SimpleNamespace(sampling_params=params)
    )
    processor._update_prompt_logprobs(
        LogprobsTensors(
            logprob_token_ids=torch.tensor([[7], [8]]),
            logprobs=torch.tensor([[-0.5], [-1.5]]),
            selected_token_ranks=torch.tensor([1, 1]),
        )
    )

    prompt_logprobs = processor.pop_prompt_logprobs()
    assert prompt_logprobs is not None
    assert len(prompt_logprobs) == 5
    assert not any(prompt_logprobs[i] for i in range(3))
    assert prompt_logprobs[3][7].logprob == -0.5
    assert prompt_logprobs[4][8].logprob == -1.5
