# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.ngram_hint_proposer import NgramHintProposer

HINT = [100, 101, 102, 103, 104, 105]


def _make_proposer(k: int = 8, max_n: int = 3, max_model_len: int = 2048):
    return NgramHintProposer(
        vllm_config=VllmConfig(
            model_config=ModelConfig(
                model="facebook/opt-125m", max_model_len=max_model_len
            ),
            speculative_config=SpeculativeConfig(
                method="ngram_hint",
                num_speculative_tokens=k,
                prompt_lookup_max=max_n,
            ),
        )
    )


def _make_input_batch(rows: list[tuple[str, list[int]]]):
    """Stub an InputBatch with one row of token ids per (req_id, tokens)."""
    width = max(len(token_ids) for _, token_ids in rows) + 4
    token_ids_cpu = np.zeros((len(rows), width), dtype=np.int32)
    for i, (_, token_ids) in enumerate(rows):
        token_ids_cpu[i, : len(token_ids)] = token_ids
    return SimpleNamespace(
        req_ids=[req_id for req_id, _ in rows],
        req_id_to_index={req_id: i for i, (req_id, _) in enumerate(rows)},
        num_tokens_no_spec=np.array(
            [len(token_ids) for _, token_ids in rows], dtype=np.int32
        ),
        token_ids_cpu=token_ids_cpu,
    )


def _make_request_state(hints):
    extra_args = {"spec_hints": hints} if hints is not None else None
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args))


def test_propose_from_hints_in_extra_args():
    proposer = _make_proposer()
    input_batch = _make_input_batch([("a", [1, 2, 100, 101, 102])])
    requests = {"a": _make_request_state([HINT])}
    assert proposer.propose(8, input_batch, [[102]], requests) == [[103, 104, 105]]


def test_no_hints():
    proposer = _make_proposer()
    input_batch = _make_input_batch([("a", [1, 2, 100, 101, 102])])
    assert proposer.propose(8, input_batch, [[102]], {}) == [[]]
    requests = {"a": _make_request_state(None)}
    assert proposer.propose(8, input_batch, [[102]], requests) == [[]]
    requests = {"a": _make_request_state([])}
    assert proposer.propose(8, input_batch, [[102]], requests) == [[]]


def test_hints_are_per_request():
    proposer = _make_proposer()
    input_batch = _make_input_batch(
        [("a", [1, 100, 101, 102]), ("b", [1, 200, 201, 202])]
    )
    requests = {
        "a": _make_request_state([HINT]),
        "b": _make_request_state([[200, 201, 202, 250, 251]]),
    }
    assert proposer.propose(8, input_batch, [[102], [202]], requests) == [
        [103, 104, 105],
        [250, 251],
    ]


def test_skip_partial_prefill():
    proposer = _make_proposer()
    input_batch = _make_input_batch([("a", [1, 2, 100, 101, 102])])
    requests = {"a": _make_request_state([HINT])}
    assert proposer.propose(8, input_batch, [[]], requests) == [[]]


def test_num_speculative_tokens_limits_the_proposal():
    proposer = _make_proposer()
    input_batch = _make_input_batch([("a", [100, 101, 102])])
    requests = {"a": _make_request_state([HINT])}
    assert proposer.propose(2, input_batch, [[102]], requests) == [[103, 104]]


def test_single_flat_hint():
    # vllm_xargs only allows a flat list, so a single sequence is accepted.
    proposer = _make_proposer()
    input_batch = _make_input_batch([("a", [1, 2, 100, 101, 102])])
    requests = {"a": _make_request_state(HINT)}
    assert proposer.propose(8, input_batch, [[102]], requests) == [[103, 104, 105]]


def test_skip_at_max_model_len():
    proposer = _make_proposer(max_model_len=5)
    input_batch = _make_input_batch([("a", [1, 2, 100, 101, 102])])
    requests = {"a": _make_request_state([HINT])}
    assert proposer.propose(8, input_batch, [[102]], requests) == [[]]
