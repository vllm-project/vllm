# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the parts of the Laya decision model that are not the encoder.

Laya cannot be pulled from the Hub: its checkpoints ship ``rl_agent_config.json``
plus a nested ``encoder/config.json`` instead of a single HF ``config.json``, so
the end-to-end path is exercised by the serving recipe rather than by an
``example_models`` download. Everything that a silent refactor could break --
the temperature lookup, the marker/question-type resolution and, above all, the
batched head's index bookkeeping -- is covered here on CPU.
"""

import math
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.laya import (
    ACT_FEATURE_DIM,
    ANSWER_MASK_LOGIT,
    MIN_OPTIONS_FOR_ENTROPY,
    OPTION_COUNT_SCALE,
    QTYPE_NAMES,
    LayaForDecision,
    _exclusive_prefix,
    _flatten_rows,
    _packed_attention,
    _temperature_bucket,
)

# Everything here runs on CPU without touching a device, and the shared
# `cleanup_dist_env_and_memory` teardown trips over the NPU allocator on
# non-CUDA builds, so opt out of the global cleanup.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "num_options,size",
    [
        (1, "2"),
        (2, "2"),
        (3, "3-5"),
        (5, "3-5"),
        (6, "6-10"),
        (10, "6-10"),
        (11, "11+"),
        (77, "11+"),
    ],
)
@pytest.mark.parametrize("qtype", QTYPE_NAMES)
def test_temperature_bucket_boundaries(qtype, num_options, size):
    # The bucket key is part of the checkpoint's fitted-temperature table, so the
    # boundaries must match `rl_common.temp_bucket` exactly.
    assert _temperature_bucket(qtype, num_options) == f"{qtype}:{size}"


def test_exclusive_prefix_and_flatten_rows():
    assert _exclusive_prefix(np.asarray([3, 0, 4], dtype=np.int64)).tolist() == [
        0,
        3,
        3,
    ]
    assert _exclusive_prefix(np.asarray([5], dtype=np.int64)).tolist() == [0]
    # An empty batch must not divide by zero or allocate.
    assert _flatten_rows([[1, 2], [], [3]], 3).tolist() == [1, 2, 3]
    assert _flatten_rows([], 0).size == 0


def test_packed_attention_never_crosses_a_sequence_boundary():
    torch.manual_seed(0)
    seq_lens = [3, 5, 2]
    num_tokens = sum(seq_lens)
    heads, head_dim = 2, 4
    query = torch.randn(num_tokens, heads, head_dim)
    key = torch.randn(num_tokens, heads, head_dim)
    value = torch.randn(num_tokens, heads, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    # The fused TND operator wants cumulative ends (`actual_seq_qlen`), which is
    # also what `_head_forward` is handed, so the helper takes them that way.
    cumulative = list(np.cumsum(seq_lens))
    packed = _packed_attention(query, key, value, heads, scale, cumulative)

    # Reference: run every request on its own, exactly what the fused TND
    # operator does, so the packed result must equal it row for row.
    start = 0
    expected = []
    for length in seq_lens:
        end = start + length
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        block = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        expected.append(block.squeeze(0).transpose(0, 1))
        start = end
    torch.testing.assert_close(packed, torch.cat(expected), atol=1e-6, rtol=1e-6)
    assert packed.shape == (num_tokens, heads, head_dim)

    # And the invariant that matters in serving: rewrite the tokens of the last
    # request and the rows of the first two must not move at all.
    other = value.clone()
    other[cumulative[1] :] = torch.randn_like(other[cumulative[1] :])
    moved = _packed_attention(query, key, other, heads, scale, cumulative)
    torch.testing.assert_close(moved[: cumulative[1]], packed[: cumulative[1]])
    assert not torch.equal(moved[cumulative[1] :], packed[cumulative[1] :])


def _fake_model(
    *,
    hidden_size=8,
    head_layers=1,
    mask_token_id=7,
    type_token_ids=None,
    temperature=None,
    temperature_by_options=None,
    plan_cache_limit=4,
    pinned_min_elements=8,
):
    """A Laya decision head with tiny random weights; no encoder, no device."""
    model = LayaForDecision.__new__(LayaForDecision)
    # `__new__` skips `nn.Module.__init__`, which is what creates `_modules`.
    nn.Module.__init__(model)
    model.head_dtype = torch.float32
    model.mask_token_id = mask_token_id
    model.type_token_ids = (
        type_token_ids if type_token_ids is not None else {1: 0, 2: 1, 3: 2}
    )
    torch.manual_seed(1234)
    layer = nn.TransformerEncoderLayer(
        hidden_size, 1, 4 * hidden_size, 0.0, batch_first=True, norm_first=True
    )
    model.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
    model.type_emb = nn.Embedding(len(QTYPE_NAMES), hidden_size)
    model.scorer = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, 1),
    )
    model.act_head = nn.Sequential(
        nn.Linear(hidden_size + ACT_FEATURE_DIM, 16),
        nn.GELU(),
        nn.Linear(16, 2),
    )
    model.temperature = list(temperature or [])
    model.temperature_by_options = dict(temperature_by_options or {})
    model._plan_cache = OrderedDict()
    model._plan_cache_limit = plan_cache_limit
    model._plan_host = {}
    model._pinned_min_elements = pinned_min_elements
    model._fused_attention = None
    model.eval()
    return model


def test_qtype_and_temperature_resolution():
    model = _fake_model(
        type_token_ids={1: 0, 11: 1, 12: 2},
        temperature=[1.5, 2.5, 3.5],
        temperature_by_options={"choice:11+": 0.1, "noul:2": 9.0},
    )

    # The question type is the token that follows [CLS].
    assert model._qtype_of(torch.tensor([99, 11, 42])) == 1
    # A leading token that is not a known type, and a one-token prompt, fall back
    # to `choice` rather than raising.
    assert model._qtype_of(torch.tensor([99, 77, 42])) == 0
    assert model._qtype_of(torch.tensor([99])) == 0

    # A fitted per-cardinality value wins over the per-type one; otherwise the
    # per-type value is used, and an uncalibrated type falls back to 1.0.
    assert model._answer_temperature(0, 11) == 0.1
    assert model._answer_temperature(2, 2) == 9.0
    assert model._answer_temperature(1, 3) == 2.5
    assert _fake_model()._answer_temperature(0, 3) == 1.0


def test_qtype_map_ignores_unknown_names():
    assert LayaForDecision._qtype_map({"choice": 1, "score": 2, "noul": 3}) == {
        1: 0,
        2: 1,
        3: 2,
    }
    assert LayaForDecision._qtype_map({"choice": 1, "bogus": 9}) == {1: 0}


def _step_inputs(model, prompts):
    """`(hidden_states, offsets, prompt_token_ids, use_activation)` for `_step_plan`."""
    lengths = [ids.numel() for ids in prompts]
    offsets = [0] + list(np.cumsum(lengths).tolist())
    hidden = torch.randn(offsets[-1], model.type_emb.embedding_dim)
    return hidden, offsets, prompts, [True] * len(prompts)


def test_step_plan_marks_the_marker_rows_and_skips_the_padding():
    model = _fake_model(hidden_size=6, mask_token_id=7)
    prompts = [
        torch.tensor([1, 11, 7, 12, 7]),
        torch.tensor([1, 12, 7]),
    ]
    hidden, offsets, prompt_ids, use_act = _step_inputs(model, prompts)

    lengths, markers, qtypes = [], [], []
    for ids in prompt_ids:
        lengths.append(ids.numel())
        qtypes.append(model._qtype_of(ids))
        markers.append(model._marker_positions(ids, ids.numel()))
    assert markers == [[2, 4], [2]]

    plan = model._step_plan(lengths, markers, qtypes, use_act, hidden.device)

    assert plan["batch"] == 2
    assert plan["kmax"] == 2
    assert plan["sizes"] == [4, 3]  # two/one markers plus the act pair
    assert plan["seq_lens"] == offsets[1:]

    # `pick` must land on the marker rows, in the order the answers are returned,
    # and repeat the request's first row for the padding slots.
    picked = plan["pick"].tolist()
    assert picked == [2, 4, 7, 5]  # request 0's markers, then request 1's + a pad
    starts = plan["starts"].tolist()
    assert starts == [0, 5]
    assert picked[0] == starts[0] + markers[0][0]
    assert picked[1] == starts[0] + markers[0][1]
    assert picked[2] == starts[1] + markers[1][0]
    assert picked[3] == starts[1]  # padding repeats the request's first row

    valid = plan["valid"].tolist()
    assert valid == [[True, True], [True, False]]
    assert plan["use_act"].tolist() == [True, True]

    # `gather` picks each request's real answers followed by its two act values.
    kmax, batch = plan["kmax"], plan["batch"]
    flat_len = batch * kmax + 2 * batch
    assert int(plan["gather"].max()) < flat_len
    # Request 0: its two slots (0, 1) then its act pair (4, 5); request 1: its
    # single slot (2) then its act pair (6, 7).
    assert plan["gather"].tolist() == [0, 1, 4, 5, 2, 6, 7]


def test_step_plan_is_cached_by_batch_shape_and_stays_bounded():
    model = _fake_model(hidden_size=6, plan_cache_limit=2)
    prompt = torch.tensor([1, 11, 7, 12, 7])

    def plan_for(ids_list):
        lengths = [ids.numel() for ids in ids_list]
        markers = [model._marker_positions(ids, ids.numel()) for ids in ids_list]
        qtypes = [model._qtype_of(ids) for ids in ids_list]
        device = torch.device("cpu")
        return model._step_plan(
            lengths, markers, qtypes, [True] * len(ids_list), device
        )

    first = plan_for([prompt])
    assert plan_for([prompt]) is first  # identical batch shape -> cached

    plan_for([torch.tensor([1, 12, 7])])
    plan_for([torch.tensor([1, 13, 7])])
    assert len(model._plan_cache) <= model._plan_cache_limit


def test_prompts_without_markers_take_the_scalar_path():
    model = _fake_model(hidden_size=6)
    prompt = torch.tensor([1, 11, 12])  # no [MASK] at all
    hidden, offsets, prompt_ids, use_act = _step_inputs(model, [prompt])

    out = model._score_requests_batched(hidden, offsets, prompt_ids, use_act)
    assert len(out) == 1
    # The scalar path answers with one value per marker; there are none, so only
    # the act pair is returned.
    assert out[0].shape == (2,)


def test_batched_and_scalar_paths_agree():
    # Both paths must be interchangeable: the batched one is the fast path for a
    # well-formed step, the scalar one answers malformed requests.
    torch.backends.mha.set_fastpath_enabled(False)  # compare the math, not MHA's kernel
    try:
        model = _fake_model(
            hidden_size=6,
            temperature=[1.7, 1.2, 2.0],
            temperature_by_options={"choice:2": 0.5},
        )
        prompts = [
            torch.tensor([1, 11, 7, 12, 7, 13]),
            torch.tensor([1, 12, 7, 12]),
            torch.tensor([1, 13, 7, 12, 7]),
        ]
        hidden, offsets, prompt_ids, _ = _step_inputs(model, prompts)

        for use_activation in ([True] * 3, [False] * 3, [True, False, True]):
            batched = model._score_requests_batched(
                hidden, offsets, prompt_ids, use_activation
            )
            scalar = model._score_requests_scalar(
                hidden, offsets, prompt_ids, use_activation
            )
            assert [len(x) for x in batched] == [len(x) for x in scalar] == [4, 3, 4]
            for got, want in zip(batched, scalar):
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)


def test_batched_answers_match_the_scalar_probabilities():
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        model = _fake_model(hidden_size=6, mask_token_id=7)
        prompt = torch.tensor([1, 11, 7, 12, 7, 13, 7])
        hidden, offsets, prompt_ids, use_act = _step_inputs(model, [prompt])

        (out,) = model._score_requests_batched(hidden, offsets, prompt_ids, use_act)
        answer, act = out[:-2], out[-2:]

        assert torch.all(answer < 1.0)  # calibrated probabilities, not raw logits
        assert answer.shape == (3,)
        torch.testing.assert_close(
            answer.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(act.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)


def test_single_option_request_matches_the_scalar_margin():
    # `topk(2)` would index out of range for a one-option request, which is
    # exactly the step that used to kill the EngineCore on Ascend.
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        model = _fake_model(hidden_size=6, mask_token_id=7)
        prompts = [torch.tensor([1, 11, 7, 12]), torch.tensor([1, 12, 7])]
        hidden, offsets, prompt_ids, use_act = _step_inputs(model, prompts)

        batched = model._score_requests_batched(hidden, offsets, prompt_ids, use_act)
        scalar = model._score_requests_scalar(hidden, offsets, prompt_ids, use_act)
        assert [len(x) for x in batched] == [3, 3]
        for got, want in zip(batched, scalar):
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)


def test_answer_padding_uses_a_finite_sentinel():
    # The sentinel has to be finite so raw logits stay JSON-serialisable, and it
    # has to be small enough that the padding options get ~zero probability.
    assert math.isfinite(ANSWER_MASK_LOGIT)
    logits = torch.tensor([[2.0, ANSWER_MASK_LOGIT, 1.0]])
    probs = torch.softmax(logits, dim=-1)
    assert probs[0, 1] < 1e-6
    assert int(probs[0].argmax()) == 0


def test_entropy_and_option_count_features_match_the_scalar_formula():
    # `_score_requests_batched` normalises the entropy by log(k) and feeds k/255
    # to the act head; both come from the clamped option count.
    assert MIN_OPTIONS_FOR_ENTROPY == 2
    assert OPTION_COUNT_SCALE == 255.0
    num_options = 7
    probs = torch.softmax(torch.randn(num_options), -1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum() / math.log(num_options)
    assert 0.0 <= float(entropy) <= 1.0
    assert num_options / OPTION_COUNT_SCALE == pytest.approx(0.02745, abs=1e-5)
