# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-streaming tokens-mode generate responses rendered from sample logprobs
kept as FlatLogprobs engine rows: byte-identical to the per-entry pydantic
path, which every other case still uses."""

import asyncio
import json
import threading
from unittest.mock import MagicMock

import numpy as np
import pybase64 as base64
import pytest
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from tests.entrypoints.scale_out.token_in_token_out.test_generate_stream import (
    MODEL_NAME,
    _build_serving_tokens,
    _make_request_output,
    _mock_engine,
)
from vllm.entrypoints.scale_out.token_in_token_out import serving as serving_mod
from vllm.entrypoints.scale_out.token_in_token_out.api_router import (
    _RenderedJSONResponse,
)
from vllm.entrypoints.scale_out.token_in_token_out.logprobs_render import (
    format_float_reprs,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateLogProbs,
    GenerateRequest,
    GenerateResponseBase,
    GenerateTokensChoice,
    GenerateTokensResponse,
    GenerateTokensStreamChoice,
    GenerateTokensStreamResponse,
    PackedTopK,
    RenderedGenerateResponse,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.logprobs import FlatLogprobs, Logprob, create_sample_logprobs
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists


def _rows(n, slots, seed=0, vocab=50_000):
    """Engine-like rows: distinct candidates; on even rows the sampled token
    (slot 0) is also one of the top-k; one -inf value. Ranks are int64."""
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(vocab, slots, replace=False) for _ in range(n)])
    if slots > 1:
        for i in range(0, n, 2):
            ids[i, 0] = ids[i, 1 + i % (slots - 1)]
    lps = (-rng.random((n, slots)) * 20).astype(np.float32)
    lps[0, -1] = -np.inf
    return ids.astype(np.int32), lps, rng.integers(0, 100, n)


def _stored(k, *steps, flat):
    """The rows of ``steps`` as the LogprobsProcessor stores them."""
    processor = LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=k,
        num_prompt_logprobs=None,
    )
    for step in steps:
        processor._update_sample_logprobs(LogprobsLists(*step))
    return processor.logprobs


def _final(outputs, finish_reason="length"):
    return RequestOutput(
        request_id="r",
        prompt=None,
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(i, "", tokens, None, logprobs, finish_reason=finish_reason)
            for i, (tokens, logprobs) in enumerate(outputs)
        ],
        finished=True,
    )


def _body(serving, k, outputs, finish_reason="length", **fields):
    """The response body as the router sends it, and whether it was
    rendered from the rows."""
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10, logprobs=k),
        model=MODEL_NAME,
        **fields,
    )
    response = serving._build_full_response(
        request, _final(outputs, finish_reason), "r", MODEL_NAME, 1700000000
    )[0]
    if isinstance(response, RenderedGenerateResponse):
        return response.body, True
    assert isinstance(response, GenerateResponseBase)
    return JSONResponse(content=response.model_dump()).body, False


def _outcome(serving, k, choices, finish_reason="length"):
    """Bodies of the list and the flat storage of ``choices``, or the error
    each raises."""
    results = []
    for flat in (False, True):
        outputs = [(tokens, _stored(k, *steps, flat=flat)) for tokens, steps in choices]
        try:
            results.append(_body(serving, k, outputs, finish_reason)[0])
        except ValueError as e:
            results.append(f"{type(e).__name__}: {e}")
    return results


@pytest.mark.parametrize("k", [0, 1, 3, 8])
@pytest.mark.parametrize("n", [1, 37, 1500])  # 1500: across render blocks
def test_rendered_bytes_equal_the_per_entry_path(k, n):
    serving = _build_serving_tokens(_mock_engine())
    rows = _rows(n, k + 1, seed=k * 7 + n)
    token_ids = rows[0][:, 0].tolist()
    legacy, _ = _body(serving, k, [(token_ids, _stored(k, rows, flat=False))])
    fast, rendered = _body(serving, k, [(token_ids, _stored(k, rows, flat=True))])
    assert rendered and fast == legacy


def _cases():
    """(name, k, choices as (token_ids, steps)) of special and irregular rows."""
    cases = []
    for k in (0, 1, 3):
        for label, value, slot in [
            ("nan sampled", np.nan, 0),
            ("nan top", np.nan, -1),
            ("+inf sampled", np.inf, 0),
            ("-0.0", -0.0, 0),
            ("1e-5", -1e-5, 0),
            ("denormal", -1e-45, -1),
            ("-9999.5", -9999.5, 0),
        ]:
            rows = _rows(5, k + 1, seed=k)
            rows[1][2, slot] = value
            cases.append((f"k={k} {label}", k, [(rows[0][:, 0].tolist(), [rows])]))
        wide = _rows(20, k + 5, seed=10 + k)  # truncated to k + 1 slots
        wide[0][3, 0] = wide[0][3, k + 3]  # sampled id only beyond k
        cases.append((f"k={k} wider", k, [(wide[0][:, 0].tolist(), [wide])]))
        a, b = _rows(7, k + 1, seed=20 + k), _rows(9, k + 1, seed=30 + k)
        both = np.concatenate([a[0], b[0]])[:, 0].tolist()
        cases.append((f"k={k} n=2", k, [(both, [a, b]), (a[0][:, 0].tolist(), [a])]))
        mismatch = _rows(4, k + 1, seed=40 + k)
        tokens = mismatch[0][:, 0].tolist()
        tokens[1] = 49_999 if tokens[1] != 49_999 else 49_998
        cases.append((f"k={k} sampled id not in slot 0", k, [(tokens, [mismatch])]))
        extra = _rows(4, k + 1, seed=50 + k)
        cases.append((f"k={k} more rows", k, [(extra[0][:3, 0].tolist(), [extra])]))
        big = _rows(5, k + 1, seed=60 + k, vocab=300_000)
        big[0][:, 0] = [262_143, 262_144, 299_999, 0, 5]  # beyond the lead table
        cases.append((f"k={k} large ids", k, [(big[0][:, 0].tolist(), [big])]))
        narrow = [_rows(3, k + 1, seed=65 + k), _rows(2, max(k, 1), seed=66 + k)]
        cases.append(
            (
                f"k={k} irregular width",
                k,
                [([i for r in narrow for i in r[0][:, 0].tolist()], narrow)],
            )
        )
    dup = _rows(4, 4, seed=70)
    dup[0][1, 2] = dup[0][1, 1]  # repeated top-k id
    cases.append(("repeated top-k", 3, [(dup[0][:, 0].tolist(), [dup])]))
    zero = _rows(4, 4, seed=75)
    zero[2][:] = 0  # rank 0 is sent as null
    cases.append(("rank 0", 3, [(zero[0][:, 0].tolist(), [zero])]))
    return cases


@pytest.mark.parametrize("name,k,choices", _cases(), ids=[c[0] for c in _cases()])
def test_special_and_irregular_rows_match_the_per_entry_path(name, k, choices):
    """The same bytes, or the same error, as the list storage."""
    serving = _build_serving_tokens(_mock_engine())
    legacy, fast = _outcome(serving, k, choices)
    assert fast == legacy


@pytest.mark.parametrize(
    "k,row,expected_top",
    [
        # Sampled id 7 (engine rank 4) also at slot 1: its entry takes
        # slot 1's value and rank, and slot 1 is not listed again.
        (3, [7, 7, 8, 9], [(7, -1.0, 1), (8, -2.0, 2), (9, -3.0, 3)]),
        # Also at slot k (the last listed slot).
        (3, [7, 8, 9, 7], [(7, -3.0, 3), (8, -1.0, 1), (9, -2.0, 2)]),
        # Outside the top-k: listed first with the engine rank, slot k cut.
        (3, [7, 8, 9, 10], [(7, -0.5, 4), (8, -1.0, 1), (9, -2.0, 2)]),
        # k=1: one entry, the sampled one.
        (1, [7, 8], [(7, -0.5, 4)]),
        (1, [7, 7], [(7, -1.0, 1)]),
        # k=0: still one top entry (max(k, 1)).
        (0, [7], [(7, -0.5, 4)]),
    ],
)
def test_sampled_entry_and_top_logprobs_follow_dict_semantics(k, row, expected_top):
    """The per-entry path builds a dict from the row: the sampled entry is
    the dict's first key with its last occurrence's value and rank, and
    top_logprobs are the first max(k, 1) items. Both paths agree, including
    rank, on these hand-checked rows."""
    serving = _build_serving_tokens(_mock_engine())
    ids = np.array([row], dtype=np.int32)
    lps = np.array([[-0.5, -1.0, -2.0, -3.0][: len(row)]], dtype=np.float32)
    ranks = np.array([4])
    legacy, fast = _outcome(serving, k, [([row[0]], [(ids, lps, ranks)])])
    assert fast == legacy
    entry = json.loads(fast)["choices"][0]["logprobs"]["content"][0]
    sampled = expected_top[0]
    assert (entry["token_id"], entry["logprob"], entry["rank"]) == sampled
    assert [
        (e["token_id"], e["logprob"], e["rank"]) for e in entry["top_logprobs"]
    ] == expected_top


@pytest.mark.parametrize("name,k,choices", _cases(), ids=[c[0] for c in _cases()])
def test_return_token_logprobs_with_top_k_matches_the_per_entry_path(
    name, k, choices, monkeypatch
):
    """With return_token_logprobs and k > 0, ``sampled`` comes from slot 0 of
    the rows and ``content`` from the fast render: the same bytes (or error)
    as the per-entry path of return_token_logprobs."""
    if k <= 0:
        return
    serving = _build_serving_tokens(_mock_engine())
    fast_render = serving_mod.render_tokens_logprobs
    results = []
    for fast in (False, True):
        monkeypatch.setattr(
            serving_mod,
            "render_tokens_logprobs",
            fast_render if fast else (lambda *args: None),
        )
        outputs = [(tokens, _stored(k, *steps, flat=True)) for tokens, steps in choices]
        try:
            results.append(_body(serving, k, outputs, return_token_logprobs=True)[0])
        except ValueError as e:
            results.append(f"{type(e).__name__}: {e}")
    assert results[1] == results[0]
    if isinstance(results[0], bytes):
        logprobs = json.loads(results[0])["choices"][0]["logprobs"]
        assert list(logprobs) == ["content", "sampled"]


def test_aborted_choice_without_tokens():
    serving = _build_serving_tokens(_mock_engine())
    legacy, fast = _outcome(serving, 3, [([], [])], finish_reason="abort")
    assert fast == legacy


def test_float_reprs_match_repr():
    """The msgspec fast path gives ``repr``'s digits for float32 values."""
    bits = np.random.default_rng(0).integers(0, 2**32, 1 << 20, dtype=np.uint64)
    values = bits.astype(np.uint32).view(np.float32)
    values = values[np.isfinite(values)].astype(np.float64)
    assert format_float_reprs(values) == [repr(v).encode() for v in values.tolist()]


def test_rendered_response_headers_match_json_response():
    content = {"a": [1, 2.5, None]}
    body = JSONResponse(content=content).body
    rendered = _RenderedJSONResponse(content=body)
    assert rendered.body == body
    assert rendered.headers.items() == JSONResponse(content=content).headers.items()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream,output_mode,logprobs,prompt_logprobs,expected",
    [
        (False, "tokens", 3, None, True),
        (False, "tokens", 0, None, True),
        (False, "tokens", 3, 2, False),
        (True, "tokens", 3, None, False),
        (False, "text", 3, None, False),
        (False, "tokens", -1, None, False),
        (False, "tokens", None, None, False),
    ],
)
async def test_flat_storage_selection(
    stream, output_mode, logprobs, prompt_logprobs, expected
):
    """Only non-streaming tokens-mode requests with top-k sample logprobs and
    no prompt logprobs, which the response returns as lists."""
    engine = _mock_engine()
    seen = []

    async def generate(engine_input, params, *args, **kwargs):
        seen.append((params.flat_logprobs, not params._detokenize_logprobs))
        yield _make_request_output(
            "r",
            [10],
            finish_reason="stop",
            finished=True,
            logprobs=None if logprobs is None else [{10: Logprob(-0.5)}],
            text="x",
        )

    engine.generate = MagicMock(side_effect=generate)
    serving = _build_serving_tokens(engine)
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(
            max_tokens=1, logprobs=logprobs, prompt_logprobs=prompt_logprobs
        ),
        model=MODEL_NAME,
        stream=stream,
        output_mode=output_mode,
    )
    out = await serving.serve_tokens(request)
    if stream:
        [chunk async for chunk in out]
    assert seen == [(expected, expected)]


def test_clients_cannot_skip_logprobs_detokenize():
    request = GenerateRequest.model_validate(
        {"token_ids": [1], "sampling_params": {"_detokenize_logprobs": False}}
    )
    assert request.sampling_params._detokenize_logprobs is True


def _full_request(n=40):
    return GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=n, logprobs=3),
        model=MODEL_NAME,
    )


async def _serve_full(serving, rows):
    """serve_tokens_full_generator for one choice of ``rows`` (k=3)."""

    async def results():
        yield _final([(rows[0][:, 0].tolist(), _stored(3, rows, flat=True))])

    return await serving.serve_tokens_full_generator(
        _full_request(len(rows[2])), results(), "r", MODEL_NAME, MagicMock()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("offload,offloaded", [(160, True), (161, False)])
async def test_large_builds_run_off_the_event_loop(monkeypatch, offload, offloaded):
    """40 rows x 4 slots = 160 entries: built in a worker thread from the
    threshold on, with the same bytes as inline."""
    serving = _build_serving_tokens(_mock_engine())
    rows = _rows(40, 4)
    threads, real = [], serving_mod.ServingTokens._build_full_response

    def recording(self, *args, **kwargs):
        threads.append(threading.current_thread())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(serving_mod.time, "time", lambda: 1700000000.0)
    inline = await asyncio.wait_for(_serve_full(serving, rows), 30)
    monkeypatch.setattr(serving_mod.ServingTokens, "_build_full_response", recording)
    monkeypatch.setattr(serving_mod, "OFFLOAD_MIN_LOGPROB_ENTRIES", offload)
    response = await asyncio.wait_for(_serve_full(serving, rows), 30)
    on_loop = threads == [threading.current_thread()]
    assert on_loop != offloaded
    if offloaded:
        assert threads[0].name.startswith("generate-response")
    assert response.body == inline.body


@pytest.mark.asyncio
async def test_two_builds_run_at_once(monkeypatch):
    serving = _build_serving_tokens(_mock_engine())
    rows = _rows(40, 4)
    monkeypatch.setattr(serving_mod, "OFFLOAD_MIN_LOGPROB_ENTRIES", 1)
    both, names = threading.Barrier(2, timeout=10), []
    real = serving_mod.ServingTokens._build_full_response

    def build(self, *args):
        names.append(threading.current_thread().name)
        both.wait()  # both builds are running at the same time
        return real(self, *args)

    monkeypatch.setattr(serving_mod.ServingTokens, "_build_full_response", build)
    first, second = await asyncio.wait_for(
        asyncio.gather(_serve_full(serving, rows), _serve_full(serving, rows)), 30
    )
    assert first.body == second.body
    assert len(set(names)) == 2


def test_sampled_is_omitted_from_stream_chunks():
    """Streaming tokens-mode chunks with logprobs carry no ``sampled`` key."""
    chunk = GenerateTokensStreamResponse(
        request_id="r",
        choices=[
            GenerateTokensStreamChoice(index=0, logprobs=GenerateLogProbs(content=[]))
        ],
    )
    assert '"logprobs":{"content":[]}' in chunk.model_dump_json()


def test_clients_cannot_select_sampled_logprobs_only():
    request = GenerateRequest.model_validate(
        {
            "token_ids": [1],
            "sampling_params": {"logprobs": 0, "_sampled_logprobs_only": True},
        }
    )
    assert request.sampling_params._sampled_logprobs_only is False


def _top_k_body(serving, k, outputs, **fields):
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10, logprobs=k),
        model=MODEL_NAME,
        return_top_k_logprobs=True,
        **fields,
    )
    response = serving._build_full_response(
        request, _final(outputs), "r", MODEL_NAME, 1700000000
    )[0]
    assert isinstance(response, RenderedGenerateResponse)
    return response.body


def _engine_rows(n, k, seed):
    """Rows as the engine emits them: a sampled id repeated in the top-k has
    the same logprob there."""
    if n == 0:
        return np.empty((0, k + 1), np.int32), np.empty((0, k + 1), np.float32), []
    ids, lps, ranks = _rows(n, k + 1, seed=seed)
    for i in range(n):
        lps[i, 1:][ids[i, 1:] == ids[i, 0]] = lps[i, 0]
    return ids, lps, ranks


@pytest.mark.parametrize("return_token_logprobs", [False, True])
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("n", [0, 1, 300])
def test_top_k_bytes_equal_the_pydantic_dump(k, n, return_token_logprobs):
    """The spliced body is what JSONResponse gives for the same values:
    ``{content: null, top_k}``, with ``sampled`` before ``top_k`` only when
    return_token_logprobs is also set."""
    serving = _build_serving_tokens(_mock_engine())
    ids, lps, ranks = _engine_rows(n, k, seed=n + k)
    lps[n // 2 :: 7, 0] = np.nan
    stored = _stored(k, (ids, lps, ranks), flat=True) if n else FlatLogprobs()
    body = _top_k_body(
        serving,
        k,
        [(ids[:, 0].tolist(), stored)],
        return_token_logprobs=return_token_logprobs,
    )

    sampled = [serving_mod._clamp_logprob(v) for v in lps[:, 0].tolist()]
    top_k = PackedTopK(
        num_positions=n,
        k=k,
        token_ids=base64.b64encode(ids[:, 1:].astype("<i4").tobytes()).decode(),
        logprobs=base64.b64encode(lps[:, 1:].astype("<f4").tobytes()).decode(),
    )
    expected = GenerateTokensResponse(
        request_id="r",
        created=1700000000,
        model=MODEL_NAME,
        usage=UsageInfo(prompt_tokens=3, completion_tokens=n, total_tokens=n + 3),
        choices=[
            GenerateTokensChoice(
                index=0,
                finish_reason="length",
                token_ids=ids[:, 0].tolist(),
                logprobs=GenerateLogProbs(
                    sampled=sampled if return_token_logprobs else None, top_k=top_k
                ),
            )
        ],
    )
    assert body == JSONResponse(content=expected.model_dump()).body
    logprobs = json.loads(body)["choices"][0]["logprobs"]
    assert list(logprobs) == (
        ["content", "sampled", "top_k"]
        if return_token_logprobs
        else ["content", "top_k"]
    )
    assert list(logprobs["top_k"]) == ["num_positions", "k", "token_ids", "logprobs"]


def test_top_k_round_trip_against_content():
    """Decoded top_k and sampled match the content entries of the same rows."""
    serving = _build_serving_tokens(_mock_engine())
    k, n = 4, 50
    ids, lps, ranks = _engine_rows(n, k, seed=3)
    lps[3, 2] = -np.inf  # raw in top_k, clamped in content
    tokens = ids[:, 0].tolist()

    def stored():
        return _stored(k, (ids, lps, ranks), flat=True)

    packed = json.loads(
        _top_k_body(serving, k, [(tokens, stored())], return_token_logprobs=True)
    )["choices"][0]["logprobs"]
    content = json.loads(_body(serving, k, [(tokens, stored())])[0])["choices"][0][
        "logprobs"
    ]["content"]

    assert packed["content"] is None
    assert packed["sampled"] == [entry["logprob"] for entry in content]
    top_k = packed["top_k"]
    assert (top_k["num_positions"], top_k["k"]) == (n, k)
    top_ids = np.frombuffer(base64.b64decode(top_k["token_ids"]), "<i4")
    top_lps = np.frombuffer(base64.b64decode(top_k["logprobs"]), "<f4")
    top_ids, top_lps = top_ids.reshape(n, k), top_lps.reshape(n, k)
    np.testing.assert_array_equal(top_ids, ids[:, 1:])
    np.testing.assert_array_equal(top_lps, lps[:, 1:])
    for i, entry in enumerate(content):
        # content lists the sampled entry first, then slots 1..k in order
        # (the sampled id's own slot merged into the first entry).
        expected = [
            (int(t), max(float(v), -9999.0))
            for t, v in zip(top_ids[i], top_lps[i])
            if t != tokens[i]
        ]
        got = [(e["token_id"], e["logprob"]) for e in entry["top_logprobs"][1:]]
        assert got == expected[: len(got)]


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"stream": True}, "stream"),
        ({"output_mode": "text"}, "output_mode"),
        ({"sampling_params": {"max_tokens": 1}}, "return_token_logprobs"),
        ({"sampling_params": {"logprobs": 0}}, "return_token_logprobs"),
        ({"sampling_params": {"logprobs": -1}}, "return_token_logprobs"),
    ],
)
def test_return_top_k_logprobs_invalid_requests(fields, message):
    body = {
        "token_ids": [1],
        "sampling_params": {"max_tokens": 1, "logprobs": 2},
        "return_top_k_logprobs": True,
        **fields,
    }
    with pytest.raises(ValidationError, match=message):
        GenerateRequest.model_validate(body)


def test_top_k_is_omitted_unless_set():
    """Responses without return_top_k_logprobs keep their schema and bytes."""
    assert GenerateLogProbs(content=[]).model_dump() == {"content": []}
    assert "top_k" not in GenerateLogProbs(sampled=[-1.0]).model_dump_json()
