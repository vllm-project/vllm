# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The comparison on fake runs built from full logits, so every KL has a known
source. CPU only."""

import itertools
import json
import math

import numpy as np
import pytest

from . import logprob_drift_eval as drift

META = {"format": drift.FORMAT, "vllm": "0", "model": "m", "setup": {}, "gpu": "g"}


def top4(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    top = np.argsort(-lp, axis=-1)[..., :4]
    return lp, top.astype(np.int32), np.take_along_axis(lp, top, axis=-1)


def generated(logits: np.ndarray, part: str) -> dict:
    """Greedy outputs from full logits [prompts, positions, vocab]."""
    _, ids, lp = top4(logits)
    return {
        f"{part}_tokens": ids[..., 0],
        f"{part}_ids": ids,
        f"{part}_lp": lp,
        f"{part}_lengths": np.full(len(logits), logits.shape[1], np.int32),
        f"{part}_text": np.array([" ".join(map(str, row)) for row in ids[..., 0]]),
    }


def fake_run(probes: np.ndarray, traffic: np.ndarray, seed: int) -> dict:
    """A run from full logits [chunks or prompts, positions, vocab]. Decode
    continues the probes, so it changes with them."""
    lp, ids, top_lp = top4(probes)
    return (
        {
            "probe_tokens": np.zeros(probes.shape[:2], np.int64),
            "probe_ids": ids,
            "probe_lp": top_lp,
            "probe_own": lp[..., 0],
            "meta": META | {"seed": seed},
        }
        | generated(probes[:, ::-1], "decode")
        | generated(traffic, "traffic")
    )


def reruns(rng, probes, traffic, seeds, probe_noise=0.01, traffic_noise=0.01):
    return [
        fake_run(
            probes + probe_noise * rng.normal(size=probes.shape),
            traffic + traffic_noise * rng.normal(size=traffic.shape),
            s,
        )
        for s in seeds
    ]


def world(rng) -> tuple[np.ndarray, np.ndarray]:
    """Probe and traffic logits for one model."""
    return rng.normal(size=(8, 16, 32)), rng.normal(size=(8, 6, 32))


REF, CAND = range(5), range(5, 9)


def test_a_consistent_shift_is_flagged_by_the_probes():
    rng = np.random.default_rng(1)
    probes, traffic = world(rng)
    shifted = probes + 0.1 * rng.normal(size=probes.shape)
    shifted[..., 0] -= 0.5  # the text's own tokens get less likely
    refs, cands = reruns(rng, probes, traffic, REF), reruns(rng, shifted, traffic, CAND)
    result = drift.compare(refs, cands)
    assert (result["verdict"], result["splits"]) == ("changed", math.comb(9, 4))
    assert result["ratios"]["probes"] > 10 and result["score"] > result["needed"]
    assert result["stats"]["same_top"] < 1 and result["stats"]["nll_change"] > 0.01
    text = drift.render(result)
    assert "Verdict: CHANGED" in text and "1 in 126" in text and "NLL +" in text
    # Fewer runs see the change too, but can't rule out chance: 1 in 100 at most.
    for n, m in (*drift.STAGES[:-1], (5, 3)):
        assert drift.compare(refs[:n], cands[:m])["verdict"] == "more"
    assert "the next stage takes 4 reference and 2 candidate" in drift.render(
        drift.compare(refs[:2], cands[:1])
    )
    again = reruns(rng, probes, traffic, range(20, 24))
    assert "Verdict: same within noise" in drift.render(drift.compare(refs, again))


def test_reruns_that_disagree_are_flagged_by_the_traffic():
    # Random KV corruption: every candidate run goes wrong in its own way.
    rng = np.random.default_rng(2)
    probes, traffic = world(rng)
    result = drift.compare(
        reruns(rng, probes, traffic, REF),
        reruns(rng, probes, traffic, CAND, traffic_noise=0.3),
    )
    assert result["verdict"] == "changed"
    assert result["ratios"]["traffic"] > 3
    assert result["ratios"]["probes"] < 2


def test_bit_identical_reruns_read_same_until_anything_differs():
    rng = np.random.default_rng(5)
    probes, traffic = world(rng)
    refs = [fake_run(probes, traffic, s) for s in REF]
    result = drift.compare(refs, [fake_run(probes, traffic, s) for s in CAND])
    assert result["ratios"] == {"probes": 0.0, "decode": 0.0, "traffic": 0.0}
    assert (result["verdict"], result["stats"]["kl"]) == ("same", 0)

    shifted = probes + 1e-3 * rng.normal(size=probes.shape)
    result = drift.compare(refs, [fake_run(shifted, traffic, s) for s in CAND])
    assert (result["ratios"]["probes"], result["verdict"]) == (math.inf, "changed")
    assert json.loads(json.dumps(drift.jsonable(result)))["score"] == "inf"


def test_at_most_one_of_the_splits_of_any_runs_separates():
    # The test's guarantee, checked exactly: taking each of the 56 splits of the
    # same 5 + 3 runs in turn as the real one, at most 1 of them doesn't read same.
    # Probe reruns land in one of a few bit-identical states, as real ones do.
    rng = np.random.default_rng(11)
    probes, traffic = world(rng)
    states = [probes + 0.01 * rng.normal(size=probes.shape) for _ in range(3)]
    runs = [
        fake_run(
            states[rng.choice(3, p=[0.6, 0.3, 0.1])],
            traffic + 0.01 * rng.normal(size=traffic.shape),
            s,
        )
        for s in range(8)
    ]
    result = drift.compare(runs[:5], runs[5:])
    matrices = {part: np.array(m) for part, m in result["matrices"].items()}
    changed = 0
    for ref in itertools.combinations(range(8), 5):
        order = [*ref, *(i for i in range(8) if i not in ref)]
        relabeled = {k: m[np.ix_(order, order)] for k, m in matrices.items()}
        changed += drift.decide(relabeled, 5)["verdict"] != "same"
    assert changed <= 1


def test_compare_rejects_runs_it_cannot_compare():
    rng = np.random.default_rng(9)
    probes, traffic = world(rng)
    refs, cands = reruns(rng, probes, traffic, REF), reruns(rng, probes, traffic, CAND)

    def with_meta(run: dict, **changes) -> dict:
        return run | {"meta": run["meta"] | changes}

    needs = "record 2 to 5 runs of the reference and fewer of the candidate"
    with pytest.raises(ValueError, match=f"4 reference and 4 candidate runs: {needs}"):
        drift.compare(refs[:4], cands)
    with pytest.raises(ValueError, match=f"6 reference and 4 candidate runs: {needs}"):
        drift.compare([*refs, *reruns(rng, probes, traffic, [20])], cands)
    with pytest.raises(ValueError, match="differ in gpu"):
        drift.compare(refs, [*cands[:3], with_meta(cands[3], gpu="other")])


def test_bucketed_kl_is_exact_when_k_covers_the_vocab_and_a_lower_bound_when_not():
    rng = np.random.default_rng(6)
    lp_a = np.log(rng.dirichlet(np.ones(6), size=(2, 5)))
    lp_b = np.log(rng.dirichlet(np.ones(6), size=(2, 5)))
    exact = (np.exp(lp_a) * (lp_a - lp_b)).sum(axis=-1)
    ids = np.broadcast_to(np.arange(6), lp_a.shape)
    np.testing.assert_allclose(
        drift.bucketed_kl(ids, lp_a, ids, lp_b), exact, rtol=1e-6
    )

    # Peaked and flat distributions over 40 ids, near each other or not.
    p = rng.dirichlet(np.full(40, 0.1), size=3000)
    q = np.where(
        rng.random((3000, 1)) < 0.5,
        rng.dirichlet(np.full(40, 0.1), size=3000),
        p * np.exp(rng.normal(0, 0.5, size=p.shape)),
    )
    p, q = np.log(p + 1e-12), np.log(q + 1e-12)
    p, q = (x - np.log(np.exp(x).sum(axis=-1, keepdims=True)) for x in (p, q))
    exact = (np.exp(p) * (p - q)).sum(axis=-1)
    top_p, top_q = (np.argsort(-x, axis=-1)[:, :5] for x in (p, q))
    bucketed = drift.bucketed_kl(
        top_p,
        np.take_along_axis(p, top_p, -1),
        top_q,
        np.take_along_axis(q, top_q, -1),
    )
    assert np.all(bucketed <= exact + 1e-9)
    assert np.all(bucketed >= -1e-12)


def test_bucketed_kl_of_identical_lists_is_exactly_zero():
    # Peaked rows, whose top 20 leave less than the floor for the rest.
    rng = np.random.default_rng(13)
    logits = 30 * rng.normal(size=(500, 1000))
    lp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    top = np.argsort(-lp, axis=-1)[:, :20]
    top_lp = np.take_along_axis(lp, top, -1).astype(np.float32)
    assert np.all(drift.bucketed_kl(top, top_lp, top, top_lp) == 0)


def test_bucketed_kl_sees_lists_that_share_no_ids():
    # b leaves a's tokens off its list, so it gives each at most e^-3.
    kl = drift.bucketed_kl(
        np.array([1, 2]),
        np.array([-0.1, -3.0]),
        np.array([3, 4]),
        np.array([-0.1, -3.0]),
    )
    assert kl > 2


def test_generated_kl_matches_a_prompt_by_prompt_loop():
    rng = np.random.default_rng(7)
    prompts, length, vocab = 40, 12, 16

    def output() -> dict:
        _, ids, lp = top4(rng.normal(size=(prompts, length, vocab)))
        lengths = rng.integers(0, length + 1, size=prompts)
        tokens = np.where(
            np.arange(length) < lengths[:, None],
            rng.integers(0, 3, size=(prompts, length)),
            -1,
        )
        return {
            "traffic_tokens": tokens,
            "traffic_ids": ids,
            "traffic_lp": lp,
            "traffic_lengths": lengths,
        }

    a, b = output(), output()
    b["traffic_tokens"][0], b["traffic_lengths"][0] = (
        a["traffic_tokens"][0],
        a["traffic_lengths"][0],
    )
    b["traffic_tokens"][1, 5:], b["traffic_lengths"][1] = (
        -1,
        min(a["traffic_lengths"][1], 5),
    )
    b["traffic_tokens"][1, :5] = a["traffic_tokens"][1, :5]

    def slow(a: dict, b: dict) -> float:
        kls = []
        for i in range(prompts):
            n = min(a["traffic_lengths"][i], b["traffic_lengths"][i])
            differ = np.nonzero(
                a["traffic_tokens"][i, :n] != b["traffic_tokens"][i, :n]
            )[0]
            stop = differ[0] + 1 if len(differ) else n
            kls.append(
                drift.bucketed_kl(
                    a["traffic_ids"][i, :stop],
                    a["traffic_lp"][i, :stop],
                    b["traffic_ids"][i, :stop],
                    b["traffic_lp"][i, :stop],
                ).mean()
                if stop
                else 0.0
            )
        return float(np.mean(kls))

    for x, y in ((a, b), (b, a)):
        fast = drift.generated_kls(x, y, "traffic").mean()
        assert fast == pytest.approx(slow(x, y), rel=1e-12)


def test_server_responses_become_arrays():
    traffic = [
        {
            "token_ids": [7, 3],
            "text": " seven three",
            "logprobs": {
                "top_logprobs": [
                    {"token_id:7": -0.1, "token_id:9": -2.5},
                    {"token_id:3": -0.3, "token_id:7": -2.0, "token_id:5": -4.0},
                ]
            },
        },
        {"token_ids": [], "text": "", "logprobs": {"top_logprobs": []}},
    ]
    run = drift.generated_arrays(traffic, "traffic", drift.TRAFFIC_TOKENS)
    assert run["traffic_lengths"].tolist() == [2, 0]
    assert run["traffic_text"].tolist() == [" seven three", ""]
    assert run["traffic_tokens"][0, :3].tolist() == [7, 3, -1]
    assert run["traffic_ids"][0, 1, :4].tolist() == [3, 7, 5, -1]
    assert run["traffic_lp"][0, 1, :3].tolist() == pytest.approx([-0.3, -2.0, -4.0])

    chunks = (
        np.arange(drift.CHUNKS * drift.CONTEXT).reshape(drift.CHUNKS, drift.CONTEXT)
        % 50
    )
    # The prompt's own token is listed though it ranks last, as vLLM does.
    probes = [
        {
            "prompt_logprobs": [None]
            + [
                {str(chunk[t]): {"logprob": -6.0}, "1": {"logprob": -0.5}}
                for t in range(1, drift.CONTEXT)
            ]
        }
        for chunk in chunks
    ]
    run = drift.probe_arrays(probes, chunks)
    first = drift.CONTEXT // 2
    assert run["probe_tokens"].tolist() == chunks[:, first:].tolist()
    assert run["probe_ids"][0, 0, 0] == 1
    # A token's own logprob is kept, -0.5 where the token is the listed 1.
    assert set(run["probe_own"].ravel().tolist()) == {-6.0, -0.5}

    with pytest.raises(ValueError, match="1 logprobs for 2"):
        drift.generated_arrays(
            [
                {
                    "token_ids": [1, 2],
                    "text": "",
                    "logprobs": {"top_logprobs": [{"token_id:1": 0.0}]},
                }
            ],
            "traffic",
            drift.TRAFFIC_TOKENS,
        )


def test_a_change_is_located():
    rng = np.random.default_rng(17)
    probes, traffic = world(rng)
    refs = reruns(rng, probes, traffic, REF)
    # One prompt's traffic goes wrong in the candidate.
    wrong = traffic.copy()
    wrong[3] = rng.normal(size=traffic[3].shape)
    result = drift.compare(refs, reruns(rng, probes, wrong, CAND))
    assert result["verdict"] == "changed"
    where = drift.render(result).splitlines()[3:]
    assert "Where: traffic" in where[0] and "most, prompt 3, whose" in where[0]
    assert where[1].startswith("  reference: ") and where[1][13:] != where[2][13:]
    # A shift in the probes is located by chunk and position.
    shifted = probes.copy()
    shifted[2, 5] += 3 * rng.normal(size=probes.shape[-1])
    cands = reruns(rng, shifted, traffic, CAND)
    runs = refs + cands
    kls = np.array([[drift.probe_kls(a, b) for b in runs] for a in runs])
    where = drift.where(kls, "probes", len(refs), refs[0], cands[0])
    assert "1 of 8 chunks" in where[0] and "most, chunk 2, at position 5" in where[0]
