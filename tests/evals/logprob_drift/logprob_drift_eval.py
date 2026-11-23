#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logprob drift evaluation for a vLLM serve endpoint.

Scores fixed passages by teacher forcing through the `prompt_logprobs` field of
`/v1/completions`, one request at a time. `record` runs on the reference build
and saves each position's own-token logprob and top-k logprobs; `compare` runs
on the candidate build and reports how far its logprobs moved from the
reference. No engine change is needed.

Usage (from the repository root):
    vllm serve <model> --port 8000          # reference build
    python -m tests.evals.logprob_drift.logprob_drift_eval record \
        --port 8000 --output reference.json
    # restart the reference build with empty caches, then add its recording;
    # repeat for a third server (see "Server restarts" in the README)
    python -m tests.evals.logprob_drift.logprob_drift_eval record \
        --port 8000 --output reference.json --append
    # stop that server, then start the candidate build on the same GPU type
    vllm serve <model> --port 8000
    python -m tests.evals.logprob_drift.logprob_drift_eval compare \
        --port 8000 --reference reference.json
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import shlex
import sys
import time
from collections.abc import Iterable

import requests

from tests.evals.gsm8k.gsm8k_eval import load_gsm8k_data

# Floor for estimated probabilities, so that a log of zero never occurs.
MIN_PROB = 1e-12
# Rounding slack: the largest value accepted as a logprob, and how far the
# returned probabilities may sum above 1.
MAX_LOGPROB = 1e-4
# `compare` may fail a build identical to the reference with chance at most 1
# in this many, over all gates. A chosen risk, not a measured value.
NOISY_BAR_FALSE_ALARM_ONE_IN = 20


class RestartNoiseError(ValueError):
    """Raised by `compare`, before scoring, when the reference's fresh servers
    disagree: the gates cannot then tell a candidate's own change from the
    noise between server starts."""


def load_passages(num_passages: int) -> list[str]:
    """GSM8K test questions with their worked answers, from the dataset the
    gsm8k eval downloads.

    Raises:
        ValueError: if more passages are requested than the test split has.

    """
    _, test_data = load_gsm8k_data()
    if num_passages > len(test_data):
        raise ValueError(
            f"{num_passages} passages requested, {len(test_data)} available"
        )
    return [
        f"Question: {item['question']}\nAnswer: {item['answer']}"
        for item in test_data[:num_passages]
    ]


def passages_sha256(passages: list[str]) -> str:
    return hashlib.sha256("\x00".join(passages).encode()).hexdigest()


def parse_prompt_logprobs(
    token_ids: list[int], prompt_logprobs: list[dict | None], top_k: int
) -> list[dict]:
    """Per scored position (every prompt token after the first): the token's
    own logprob and the top-k `[token_id, logprob]` pairs, highest first.

    Raises:
        ValueError: if the entry count differs from the token count, or a
            position lacks the own-token logprob, a rank or `top_k` logprobs,
            or its values are not logprobs.

    """
    if len(prompt_logprobs) != len(token_ids):
        raise ValueError(
            f"{len(prompt_logprobs)} prompt_logprobs entries for "
            f"{len(token_ids)} prompt tokens"
        )
    positions = []
    for pos in range(1, len(token_ids)):
        entry = prompt_logprobs[pos]
        if not entry:
            raise ValueError(f"position {pos}: the server returned no logprobs")
        logprobs = {int(tok): float(val["logprob"]) for tok, val in entry.items()}
        if not all(math.isfinite(lp) for lp in logprobs.values()):
            raise ValueError(
                f"position {pos}: the server returned a non-finite logprob"
            )
        if max(logprobs.values()) > MAX_LOGPROB:
            raise ValueError(
                f"position {pos}: the server returned values above {MAX_LOGPROB}, "
                "which are not logprobs; serve with the default --logprobs-mode"
            )
        if sum(math.exp(lp) for lp in logprobs.values()) > 1.0 + MAX_LOGPROB:
            raise ValueError(
                f"position {pos}: the returned values sum to more than probability "
                "1, so they are not logprobs; serve with the default --logprobs-mode"
            )
        own_token = token_ids[pos]
        if own_token not in logprobs:
            raise ValueError(
                f"position {pos}: the server returned no logprob for the prompt "
                f"token {own_token}, so the passage's NLL cannot be computed"
            )
        # The server returns its top-k plus the prompt token. A prompt token
        # outside the top-k that ties the k-th logprob has a rank above k, so
        # filtering on rank keeps the server's own k-th token.
        ranks = {}
        for tok, val in entry.items():
            if val.get("rank") is None:
                raise ValueError(f"position {pos}: a logprob entry has no rank")
            ranks[int(tok)] = int(val["rank"])
        top = sorted(
            ((tok, lp) for tok, lp in logprobs.items() if ranks[tok] <= top_k),
            key=lambda kv: (-kv[1], ranks[kv[0]]),
        )[:top_k]
        if len(top) < top_k:
            raise ValueError(
                f"position {pos}: {len(top)} logprobs returned for top-k "
                f"{top_k}; lower --top-k or raise the server's --max-logprobs"
            )
        positions.append({"own": logprobs[own_token], "top": [list(t) for t in top]})
    return positions


def score_passages(
    base_url: str, model: str, passages: list[str], top_k: int, timeout: float
) -> list[dict]:
    """Teacher-forced scores of each passage, one request at a time.

    Raises:
        ValueError: if the server answers with an error or a response the
            eval cannot use.
        requests.RequestException: if a request fails.

    """
    scored = []
    for text in passages:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": text,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": top_k,
                "return_token_ids": True,
            },
            timeout=timeout,
        )
        if not response.ok:
            raise ValueError(
                f"the server answered {response.status_code}: {response.text[:500]}"
            )
        choice = response.json()["choices"][0]
        token_ids = choice.get("prompt_token_ids")
        prompt_logprobs = choice.get("prompt_logprobs")
        if not token_ids or prompt_logprobs is None:
            raise ValueError(
                "the server returned no prompt_token_ids or prompt_logprobs; it "
                "needs the `return_token_ids` and `prompt_logprobs` fields"
            )
        positions = parse_prompt_logprobs(token_ids, prompt_logprobs, top_k)
        scored.append({"token_ids": token_ids, "positions": positions})
    return scored


def kl_estimate(ref_top: list[list], cand_top: list[list]) -> float:
    """Estimated KL(reference || candidate) in nats, over the reference's top-k
    tokens plus one bucket for the rest of the vocabulary.

    A reference token missing from the candidate's top-k gets an equal share
    of the candidate's remaining mass (1 minus its top-k probabilities), split
    with one more share for the rest of the vocabulary and capped at the
    candidate's k-th probability. The tail bucket is then 1 minus the
    candidate's probabilities of the reference's top-k. The share is a guess, so
    the result is neither an upper nor a lower bound on the full-vocabulary KL.
    It can even be negative.

    A missing token's share or the tail below `MIN_PROB` is raised to it, as
    when the candidate's top-k holds all its mass. Each such term then adds at
    most -log(MIN_PROB) = 27.6 nats times its reference probability.
    """
    # The floor below would make equal lists read as a tiny nonzero KL.
    if ref_top == cand_top:
        return 0.0
    cand = {tok: lp for tok, lp in cand_top}
    missing = [tok for tok, _ in ref_top if tok not in cand]
    missing_logprob = 0.0
    if missing:
        cand_rest = max(1.0 - sum(math.exp(lp) for _, lp in cand_top), 0.0)
        share = min(math.exp(cand_top[-1][1]), cand_rest / (len(missing) + 1))
        missing_logprob = math.log(max(share, MIN_PROB))
    kl = ref_mass = cand_mass = 0.0
    for tok, ref_lp in ref_top:
        cand_lp = cand.get(tok, missing_logprob)
        kl += math.exp(ref_lp) * (ref_lp - cand_lp)
        ref_mass += math.exp(ref_lp)
        cand_mass += math.exp(cand_lp)
    ref_tail = 1.0 - ref_mass
    if ref_tail > 0.0:
        cand_tail = max(1.0 - cand_mass, MIN_PROB)
        kl += ref_tail * (math.log(ref_tail) - math.log(cand_tail))
    return kl


def compare_passages(reference: list[dict], candidate: list[dict]) -> dict:
    """Drift of `candidate` from `reference`, both from `score_passages`.

    Raises:
        ValueError: if the passages differ in count, tokens or positions; a
            top-k is empty, or the reference's is unsorted or shorter than the
            candidate's; a logprob is not finite or above `MAX_LOGPROB`; or
            there is nothing to compare (no positions, or a reference NLL of
            0).

    """
    if len(reference) != len(candidate):
        raise ValueError(
            f"{len(reference)} reference passages, {len(candidate)} candidate"
        )
    num = identical = 0
    ref_nll = cand_nll = top1 = overlap = kl = 0.0
    for ref, cand in zip(reference, candidate):
        if ref["token_ids"] != cand["token_ids"]:
            raise ValueError(
                "the passages tokenize differently: compare the same model and "
                "tokenizer"
            )
        if len(ref["positions"]) != len(cand["positions"]):
            raise ValueError(
                f"{len(ref['positions'])} reference positions, "
                f"{len(cand['positions'])} candidate"
            )
        for ref_pos, cand_pos in zip(ref["positions"], cand["positions"]):
            ref_top, cand_top = ref_pos["top"], cand_pos["top"]
            if not ref_top or not cand_top:
                raise ValueError(f"position {num}: an empty top-k")
            if len(ref_top) < len(cand_top):
                raise ValueError(
                    f"position {num}: {len(ref_top)} reference top-k logprobs, "
                    f"{len(cand_top)} candidate"
                )
            if any(a[1] < b[1] for a, b in zip(ref_top, ref_top[1:])):
                raise ValueError(f"position {num}: the reference top-k is not sorted")
            # Check both sides: the reference file may hold any number (json.load
            # accepts NaN), and every value must be a finite logprob of at most
            # MAX_LOGPROB.
            values = [ref_pos["own"], cand_pos["own"]]
            values += [lp for _, lp in ref_top] + [lp for _, lp in cand_top]
            if not all(math.isfinite(v) and v <= MAX_LOGPROB for v in values):
                raise ValueError(
                    f"position {num}: a logprob is not finite or above {MAX_LOGPROB}"
                )
            num += 1
            identical += ref_pos == cand_pos
            ref_nll -= ref_pos["own"]
            cand_nll -= cand_pos["own"]
            # Agreement when a token tied for first in the reference is also
            # tied for first in the candidate.
            ref_first = {tok for tok, lp in ref_top if lp == ref_top[0][1]}
            cand_first = {tok for tok, lp in cand_top if lp == cand_top[0][1]}
            top1 += bool(ref_first & cand_first)
            ref_ids = {tok for tok, _ in ref_top}
            overlap += len(ref_ids & {tok for tok, _ in cand_top}) / len(ref_ids)
            kl += kl_estimate(ref_top, cand_top)
    if num == 0:
        raise ValueError("no scored positions")
    if ref_nll == 0:
        raise ValueError("the reference's total NLL is 0; nothing to compare against")
    return {
        "positions": num,
        "identical_positions": identical,
        "mean_nll_reference": ref_nll / num,
        "mean_nll_candidate": cand_nll / num,
        "nll_rel_change": (cand_nll - ref_nll) / ref_nll,
        "top1_agreement": top1 / num,
        "topk_overlap": overlap / num,
        "kl_estimate": kl / num,
    }


def rerun_noise(runs: Iterable[list[dict]]) -> dict:
    """The worst drift between the first scoring pass and each later one: the
    largest absolute NLL change and KL estimate, and the lowest top-1
    agreement. `identical` is true when every later pass matched the first.
    Later passes are read one at a time, so `runs` can be a generator.

    Raises:
        ValueError: if fewer than 2 passes are given.

    """
    runs = iter(runs)
    first = next(runs, None)
    drifts = [] if first is None else [compare_passages(first, run) for run in runs]
    if not drifts:
        raise ValueError("rerun noise needs at least 2 scoring passes")
    return {
        "num_runs": len(drifts) + 1,
        "identical": all(d["identical_positions"] == d["positions"] for d in drifts),
        "max_abs_nll_rel_change": max(abs(d["nll_rel_change"]) for d in drifts),
        "min_top1_agreement": min(d["top1_agreement"] for d in drifts),
        "max_kl_estimate": max(abs(d["kl_estimate"]) for d in drifts),
    }


def check_noise_sample(
    noise: dict,
    max_nll_rel_change: float = 0.0,
    min_top1_agreement: float = 1.0,
    max_kl_estimate: float = 0.0,
) -> None:
    """Refuse a bar taken from too few noisy reruns.

    A gate whose tolerance is not looser than the measured noise takes its bar
    from the largest of the `num_runs - 1` rerun drifts. A candidate that
    scores like one more rerun adds one more draw of the same distribution.
    Each of the `num_runs` draws is then the largest with chance 1 / num_runs,
    whatever the distribution. Over `g` such gates, a failure has chance at most
    g / num_runs. So `num_runs` must be at least
    g * NOISY_BAR_FALSE_ALARM_ONE_IN: 20 for one gate, 60 for all three.

    Raises:
        ValueError: if a noise value is not finite, or the reference has
            fewer passes than the gates whose bar comes from the noise need.

    """
    values = [
        noise["max_abs_nll_rel_change"],
        noise["min_top1_agreement"],
        noise["max_kl_estimate"],
    ]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("the reference's rerun noise has a non-finite value")
    # A tolerance equal to a nonzero noise value is a bar set by the noise too.
    nll, top1, kl = values
    noisy_gates = (
        (nll > 0.0 and nll >= max_nll_rel_change)
        + (top1 < 1.0 and top1 <= min_top1_agreement)
        + (kl > 0.0 and kl >= max_kl_estimate)
    )
    needed = noisy_gates * NOISY_BAR_FALSE_ALARM_ONE_IN
    if noise["num_runs"] < needed:
        raise ValueError(
            f"the reference's {noise['num_runs']} passes differ from each other, "
            f"and {noisy_gates} gate(s) would take their bar from the worst of "
            "their drifts; a candidate identical to the reference would fail "
            f"with chance up to {noisy_gates}/{noise['num_runs']}. Record with "
            f"--num-runs {needed} or more, or give each such gate a looser "
            "tolerance than the measured noise: above it for the NLL change and "
            "KL estimate, below it for top-1 agreement"
        )


def check_gates(
    drift: dict,
    noise: dict,
    max_nll_rel_change: float = 0.0,
    min_top1_agreement: float = 1.0,
    max_kl_estimate: float = 0.0,
) -> list[str]:
    """Failed gates, empty when the candidate passes.

    Each bar is the looser of the given tolerance and the reference's own rerun
    noise: the larger for the absolute NLL change and KL estimate, the smaller
    for top-1 agreement. Each check is negated, as in `not value <= bar`, so a NaN
    drift fails it. A NaN noise value does not; `check_noise_sample` rejects it.
    """
    failures = []
    nll_bar = max(max_nll_rel_change, noise["max_abs_nll_rel_change"])
    if not abs(drift["nll_rel_change"]) <= nll_bar:
        failures.append(
            f"mean NLL changed by {drift['nll_rel_change']:+.4%}, bar {nll_bar:.4%}"
        )
    top1_bar = min(min_top1_agreement, noise["min_top1_agreement"])
    if not drift["top1_agreement"] >= top1_bar:
        failures.append(
            f"top-1 agreement {drift['top1_agreement']:.4%}, bar {top1_bar:.4%}"
        )
    kl_bar = max(max_kl_estimate, noise["max_kl_estimate"])
    if not abs(drift["kl_estimate"]) <= kl_bar:
        failures.append(
            f"estimated KL {drift['kl_estimate']:.3g} nats, bar {kl_bar:.3g}"
        )
    return failures


def recording_distance(a: list[dict], b: list[dict]) -> float:
    """The mean per-position KL estimate between two recordings, averaged over
    both directions.

    Unused here. A comparison over repeated server starts can build on it (see
    "Server restarts" in the README). It is symmetric and zero for identical
    recordings, but it is not a metric. The KL estimate covers only the top-k,
    so it can be slightly negative. Two top-k lists that differ only in the
    order of equal logprobs give zero.
    """
    return (
        compare_passages(a, b)["kl_estimate"] + compare_passages(b, a)["kl_estimate"]
    ) / 2


def server_recordings(reference: dict) -> list[list[dict]]:
    """The reference's recordings, one per server: the first server's, then the
    servers `record --append` added. A reference from one server has one."""
    return [reference["passages"]] + [
        s["passages"] for s in reference.get("other_servers", [])
    ]


def worst_rerun_noise(
    reference: dict,
    max_nll_rel_change: float = 0.0,
    min_top1_agreement: float = 1.0,
    max_kl_estimate: float = 0.0,
) -> dict:
    """The rerun noise the gates use: per value, the worst over the
    reference's servers (largest NLL change and KL, lowest top-1).

    A server holds a bar when it has the worst value of a gate whose bar
    comes from the noise under these tolerances (as in `check_noise_sample`).
    `num_runs` is the fewest passes among those servers, because a bar from
    noise is only as reliable as the server that measured it. This assumes
    the servers of one state share a rerun-noise distribution; nothing here
    checks that.

    Raises:
        ValueError: if any server's rerun noise has a non-finite value.

    """
    noises = [reference["rerun_noise"]] + [
        s["rerun_noise"] for s in reference.get("other_servers", [])
    ]
    # max and min drop a NaN that is not the first element, so check each
    # server before aggregating.
    for n in noises:
        values = (
            n["max_abs_nll_rel_change"],
            n["min_top1_agreement"],
            n["max_kl_estimate"],
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("the reference's rerun noise has a non-finite value")
    worst = {
        "max_abs_nll_rel_change": max(n["max_abs_nll_rel_change"] for n in noises),
        "min_top1_agreement": min(n["min_top1_agreement"] for n in noises),
        "max_kl_estimate": max(n["max_kl_estimate"] for n in noises),
    }
    sets_bar = {
        "max_abs_nll_rel_change": worst["max_abs_nll_rel_change"] > 0.0
        and worst["max_abs_nll_rel_change"] >= max_nll_rel_change,
        "min_top1_agreement": worst["min_top1_agreement"] < 1.0
        and worst["min_top1_agreement"] <= min_top1_agreement,
        "max_kl_estimate": worst["max_kl_estimate"] > 0.0
        and worst["max_kl_estimate"] >= max_kl_estimate,
    }
    holders = [
        n for n in noises if any(sets_bar[k] and n[k] == worst[k] for k in worst)
    ] or noises
    return {
        "num_runs": min(n["num_runs"] for n in holders),
        "identical": all(n["identical"] for n in noises),
        **worst,
    }


def restart_states(reference: dict) -> list[list[int]]:
    """Group the reference's servers (numbered from 1) into states: servers
    whose recordings are bit-identical at every position share a state."""
    states: list[tuple[list[dict], list[int]]] = []
    for i, recording in enumerate(server_recordings(reference), start=1):
        for first, members in states:
            drift = compare_passages(first, recording)
            if drift["identical_positions"] == drift["positions"]:
                members.append(i)
                break
        else:
            states.append((recording, [i]))
    return [members for _, members in states]


def append_recording(reference: dict, new: dict) -> dict:
    """Add another server's recording (a `record` result) to a reference.

    Raises:
        ValueError: if the new recording is not of the same model, vLLM
            version, server arguments, passages and top-k.

    """
    ref, add = reference["metadata"], new["metadata"]
    for key in ("model", "vllm_version", "num_passages", "passages_sha256", "top_k"):
        if ref.get(key) != add.get(key):
            raise ValueError(
                f"cannot append: {key} is {add.get(key)!r}, the reference's is "
                f"{ref.get(key)!r}; every server of a reference runs the same build "
                "and settings"
            )
    if shlex.split(ref.get("server_args") or "") != shlex.split(
        add.get("server_args") or ""
    ):
        raise ValueError("cannot append: the --server-args differ from the reference's")
    others = reference.get("other_servers", [])
    return reference | {
        "other_servers": others
        + [
            {
                "metadata": new["metadata"],
                "rerun_noise": new["rerun_noise"],
                "passages": new["passages"],
            }
        ]
    }


def server_info(base_url: str, model: str | None, timeout: float) -> dict:
    """The served model (the first one when `model` is None) and vLLM version."""
    if model is None:
        models = requests.get(f"{base_url}/v1/models", timeout=timeout)
        models.raise_for_status()
        model = models.json()["data"][0]["id"]
    version = requests.get(f"{base_url}/version", timeout=timeout)
    version.raise_for_status()
    return {"model": model, "vllm_version": version.json().get("version")}


def record(
    base_url: str,
    model: str | None = None,
    num_passages: int = 64,
    top_k: int = 20,
    num_runs: int = 2,
    server_args: str | None = None,
    timeout: float = 600,
) -> dict:
    """Score the passages `num_runs` times; keep the first pass as the
    reference and, as the rerun noise, the worst drift of any later pass from
    it.

    Raises:
        ValueError: if a count is out of range or the server's response is
            not usable.

    """
    if num_runs < 2:
        raise ValueError("num_runs must be at least 2 to measure rerun noise")
    if num_passages < 1 or top_k < 1:
        raise ValueError("num_passages and top_k must be at least 1")
    info = server_info(base_url, model, timeout)
    passages = load_passages(num_passages)
    tic = time.perf_counter()

    def score() -> list[dict]:
        return score_passages(base_url, info["model"], passages, top_k, timeout)

    # Keep only the first pass; each later one is compared and dropped.
    first = score()
    noise = rerun_noise(
        itertools.chain([first], (score() for _ in range(num_runs - 1)))
    )
    return {
        "metadata": info
        | {
            "server_args": server_args,
            "num_passages": num_passages,
            "passages_sha256": passages_sha256(passages),
            "top_k": top_k,
            "positions": sum(len(p["positions"]) for p in first),
            "latency": time.perf_counter() - tic,
        },
        "rerun_noise": noise,
        "passages": first,
    }


def compare(
    base_url: str,
    reference: dict,
    model: str | None = None,
    server_args: str | None = None,
    timeout: float = 600,
    **gates: float,
) -> dict:
    """Score the reference's passages on the candidate server and gate the
    drift. `gates` are the tolerances of `check_gates`.

    Raises:
        ValueError: if a tolerance is out of range, the server arguments,
            model or dataset differ from the reference's, the reference's
            rerun noise is not finite or has too few passes, or the server's
            response is not usable.
        RestartNoiseError: if the reference's servers are not bit-identical
            (checked before scoring).
        KeyError: if the reference lacks a field the comparison needs.

    """
    gates = {
        "max_nll_rel_change": 0.0,
        "min_top1_agreement": 1.0,
        "max_kl_estimate": 0.0,
    } | gates
    if not 0.0 <= gates["min_top1_agreement"] <= 1.0:
        raise ValueError("min_top1_agreement must be between 0 and 1")
    for name in ("max_nll_rel_change", "max_kl_estimate"):
        if not gates[name] >= 0.0:
            raise ValueError(f"{name} must not be negative")
    ref_meta = reference["metadata"]
    # An empty string is the same as no arguments.
    ref_args = ref_meta.get("server_args") or None
    server_args = server_args or None
    if (ref_args is None) != (server_args is None) or (
        ref_args is not None and shlex.split(ref_args) != shlex.split(server_args)
    ):
        raise ValueError(
            f"the candidate's --server-args ({server_args!r}) differ from the "
            f"reference's ({ref_args!r}); give the same arguments to both, or to "
            "neither"
        )
    # A reference that is wrong for this run is an input error, not restart
    # noise, so these checks come first.
    passages = load_passages(ref_meta["num_passages"])
    if passages_sha256(passages) != ref_meta["passages_sha256"]:
        raise ValueError("the dataset differs from the one the reference scored")
    info = server_info(base_url, model, timeout)
    if info["model"] != ref_meta["model"]:
        raise ValueError(
            f"the candidate serves {info['model']}, the reference was recorded "
            f"on {ref_meta['model']}"
        )
    noise = worst_rerun_noise(reference, **gates)
    states = restart_states(reference)
    if len(states) > 1:
        reruns = (
            "; some servers' own passes differ too, so rerun noise may account "
            "for part of this"
            if not noise["identical"]
            else ""
        )
        raise RestartNoiseError(
            f"the reference's {sum(map(len, states))} fresh servers fall into "
            f"{len(states)} states ({'; '.join(map(str, states))}): its own "
            "restarts differ, so a candidate would fail the gates without any "
            "change of its own. Telling a change from this noise needs a "
            "comparison over repeated server starts, which this eval does not "
            f'make; see "Server restarts" in the README{reruns}'
        )
    check_noise_sample(noise, **gates)
    scored = score_passages(
        base_url, info["model"], passages, ref_meta["top_k"], timeout
    )
    drift = compare_passages(reference["passages"], scored)
    failures = check_gates(drift, noise, **gates)
    return {
        "reference": ref_meta,
        "candidate": info | {"server_args": server_args},
        "rerun_noise": noise,
        "reference_servers": len(states[0]),
        "drift": drift,
        "gates": gates,
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Logprob drift eval for vLLM serve")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    record_parser = subparsers.add_parser(
        "record", help="Score the passages on the reference server"
    )
    compare_parser = subparsers.add_parser(
        "compare", help="Score them on the candidate server and gate the drift"
    )
    for sub in (record_parser, compare_parser):
        sub.add_argument("--host", type=str, default="http://127.0.0.1")
        sub.add_argument("--port", type=int, default=8000)
        sub.add_argument(
            "--model", type=str, help="Served model name (default: the first one)"
        )
        sub.add_argument(
            "--server-args",
            type=str,
            help=(
                "The server's `vllm serve` arguments as you typed them; record "
                "stores them and compare requires the same ones. They are not "
                "read from the server"
            ),
        )
        sub.add_argument("--request-timeout-seconds", type=float, default=600)
    record_parser.add_argument("--num-passages", type=int, default=64)
    record_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Logprobs per position; at most the server's --max-logprobs",
    )
    record_parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help=(
            "Scoring passes; the drift between them is the rerun noise. If the "
            f"passes differ, compare needs {NOISY_BAR_FALSE_ALARM_ONE_IN} for "
            "each gate whose bar the noise sets (up to 3 gates)"
        ),
    )
    record_parser.add_argument("--output", type=str, required=True)
    record_parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Add this server's recording to the reference already in --output "
            "instead of writing a new one. Use it on freshly started servers of "
            "the same build, with empty caches"
        ),
    )
    compare_parser.add_argument("--reference", type=str, required=True)
    compare_parser.add_argument(
        "--max-nll-rel-change",
        type=float,
        default=0.0,
        help="Allowed absolute relative change of the mean NLL, when above the "
        "rerun noise",
    )
    compare_parser.add_argument(
        "--min-top1-agreement",
        type=float,
        default=1.0,
        help="Lowest allowed top-1 agreement, when below the rerun noise's",
    )
    compare_parser.add_argument(
        "--max-kl-estimate",
        type=float,
        default=0.0,
        help="Allowed estimated KL in nats per position, when above the rerun noise",
    )
    compare_parser.add_argument("--save-results", type=str)
    args = parser.parse_args()
    base_url = f"{args.host}:{args.port}"

    try:
        if args.mode == "record":
            result = record(
                base_url,
                model=args.model,
                num_passages=args.num_passages,
                top_k=args.top_k,
                num_runs=args.num_runs,
                server_args=args.server_args,
                timeout=args.request_timeout_seconds,
            )
            # This server's own noise, before appending to the reference.
            server_noise = result["rerun_noise"]
            if args.append:
                with open(args.output) as f:
                    result = append_recording(json.load(f), result)
            servers = len(server_recordings(result))
            states = restart_states(result)
            # A partial write would lose the servers already in the reference.
            tmp = f"{args.output}.tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(result, f)
                os.replace(tmp, args.output)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            print(f"Positions: {result['metadata']['positions']}")
            print(
                f"Servers in the reference: {servers}, in {len(states)} state(s)"
                + ("" if len(states) == 1 else f": {states}")
            )
            print(f"This server's rerun noise: {json.dumps(server_noise)}")
            print(f"Reference saved to {args.output}")
            try:
                check_noise_sample(worst_rerun_noise(result))
            except ValueError as e:
                print(f"Warning: with the default tolerances, {e}", file=sys.stderr)
            if servers == 1:
                print(
                    "Note: one server; record on 2 more freshly started servers "
                    "with --append to check restart noise",
                    file=sys.stderr,
                )
            if len(states) > 1:
                # The recordings are kept for a comparison over repeated
                # starts; compare refuses this reference (exit 3).
                print(
                    f"Restart noise: the reference's servers are in {len(states)} "
                    "states; compare would stop with exit 3",
                    file=sys.stderr,
                )
                return 3
            return 0

        with open(args.reference) as f:
            reference = json.load(f)
        result = compare(
            base_url,
            reference,
            model=args.model,
            server_args=args.server_args,
            timeout=args.request_timeout_seconds,
            max_nll_rel_change=args.max_nll_rel_change,
            min_top1_agreement=args.min_top1_agreement,
            max_kl_estimate=args.max_kl_estimate,
        )
    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
        OSError,
        requests.RequestException,
    ) as e:
        if isinstance(e, RestartNoiseError):
            print(f"Restart noise: {e}", file=sys.stderr)
            return 3
        print(f"Error: {e}", file=sys.stderr)
        return 2

    drift = result["drift"]
    print("\nResults:")
    for side in ("reference", "candidate"):
        meta = result[side]
        print(f"{side.capitalize()}: {meta['model']}, vLLM {meta.get('vllm_version')}")
    print(f"Positions: {drift['positions']}")
    print(f"Identical positions: {drift['identical_positions']}")
    print(f"Mean NLL change: {drift['nll_rel_change']:+.4%}")
    print(f"Top-1 agreement: {drift['top1_agreement']:.4%}")
    print(f"Top-k overlap: {drift['topk_overlap']:.4f}")
    print(f"KL estimate (not a bound): {drift['kl_estimate']:.4g} nats")
    print(
        "  A reference token outside the candidate's top-k gets a share of the "
        f"candidate's remaining mass, at least {MIN_PROB:g}; it adds at most "
        f"{-math.log(MIN_PROB):.1f} nats times its reference probability"
    )
    print(f"Reference servers in agreement: {result['reference_servers']}")
    if result["reference_servers"] == 1:
        print(
            "  Restart noise unchecked: the reference comes from one server "
            "(record --append adds more)"
        )
    for failure in result["failures"]:
        print(f"FAILED: {failure}")
    if result["failures"]:
        print(
            "  One failing server can be a rare restart state: start the "
            "candidate again with empty caches and compare once more. A second "
            "failure makes a change likely but does not rule out restart noise"
        )
    if args.save_results:
        try:
            with open(args.save_results, "w") as f:
                json.dump(result, f, indent=2)
        except OSError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        print(f"Results saved to {args.save_results}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
