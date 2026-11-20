# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logprob drift eval: metric arithmetic on synthetic logprobs, and a
record/compare round trip against one vLLM server.

Usage:
pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py \
    --config-list-file=configs/models-small.txt
"""

import json
import math
import random
import shlex
import sys

import pytest
import yaml

from tests.utils import RemoteOpenAIServer

from . import logprob_drift_eval
from .logprob_drift_eval import (
    NOISY_BAR_FALSE_ALARM_ONE_IN,
    check_gates,
    check_noise_sample,
    compare,
    compare_passages,
    kl_estimate,
    parse_prompt_logprobs,
    record,
    rerun_noise,
)

VOCAB_SIZE = 50
# Server startup budget, as the gsm8k eval uses.
DEFAULT_STARTUP_MAX_WAIT_SECONDS = 1200
TOP_K = 5


def _scored(temperature: float = 1.0) -> list[dict]:
    """Passages as `score_passages` returns them, parsed from server-style
    prompt_logprobs (top-k plus the prompt token, keyed by token id string)."""
    rng = random.Random(0)
    passages = []
    for _ in range(3):
        token_ids = [rng.randrange(VOCAB_SIZE) for _ in range(12)]
        entries: list[dict | None] = [None]
        for tok in token_ids[1:]:
            logits = [rng.gauss(0.0, 2.0) / temperature for _ in range(VOCAB_SIZE)]
            log_z = math.log(sum(math.exp(x) for x in logits))
            ranked = sorted(range(VOCAB_SIZE), key=lambda t: -logits[t])
            entries.append(
                {
                    str(t): {"logprob": logits[t] - log_z, "rank": ranked.index(t) + 1}
                    for t in set(ranked[:TOP_K]) | {tok}
                }
            )
        positions = parse_prompt_logprobs(token_ids, entries, TOP_K)
        passages.append({"token_ids": token_ids, "positions": positions})
    return passages


def test_identical_logprobs_pass_with_exact_zeros():
    reference, candidate = _scored(), _scored()
    drift = compare_passages(reference, candidate)
    assert drift["identical_positions"] == drift["positions"] == 3 * 11
    assert drift["nll_rel_change"] == 0.0
    assert drift["kl_estimate"] == 0.0
    assert drift["top1_agreement"] == 1.0
    assert drift["topk_overlap"] == 1.0
    noise = rerun_noise([reference, _scored()])
    assert noise["identical"] and noise["max_abs_nll_rel_change"] == 0.0
    assert check_gates(drift, noise) == []


def test_shifted_distribution_fails():
    reference = _scored()
    noise = rerun_noise([reference, _scored()])
    drift = compare_passages(reference, _scored(temperature=1.5))
    assert drift["nll_rel_change"] != 0.0
    assert drift["kl_estimate"] > 0.0
    # A temperature change keeps every ranking, so top-1 agreement stays at 1.
    assert drift["top1_agreement"] == 1.0
    failures = check_gates(drift, noise)
    assert [f.split()[0] for f in failures] == ["mean", "estimated"]
    # Tolerances at the measured drift accept it.
    assert (
        check_gates(
            drift,
            noise,
            max_nll_rel_change=abs(drift["nll_rel_change"]),
            max_kl_estimate=drift["kl_estimate"],
        )
        == []
    )


def _one_position(top: list[list], own_token: int) -> list[dict]:
    own = dict((tok, lp) for tok, lp in top)[own_token]
    return [{"token_ids": [0, own_token], "positions": [{"own": own, "top": top}]}]


def test_top1_tie_counts_as_agreement():
    reference = _one_position([[1, -0.5], [2, -1.0]], own_token=1)
    tied = _one_position([[2, -0.6], [1, -0.6]], own_token=1)
    behind = _one_position([[2, -0.5], [1, -0.7]], own_token=1)
    assert compare_passages(reference, tied)["top1_agreement"] == 1.0
    assert compare_passages(reference, behind)["top1_agreement"] == 0.0


def test_unequal_position_counts_raise():
    reference = _one_position([[1, -0.5], [2, -1.0]], own_token=1)
    candidate = [{"token_ids": reference[0]["token_ids"], "positions": []}]
    with pytest.raises(ValueError, match="1 reference positions, 0 candidate"):
        compare_passages(reference, candidate)


def test_kl_estimate_reference_token_outside_candidate_top_k():
    reference = _one_position([[1, math.log(0.6)], [2, math.log(0.3)]], own_token=1)
    candidate = _one_position([[1, math.log(0.6)], [3, math.log(0.3)]], own_token=1)
    drift = compare_passages(reference, candidate)
    # Token 2 is outside the candidate's top-k. It gets one of two equal shares
    # of the candidate's remaining 0.1, so 0.05. The candidate's tail is what
    # the reference's top-k leaves: 1 - 0.6 - 0.05 = 0.35.
    expected = 0.3 * math.log(0.3 / 0.05) + 0.1 * math.log(0.1 / 0.35)
    assert drift["kl_estimate"] == pytest.approx(expected)
    assert drift["topk_overlap"] == 0.5


def test_gates_widen_to_rerun_noise():
    reference = _scored()
    drift = compare_passages(reference, _scored(temperature=1.5))
    noise = {
        "max_abs_nll_rel_change": abs(drift["nll_rel_change"]),
        "min_top1_agreement": 1.0,
        "max_kl_estimate": drift["kl_estimate"],
    }
    assert check_gates(drift, noise) == []


def test_values_above_zero_raise():
    entries = [None, {"7": {"logprob": 3.5, "rank": 1}}]
    with pytest.raises(ValueError, match="not logprobs"):
        parse_prompt_logprobs([5, 7], entries, top_k=1)


def test_non_finite_logprob_raises():
    entries = [None, {"7": {"logprob": float("-inf"), "rank": 1}}]
    with pytest.raises(ValueError, match="non-finite"):
        parse_prompt_logprobs([5, 7], entries, top_k=1)


def test_kl_gate_compares_magnitudes():
    drift = {"nll_rel_change": 0.0, "top1_agreement": 1.0, "kl_estimate": -0.005}
    noise = {
        "max_abs_nll_rel_change": 0.0,
        "min_top1_agreement": 1.0,
        "max_kl_estimate": 0.01,
    }
    assert check_gates(drift, noise) == []
    drift["kl_estimate"] = -0.02
    assert [f.split()[0] for f in check_gates(drift, noise)] == ["estimated"]


def test_values_summing_above_one_raise():
    entries = [
        None,
        {"7": {"logprob": -0.1, "rank": 1}, "8": {"logprob": -0.2, "rank": 2}},
    ]
    with pytest.raises(ValueError, match="more than probability"):
        parse_prompt_logprobs([5, 7], entries, top_k=2)


def test_missing_own_token_logprob_raises():
    entries = [None, {"3": {"logprob": -0.1, "rank": 1}}]
    with pytest.raises(ValueError, match="no logprob for the prompt token 7"):
        parse_prompt_logprobs([5, 7], entries, top_k=1)


def test_record_rejects_non_positive_counts():
    with pytest.raises(ValueError, match="must be at least 1"):
        record("http://127.0.0.1:1", model="m", num_passages=0)
    with pytest.raises(ValueError, match="must be at least 1"):
        record("http://127.0.0.1:1", model="m", top_k=0)


def test_top_k_tie_keeps_the_servers_kth_token():
    # The prompt token 3 ties token 9 at -0.7, but the server ranks it 2,
    # outside the top-1. The parsed top-1 must still be token 9.
    entries = [
        None,
        {"9": {"logprob": -0.7, "rank": 1}, "3": {"logprob": -0.7, "rank": 2}},
    ]
    (position,) = parse_prompt_logprobs([5, 3], entries, top_k=1)
    assert position["top"] == [[9, -0.7]]
    assert position["own"] == -0.7


def test_missing_rank_raises():
    entries = [None, {"7": {"logprob": -0.1}}]
    with pytest.raises(ValueError, match="no rank"):
        parse_prompt_logprobs([5, 7], entries, top_k=1)


def test_no_logprobs_at_a_position_raises():
    with pytest.raises(ValueError, match="position 1: the server returned no"):
        parse_prompt_logprobs([5, 7], [None, None], top_k=1)


def test_fewer_than_top_k_logprobs_raise():
    entries = [None, {"7": {"logprob": -0.1, "rank": 1}}]
    with pytest.raises(ValueError, match="1 logprobs returned for top-k 2"):
        parse_prompt_logprobs([5, 7], entries, top_k=2)


def test_nan_in_reference_raises():
    reference = _one_position([[1, -0.5], [2, -1.0]], own_token=1)
    # json.load reads the token NaN as a float NaN.
    reference = json.loads(json.dumps(reference).replace("-0.5", "NaN"))
    candidate = _one_position([[1, -0.5], [2, -1.0]], own_token=1)
    with pytest.raises(ValueError, match="not finite"):
        compare_passages(reference, candidate)


def test_reference_logprob_above_max_raises():
    # A hand-edited value above 709 would otherwise overflow math.exp.
    reference = _one_position([[1, 1e5], [2, -1.0]], own_token=1)
    candidate = _one_position([[1, -0.5], [2, -1.0]], own_token=1)
    with pytest.raises(ValueError, match="above"):
        compare_passages(reference, candidate)


def test_reference_top_k_shorter_than_candidate_raises():
    reference = _one_position([[1, -0.5]], own_token=1)
    candidate = _one_position([[1, -0.5], [2, -1.0]], own_token=1)
    with pytest.raises(ValueError, match="1 reference top-k logprobs, 2 candidate"):
        compare_passages(reference, candidate)


def test_kl_estimate_is_zero_for_equal_lists_with_a_tiny_tail():
    # The tail is 1e-13 on both sides, below the MIN_PROB floor.
    top = [[1, -1e-13]]
    assert kl_estimate(top, [list(t) for t in top]) == 0.0


def test_unsorted_reference_top_k_raises():
    reference = _one_position([[2, -1.0], [1, -0.5]], own_token=1)
    candidate = _one_position([[1, -0.5], [2, -1.0]], own_token=1)
    with pytest.raises(ValueError, match="not sorted"):
        compare_passages(reference, candidate)


def test_rerun_noise_reads_a_generator():
    runs = [_scored(), _scored(), _scored(temperature=1.5)]
    assert rerun_noise(iter(runs)) == rerun_noise(runs)
    assert rerun_noise(iter(runs))["num_runs"] == 3
    with pytest.raises(ValueError, match="at least 2"):
        rerun_noise(iter(runs[:1]))


def test_nan_drift_fails_every_gate():
    nan = float("nan")
    noise = {
        "max_abs_nll_rel_change": 0.0,
        "min_top1_agreement": 1.0,
        "max_kl_estimate": 0.0,
    }
    drift = {"nll_rel_change": nan, "top1_agreement": nan, "kl_estimate": nan}
    assert len(check_gates(drift, noise)) == 3


def test_top1_gate_fails():
    drift = {"nll_rel_change": 0.0, "top1_agreement": 0.9, "kl_estimate": 0.0}
    noise = {
        "max_abs_nll_rel_change": 0.0,
        "min_top1_agreement": 1.0,
        "max_kl_estimate": 0.0,
    }
    failures = check_gates(drift, noise)
    assert len(failures) == 1 and failures[0].startswith("top-1")


def test_kl_estimate_floors_missing_probability():
    # The candidate's top-k holds all its mass, so a reference token outside
    # it gets the MIN_PROB floor instead of log(0).
    ref_top = [[1, math.log(0.5)], [2, math.log(0.5)]]
    cand_top = [[1, math.log(0.5)], [3, math.log(0.5)]]
    expected = 0.5 * (math.log(0.5) - math.log(logprob_drift_eval.MIN_PROB))
    assert kl_estimate(ref_top, cand_top) == pytest.approx(expected)


def _noisy_runs(rng: random.Random, num_runs: int, scale: float) -> list:
    """`num_runs` scoring passes of one server whose logprobs move by
    independent Gaussian noise of standard deviation `scale` between passes."""
    base = _scored()
    runs = []
    for _ in range(num_runs):
        run = []
        for passage in base:
            positions = [
                {
                    "own": pos["own"] + rng.gauss(0.0, scale),
                    # Kept sorted, highest first, as the server returns it.
                    "top": sorted(
                        ([tok, lp + rng.gauss(0.0, scale)] for tok, lp in pos["top"]),
                        key=lambda t: -t[1],
                    ),
                }
                for pos in passage["positions"]
            ]
            run.append({"token_ids": passage["token_ids"], "positions": positions})
        runs.append(run)
    return runs


@pytest.mark.parametrize("num_runs", [2, 4])
def test_rerun_bar_fails_an_identical_build_about_once_in_num_runs(num_runs):
    """Why check_noise_sample exists: with noisy reruns, the NLL gate fails a
    candidate drawn like a rerun about 1 / num_runs of the time."""
    rng = random.Random(1)
    trials = 400
    fails = 0
    for _ in range(trials):
        *runs, candidate = _noisy_runs(rng, num_runs + 1, scale=1e-3)
        drift = compare_passages(runs[0], candidate)
        failures = check_gates(drift, rerun_noise(runs))
        fails += any(f.startswith("mean") for f in failures)
    # 0.125 is 5 standard deviations (0.025) of the rate at num_runs=2 over 400
    # trials.
    assert abs(fails / trials - 1 / num_runs) < 0.125


def _noise(identical: bool, num_runs: int, nll: float = 0.01, kl: float = 0.0) -> dict:
    return {
        "num_runs": num_runs,
        "identical": identical,
        "max_abs_nll_rel_change": 0.0 if identical else nll,
        "min_top1_agreement": 1.0,
        "max_kl_estimate": 0.0 if identical else kl,
    }


def test_noisy_bar_needs_enough_runs():
    one_in = NOISY_BAR_FALSE_ALARM_ONE_IN
    check_noise_sample(_noise(identical=True, num_runs=2))
    with pytest.raises(ValueError, match="up to 1/2"):
        check_noise_sample(_noise(identical=False, num_runs=2))
    check_noise_sample(_noise(identical=False, num_runs=one_in))
    # A nonzero `kl` makes the KL gate noisy too, so two gates need 40 passes.
    two_gates = _noise(identical=False, num_runs=2 * one_in - 1, kl=1e-6)
    with pytest.raises(ValueError, match=f"--num-runs {2 * one_in} "):
        check_noise_sample(two_gates)
    check_noise_sample(_noise(identical=False, num_runs=2 * one_in, kl=1e-6))
    # With a tolerance above the noise, the bar does not come from the sampled
    # maximum, so 2 passes are enough. A tolerance equal to the noise still
    # takes its bar from the noise, so 2 passes are refused.
    check_noise_sample(_noise(identical=False, num_runs=2), max_nll_rel_change=0.02)
    with pytest.raises(ValueError, match="up to 1/2"):
        check_noise_sample(_noise(identical=False, num_runs=2), max_nll_rel_change=0.01)
    nan_noise = _noise(False, one_in, nll=math.nan)
    with pytest.raises(ValueError, match="non-finite"):
        check_noise_sample(nan_noise)
    # The top-1 gate is looser with a lower tolerance, not a higher one.
    top1_noise = _noise(identical=False, num_runs=2, nll=0.0) | {
        "min_top1_agreement": 0.9
    }
    with pytest.raises(ValueError, match="below it for top-1"):
        check_noise_sample(top1_noise)
    check_noise_sample(top1_noise, min_top1_agreement=0.85)


def _reference(server_args: str | None, noise: dict | None = None) -> dict:
    passages = _scored()
    return {
        "metadata": {
            "model": "m",
            "server_args": server_args,
            "num_passages": len(passages),
            "passages_sha256": "unused",
            "top_k": TOP_K,
        },
        "rerun_noise": noise or _noise(identical=True, num_runs=2),
        "passages": passages,
    }


@pytest.fixture
def fake_server(monkeypatch):
    """Replaces the network and the dataset download with synthetic data;
    the returned list is the queue of scoring passes the server answers."""
    scores: list = []
    monkeypatch.setattr(logprob_drift_eval, "load_passages", lambda n: ["p"] * n)
    monkeypatch.setattr(logprob_drift_eval, "passages_sha256", lambda p: "unused")
    monkeypatch.setattr(
        logprob_drift_eval,
        "server_info",
        lambda url, model, timeout: {"model": "m", "vllm_version": "x"},
    )
    monkeypatch.setattr(
        logprob_drift_eval, "score_passages", lambda *args: scores.pop(0)
    )
    return scores


@pytest.mark.parametrize(
    "ref_args, cand_args",
    [(None, "--kv-cache-dtype fp8"), ("--max-model-len 4096", None), ("-a 1", "-a 2")],
)
def test_server_args_must_match_on_both_sides(fake_server, ref_args, cand_args):
    with pytest.raises(ValueError, match="to both, or to neither"):
        compare("http://unused", _reference(ref_args), server_args=cand_args)


def test_server_args_compare_as_split_arguments(fake_server):
    fake_server.append(_scored())
    result = compare("http://unused", _reference("-a  1"), server_args="-a 1")
    assert result["passed"]
    fake_server.append(_scored())
    assert compare("http://unused", _reference(""), server_args=None)["passed"]


@pytest.mark.parametrize("gate", ["max_nll_rel_change", "max_kl_estimate"])
def test_negative_tolerance_raises(fake_server, gate):
    with pytest.raises(ValueError, match="must not be negative"):
        compare("http://unused", _reference(None), **{gate: -0.5})


def test_compare_refuses_a_noisy_two_run_reference(fake_server):
    reference = _reference(None, _noise(identical=False, num_runs=2))
    with pytest.raises(ValueError, match="--num-runs 20"):
        compare("http://unused", reference)


def _run_main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["logprob_drift_eval", *argv])
    return logprob_drift_eval.main()


def test_main_exit_codes(fake_server, monkeypatch, tmp_path):
    ref_file = tmp_path / "reference.json"
    fake_server.extend([_scored(), _scored()])
    assert _run_main(monkeypatch, "record", "--output", str(ref_file)) == 0
    fake_server.append(_scored())
    assert _run_main(monkeypatch, "compare", "--reference", str(ref_file)) == 0
    fake_server.append(_scored(temperature=1.5))
    assert _run_main(monkeypatch, "compare", "--reference", str(ref_file)) == 1
    missing = str(tmp_path / "missing.json")
    assert _run_main(monkeypatch, "compare", "--reference", missing) == 2
    # Metadata the gates do not use may be missing.
    reference = json.loads(ref_file.read_text())
    del reference["metadata"]["vllm_version"]
    ref_file.write_text(json.dumps(reference))
    fake_server.append(_scored())
    assert _run_main(monkeypatch, "compare", "--reference", str(ref_file)) == 0
    # A malformed reference is an error, not a drift.
    reference = json.loads(ref_file.read_text())
    reference["passages"][0]["positions"][0]["top"] = []
    ref_file.write_text(json.dumps(reference))
    fake_server.append(_scored())
    assert _run_main(monkeypatch, "compare", "--reference", str(ref_file)) == 2


def test_record_warns_when_compare_would_refuse(
    fake_server, monkeypatch, tmp_path, capsys
):
    fake_server.extend(_noisy_runs(random.Random(0), 2, scale=1e-3))
    ref_file = tmp_path / "reference.json"
    assert _run_main(monkeypatch, "record", "--output", str(ref_file)) == 0
    noise = json.loads(ref_file.read_text())["rerun_noise"]
    noisy_gates = (
        (noise["max_abs_nll_rel_change"] > 0)
        + (noise["min_top1_agreement"] < 1)
        + (noise["max_kl_estimate"] > 0)
    )
    needed = noisy_gates * NOISY_BAR_FALSE_ALARM_ONE_IN
    assert noisy_gates and f"--num-runs {needed} " in capsys.readouterr().err


def test_logprob_drift_correctness(config_filename):
    """Record and compare on one server: the drift must pass the gates. The
    candidate is then one more pass of the same process, which is the case the
    gates are built for. A server restart is not part of this test."""
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))
    server_args = shlex.split(eval_config.get("server_args", ""))
    server_args.append("--disable-uvicorn-access-log")
    startup_max_wait_seconds = eval_config.get(
        "startup_max_wait_seconds", DEFAULT_STARTUP_MAX_WAIT_SECONDS
    )
    env_dict = dict(eval_config.get("env") or {})
    env_dict["VLLM_ENGINE_READY_TIMEOUT_S"] = str(int(startup_max_wait_seconds))

    with RemoteOpenAIServer(
        eval_config["model_name"],
        server_args,
        env_dict=env_dict,
        max_wait_seconds=startup_max_wait_seconds,
    ) as remote_server:

        def record_runs(num_runs: int) -> dict:
            return record(
                remote_server.url_root,
                model=eval_config["model_name"],
                num_passages=eval_config["num_passages"],
                top_k=eval_config["top_k"],
                num_runs=num_runs,
                server_args=shlex.join(server_args),
            )

        reference = record_runs(2)
        if not reference["rerun_noise"]["identical"]:
            # Enough passes for all three gates to take their bar from the noise.
            reference = record_runs(3 * NOISY_BAR_FALSE_ALARM_ONE_IN)
        result = compare(
            remote_server.url_root,
            reference,
            model=eval_config["model_name"],
            server_args=shlex.join(server_args),
        )

    drift = result["drift"]
    print(f"Logprob drift results for {eval_config['model_name']}:")
    print(f"  Rerun noise: {result['rerun_noise']}")
    print(f"  Positions: {drift['positions']}")
    print(f"  Mean NLL change: {drift['nll_rel_change']:+.4%}")
    print(f"  Top-1 agreement: {drift['top1_agreement']:.4%}")
    assert result["passed"], result["failures"]
