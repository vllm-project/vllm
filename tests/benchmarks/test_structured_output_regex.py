# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from pathlib import Path


def test_regex_correct_rate_requires_a_full_match(monkeypatch):
    monkeypatch.syspath_prepend(Path(__file__).resolve().parents[2] / "benchmarks")
    from benchmark_serving_structured_output import evaluate

    args = Namespace(structure_type="regex", regex=r"\w+@\w+\.com\n")
    email = "alan@enigma.com\n"
    results = [
        {"expected": None, "generated": email},
        {"expected": None, "generated": email + "extra text"},
    ]

    assert evaluate(results, args) == 50.0
    assert [result["correctness"] for result in results] == [True, False]
