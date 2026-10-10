# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the existing System One gateway against a running Nemotron backend."""

import argparse
import json
import math
import os
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8011")
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    cases = {
        "choice": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments and payouts",
                    "technical": "Infrastructure outages",
                },
            }
        },
        "noul": {
            "urgent": {"type": "noul", "instructions": "Has this lasted multiple days?"}
        },
        "score": {
            "severity": {
                "type": "score",
                "instructions": "How disruptive is this problem?",
                "criteria": [
                    "No problem",
                    "Minor inconvenience",
                    "Blocked payments for multiple days",
                ],
            }
        },
        "twelve_questions": {
            f"q{i}": {
                "type": "noul",
                "instructions": "Have payouts failed for multiple days?",
            }
            for i in range(12)
        },
    }
    results = {}
    for name, questions in cases.items():
        body = {
            "model": "jev-latest",
            "state": "Customer payouts have failed for three days.",
            "questions": questions,
        }
        started = time.perf_counter()
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/systemone",
            headers=headers,
            data=json.dumps(body).encode(),
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        elapsed = (time.perf_counter() - started) * 1000
        assert result["answers"].keys() == questions.keys()
        for answer in result["answers"].values():
            if answer["type"] == "noul":
                assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1
            else:
                probabilities = answer["probabilities"]
                assert all(
                    math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()
                )
                assert math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)
        results[name] = {"ms": round(elapsed, 2), "answers": result["answers"]}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
