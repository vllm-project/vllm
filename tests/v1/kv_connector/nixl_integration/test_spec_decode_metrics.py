# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assert speculative decoding stays engaged across the PD + offload path.

Speculative decoding is output-preserving, so an accuracy test alone cannot
tell whether it is actually running: if drafts silently stop being accepted
across the NIXL transfer + CPU offload path, decoding falls back to one token
per step and the output (and gsm8k accuracy) is unchanged.

After traffic has flowed through the pipeline, the decode server's spec-decode
counters must therefore show a healthy mean acceptance length, i.e. drafts were
proposed and a meaningful fraction was accepted.

Environment variables:
    SERVER_HOST  - decode server host (default: 127.0.0.1)
    DECODE_PORT  - decode server port to scrape /metrics from (default: 8200)
"""

import os
from urllib.request import urlopen

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
DECODE_PORT = os.environ.get("DECODE_PORT", "8200")

MIN_ACCEPTANCE_LENGTH = 1.3


def _fetch_counter(metric_name: str) -> float:
    """Return a single counter value from the decode server's /metrics."""
    url = f"http://{SERVER_HOST}:{DECODE_PORT}/metrics"
    body = urlopen(url).read().decode()
    for line in body.split("\n"):
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            return float(line.rsplit(" ", 1)[-1])
    raise ValueError(f"Metric {metric_name} not found in decode /metrics")


def test_spec_decode_engaged():
    """Speculative decoding must remain active through the combined pipeline."""
    n_drafts = _fetch_counter("vllm:spec_decode_num_drafts_total")
    n_accepted = _fetch_counter("vllm:spec_decode_num_accepted_tokens_total")

    assert n_drafts > 0, "No speculative drafts were generated on the decode server"

    mean_acceptance_length = 1 + (n_accepted / n_drafts)
    print(
        f"\nspec decode: drafts={n_drafts:.0f} accepted={n_accepted:.0f} "
        f"mean acceptance length={mean_acceptance_length:.3f}"
    )
    assert mean_acceptance_length > MIN_ACCEPTANCE_LENGTH, (
        f"Mean acceptance length {mean_acceptance_length:.3f} <= "
        f"{MIN_ACCEPTANCE_LENGTH}: speculative decoding degraded; drafts are "
        "not being accepted through the NIXL transfer + CPU offload path"
    )
