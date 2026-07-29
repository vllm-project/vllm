# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import requests
from prometheus_client.parser import text_string_to_metric_families


def assert_no_nan_logits(metrics_url: str) -> None:
    """Fail if the server reported any requests with NaNs in logits."""
    response = requests.get(metrics_url, timeout=30)
    response.raise_for_status()
    samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == "vllm:corrupted_requests_total"
    ]
    assert samples, "NaN logits metric is missing"

    num_corrupted_requests = sum(sample.value for sample in samples)
    assert num_corrupted_requests == 0, (
        f"Detected {num_corrupted_requests:g} requests with NaNs in logits: "
        f"{[(sample.labels, sample.value) for sample in samples]}"
    )
