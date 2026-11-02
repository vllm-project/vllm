# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that decoders behind p2p_connector_proxy.py pull KV over the p2p tier.

gsm8k accuracy alone passes even when the decoder silently recomputes the
prompt, so this sends fully unique prompts through the proxy and asserts the
decoders' external prefix cache hit counter grew by most of the prompt blocks.

With EXPECT_TRANSFER=0 it asserts the opposite: no KV moved. That pins the
safe fallback for deployments the tier must refuse (e.g. mixed TP with the
direct layout), where pairing would hand the decoder unusable bytes.

Driven by run_accuracy_test.sh via env vars:
    TEST_MODEL, PROXY_URL, DECODE_METRICS_URLS (comma-separated),
    DECODE_BLOCK_SIZE, EXPECT_TRANSFER.
"""

import os
import random
import time
import uuid

import openai
import requests
from prometheus_client.parser import text_string_to_metric_families

MODEL_NAME = os.environ.get("TEST_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:8192/v1")
DECODE_METRICS_URLS = os.environ.get(
    "DECODE_METRICS_URLS", "http://localhost:8200/metrics"
).split(",")
BLOCK_SIZE = int(os.environ.get("DECODE_BLOCK_SIZE", "128"))
NUM_PROMPTS = int(os.environ.get("NUM_TRANSFER_PROMPTS", "8"))
# Fraction of each prompt's full blocks that must come from the external tier.
MIN_HIT_FRACTION = 0.5
EXPECT_TRANSFER = os.environ.get("EXPECT_TRANSFER", "1") == "1"
HITS_METRIC = "vllm:external_prefix_cache_hits"

WORDS = [
    "apple",
    "river",
    "cloud",
    "stone",
    "forest",
    "lamp",
    "ocean",
    "garden",
    "window",
    "silver",
    "bridge",
    "mountain",
    "candle",
    "paper",
    "engine",
    "harbor",
    "violet",
    "meadow",
    "castle",
    "thunder",
]


def _unique_prompt() -> str:
    # The uuid leads the prompt so no two prompts (or gsm8k prompts) share a
    # prefix block that the decoder's own GPU prefix cache could serve.
    tag = uuid.uuid4().hex
    rng = random.Random(tag)
    body = " ".join(rng.choice(WORDS) for _ in range(5 * BLOCK_SIZE))
    return f"{tag} {body}"


def _external_hits() -> float:
    total = 0.0
    for url in DECODE_METRICS_URLS:
        text = requests.get(url, timeout=30).text
        for family in text_string_to_metric_families(text):
            if family.name != HITS_METRIC:
                continue
            for sample in family.samples:
                if sample.name == f"{HITS_METRIC}_total":
                    total += sample.value
    return total


def test_decoder_pulls_kv_over_p2p():
    client = openai.OpenAI(api_key="EMPTY", base_url=PROXY_URL)
    before = _external_hits()

    expected_tokens = 0
    for _ in range(NUM_PROMPTS):
        completion = client.completions.create(
            model=MODEL_NAME,
            prompt=_unique_prompt(),
            max_tokens=4,
            temperature=0.0,
        )
        assert completion.usage is not None
        prompt_tokens = completion.usage.prompt_tokens
        # The last prompt token is always recomputed, so only full blocks
        # before it can be loaded from the external tier.
        expected_tokens += (prompt_tokens - 1) // BLOCK_SIZE * BLOCK_SIZE

    assert expected_tokens > 0, "prompts too short to span a full block"

    if not EXPECT_TRANSFER:
        # Let any late counter update land before asserting nothing moved.
        time.sleep(5)
        delta = _external_hits() - before
        print(f"external prefix cache hits: +{delta:.0f} tokens (expected 0)")
        assert delta == 0, (
            f"decoders loaded {delta:.0f} prompt tokens from the external tier "
            "in a deployment the p2p tier is expected to refuse"
        )
        return

    min_hits = MIN_HIT_FRACTION * expected_tokens

    # Counters are published from scheduler stats; give the last step a moment.
    deadline = time.monotonic() + 15
    delta = _external_hits() - before
    while delta < min_hits and time.monotonic() < deadline:
        time.sleep(1)
        delta = _external_hits() - before

    print(
        f"external prefix cache hits: +{delta:.0f} tokens "
        f"(full-block prompt tokens: {expected_tokens}, need >= {min_hits:.0f})"
    )
    assert delta >= min_hits, (
        f"decoders loaded only {delta:.0f} of {expected_tokens} prompt tokens "
        "from the external tier; KV was recomputed instead of pulled over p2p"
    )
