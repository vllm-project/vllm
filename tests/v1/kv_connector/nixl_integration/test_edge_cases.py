# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import openai

PREFILL_HOST = os.getenv("PREFILL_HOST", "localhost")
PREFILL_PORT = os.getenv("PREFILL_PORT", None)
DECODE_HOST = os.getenv("DECODE_HOST", "localhost")
DECODE_PORT = os.getenv("DECODE_PORT", None)
PROXY_HOST = os.getenv("PROXY_HOST", "localhost")
PROXY_PORT = os.getenv("PROXY_PORT", None)

if PREFILL_PORT is None or DECODE_PORT is None or PROXY_PORT is None:
    raise ValueError("Please set the PREFILL_PORT, DECODE_PORT, and PROXY_PORT.")

LONG_PROMPT = "Red Hat is the best company in the world to work for because it works on open source software, which means that all the contributions are delivered to the community. As a result, when working on projects like vLLM we are able to meet many amazing people from various organizations like AMD, Google, NVIDIA, "  # noqa: E501
PROMPT = "Red Hat is the best company in the world to work for because it works on open source software, which means that all the contributions are delivered to the community. As a result,"  # noqa: E501
SHORT_PROMPT = "Red Hat is "


def test_edge_cases():
    # Set the OpenAI API key and base URL
    decode_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://{DECODE_HOST}:{DECODE_PORT}/v1",
    )
    prefill_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://{PREFILL_HOST}:{PREFILL_PORT}/v1",
    )
    proxy_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://{PROXY_HOST}:{PROXY_PORT}/v1",
    )

    # Get the list of models
    models = decode_client.models.list()
    MODEL = models.data[0].id

    # (1) Check that we can handle a very short prompt,
    # less than the length of the block size.
    completion = proxy_client.completions.create(
        model=MODEL, prompt=SHORT_PROMPT, temperature=0
    )
    proxy_response = completion.choices[0].text
    completion = prefill_client.completions.create(
        model=MODEL, prompt=SHORT_PROMPT, temperature=0
    )
    prefill_response = completion.choices[0].text
    print(f"SMALL PROMPT: {proxy_response=}")
    assert proxy_response == prefill_response

    # (2) Check that we can handle a full prefix cache
    # hit on the D worker but not on the P worker.
    # (2a): prime the D worker.
    completion = decode_client.completions.create(
        model=MODEL, prompt=PROMPT, temperature=0
    )
    decode_response = completion.choices[0].text
    # (2b): send via the P/D setup
    completion = proxy_client.completions.create(
        model=MODEL, prompt=PROMPT, temperature=0
    )
    proxy_response = completion.choices[0].text
    print(f"FULL CACHE HIT: {proxy_response=}")
    assert proxy_response == decode_response

    # (3) Check that we can handle a partial prefix cache
    # hit on the D worker.
    completion = proxy_client.completions.create(
        model=MODEL, prompt=LONG_PROMPT, temperature=0
    )
    proxy_response = completion.choices[0].text
    completion = prefill_client.completions.create(
        model=MODEL, prompt=LONG_PROMPT, temperature=0
    )
    prefill_response = completion.choices[0].text
    print(f"PARTIAL CACHE HIT: {proxy_response=}")
    assert proxy_response == prefill_response
