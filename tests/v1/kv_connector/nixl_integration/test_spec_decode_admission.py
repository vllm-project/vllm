# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the async KV load admission margin with spec decode.

The decode server is started with spec decode enabled and
`--num-gpu-blocks-override 5` (one null block + 4 usable blocks of
`BLOCK_SIZE` tokens). A remote-prefilled prompt that fills exactly those 4
blocks and whose spec decode step still fits in its last block must be
admitted; before the fix the scheduler reserved a flat extra block for the
spec step and retried the request forever (no running requests to free any).
"""

import os

import requests

PREFILL_HOST = os.getenv("PREFILL_HOST", "localhost")
PREFILL_PORT = os.getenv("PREFILL_PORT", None)
DECODE_HOST = os.getenv("DECODE_HOST", "localhost")
DECODE_PORT = os.getenv("DECODE_PORT", None)

if PREFILL_PORT is None or DECODE_PORT is None:
    raise ValueError("Please set the PREFILL_PORT and DECODE_PORT.")

# Must match the decode server args in run_edge_case_test.sh.
BLOCK_SIZE = 32
NUM_USABLE_BLOCKS = 4
NUM_SPEC_TOKENS = 1
# Fill all usable blocks, leaving room in the last one for the spec decode
# step (1 bonus token + NUM_SPEC_TOKENS draft tokens).
PROMPT_LEN = NUM_USABLE_BLOCKS * BLOCK_SIZE - (1 + NUM_SPEC_TOKENS)
DECODE_TIMEOUT_S = 60


def _completions_url(host: str, port: str) -> str:
    return f"http://{host}:{port}/v1/completions"


def test_async_load_admits_request_filling_free_blocks():
    model = requests.get(f"http://{DECODE_HOST}:{DECODE_PORT}/v1/models").json()
    model = model["data"][0]["id"]
    prompt = [(1000 + 7 * i) % 30000 for i in range(PROMPT_LEN)]

    prefill = requests.post(
        _completions_url(PREFILL_HOST, PREFILL_PORT),
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        },
        timeout=DECODE_TIMEOUT_S,
    ).json()
    assert prefill["usage"]["prompt_tokens"] == PROMPT_LEN
    kv_transfer_params = prefill["kv_transfer_params"]
    assert kv_transfer_params["remote_block_ids"]

    try:
        decode = requests.post(
            _completions_url(DECODE_HOST, DECODE_PORT),
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 1 + NUM_SPEC_TOKENS,
                "temperature": 0,
                "kv_transfer_params": kv_transfer_params,
            },
            timeout=DECODE_TIMEOUT_S,
        ).json()
    except requests.Timeout:
        raise AssertionError(
            f"decode did not admit the {PROMPT_LEN}-token remote-prefilled "
            f"request within {DECODE_TIMEOUT_S}s"
        ) from None
    assert decode["usage"]["prompt_tokens"] == PROMPT_LEN
    assert decode["usage"]["completion_tokens"] == 1 + NUM_SPEC_TOKENS
