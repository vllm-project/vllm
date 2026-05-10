# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for issue #42019:
`prompt_logprobs` must not depend on request order when prefix caching is on.

Root cause: LogprobsTensors.empty_cpu() uses torch.empty (uninitialized memory).
When prefix cache hits N tokens, positions [0:N] are never written, leaving
stale memory from prior requests. Fix: zero out cached positions on creation.
"""
import contextlib
import gc

import pytest
import torch

from vllm import LLM, SamplingParams

PROMPTS = [
    [1, 10408, 15, 3312, 16315, 7519, 47932, 247, 16204, 275,
     4255, 20098, 19083, 15, 2064, 6505, 347, 11853, 665, 1978,
     368, 1977, 432, 4076, 8737, 13, 12868, 342, 326, 28148, 2929, 13],
    [2, 187, 6759, 16, 681, 16, 12929, 316, 14, 88, 1087, 16, 73,
     1976, 16, 73, 15, 5581, 2, 1387, 4311, 187, 7330, 14, 9150,
     9283, 608, 14, 2420, 187, 7330, 14],
    [3, 16440, 323, 368, 24174, 634, 12108, 13, 38857, 17087, 294,
     4399, 19083, 15, 2064, 368, 971, 11853, 14565, 1978, 368, 1977,
     432, 634, 9781, 13, 403, 1469, 281, 320, 8261, 13],
]


def _score(llm, order):
    params = SamplingParams(
        n=1, max_tokens=1, temperature=0.0,
        prompt_logprobs=0, detokenize=False,
    )
    outputs = llm.generate([PROMPTS[i] for i in order], params)
    by_first = {}
    for ro in outputs:
        vals = [0.0]
        for lp_dict, tok_id in zip(
            ro.prompt_logprobs[1:], ro.prompt_token_ids[1:]
        ):
            vals.append(float(lp_dict[tok_id].logprob))
        by_first[int(ro.prompt_token_ids[0])] = torch.tensor(vals)
    return by_first


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_prompt_logprobs_order_independent(enable_prefix_caching):
    """prompt_logprobs must be identical regardless of request order."""
    llm = LLM(
        "EleutherAI/pythia-14m",
        dtype="float32",
        gpu_memory_utilization=0.5,
        enable_prefix_caching=enable_prefix_caching,
    )
    try:
        ref = _score(llm, (0, 1, 2))
        shuffled = _score(llm, (2, 0, 1))

        for first_token in sorted(ref):
            diff = (shuffled[first_token] - ref[first_token]).abs()
            assert diff.max().item() == 0.0, (
                f"prompt starting with token {first_token}: "
                f"max diff={diff.max().item():.6f} at pos={diff.argmax().item()} "
                f"(enable_prefix_caching={enable_prefix_caching})"
            )
    finally:
        with contextlib.suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
