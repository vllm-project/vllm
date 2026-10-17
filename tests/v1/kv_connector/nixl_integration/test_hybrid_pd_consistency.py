# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-only vs P->D consistency for hybrid (Mamba + attention) models.

For hybrid models the prefiller stops one token short and the decoder
recomputes it with the decode kernel (``_prefill_backoff``), so P->D output is
not bit-identical to a single instance. Exact-match assertions (as in
``test_edge_cases.py``) are therefore wrong here; instead the top-1 logprobs of
the first generated tokens must stay close. Corrupted or missing transferred
state (KDA recurrent/conv state, MLA latent, kpool tail) moves them by several
nats, while the kernel-path difference is a few hundredths.
"""

import os

import openai

PREFILL_PORT = os.environ["PREFILL_PORT"]
DECODE_PORT = os.environ["DECODE_PORT"]
PROXY_PORT = os.environ["PROXY_PORT"]
HOST = os.getenv("PD_HOST", "localhost")

# Loose enough for the decode-vs-prefill kernel difference (observed <0.25 nats
# on a healthy P->D path), tight enough that a missing or stale transferred
# state (KDA recurrent/conv, MLA latent, kpool tail) is still flagged.
MAX_LOGPROB_DELTA = 1.5
NUM_TOKENS = 8
# ~6500 tokens: longer than index_topk (2048) and one 4352-token auto block.
LONG_PROMPT_WORDS = 2000


def _client(port: str) -> openai.OpenAI:
    return openai.OpenAI(api_key="MY_KEY", base_url=f"http://{HOST}:{port}/v1")


def _logprobs(client: openai.OpenAI, model: str, prompt: str) -> list[float]:
    choice = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=NUM_TOKENS,
        logprobs=1,
        seed=0,
    ).choices[0]
    return choice.logprobs.token_logprobs


def test_pd_logprobs_match_single_instance():
    prefill, decode, proxy = (
        _client(PREFILL_PORT),
        _client(DECODE_PORT),
        _client(PROXY_PORT),
    )
    model = decode.models.list().data[0].id
    prompts = {
        "short": "Red Hat is ",
        "mid": "The best part about working on vLLM is that I got to meet "
        "so many people across various different organizations like",
        "long": " ".join(f"word{i}" for i in range(LONG_PROMPT_WORDS)),
    }
    for name, prompt in prompts.items():
        lp_p = _logprobs(prefill, model, prompt)
        lp_d = _logprobs(decode, model, prompt)
        lp_x = _logprobs(proxy, model, prompt)
        delta_pd_single = max(abs(a - b) for a, b in zip(lp_p, lp_d))
        delta_proxy = max(abs(a - b) for a, b in zip(lp_p[:4], lp_x[:4]))
        print(
            f"[{name}] P-vs-D single max|dlogprob|={delta_pd_single:.3f} "
            f"P-vs-proxy first-4 max|dlogprob|={delta_proxy:.3f}"
        )
        assert len(lp_x) == NUM_TOKENS
        assert delta_proxy < MAX_LOGPROB_DELTA, (
            f"{name}: P->D logprobs drifted by {delta_proxy:.3f}"
        )
