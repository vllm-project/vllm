"""Qwen2.5-0.5B reranker-style target-token-scoring demo.

This demo mirrors the relevance-scoring pattern from rtp-llm's
``Qwen3RerankerHandler`` (which projects the last hidden state through the full
``[V, H]`` LM Head and then slices two ``yes``/``no`` columns), but uses the
compact target-token-scoring path: the two candidate weight rows are
``index_select``-ed first, so the full-vocab MatMul never runs.

It runs two checks:

1. **Equivalence**: ``compact_logits[:, j] == full_vocab_logits[:, target_ids[j]]``
   -- proves only irrelevant columns were dropped, not the candidate math.
2. **Caller aggregation**: shows how a caller turns the ordered target-set
   logprobs into a relevance score outside the runtime (the runtime carries
   only candidate values; it never encodes a score formula).

It also includes a NumPy-only fallback path for environments without torch,
matching the case doc's host-math variant.

For a real Qwen2.5-0.5B model served by vLLM, enable the engine flag and opt
each scoring request into target-set semantics via ``logprob_token_ids`` plus
``target_token_scoring_normalization="target_set"``:

.. code-block:: python

    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-0.5B", target_token_scoring=True)
    # yes/no token ids from the Qwen2.5 tokenizer
    sp = SamplingParams(
        temperature=0, max_tokens=1, logprob_token_ids=[yes_id, no_id],
        target_token_scoring_normalization="target_set",
    )
    outputs = llm.generate(prompts, sp)
"""

from __future__ import annotations

import numpy as np

TARGET_TOKEN_IDS = [408, 578]  # fictional yes/no ids; must be < vocab below


def _torch_path() -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    batch, hidden, vocab = 4, 16, 1000
    weight = torch.randn(vocab, hidden)
    bias = torch.randn(vocab)
    hidden_states = torch.randn(batch, hidden)
    target_ids = torch.as_tensor(TARGET_TOKEN_IDS, dtype=torch.long)

    full_logits = F.linear(hidden_states, weight, bias)  # [B, V]
    expected = full_logits[:, target_ids]  # the two columns the caller wants

    sel_w = weight.index_select(0, target_ids).contiguous()  # [K, H]
    sel_b = bias.index_select(0, target_ids)
    compact_logits = F.linear(hidden_states, sel_w, sel_b)  # [B, K]

    assert torch.allclose(compact_logits, expected, atol=1e-5), (
        "compact logits must match the full-vocab candidate columns"
    )
    print("[torch] equivalence OK: compact[:, j] == full[:, target_ids[j]]")

    # Stable target-set log-softmax (the runtime does this on-device in FP32).
    f = compact_logits.float()
    m = f.amax(dim=-1, keepdim=True)
    shifted = f - m
    logprobs = shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)
    probs = logprobs.exp()

    # Caller-side aggregation: pick the "yes" column as the relevance score.
    yes_index = 0  # TARGET_TOKEN_IDS[0] is the "yes" id in this fiction
    scores = probs[:, yes_index]
    print(f"[torch] target-set probs:\n{probs}")
    print(f"[torch] relevance scores (yes prob): {scores.tolist()}")


def _numpy_path() -> None:
    rng = np.random.default_rng(0)
    batch, hidden, vocab = 4, 16, 1000
    weight = rng.standard_normal((vocab, hidden))
    bias = rng.standard_normal(vocab)
    hidden_states = rng.standard_normal((batch, hidden))
    target_ids = np.asarray(TARGET_TOKEN_IDS, dtype=np.int64)

    full_logits = hidden_states @ weight.T + bias  # [B, V]
    expected = full_logits[:, target_ids]

    sel_w = weight[target_ids]  # [K, H]
    sel_b = bias[target_ids]
    compact_logits = hidden_states @ sel_w.T + sel_b  # [B, K]
    assert np.allclose(compact_logits, expected, atol=1e-5)
    print("[numpy] equivalence OK: compact[:, j] == full[:, target_ids[j]]")

    row_max = compact_logits.max(axis=1, keepdims=True)
    shifted = compact_logits - row_max
    logprobs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(logprobs)
    print(f"[numpy] target-set probs:\n{probs}")
    print(f"[numpy] relevance scores (yes prob): {probs[:, 0].tolist()}")


def main() -> None:
    try:
        _torch_path()
    except ImportError:
        print("torch not available; running NumPy-only fallback")
        _numpy_path()
        return
    # Also run the NumPy path to show the host-math variant agrees.
    _numpy_path()


if __name__ == "__main__":
    main()
