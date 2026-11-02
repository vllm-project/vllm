# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read the hidden state a token's logits were computed from, in the same request.

A generation request can ask for, per generated token, the hidden state the
model's `compute_logits` turned into that token's logits (for most decoder
models: the last layer's output after the final norm). It comes back in
`CompletionOutput.last_hidden_states`, shaped [num_generated_tokens, hidden_size]
in the model's dtype, from the very forward pass that produced the tokens and
their logprobs.

Typical use: a small head trained on top of the served model (a linear probe,
a classifier over a fixed label set, a calibration layer) that needs the model's
scores and its hidden state for the same input. Without this output the hidden
state takes a second pass through a pooling engine, which runs other kernels
and needs its own copy of the weights.

How it works:
  1. Start the engine with ``--enable-return-last-hidden-states`` (or
     ``LLM(..., enable_return_last_hidden_states=True)``). Without it, requests
     that ask are rejected; with it, requests that do not ask are unchanged.
  2. Set ``SamplingParams.return_last_hidden_states=True`` on the requests that
     need it. The rows arrive with the final output.

Requires Model Runner V2. Not supported with speculative decoding, pipeline
parallelism or context parallelism.

This example scores a fixed label set in one request (the label log-probabilities,
renormalized over the labels) and applies a linear head to the hidden state of
that same forward:
    p = softmax(label_logprobs + h @ A + c)

Usage:
    python examples/generate/last_hidden_states_offline.py
    python examples/generate/last_hidden_states_offline.py --model Qwen/Qwen3-0.6B
"""

import argparse

import torch

from vllm import LLM, SamplingParams

PROMPT = (
    "Review: The battery died after two days and support never answered.\n"
    "Is the review positive or negative? Answer:"
)
LABELS = [" positive", " negative"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    args = parser.parse_args()

    llm = LLM(args.model, enable_return_last_hidden_states=True)
    tokenizer = llm.get_tokenizer()
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in LABELS
    ]

    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprob_token_ids=label_ids,  # score exactly the labels
        return_last_hidden_states=True,
    )
    completion = llm.generate([PROMPT], params)[0].outputs[0]

    # The label distribution, renormalized over the labels.
    label_logprobs = torch.log_softmax(
        torch.tensor([completion.logprobs[0][t].logprob for t in label_ids]), dim=-1
    )
    hidden = completion.last_hidden_states[0].float()  # [hidden_size]
    print(f"label log-probabilities: {dict(zip(LABELS, label_logprobs.tolist()))}")
    print(f"hidden state: {tuple(hidden.shape)}, {completion.last_hidden_states.dtype}")

    # A head trained elsewhere on (label_logprobs, hidden) pairs; random here.
    generator = torch.Generator().manual_seed(0)
    A = 1e-3 * torch.randn(hidden.shape[0], len(LABELS), generator=generator)
    c = torch.zeros(len(LABELS))
    probs = torch.softmax(label_logprobs + hidden @ A + c, dim=-1)
    print(f"head probabilities: {dict(zip(LABELS, probs.tolist()))}")


if __name__ == "__main__":
    main()
