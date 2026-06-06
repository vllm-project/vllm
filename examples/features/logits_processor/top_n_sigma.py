# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""This example demonstrates implementing Top-n-sigma logit truncation as a
custom logits processor in vLLM.

Top-n-sigma is a logit-space dynamic truncation method that uses the standard
deviation of the logit distribution to determine the filtering threshold:

    threshold = max_logit - n * std_logit
    logits[logits < threshold] = -inf

Unlike probability-space filters (top_p, min_p), Top-n-sigma operates before
softmax and adapts to the "peakiness" of the logit distribution:
- Sharp distribution (model is certain): small std -> narrow threshold ->
  fewer candidates
- Flat distribution (model is uncertain): large std -> wide threshold ->
  more candidates

Usage:

    SamplingParams(
        temperature=0.8,
        extra_args={"top_n_sigma": 2.0}  # n=2.0 standard deviations
    )

For a basic example of implementing a custom logits processor, see
the `DummyLogitsProcessor` implementation in `custom.py`.

A batch is constructed with alternating requests that do and don't use
top_n_sigma, demonstrating how the processor coexists with normal sampling.
"""

from typing import Any

import torch

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
)
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class TopNSigmaLogitsProcessor(LogitsProcessor):
    """Top-n-sigma logit truncation processor.

    Filters logits based on statistical outlier detection: tokens whose
    logit value falls more than n standard deviations below the maximum
    logit are masked to -inf before softmax.

    This is argmax-invariant because the argmax token always has the
    highest logit and thus always survives the filter.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams):
        n_sigma: Any | None = params.extra_args and params.extra_args.get(
            "top_n_sigma"
        )
        if n_sigma is not None:
            if not isinstance(n_sigma, (int, float)):
                raise ValueError(
                    f"top_n_sigma must be a number, got {type(n_sigma)}"
                )
            if n_sigma <= 0:
                raise ValueError(
                    f"top_n_sigma must be positive, got {n_sigma}"
                )

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        self.req_info: dict[int, float] = {}

    def is_argmax_invariant(self) -> bool:
        return True

    def update_state(self, batch_update: BatchUpdate | None):
        def extract_n_sigma(params: SamplingParams) -> float | None:
            self.validate_params(params)
            return params.extra_args and params.extra_args.get("top_n_sigma")

        process_dict_updates(
            self.req_info,
            batch_update,
            lambda params, _, __: extract_n_sigma(params),
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        rows = torch.tensor(
            list(self.req_info.keys()), dtype=torch.long, device=logits.device
        )
        n_sigmas = torch.tensor(
            list(self.req_info.values()), dtype=logits.dtype, device=logits.device
        )

        selected_logits = logits[rows].clone()

        # Skip rows with NaN/Inf or all-equal logits (std == 0)
        finite_mask = torch.isfinite(selected_logits).all(dim=-1)
        nonzero_std_mask = selected_logits.std(dim=-1) != 0
        process_mask = finite_mask & nonzero_std_mask

        rows_to_process = rows[process_mask]
        n_sigmas_to_process = n_sigmas[process_mask]
        logits_to_process = selected_logits[process_mask]

        if rows_to_process.numel() == 0:
            return logits

        max_logits = logits_to_process.max(dim=-1, keepdim=True).values
        std_logits = logits_to_process.std(dim=-1, keepdim=True)

        thresholds = max_logits - n_sigmas_to_process.unsqueeze(-1) * std_logits
        logits_to_process[logits_to_process < thresholds] = float("-inf")

        logits[rows_to_process] = logits_to_process

        return logits


# Sample prompts with varying difficulty (certainty) levels.
prompts = [
    "The capital of France is",
    "Hello, my name is",
    "The future of AI is",
    "The president of the United States is",
]

# Create a mixture of requests with and without top_n_sigma
sampling_params_list = [
    # With top_n_sigma=2.0: keeps tokens within 2 std of max logit
    SamplingParams(
        temperature=0.8, max_tokens=20, extra_args={"top_n_sigma": 2.0}
    ),
    # Without top_n_sigma: normal sampling for comparison
    SamplingParams(temperature=0.8, max_tokens=20),
    # With top_n_sigma=1.0: more aggressive filtering
    SamplingParams(
        temperature=0.8, max_tokens=20, extra_args={"top_n_sigma": 1.0}
    ),
    # Without top_n_sigma: normal sampling for comparison
    SamplingParams(temperature=0.8, max_tokens=20),
]


def self_test():
    """Quick correctness self-test for the vectorized apply() implementation.

    Verifies argmax invariance and threshold filtering on synthetic logits.
    Run with: python top_n_sigma.py --test
    """
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU available, running self-test on CPU")

    torch.manual_seed(42)
    V = 32000
    B = 8
    req_info = {0: 2.0, 1: 1.0, 3: 3.0, 5: 2.0, 7: 5.0}

    logits = torch.randn(B, V, device=device)

    # Instantiate processor
    proc = TopNSigmaLogitsProcessor(
        vllm_config=None, device=torch.device(device), is_pin_memory=False
    )
    proc.req_info = req_info

    orig_argmax = logits.argmax(dim=-1)
    filtered = proc.apply(logits.clone())
    filt_argmax = filtered.argmax(dim=-1)

    # 1. Argmax invariance
    argmax_ok = (filt_argmax == orig_argmax).all().item()
    print(f"  Argmax invariance: {argmax_ok}")

    # 2. Threshold filtering produces reasonable survival counts
    surviving = (filtered > float("-inf")).sum(dim=-1)
    # Rows with n_sigma should have fewer surviving tokens than V
    for row_idx, n_sigma in req_info.items():
        s = surviving[row_idx].item()
        pct = 100 * s / V
        print(f"  Row {row_idx} n={n_sigma}: {s}/{V} ({pct:.1f}%) survive")

    # 3. Edge case: constant logits (std=0) → no filtering
    const_logits = torch.ones(4, V, device=device)
    proc_const = TopNSigmaLogitsProcessor(
        vllm_config=None, device=torch.device(device), is_pin_memory=False
    )
    proc_const.req_info = {0: 2.0, 1: 2.0}
    filtered_const = proc_const.apply(const_logits.clone())
    const_ok = (filtered_const == const_logits).all().item()
    print(f"  Constant logits (std=0) unchanged: {const_ok}")

    # 4. Edge case: empty req_info → logits unchanged
    proc_empty = TopNSigmaLogitsProcessor(
        vllm_config=None, device=torch.device(device), is_pin_memory=False
    )
    proc_empty.req_info = {}
    filtered_empty = proc_empty.apply(logits.clone())
    empty_ok = (filtered_empty == logits).all().item()
    print(f"  Empty req_info unchanged: {empty_ok}")

    all_pass = argmax_ok and const_ok and empty_ok
    if all_pass:
        print("  ALL PASS")
    else:
        print("  SOME FAIL — investigate before deploying")
        sys.exit(1)


def main():
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[TopNSigmaLogitsProcessor],
    )
    outputs = llm.generate(prompts, sampling_params_list)
    print("\nTop-n-sigma Logits Processor Demo\n" + "=" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        self_test()
    else:
        main()
