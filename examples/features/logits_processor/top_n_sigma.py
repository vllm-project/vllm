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
        n_sigma = (params.extra_args or {}).get("top_n_sigma")
        if n_sigma is not None and (
            not isinstance(n_sigma, (int, float)) or n_sigma <= 0
        ):
            raise ValueError(
                f"top_n_sigma must be a positive number, got {n_sigma!r}"
            )

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.req_info: dict[int, float] = {}
        self._cached_rows_cpu: torch.Tensor | None = None
        self._cached_n_sigmas_cpu: torch.Tensor | None = None

    def is_argmax_invariant(self) -> bool:
        return True

    def update_state(self, batch_update: BatchUpdate | None):
        def extract_n_sigma(params: SamplingParams) -> float | None:
            self.validate_params(params)
            return (params.extra_args or {}).get("top_n_sigma")

        needs_update = process_dict_updates(
            self.req_info,
            batch_update,
            lambda params, _, __: extract_n_sigma(params),
        )

        # Only rebuild CPU caches when dict actually changed.
        if needs_update:
            if self.req_info:
                self._cached_rows_cpu = torch.tensor(
                    list(self.req_info.keys()), dtype=torch.long,
                    pin_memory=self.is_pin_memory,
                )
                self._cached_n_sigmas_cpu = torch.tensor(
                    list(self.req_info.values()), dtype=torch.float32,
                    pin_memory=self.is_pin_memory,
                )
            else:
                self._cached_rows_cpu = None
                self._cached_n_sigmas_cpu = None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._cached_rows_cpu is None:
            return logits

        rows = self._cached_rows_cpu.to(
            device=logits.device, non_blocking=True,
        )
        n_sigmas = self._cached_n_sigmas_cpu.to(
            device=logits.device, dtype=logits.dtype, non_blocking=True,
        )

        selected_logits = logits[rows]

        # Skip rows with NaN/Inf or all-equal logits (std == 0)
        finite_mask = torch.isfinite(selected_logits).all(dim=-1)
        std_all = selected_logits.std(dim=-1)
        nonzero_std_mask = std_all != 0
        process_mask = finite_mask & nonzero_std_mask

        logits_to_process = selected_logits[process_mask]

        if logits_to_process.numel() == 0:
            return logits

        rows_to_process = rows[process_mask]
        n_sigmas_to_process = n_sigmas[process_mask]
        max_logits = logits_to_process.max(dim=-1, keepdim=True).values
        std_logits = std_all[process_mask].unsqueeze(-1)

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
    SamplingParams(temperature=1.0, max_tokens=20),
    # With top_n_sigma=1.0: more aggressive filtering
    SamplingParams(
        temperature=0.8, max_tokens=20, extra_args={"top_n_sigma": 1.0}
    ),
    # Without top_n_sigma: normal sampling for comparison
    SamplingParams(temperature=0.8, max_tokens=20),
]


def main():
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[TopNSigmaLogitsProcessor],
    )
    # Ordered by config: [s0_p0, s0_p1, ..., s1_p0, s1_p1, ...]
    all_params = [s for s in sampling_params_list for _ in prompts]
    all_prompts = prompts * len(sampling_params_list)

    outputs = llm.generate(all_prompts, all_params)
    config_labels = [
        "top_n_sigma=2.0 (temp=0.8)",
        "baseline temp=1.0 (no filter)",
        "top_n_sigma=1.0 (temp=0.8)",
        "baseline temp=0.8 (no filter)",
    ]
    n_prompts = len(prompts)

    print("\nTop-n-sigma Logits Processor Demo\n" + "=" * 60)
    for cfg_idx, label in enumerate(config_labels):
        print(f"\n[{label}]")
        print("-" * 60)
        for p_idx in range(n_prompts):
            out = outputs[cfg_idx * n_prompts + p_idx]
            print(f"Prompt:  {out.prompt!r}")
            print(f"Output:  {out.outputs[0].text!r}")
            print()


if __name__ == "__main__":
    main()
