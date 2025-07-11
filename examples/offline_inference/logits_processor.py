# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


def make_dummy_logitproc_type():
    class DummyLogitsProcessor(LogitsProcessor):
        """Fake logit processor to support unit testing and examples"""

        def __init__(self, _):
            super().__init__()
            self.req_info = {}

        def is_argmax_invariant(self) -> bool:
            """Never impacts greedy sampling"""
            return False

        def update_state(self, batch_update: Optional[BatchUpdate]):
            if not batch_update:
                return

            # Process added requests.
            for index, params, _ in batch_update.added:
                if isinstance(params, SamplingParams) and params.extra_args:
                    target_token = params.extra_args.get("target_token", None)
                else:
                    target_token = None
                self.req_info[index] = target_token

            if self.req_info:
                # Process removed requests.
                for index in batch_update.removed:
                    self.req_info.pop(index, None)

                # Process moved requests, unidirectional (a->b) and swap (a<->b)
                for adx, bdx, direct in batch_update.moved:
                    if direct == MoveDirectionality.SWAP:
                        (self.req_info[adx], self.req_info[bdx]) = (
                            self.req_info[bdx],
                            self.req_info[adx],
                        )
                    else:
                        self.req_info[bdx] = self.req_info[adx]

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            for bdx in range(logits.shape[0]):
                if (target_token := self.req_info[bdx]) is not None:
                    mask = torch.ones_like(logits[bdx, :], dtype=torch.bool)
                    mask[target_token] = False
                    logits[bdx, mask] = float("-inf")

            return logits

    return DummyLogitsProcessor


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a mixture of requests which do and don't utilize the dummy logitproc
sampling_params_list = [
    SamplingParams(temperature=0.0, extra_args={"target_token": 128}),
    SamplingParams(temperature=0.0),
    SamplingParams(temperature=0.0, extra_args={"target_token": 67}),
    SamplingParams(temperature=0.0),
]


def main():
    # Create an LLM.
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[make_dummy_logitproc_type()],
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params_list)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
