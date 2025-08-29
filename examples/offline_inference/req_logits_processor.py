# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""This example demonstrates wrapping a request-level logits processor to be
compatible with vLLM's batch-level logits processing

For testing purposes, a dummy logits processor is employed which, if
`target_token` is passed as a keyword argument to `SamplingParams.extra_args`,
will mask out all tokens except `target_token`. This logits processor can be
applied to a vector of logits associated with a single decode step for a single
request. The logits processor cannot be applied to a request which does not
pass in a `target_token` custom argument.

The request-level dummy logits processor is wrapped to create a batch-level
logits processor, which can apply the logits processor to output logits from
all requests in the persistent batch in a given decode step.

A batch is constructed with `temperature=0.0` and 50% of requests specifying
`target_token`, and for these requests - and *only* these requests - we
expect the `target_token` to be decoded in each step, yielding an output
similar to that shown below:

Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    " ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '"
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    " not a racist. He is a racist.\nHe's a racist because he"
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' also also also also also also also also also also also also also
             also also also'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    ' in the hands of the people.\n\nThe future of AI is in the'
------------------------------------------------------------
"""

from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


def get_req_dummy_logits_processor(
    params: SamplingParams,
) -> Optional[RequestLogitsProcessor]:
    """Fake example of a request-level logits processor implementation.

    This function returns a new request-level logits processor, customized
    to the sampling params associated with a particular request.

    Returns None if the logits processor should not be applied to the
    particular request. To use the logits processor the request must have
    a "target_token" custom argument with an integer value.

    Args:
      params: per-request sampling params

    Returns:
      `Callable` request logits processor, or None
    """
    if not params.extra_args or "target_token" not in params.extra_args:
        return None
    target_token = params.extra_args["target_token"]

    def req_dummy_logits_processor(
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """The request-level logits processor masks out all logits except the
        token id identified by target_token"""
        val_to_keep = logits[target_token]
        logits[:] = float("-inf")
        logits[target_token] = val_to_keep
        return logits

    return req_dummy_logits_processor


class DummyLogitsProcessor(AdapterLogitsProcessor):
    """Example of wrapping a fake request-level logit processor to create a
    batch-level logits processor"""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        return False

    def req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        return get_req_dummy_logits_processor(params)


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
        logits_processors=[DummyLogitsProcessor],
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
