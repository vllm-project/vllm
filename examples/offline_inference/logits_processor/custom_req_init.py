# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""This example demonstrates a special case of wrapping a request-level logits
processor, namely the case where it is necessary to utilize engine config or
environment info passed to the constructor. The subclass must override the
wrapper base class `__init__()` method to access the engine config, the device
identifier, or the flag which indicates whether pinned memory is available.

For demo purposes, a request-level dummy logits processor is employed which
causes the same token (`target_token`) to be decoded in each step. The
request-level dummy logits processor is wrapped to create a batch-level logits
processor, which can apply the logits processor to output logits from all
requests in the persistent batch in a given decode step.

The wrapped dummy logits processor below models a scenario where we must
disable the logits processor on non-"cuda" platforms. The wrapper base class
`__init__()` is overridden in order to check this condition and set a flag.

A batch is constructed with `temperature=0.0` and 50% of requests specifying
`target_token`, and for these requests - and *only* these requests - we
expect that on a "cuda" device the output will look something like:

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

which indicates that the logits processor is running. However, on a non-"cuda"
device, the first and third requests would not repeat the same token.
"""

from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

logger = init_logger(__name__)


class DummyPerReqLogitsProcessor:
    """The request-level logits processor masks out all logits except the
    token id identified by `target_token`"""

    def __init__(self, target_token: int) -> None:
        """Specify `target_token`"""
        self.target_token = target_token

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        val_to_keep = logits[self.target_token].item()
        logits[:] = float("-inf")
        logits[self.target_token] = val_to_keep
        return logits


class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Example of overriding the wrapper class `__init__()` in order to utilize
    info about the device type"""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self.is_cuda = device.type == "cuda"

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """This method returns a new request-level logits processor, customized
        to the `target_token` value associated with a particular request.

        Returns None if the logits processor should not be applied to the
        particular request. To use the logits processor the request must have
        a "target_token" custom argument with an integer value, and the device
        must be "cuda"-type

        Args:
          params: per-request sampling params

        Returns:
          `Callable` request logits processor, or None
        """
        if (
            not self.is_cuda
            or (
                target_token := params.extra_args
                and params.extra_args.get("target_token")
            )
            is None
        ):
            return None
        if not isinstance(target_token, int):
            logger.warning(
                "target_token value %s is not int; not applying logits"
                " processor to request.",
                target_token,
            )
            return None
        return DummyPerReqLogitsProcessor(target_token)


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
        logits_processors=[WrappedPerReqLogitsProcessor],
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
