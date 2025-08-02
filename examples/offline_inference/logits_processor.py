# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""This example demonstrates instantiating vLLM with a custom logits processor
class object.

For a basic example of implementing a custom logits processor, see
the `DummyLogitsProcessor` implementation in `vllm/test_utils.py`.

For testing purposes, a dummy logits processor is employed which, if
`target_token` is passed as a keyword argument to `SamplingParams.extra_args`,
will mask out all tokens except `target_token`.

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

from vllm import LLM, SamplingParams
from vllm.test_utils import DummyLogitsProcessor

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
