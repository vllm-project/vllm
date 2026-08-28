# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm import LLM, SamplingParams
from vllm.renderers.inputs.preprocess import parse_model_prompt

# Passing a raw string directly to generate() still works, but triggers
# a deprecation warning pointing at Renderer.render_cmpl()/render_chat().
# This example shows the replacement path.
prompt = "Hello, my name is"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(model="facebook/opt-125m")

    # Manually render the prompt instead of passing a raw string,
    # avoiding the raw-prompt deprecation warning.
    parsed_prompt = parse_model_prompt(llm.llm_engine.model_config, prompt)
    tok_params = llm.renderer.default_cmpl_tok_params.with_kwargs()
    (engine_input,) = llm.renderer.render_cmpl([parsed_prompt], tok_params)

    # Generate text from the pre-rendered prompt.
    outputs = llm.generate(engine_input, sampling_params)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()