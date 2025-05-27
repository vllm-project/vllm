# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to run offline inference with an EAGLE speculative
decoding model on neuron. To use EAGLE speculative decoding, you must use
a draft model that is specifically fine-tuned for EAGLE speculation.
Additionally, to use EAGLE with NxD Inference, the draft model must include
the LM head weights from the target model. These weights are shared between
the draft and target model.
"""

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "What is annapurna labs?",
]


def main():
    # Create a sampling params object.
    sampling_params = SamplingParams(top_k=1, max_tokens=500, ignore_eos=True)

    # Create an LLM.
    llm = LLM(
        model="/home/ubuntu/model_hf/Meta-Llama-3.1-70B-Instruct",
        speculative_config={
            "model": "/home/ubuntu/model_hf/Llama-3.1-70B-Instruct-EAGLE-Draft",
            "num_speculative_tokens": 5,
            "max_model_len": 2048,
        },
        max_num_seqs=4,
        # The max_model_len and block_size arguments are required to be same as
        # max sequence length when targeting neuron device.
        # Currently, this is a known limitation in continuous batching support
        # in neuronx-distributed-inference.
        max_model_len=2048,
        block_size=2048,
        # The device can be automatically detected when AWS Neuron SDK is installed.
        # The device argument can be either unspecified for automated detection,
        # or explicitly assigned.
        device="neuron",
        tensor_parallel_size=32,
        override_neuron_config={
            "enable_eagle_speculation": True,
            "enable_fused_speculation": True,
        },
    )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, \n\n\n\ Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
