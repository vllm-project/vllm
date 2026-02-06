#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Offline batched inference with a Step-3 style text model that uses the
hybrid attention layer.

This example mirrors ``examples/offline_inference/basic/basic.py`` but is
intended for models whose config is based on ``Step3TextConfig`` and that
enable the hybrid attention path via the ``use_hybrid_step3_attn`` flag.

To use this script, point ``model_name`` at a checkpoint whose Hugging Face
config has::

    {
        "model_type": "step3_text",
        "architectures": ["Step3TextForCausalLM"],
        "use_hybrid_step3_attn": true,
        ...
    }

The vLLM model registry will automatically route such models to
``Step3TextForCausalLM``, which internally wires up ``HybridAttentionLayer``
and ``HybridSSMAdapter`` as defined in:

- ``vllm/model_executor/models/step3_text.py``
- ``vllm/model_executor/layers/hybrid_attn_layer.py``
- ``vllm/model_executor/layers/hybrid_ssm_adapter.py``
"""

from vllm import LLM, SamplingParams

# Replace this with the actual Step-3 hybrid text model you want to run.
# The only requirement is that its HF config matches the expectations
# documented above (model_type ``step3_text`` with hybrid attention enabled).
model_name = "your-org/step3-text-hybrid"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main() -> None:
    # Create an LLM that will use the Step3TextForCausalLM implementation
    # under the hood when the model config has model_type ``step3_text``.
    llm = LLM(model=model_name)

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("\nGenerated Outputs (Step3Text hybrid attention):\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()


