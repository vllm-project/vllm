# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams


def test_mistral():
    llm = LLM(model="mistralai/Mistral-7B-v0.1",
              tensor_parallel_size=2,
              max_num_seqs=4,
              max_model_len=128,
              use_v2_block_manager=True,
              override_neuron_config={
                  "sequence_parallel_enabled": False,
                  "skip_warmup": True
              })

    # Send more prompts than the compiled batch size (4) and request
    # varying generation lengths to test accuracy related to Neuron
    # specific sequence id sorting.
    prompts = [
        "The president of the United States is",
        "The capital of France is",
        "What is Annapurna labs?",
        "I believe the meaning of life is",
        "Tell me a story about a brave knight",
        "Hello, my name is Llama",
    ]

    sampling_params = [
        SamplingParams(top_k=1, max_tokens=10),
        SamplingParams(top_k=1, max_tokens=20),
        SamplingParams(top_k=1, max_tokens=30),
        SamplingParams(top_k=1, max_tokens=40),
        SamplingParams(top_k=1, max_tokens=50),
        SamplingParams(top_k=1, max_tokens=60)
    ]

    outputs = llm.generate(prompts, sampling_params)

    expected_outputs = [
        " the most powerful person in the world. He is",
        " a city of many faces. It is a city of history, culture, art, "
        "fashion, and",
        "\n\nAnnapurna Labs is a semiconductor company that was founded "
        "in 2013 by Amazon. The company is",
        " to be happy.\n\nI believe that happiness is a choice.\n\nI "
        "believe that happiness is a state of mind.\n\nI believe that "
        "happiness is a journey.\n\nI believe",
        " who rescued a princess from a dragon.\n\nTell me a story about"
        " a princess who rescued herself from a dragon.\n\nTell me a "
        "story about a princess who rescued herself from a dragon and "
        "then rescued a knight from",
        " and I am a 10 year old male. I am a very friendly and "
        "affectionate boy who loves to be around people. I am a very "
        "active boy who loves to play and run around. I am a very smart "
        "boy who loves to learn new things. I am a very loyal boy"
    ]

    for expected_output, output in zip(expected_outputs, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        assert (expected_output == generated_text)

    print("Neuron Mistral test passed.")
