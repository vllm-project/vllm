# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams


def test_v1_chunked_prefill():
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    llm = LLM(
        model=model_path,
        max_num_seqs=8,
        max_model_len=512,
        max_num_batched_tokens=128,  # chunk size
        block_size=32,
        tensor_parallel_size=2,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        override_neuron_config={
            "is_block_kv_layout": True,
            "sequence_parallel_enabled": True,
            "chunked_prefill_config": {
                "max_num_seqs": 8,
                "kernel_q_tile_size": 128,
                "kernel_kv_tile_size": 4096,
            },
            "skip_warmup": True,
            "save_sharded_checkpoint": True,
            "logical_nc_config": 1,
        },
    )

    prompts = [
        "The president of the United States is",
        "The capital of France is",
        ("It is not the critic who counts; not the man who points out how the "
         "strong man stumbles, or where the doer of deeds could have done them "
         "better. The credit belongs to the man who is actually in the arena, "
         "whose face is marred by dust and sweat and blood; who strives "
         "valiantly; who errs, who comes short again and again, because there "
         "is no effort without error and shortcoming; but who does actually "
         "strive to do the deeds; who knows great enthusiasms, the great "
         "devotions; who spends himself in a worthy cause; who at the best "
         "knows"),
        ("Do not go gentle into that good night, Old age should burn and rave "
         "at close of day; Rage, rage against the dying of the light. Though "
         "wise men at their end know dark is right, Because their words had "
         "forked no lightning they Do not go gentle into that good night. Good "
         "men, the last wave by, crying how bright Their frail deeds might have"
         " danced in a green bay, Rage, rage against the dying of the light. "
         "Wild men who caught and sang the sun in flight, And learn, too late, "
         "they grieved it on its way, Do not go gentle into that good night. "
         "Grave men, near death, who see with blinding sight Blind eyes could "
         "blaze like meteors and be gay, Rage, rage against the dying of the "
         "light. And you, my father, there on the sad height, Curse, bless, me "
         "now with your fierce tears, I pray. Do not go gentle into that good "
         "night. Rage, rage against the dying of the light."),
    ]
    sampling_params = SamplingParams(max_tokens=30, top_k=1)

    outputs = llm.generate(prompts, sampling_params)

    expected_outputs = [
        " a man named Donald Trump.\n\n2. B. The president of the United States"
        " is a man named Donald Trump.\n\n3. C",
        " Paris.\n\n2. B. The capital of France is Paris.\n\n3. C. The capital"
        " of France is Paris.\n\n",
        "ends the triumph of high achievement, and at worst, if he fails, at "
        "least he fails while daring greatly, so that his place shall",
        " Rage, rage against the dying of the light. Rage, rage against the "
        "dying of the light. Rage, rage against"
    ]

    for expected_output, output in zip(expected_outputs, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        assert (expected_output == generated_text)

    print("Neuron V1 chunked prefill test passed.")
