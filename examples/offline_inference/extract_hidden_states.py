from vllm import LLM, SamplingParams

# Example: Using the custom "extract_hidden_states" speculator method
# This method uses the ExampleHiddenStatesConnector to extract and save hidden states

llm = LLM(
    model="Qwen/Qwen3-8B",  # Your target model
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [  # Target model layer indices
                    1,
                    2,
                    3,
                    4,
                ],
            }
        },
    },
    enforce_eager=True,
    kv_transfer_config={
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": "/tmp/hidden_states",
        },
    },
)

prompts = ["Generate a sentence with hidden states"]
sampling_params = SamplingParams(max_tokens=1)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print("\nPrompt:", output.prompt)
    print("Prompt token ids:", output.prompt_token_ids)
    print(
        "Prompt hidden states path:",
        output.kv_transfer_params.get("hidden_states_path"),
    )
