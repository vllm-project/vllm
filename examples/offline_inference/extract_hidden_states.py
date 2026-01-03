from safetensors import safe_open

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

    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    assert hidden_states_path is not None
    print("Prompt hidden states path:", hidden_states_path)

    with safe_open(hidden_states_path, "pt") as f:
        token_ids = f.get_tensor("token_ids")
        hidden_states = f.get_tensor("hidden_states")

        print("Extracted token ids:", token_ids)  # Matches prompt token ids
        print(
            "Extracted hidden states shape:", hidden_states.shape
        )  # [num_hidden_layers, prompt len, hidden size]
        print("Extracted hidden states:", hidden_states)
