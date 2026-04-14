# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import tempfile

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)

# Example: Using the custom "extract_hidden_states" speculator method and
# ExampleHiddenStatesConnector to extract and save hidden states from vllm

with tempfile.TemporaryDirectory() as tmpdirname:
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
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": tmpdirname,
            },
        ),
    )

    prompts = ["Generate a sentence with hidden states", "Write a python function"]
    sampling_params = SamplingParams(max_tokens=1)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print("\nPrompt:", output.prompt)
        print("Prompt token ids:", output.prompt_token_ids)

        hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
        assert hidden_states_path is not None
        print("Prompt hidden states path:", hidden_states_path)

        obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
        token_ids = obj["token_ids"]
        hidden_states = obj["hidden_states"]

        print("Extracted token ids:", token_ids)  # Matches prompt token ids
        print(
            "Extracted hidden states shape:", hidden_states.shape
        )  # [prompt_len, num_extracted_layers, hidden_size]
        print("Extracted hidden states:", hidden_states)

        # Clean up hidden state files
        lock_path = hidden_states_path + ".lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)
        if os.path.exists(hidden_states_path):
            os.remove(hidden_states_path)
