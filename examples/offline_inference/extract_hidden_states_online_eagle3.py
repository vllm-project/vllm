# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract hidden states for eagle3 training using OnlineHiddenStatesConnector.

Captures hidden states from specific layers of Qwen3-8B that are needed
to train an Eagle3 drafter model (e.g. AngelSlim/Qwen3-8B_eagle3).
The layer IDs match the eagle3 drafter's expected auxiliary hidden state
layers.  Output is written asynchronously with zstd compression.

This is the data collection step — the captured hidden states and token
IDs are used as training data for distilling an Eagle3 speculative
decoding drafter.

Usage:
    python examples/offline_inference/extract_hidden_states_online_eagle3.py
"""

import os
import tempfile
import time

from safetensors import safe_open

from vllm import LLM, SamplingParams


def load_safetensors(path: str) -> dict:
    """Load tensors from a safetensors file, handling .zst compression."""
    actual_path = path
    if os.path.exists(path + ".zst"):
        import zstandard as zstd

        actual_path = path + ".tmp.safetensors"
        dctx = zstd.ZstdDecompressor()
        with open(path + ".zst", "rb") as f:
            raw = dctx.decompress(f.read())
        with open(actual_path, "wb") as f:
            f.write(raw)

    with safe_open(actual_path, "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


# Layer IDs matching the Eagle3 drafter config for Qwen3-8B.
# These are the target model layers whose hidden states the Eagle3
# drafter uses as auxiliary inputs.
EAGLE3_LAYER_IDS = [1, 2, 3, 4]

with tempfile.TemporaryDirectory() as tmpdirname:
    llm = LLM(
        model="Qwen/Qwen3-8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": EAGLE3_LAYER_IDS,
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "OnlineHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": tmpdirname,
                "use_compression": True,
                "compression_level": 3,
            },
        },
    )

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a quicksort implementation in Python.",
        "What are the main causes of climate change?",
    ]
    sampling_params = SamplingParams(max_tokens=1)
    outputs = llm.generate(prompts, sampling_params)

    # Allow background writer to flush
    time.sleep(2)

    for output in outputs:
        print("\nPrompt:", output.prompt)
        print("Prompt token ids:", output.prompt_token_ids)

        hidden_states_path = output.kv_transfer_params.get(
            "hidden_states_path"
        )
        assert hidden_states_path is not None

        tensors = load_safetensors(hidden_states_path)
        hs = tensors["hidden_states"]
        print(f"Hidden states shape: {hs.shape}")
        # Shape: [prompt_len, num_eagle3_layers, hidden_size]
        print(f"  prompt_len={hs.shape[0]}, "
              f"num_layers={hs.shape[1]}, "
              f"hidden_size={hs.shape[2]}")
