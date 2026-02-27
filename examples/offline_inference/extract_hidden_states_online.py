# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract hidden states using OnlineHiddenStatesConnector.

Uses the "extract_hidden_states" speculator method with the async online
connector to capture hidden states from a target model.  The online
connector writes safetensors files in a background thread using async
GPU→CPU copies, keeping disk I/O off the forward-pass critical path.
Output is optionally compressed with zstd.

Usage:
    python examples/offline_inference/extract_hidden_states_online.py
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


with tempfile.TemporaryDirectory() as tmpdirname:
    llm = LLM(
        model="Qwen/Qwen3-8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4],
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
        "Generate a sentence with hidden states",
        "Write a python function",
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
        print("Extracted token ids:", tensors["token_ids"])
        print("Extracted hidden states shape:", tensors["hidden_states"].shape)
        # Shape: [prompt_len, num_layers, hidden_size]
