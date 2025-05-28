# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import tempfile

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from vllm import LLM, SamplingParams


def patch_eagle_draft_with_lm_head(target_model_id: str,
                                   draft_model_id: str) -> str:
    # In NxDI, draft model checkpoint must include lm_head weights from target
    # model. For more details see https://awsdocs-neuron.readthedocs-hosted.com
    # /en/latest/libraries/nxd-inference/developer_guides/feature-guide.html
    # #eagle-checkpoint-compatibility
    final_draft_dir = "/tmp/patched_eagle_draft"

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = snapshot_download(repo_id=target_model_id,
                                       local_dir=os.path.join(
                                           tmp_dir, "target"))
        draft_dir = snapshot_download(repo_id=draft_model_id,
                                      local_dir=os.path.join(tmp_dir, "draft"))

        lm_head_key = "lm_head.weight"
        index_path = os.path.join(target_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard_name = index["weight_map"][lm_head_key]
        target_safetensor_path = os.path.join(target_dir, shard_name)

        with safe_open(target_safetensor_path, framework="pt") as f:
            target_lm_head = f.get_tensor(lm_head_key)

        draft_path = os.path.join(draft_dir, "pytorch_model.bin")
        draft_state_dict = torch.load(draft_path, map_location="cpu")
        draft_state_dict[lm_head_key] = target_lm_head.to(torch.float16)
        torch.save(draft_state_dict, draft_path)

        shutil.copytree(draft_dir, final_draft_dir, dirs_exist_ok=True)

    return final_draft_dir


def test_eagle():
    patched_draft_path = patch_eagle_draft_with_lm_head(
        target_model_id="meta-llama/Llama-2-7b-hf",
        draft_model_id="yuhuili/EAGLE-llama2-chat-7B")
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        speculative_config={
            "model": patched_draft_path,
            "num_speculative_tokens": 5,
            "max_model_len": 128
        },
        max_num_seqs=1,
        max_model_len=128,
        tensor_parallel_size=2,
        override_neuron_config={
            "enable_eagle_speculation": True,
            "enable_fused_speculation": True,
            "fused_qkv": True
        },
    )
    prompts = [
        "The president of the United States is",
    ]
    outputs = llm.generate(prompts, SamplingParams(top_k=1))
    expected_output = " the head of state and head of government of " \
    "the United States. The president direct"

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        assert (expected_output == generated_text)

    print("Neuron Eagle speculation test passed.")
