# SPDX-License-Identifier: Apache-2.0

import shutil
from os import path
from tempfile import TemporaryDirectory
from typing import List, Tuple

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from vllm.lora.request import LoRARequest

from ..models.utils import check_outputs_equal

ULTRAVOX_MODEL_NAME = "fixie-ai/ultravox-v0_3"
LLMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

VLLM_PLACEHOLDER = "<|reserved_special_token_0|>"

PROMPT = "Tell me about a Fool's mate move in 20 words. Provide the moves!"


def llama3_1_8b_chess_lora_path():
    return snapshot_download(
        repo_id="mkopecki/chess-lora-adapter-llama-3.1-8b")


# can't use llama lora adapter without module name transformation
# because ultravox nest language model
def transform_module_names_for_ultravox(state_dict):
    transformed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("base_model.model",
                              "base_model.model.language_model")
        transformed_state_dict[new_key] = value
    return transformed_state_dict


def mk_llama3_1_8b_ultravox_chess_lora(source_repo, target_path):
    tensor_file = "adapter_model.safetensors"
    state_dict = load_file(path.join(source_repo, tensor_file))
    transformed_state_dict = transform_module_names_for_ultravox(state_dict)

    save_file(transformed_state_dict, path.join(target_path, tensor_file))

    config_file = "adapter_config.json"
    shutil.copyfile(path.join(source_repo, config_file),
                    path.join(target_path, config_file))
    return target_path


def _get_prompt(audio_count, question, placeholder, model_name) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    placeholder = f"{placeholder}\n" * audio_count

    return tokenizer.apply_chat_template([{
        'role': 'user',
        'content': f"{placeholder}{question}"
    }],
                                         tokenize=False,
                                         add_generation_prompt=True)


def test_ultravox_lora(vllm_runner):
    """
    TODO: Train an Ultravox LoRA instead of using a Llama LoRA.
    """
    # Workaround to prevent device mismatch in Whisper.
    # Can be removed when it is fixed upstream in transformer
    # https://github.com/huggingface/transformers/pull/35866
    torch.set_default_device("cpu")

    llama3_1_8b_chess_lora = llama3_1_8b_chess_lora_path()
    with TemporaryDirectory() as temp_ultravox_lora_dir:
        llama3_1_8b_ultravox_chess_lora = mk_llama3_1_8b_ultravox_chess_lora(
            llama3_1_8b_chess_lora, temp_ultravox_lora_dir)
        with vllm_runner(
                ULTRAVOX_MODEL_NAME,
                enforce_eager=True,
                max_num_seqs=2,
                enable_lora=True,
                max_loras=1,
                max_lora_rank=128,
                dtype="bfloat16",
                max_model_len=1024,
        ) as vllm_model:
            ultravox_outputs: List[Tuple[
                List[int], str]] = vllm_model.generate_greedy(
                    [
                        _get_prompt(0, PROMPT, VLLM_PLACEHOLDER,
                                    ULTRAVOX_MODEL_NAME)
                    ],
                    256,
                    lora_request=LoRARequest(str(1), 1,
                                             llama3_1_8b_ultravox_chess_lora),
                )

    # run llama with and without lora to compare outputs with above
    with vllm_runner(
            LLMA_MODEL_NAME,
            enforce_eager=True,
            max_num_seqs=2,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=128,
            dtype="bfloat16",
            max_model_len=1024,
    ) as vllm_model:
        llama_outputs: List[Tuple[List[int], str]] = (
            vllm_model.generate_greedy(
                [_get_prompt(0, PROMPT, VLLM_PLACEHOLDER, LLMA_MODEL_NAME)],
                256,
                lora_request=LoRARequest(str(1), 1, llama3_1_8b_chess_lora),
            ))

    check_outputs_equal(
        outputs_0_lst=ultravox_outputs,
        outputs_1_lst=llama_outputs,
        name_0="ultravox",
        name_1="llama",
    )
