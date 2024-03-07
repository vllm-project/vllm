import os
from typing import List, Optional
import peft
import pytest
from random import sample
from transformers import AutoModelForCausalLM

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "meta-llama/Llama-2-7b-hf"
PROMPTS = [
    "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This function should describe the target string accurately and the\nfunction must be one of the following ['inform', 'request', 'give_opinion',\n'confirm', 'verify_attribute', 'suggest', 'request_explanation',\n'recommend', 'request_attribute'].\n\nThe attributes must be one of the following:\n['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating',\n'genres', 'player_perspective', 'has_multiplayer', 'platforms',\n'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier'] [/system] [user] Here is the target sentence:\nSpellForce 3 is a pretty bad game. The developer Grimlore Games is clearly a bunch of no-talent hacks, and 2017 was a terrible year for games anyway. [/user] [assistant]",
    "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This function should describe the target string accurately and the\nfunction must be one of the following ['inform', 'request', 'give_opinion',\n'confirm', 'verify_attribute', 'suggest', 'request_explanation',\n'recommend', 'request_attribute'].\n\nThe attributes must be one of the following:\n['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating',\n'genres', 'player_perspective', 'has_multiplayer', 'platforms',\n'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier'] [/system] [user] Here is the target sentence:\nI wanted to like Grimlore Games' 2017 entry, but in SpellForce 3 they just didn't get anything right. [/user] [assistant]",
    "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This function should describe the target string accurately and the\nfunction must be one of the following ['inform', 'request', 'give_opinion',\n'confirm', 'verify_attribute', 'suggest', 'request_explanation',\n'recommend', 'request_attribute'].\n\nThe attributes must be one of the following:\n['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating',\n'genres', 'player_perspective', 'has_multiplayer', 'platforms',\n'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier'] [/system] [user] Here is the target sentence:\nBioShock is a good role-playing, action-adventure, shooter that released for PlayStation, Xbox, and PC in 2007. It is available on Steam, and it has a Mac release but not a Linux release. [/user] [assistant]",
]


def get_lora_model(model_id: str, target_modules: List[str], rank: int):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    lora_config = peft.tuners.lora.LoraConfig(target_modules, rank)
    lora_model = peft.PeftModel(model, lora_config)
    return lora_model


def do_sample(llm,
              lora_path: Optional[str] = None,
              lora_id: Optional[int] = None,
              logprobs: int = 0,
              n_tokens: int = 256):
    prompts = PROMPTS
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=256,
                                          logprobs=logprobs,
                                          stop=["[/assistant]"])
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
    generated_texts = []
    generated_logprobs = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        generated_logprobs.append([
            list(logprob.keys()) for logprob in output.outputs[0].logprobs
        ][:n_tokens])
    return generated_logprobs if logprobs else generated_texts


SUPPORTED_MODULES = [
    "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
    "lm_head"
]
TARGET_MODULES_LIST = []
for length in range(2, 6):
    TARGET_MODULES_LIST.extend(
        [sample(SUPPORTED_MODULES, length) for _ in range(3)])


# Test the functionality when layer and rank are varied.
# Also verify the reference used below is always the same.
@pytest.mark.parametrize("target_modules", TARGET_MODULES_LIST)
@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_layer_variation_verify_reference(target_modules, rank, tmpdir):
    llm = vllm.LLM(MODEL_PATH,
                   enable_lora=True,
                   max_num_seqs=16,
                   max_loras=4,
                   tensor_parallel_size=4,
                   worker_use_ray=True)
    model = get_lora_model(MODEL_PATH, target_modules, rank)
    tmp_dir_lora = os.path.join(tmpdir, "tmp_dir_lora")
    model.save_pretrained(tmp_dir_lora)
    merged_probs = do_sample(llm, tmp_dir_lora, 1, logprobs=5, n_tokens=1)
    reference_id_sets = [set(prob[0]) for prob in merged_probs]
    breakpoint()
    assert reference_id_sets == [{450, 13, 306, 11221, 2266},
                                 {450, 13, 306, 11221, 2266},
                                 {450, 13, 306, 11221, 2266}]


# Test the correctness when layer and rank are varied
@pytest.mark.parametrize("target_modules", TARGET_MODULES_LIST)
@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_layer_variation_correctness(target_modules, rank, tmp_path):
    reference_id_sets = [{450, 13, 306, 11221, 2266},
                         {450, 13, 306, 11221, 2266},
                         {450, 13, 306, 11221, 2266}]

    model = get_lora_model(MODEL_PATH, target_modules, rank)
    tmp_dir_merged = os.path.join(tmp_path, "tmp_dir_merged")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(tmp_dir_merged)

    llm = vllm.LLM(tmp_dir_merged,
                   tokenizer=MODEL_PATH,
                   enable_lora=False,
                   max_num_seqs=16,
                   tensor_parallel_size=4,
                   worker_use_ray=True)
    probs = do_sample(llm, logprobs=5, n_tokens=1)
    # for the first token, verify the top-5 tokens are identical
    id_sets = [set(prob[0]) for prob in probs]
    assert id_sets == reference_id_sets
