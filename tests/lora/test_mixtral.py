from typing import List

import pytest
import torch

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int,
              prompts: List[str]) -> List[str]:

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=256)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
    generated_texts: List[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.parametrize("tp_size", [4])
def test_mixtral_lora(mixtral_lora_files, tp_size):
    """Original test, the LoRA model has the common target modules, not all"""
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    prompts = [
        "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This function should describe the target string accurately and the\nfunction must be one of the following ['inform', 'request', 'give_opinion',\n'confirm', 'verify_attribute', 'suggest', 'request_explanation',\n'recommend', 'request_attribute'].\n\nThe attributes must be one of the following:\n['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating',\n'genres', 'player_perspective', 'has_multiplayer', 'platforms',\n'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier'] [/system] [user] Here is the target sentence:\nSpellForce 3 is a pretty bad game. The developer Grimlore Games is clearly a bunch of no-talent hacks, and 2017 was a terrible year for games anyway. [/user] [assistant]",  # noqa: E501
        "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This function should describe the target string accurately and the\nfunction must be one of the following ['inform', 'request', 'give_opinion',\n'confirm', 'verify_attribute', 'suggest', 'request_explanation',\n'recommend', 'request_attribute'].\n\nThe attributes must be one of the following:\n['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating',\n'genres', 'player_perspective', 'has_multiplayer', 'platforms',\n'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier'] [/system] [user] Here is the target sentence:\nI wanted to like Grimlore Games' 2017 entry, but in SpellForce 3 they just didn't get anything right. [/user] [assistant]",  # noqa: E501
        "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This function should describe the target string accurately and the\nfunction must be one of the following ['inform', 'request', 'give_opinion',\n'confirm', 'verify_attribute', 'suggest', 'request_explanation',\n'recommend', 'request_attribute'].\n\nThe attributes must be one of the following:\n['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating',\n'genres', 'player_perspective', 'has_multiplayer', 'platforms',\n'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier'] [/system] [user] Here is the target sentence:\nBioShock is a good role-playing, action-adventure, shooter that released for PlayStation, Xbox, and PC in 2007. It is available on Steam, and it has a Mac release but not a Linux release. [/user] [assistant]",  # noqa: E501
    ]

    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        distributed_executor_backend="ray",
        tensor_parallel_size=tp_size,
    )

    expected_lora_output = [
        "give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])",  # noqa: E501
        "give_opinion(name[SpellForce 3], developer[Grimlore Games], release_year[2017], rating[poor])",  # noqa: E501
        "inform(name[BioShock], release_year[2007], rating[good], genres[action-adventure, role-playing, shooter], platforms[PlayStation, Xbox, PC], available_on_steam[yes], has_linux_release[no], has_mac_release[yes])",  # noqa: E501
    ]
    assert do_sample(llm, mixtral_lora_files, lora_id=1,
                     prompts=prompts) == expected_lora_output
    assert do_sample(llm, mixtral_lora_files, lora_id=2,
                     prompts=prompts) == expected_lora_output


@pytest.mark.parametrize("tp_size", [4])
def test_mixtral_lora_all_target_modules(mixtral_lora_files_all_target_modules,
                                         tp_size):
    """This LoRA model has all supported Mixtral target modules"""

    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    prompts = [
        "Question: What happens if you touch the eyes of a blind man?\nChoices:\nB: The man's sight will be restored.\nA: Nothing happens if you touch the eyes of a blind man.\nC: The man will get his eyesight back.\nD: The man will no longer be blind.\nAnswer:",  # noqa: E501
        "Question: Which will increase the energy of motion of water molecules?\nChoices:\nA: add heat\nB: add ice\nC: remove some of the water\nD: place the water in the freezer\nAnswer:",  # noqa: E501
        "Since Craig threw aluminum cans in the trash and Benjamin recycled, _ was environmentally irresponsible.\nChoices:\n1: Craig\n2: Benjamin\nAnswer:",  # noqa: E501
    ]

    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        distributed_executor_backend="ray",
        tensor_parallel_size=tp_size,
        max_lora_rank=32,
    )

    expected_lora_output = [
        "A: Nothing happens if you touch the eyes of a blind man.",
        "A: add heat",
        "1: Craig",
    ]

    assert do_sample(llm,
                     mixtral_lora_files_all_target_modules,
                     lora_id=1,
                     prompts=prompts) == expected_lora_output
    assert do_sample(llm,
                     mixtral_lora_files_all_target_modules,
                     lora_id=2,
                     prompts=prompts) == expected_lora_output
