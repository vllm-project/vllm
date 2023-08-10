from vllm import LLM, SamplingParams
from fastchat.model import get_conversation_template, load_model
import json
import re
import os
import argparse


def load_test_cases():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data/prompt.json")
    with open(data_path, 'r') as f:
        objs = json.load(f)
    return objs


def parse_response_num(output):
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result")
        response_number = -1
    return response_number


def test_long_prompt(model_name_or_path):

    def prepare_prompt(test_case):
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], test_case["prompt"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    test_cases = load_test_cases()
    max_tokens = 100

    hf_model, tokenizer = load_model(model_name_or_path, device="cuda")
    hf_reponse_nums = []
    for idx, test_case in enumerate(test_cases):
        # prepare prompt
        prompt = prepare_prompt(test_case)
        input = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = input.input_ids.size()[-1]

        hf_generated = hf_model.generate(
            input.input_ids, max_new_tokens=max_tokens)[0][prompt_length:]
        hf_text = tokenizer.batch_decode([hf_generated],
                                         skip_special_tokens=True)[0]
        hf_reponse_nums.append(parse_response_num(hf_text))
    del hf_model

    vllm_model = LLM(model=model_name_or_path,
                     rope_scaling={
                         "type": "linear",
                         "factor": 8.0
                     })
    vllm_sample_params = SamplingParams(max_tokens=max_tokens, temperature=0)
    vllm_response_nums = []
    for idx, test_case in enumerate(test_cases):
        prompt = prepare_prompt(test_case)
        vllm_generated = vllm_model.generate(
            prompt, sampling_params=vllm_sample_params)[0].outputs[0]
        vllm_text = vllm_generated.text
        vllm_response_nums.append(parse_response_num(vllm_text))

    correct = 0
    for test_case, hf_response_num, vllm_response_num in zip(
            test_cases, hf_reponse_nums, vllm_response_nums):
        if hf_response_num != vllm_response_num:
            print(
                f"[Error] answer: {test_case['expected_number']},",
                f"hf generated: {hf_response_num}, vllm generated: {vllm_response_num}"
            )
        else:
            correct += 1
    print(f"Correct: {correct}/{len(test_cases)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path",
                        type=str,
                        help="model path",
                        default="lmsys/longchat-7b-16k")
    args = parser.parse_args()

    test_long_prompt(args.model_name_or_path)
