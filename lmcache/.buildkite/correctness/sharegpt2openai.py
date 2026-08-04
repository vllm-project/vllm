# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import json

VLLM_TEMPLATE = {
    "model": "Qwen3/Qwen3-32B",
    "frequency_penalty": 0.0,
    "logit_bias": None,
    "logprobs": False,
    "top_logprobs": 0,
    "max_tokens": 2048,
    "max_completion_tokens": 2048,
    "n": 1,
    "presence_penalty": 0.0,
    "response_format": None,
    "seed": None,
    "stop": [],
    "stream": True,
    "stream_options": None,
    "temperature": 0.01,
    "top_p": 1.0,
    "tools": None,
    "tool_choice": "none",
    "reasoning_effort": None,
    "include_reasoning": True,
    "parallel_tool_calls": False,
    "user": None,
    "best_of": None,
    "use_beam_search": False,
    "top_k": -1,
    "min_p": None,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "stop_token_ids": [],
    "include_stop_str_in_output": False,
    "ignore_eos": False,
    "min_tokens": 0,
    "skip_special_tokens": True,
    "spaces_between_special_tokens": True,
    "priority": 0,
    "truncate_prompt_tokens": None,
    "prompt_logprobs": None,
    "allowed_token_ids": None,
    "bad_words": [],
    "order": "routine",
    "lora_type": None,
    "reasoning": False,
    "echo": False,
    "add_generation_prompt": True,
    "continue_final_message": False,
    "add_special_tokens": False,
    "documents": None,
    "chat_template": None,
    "chat_template_kwargs": {"enable_thinking": False},
    "mm_processor_kwargs": None,
    "guided_json": None,
    "guided_regex": None,
    "guided_choice": None,
    "guided_grammar": None,
    "structural_tag": None,
    "guided_decoding_backend": None,
    "guided_whitespace_pattern": None,
    "kv_transfer_params": None,
    "vllm_xargs": None,
}


def convert_sharegpt_to_vllm(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = {}

    for item in data:
        conv = item["conversations"]

        # # Delete the gpt's response at the end
        if len(conv) > 0 and conv[-1]["from"] == "gpt":
            conv = conv[:-1]

        messages = []
        for c in conv:
            role = "user" if c["from"] == "human" else "assistant"
            content = c["value"]
            messages.append({"role": role, "content": content})

        file_key = f"{item['id']}"

        out_entry = VLLM_TEMPLATE.copy()
        out_entry["messages"] = messages

        output[file_key] = out_entry

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT JSON dataset to vLLM format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input ShareGPT JSON file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the output vLLM JSON file.",
    )

    args = parser.parse_args()

    convert_sharegpt_to_vllm(args.input, args.output)
    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
