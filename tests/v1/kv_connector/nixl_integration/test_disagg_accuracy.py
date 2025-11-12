# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os
import time

import openai
import requests

MAX_OUTPUT_LEN = 30

SAMPLE_PROMPTS = (
    "Red Hat is the best company in the world to work for because it works on "
    "open source software, which means that all the contributions are "
    "delivered to the community. As a result, when working on projects like "
    "vLLM we are able to meet many amazing people from various organizations "
    "like AMD, Google, NVIDIA, ",
    "We hold these truths to be self-evident, that all men are created equal, "
    "that they are endowed by their Creator with certain unalienable Rights, "
    "that among these are Life, Liberty and the pursuit of Happiness.--That "
    "to secure these rights, Governments are instituted among Men, deriving "
    "their just powers from the consent of the governed, ",
)


def check_vllm_server(url: str, timeout=5, retries=3) -> bool:
    """
    Checks if the vLLM server is ready by sending a GET request to the
    /health endpoint.

    Args:
        url (str): The base URL of the vLLM server.
        timeout (int): Timeout in seconds for the request.
        retries (int): Number of retries if the server is not ready.

    Returns:
        bool: True if the server is ready, False otherwise.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True
            else:
                print(
                    f"Attempt {attempt + 1}: Server returned status code "
                    "{response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error connecting to server: {e}")
        time.sleep(1)  # Wait before retrying
    return False


def run_simple_prompt(
    base_url: str, model_name: str, input_prompt: str, use_chat_endpoint: bool
) -> str:
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url)
    if use_chat_endpoint:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": input_prompt}]}
            ],
            max_completion_tokens=MAX_OUTPUT_LEN,
            temperature=0.0,
            seed=42,
        )
        return completion.choices[0].message.content
    else:
        completion = client.completions.create(
            model=model_name,
            prompt=input_prompt,
            max_tokens=MAX_OUTPUT_LEN,
            temperature=0.0,
            seed=42,
        )

        return completion.choices[0].text


def main():
    """
    This script demonstrates how to accept two optional string arguments
    ("service_url" and "file_name") from the command line, each with a
    default value of an empty string, using the argparse module.
    """
    parser = argparse.ArgumentParser(description="vLLM client script")

    parser.add_argument(
        "--service_url",  # Name of the first argument
        type=str,
        required=True,
        help="The vLLM service URL.",
    )

    parser.add_argument(
        "--model_name",  # Name of the first argument
        type=str,
        required=True,
        help="model_name",
    )

    parser.add_argument(
        "--mode",  # Name of the second argument
        type=str,
        default="baseline",
        help="mode: baseline==non-disagg, or disagg",
    )

    parser.add_argument(
        "--file_name",  # Name of the second argument
        type=str,
        default=".vllm_output.txt",
        help="the file that saves the output tokens ",
    )

    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    if args.mode == "baseline":
        # non-disagg
        health_check_url = f"{args.service_url}/health"
    else:
        # disagg proxy
        health_check_url = f"{args.service_url}/healthcheck"
        if not os.path.exists(args.file_name):
            raise ValueError(
                f"In disagg mode, the output file {args.file_name} from "
                "non-disagg. baseline does not exist."
            )

    service_url = f"{args.service_url}/v1"

    if not check_vllm_server(health_check_url):
        raise RuntimeError(f"vllm server: {args.service_url} is not ready yet!")

    output_strs = dict()
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        use_chat_endpoint = i % 2 == 1
        output_str = run_simple_prompt(
            base_url=service_url,
            model_name=args.model_name,
            input_prompt=prompt,
            use_chat_endpoint=use_chat_endpoint,
        )
        print(f"Prompt: {prompt}, output: {output_str}")
        output_strs[prompt] = output_str

    if args.mode == "baseline":
        # baseline: save outputs
        try:
            with open(args.file_name, "w") as json_file:
                json.dump(output_strs, json_file, indent=4)
        except OSError as e:
            print(f"Error writing to file: {e}")
            raise
    else:
        # disagg. verify outputs
        baseline_outputs = None
        try:
            with open(args.file_name) as json_file:
                baseline_outputs = json.load(json_file)
        except OSError as e:
            print(f"Error writing to file: {e}")
            raise
        assert isinstance(baseline_outputs, dict)
        assert len(baseline_outputs) == len(output_strs)
        for prompt, output in baseline_outputs.items():
            assert prompt in output_strs, f"{prompt} not included"
            assert output == output_strs[prompt], (
                f"baseline_output: {output} != PD output: {output_strs[prompt]}"
            )


if __name__ == "__main__":
    main()
