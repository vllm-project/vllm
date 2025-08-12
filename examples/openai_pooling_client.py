"""
Example online usage of Pooling API.

Run `vllm serve <model> --task <embed|classify|reward|score>`
to start up the server in vLLM.
"""
import argparse
import pprint

import requests


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model",
                        type=str,
                        default="jason9693/Qwen2.5-1.5B-apeach")

    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/pooling"
    model_name = args.model

    # Input like Completions API
    prompt = {"model": model_name, "input": "vLLM is great!"}
    pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Pooling Response:")
    pprint.pprint(pooling_response.json())

    # Input like Chat API
    prompt = {
        "model":
        model_name,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "vLLM is great!"
            }],
        }]
    }
    pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Pooling Response:")
    pprint.pprint(pooling_response.json())
