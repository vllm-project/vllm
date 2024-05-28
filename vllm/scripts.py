# The CLI entrypoint to vLLM.
import argparse
import requests
from typing import Dict, Any

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (CompletionRequest, ChatCompletionRequest)

COMPLETE_ROUTE = "/v1/completions"
CHAT_COMPLETE_ROUTE = "/v1/chat/completions"
LIST_MODELS_ROUTE = "/v1/models"

def main():
    parser = argparse.ArgumentParser(description="vLLM CLI")
    subparsers = parser.add_subparsers(required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the vLLM OpenAI Compatible API server",
        usage="vllm serve <model_tag> [options]")
    # Override the `--model` optional argument, make it positional.
    serve_parser.add_argument(
        "model-tag",
        type=str,
        help="The model tag to serve")
    serve_parser = make_arg_parser(serve_parser)
    serve_parser.set_defaults(func=run_server)
    
    complete_parser = subparsers.add_parser(
        "complete",
        help="Generate text completions based on the given prompt",
        usage="vllm complete <complete_prompt> [options]")
    complete_parser.add_argument("--url",
                                 type=str,
                                 default=None,
                                 help="url of the running OpenAI-Compatible RESTful API server")
    complete_parser.add_argument("--complete-model-name",
                                type=str, default=None,
                                help="the model name used in prompt completion")
    complete_parser.set_defaults(func=interactive_cli)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def interactive_cli(args: argparse.Namespace) -> None:
    if args.command == "complete":
        complete(args)
    elif args.command == "chat":
        chat(args)


def complete(args: argparse.Namespace) -> None:
    while True:
        input_prompt = input("Please enter prompt to complete: ")
        if args.url is not None:
            url = args.url
        else:
            url = f"http://{args.host}:{args.port}"

        create_completion_url = url + COMPLETE_ROUTE
        create_completion_headers = {
            "Content-Type": "application/json"
        }

        if args.complete_model_name:
            complete_model_name = args.complete_model_name
        else:
            list_models_url = url + LIST_MODELS_ROUTE
            complete_model_name = _default_model(list_models_url)
        
        completion_request_data = CompletionRequest(
            model=complete_model_name,
            prompt=input_prompt
        ).model_dump()
        
        try:
            completion_response = requests.post(url=create_completion_url,
                                                headers=create_completion_headers,
                                                json=completion_request_data)
            completion_response.raise_for_status()
            completion = completion_response.json()
            completion = _post_response_dict(create_completion_url,
                                            create_completion_headers,
                                            completion_request_data)
            choice = completion.get("choices", [])[0].get("text", "No completion found.")
            print(f"Response Content: {choice}")
        except Exception as err:
            print(f"Error occurred in prompt completion: {err}")


def chat(args: argparse.Namespace) -> None:
    pass



def _default_model(url: str) -> str:
    try:
        models_dict = _get_response_dict(url)
        if "data" in models_dict and models_dict["data"]:
            return models_dict["data"][0]["id"]
        else:
            raise Exception("No models available for completion.")
    except Exception as err:
        raise Exception(f"Error occurred when getting default model for completion: {err}")

    
def _get_response_dict(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        raise Exception(f"HTTP error occurred: {http_err}")
    except requests.exceptions.JSONDecodeError as json_err:
        raise Exception(f"Invalid JSON response: {json_err}")
    except Exception as err:
        raise Exception(f"Error occurred: {err}")
    
def _post_response_dict(
        url: str,
        headers: Dict[str, str],
        json_data: Dict[str, Any]
    ) -> Dict[str, Any]:
    try:
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        raise Exception(f"HTTP error occurred: {http_err}")
    except requests.exceptions.JSONDecodeError as json_err:
        raise Exception(f"Invalid JSON response: {json_err}")
    except Exception as err:
        raise Exception(f"Error occurred: {err}")


if __name__ == "__main__":
    main()
