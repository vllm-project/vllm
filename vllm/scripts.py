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
        help="Generate text completions based on the given prompt via the running API server",
        usage="vllm complete [options]")
    complete_parser.add_argument("--url",
                                 type=str,
                                 default=None,
                                 help="url of the running OpenAI-Compatible RESTful API server")
    complete_parser.add_argument("--complete-model-name",
                                type=str, default=None,
                                help="the model name used in prompt completion")
    complete_parser.set_defaults(func=interactive_cli)

    chat_parser = subparsers.add_parser(
        "chat",
        help="Generate chat completions via the running API server",
        usage="vllm chat [options]"
    )
    chat_parser.add_argument("--url",
                                 type=str,
                                 default=None,
                                 help="url of the running OpenAI-Compatible RESTful API server")
    chat_parser.add_argument("--chat-model-name",
                                type=str, default=None,
                                help="the model name used in chat completions")
    chat_parser.add_argument("--generation-prompt",
                             type=str,
                             help="the generation prompt to be added to the chat template")
    chat_parser.set_defaults(func=interactive_cli)

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
        input_prompt = input("Please enter prompt to complete:")

        base_url = getattr(args, "url", f"http://{args.host}:{args.port}")

        create_completion_url = base_url + COMPLETE_ROUTE
        create_completion_headers = {
            "Content-Type": "application/json"
        }

        if args.complete_model_name:
            complete_model_name = args.complete_model_name
        else:
            complete_model_name = _default_model(base_url)
        
        completion_request_data = CompletionRequest(
            model=complete_model_name,
            prompt=input_prompt
        ).model_dump()
        
        try:
            completion = _post_response_dict(create_completion_url,
                                            create_completion_headers,
                                            completion_request_data)
            choice = completion.get("choices", [])[0].get("text", "No completion found.")
            print(f"Response Content: {choice}")
        except Exception as err:
            print(f"Error occurred in prompt completion: {err}")
            return


def chat(args: argparse.Namespace) -> None:
    conversation = []
    while True:
        input_message = input("Please enter a message for the chat model:")
        message = {
            "role": "user",
            "content": input_message
        }
        if args.generation_prompt:
            message["generation prompt"] = args.generation_prompt
        conversation.append(message)

        base_url = getattr(args, "url", f"http://{args.host}:{args.port}")

        create_chat_completion_url = base_url + CHAT_COMPLETE_ROUTE
        create_chat_completion_headers = {
            "Content-Type": "application/json"
        }

        if args.chat_model_name:
            chat_model_name = args.chat_model_name
        else:
            chat_model_name = _default_model(base_url)

        chat_request_data = ChatCompletionRequest(
            messages=conversation,
            model=chat_model_name
        ).model_dump()

        try:
            chat_completion = _post_response_dict(create_chat_completion_url,
                                            create_chat_completion_headers,
                                            chat_request_data)
            choice = (
                chat_completion.get("content", {})
                               .get("choices", [])[0]
                               .get("message", {})
                               .get("content", "No chat completion found.")
            )
            print(f"Response Content: {choice}")
        except Exception as err:
            print(f"Error occurred in chat completion: {err}")
            return


def _default_model(base_url: str) -> str:
    list_models_url = base_url + LIST_MODELS_ROUTE
    try:
        models_dict = _get_response_dict(list_models_url)
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
