# The CLI entrypoint to vLLM.
import argparse
from openai import OpenAI
import os
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
                                 default="http://localhost:8000",
                                 help="url of the running OpenAI-Compatible RESTful API server")
    complete_parser.add_argument("--model-name",
                                type=str, default=None,
                                help="the model name used in prompt completion, \
                                    default to the first model in list models API call.")
    complete_parser.add_argument("--openai-api-key",
                                 type=str,
                                 default=None,
                                 help="API key for OpenAI services. If provided, this api key will \
                                    overwrite the api key obtained through environment variables.")
    complete_parser.set_defaults(func=interactive_cli)

    chat_parser = subparsers.add_parser(
        "chat",
        help="Generate chat completions via the running API server",
        usage="vllm chat [options]"
    )
    chat_parser.add_argument("--url",
                                 type=str,
                                 default="http://localhost:8000",
                                 help="url of the running OpenAI-Compatible RESTful API server")
    chat_parser.add_argument("--model-name",
                                type=str, default=None,
                                help="the model name used in chat completions, \
                                    default to the first model in list models API call.")
    chat_parser.add_argument("--system-prompt",
                             type=str,
                             help="the system prompt to be added to the chat template, used for \
                                 methods that support system prompts.")
    chat_parser.set_defaults(func=interactive_cli)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def interactive_cli(args: argparse.Namespace) -> None:
    base_url = getattr(args, "url")
    openai_api_key = getattr(args, "openai-api-key", os.environ.get("OPENAI_API_KEY"))
    create_completion_url = base_url + COMPLETE_ROUTE
    create_chat_completion_url = base_url + CHAT_COMPLETE_ROUTE

    http_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    openai_client = OpenAI(api_key=openai_api_key, default_headers=http_headers)

    if args.model_name:
        model_name = args.model_name
    else:
        available_models = openai_client.models.list()
        model_name = available_models["data"][0]["id"]

    if args.command == "complete":
        complete(create_completion_url, model_name, openai_client)
    elif args.command == "chat":
        chat(create_chat_completion_url, model_name, openai_client)


def complete(create_completion_url: str, model_name: str, client: OpenAI) -> None:
    while True:
        input_prompt = input("Please enter prompt to complete:\n>")
        
        completion = client.completions.create(
            model=model_name,
            prompt=input_prompt
        )
        choice = completion.get("choices", [])[0].get("text", "No completion found.")
        print(f"Response Content: {choice}")


def chat(create_chat_completion_url: str, model_name: str, client: OpenAI) -> None:
    conversation = []
    while True:
        input_message = input("Please enter a message for the chat model:\n>")
        message = {
            "role": "user",
            "content": input_message
        }
        conversation.append(message)

        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=conversation
        )

        response_message = (
            chat_completion.get("content", {})
                            .get("choices", [])[0]
                            .get("message", {})
        )
        choice = response_message.get("content", "No response message found.")

        conversation.append(response_message)
        print(f"Response Content: {choice}")


if __name__ == "__main__":
    main()
