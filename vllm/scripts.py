# The CLI entrypoint to vLLM.
import argparse
import os
import signal
import sys
from typing import Optional

from openai import OpenAI

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser


def registrer_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def serve(args: argparse.Namespace) -> None:
    # EngineArgs expects the model name to be passed as --model.
    if args.model is not None and args.model == args.model_tag:
        raise ValueError(
            "The --model argument is not supported for the serve command. "
            "Use positional argument [model_tag] instead.")
    args.model = args.model_tag

    run_server(args)


def interactive_cli(args: argparse.Namespace) -> None:
    registrer_signal_handlers()

    base_url = args.url
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    if args.model_name:
        model_name = args.model_name
    else:
        available_models = openai_client.models.list()
        model_name = available_models.data[0].id

    print(f"Using model: {model_name}")

    if args.command == "complete":
        complete(model_name, openai_client)
    elif args.command == "chat":
        chat(args.system_prompt, model_name, openai_client)


def complete(model_name: str, client: OpenAI) -> None:
    print("Please enter prompt to complete:")
    buffer = ""
    while True:
        input_prompt = input("> ")

        completion = client.completions.create(model=model_name,
                                               prompt=buffer + input_prompt)
        output = completion.choices[0].text
        print(output)

        buffer += input_prompt + "\n" + output + "\n"


def chat(system_prompt: Optional[str], model_name: str,
         client: OpenAI) -> None:
    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    print("Please enter a message for the chat model:")
    while True:
        input_message = input("> ")
        message = {"role": "user", "content": input_message}
        conversation.append(message)

        chat_completion = client.chat.completions.create(model=model_name,
                                                         messages=conversation)

        response_message = chat_completion.choices[0].message
        output = response_message.content

        conversation.append(response_message)
        print(output)


def _add_query_options(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="url of the running OpenAI-Compatible RESTful API server")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=("The model name used in prompt completion, default to "
              "the first model in list models API call."))
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for OpenAI services. If provided, this api key "
            "will overwrite the api key obtained through environment variables."
        ))
    return parser


def main():
    parser = argparse.ArgumentParser(description="vLLM CLI")
    subparsers = parser.add_subparsers(required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the vLLM OpenAI Compatible API server",
        usage="vllm serve <model_tag> [options]")
    serve_parser.add_argument("model_tag",
                              type=str,
                              help="The model tag to serve")
    serve_parser = make_arg_parser(serve_parser)
    serve_parser.set_defaults(dispatch_function=serve)

    complete_parser = subparsers.add_parser(
        "complete",
        help=("Generate text completions based on the given prompt "
              "via the running API server"),
        usage="vllm complete [options]")
    _add_query_options(complete_parser)
    complete_parser.set_defaults(dispatch_function=interactive_cli,
                                 command="complete")

    chat_parser = subparsers.add_parser(
        "chat",
        help="Generate chat completions via the running API server",
        usage="vllm chat [options]")
    _add_query_options(chat_parser)
    chat_parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=("The system prompt to be added to the chat template, "
              "used for models that support system prompts."))
    chat_parser.set_defaults(dispatch_function=interactive_cli, command="chat")

    args = parser.parse_args()
    # One of the sub commands should be executed.
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
