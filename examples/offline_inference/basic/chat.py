# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from utils import add_sampling_params_args, del_sampling_params_args

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


def main(args: Namespace):
    chat_template_path = args.chat_template_path
    del args.chat_template_path

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    args = del_sampling_params_args(args)

    # Create an LLM.
    llm = LLM(**vars(args))

    def print_outputs(outputs):
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Generated text: {generated_text!r}")
        print("-" * 80)

    print("=" * 80)

    # In this script, we demonstrate how to pass input to the chat method:
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content":
            "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
    print_outputs(outputs)

    # You can run batch inference with llm.chat API
    conversations = [conversation for _ in range(10)]

    # We turn on tqdm progress bar to verify it's indeed running batch inference
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    print_outputs(outputs)

    # A chat template can be optionally supplied.
    # If not, the model will use its default chat template.
    if chat_template_path:
        with open(chat_template_path) as f:
            chat_template = f.read()

        outputs = llm.chat(
            conversations,
            sampling_params,
            use_tqdm=False,
            chat_template=chat_template,
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser = add_sampling_params_args(parser)
    # Set example specific arguments
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--chat-template-path", type=str, default=None)
    args = parser.parse_args()
    main(args)
