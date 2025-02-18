# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


def main(args: dict):
    chat_template_path = args.pop("chat_template_path")

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.pop("max_tokens"),
        temperature=args.pop("temperature"),
        top_p=args.pop("top_p"),
        top_k=args.pop("top_k"),
    )

    # Create an LLM.
    llm = LLM(**args)

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
    # Add engine args
    ea_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(ea_group)
    ea_group.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    # Add sampling params
    sp = SamplingParams()
    sp_group = parser.add_argument_group("Sampling parameters")
    sp_group.add_argument("--max-tokens", type=int, default=sp.max_tokens)
    sp_group.add_argument("--temperature", type=float, default=sp.temperature)
    sp_group.add_argument("--top-p", type=float, default=sp.top_p)
    sp_group.add_argument("--top-k", type=int, default=sp.top_k)
    # Add example params
    parser.add_argument("--chat-template-path", type=str, default=None)
    args: dict = vars(parser.parse_args())
    main(args)
