# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from eval_utils import (
    add_common_benchmark_args,
    get_message,
    load_benchmark_config,
    load_benchmark_dataset,
    run_benchmark,
)
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: dict):
    # Pop sampling arguments
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Pop benchmark specific arguments
    split = args.pop("split")
    subject = args.pop("subject")
    max_samples = args.pop("max_samples")
    output_path = args.pop("output_path")
    config_path = args.pop("config_path")
    seed = args.pop("seed")
    batch_size = args.pop("batch_size")

    # Create an LLM with remaining args
    print("Loading vLLM model...")
    args["disable_mm_preprocessor_cache"] = True
    llm = LLM(**args)

    # Load tokenizer for chat template
    model_name = args.get("model")
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create sampling params using the LLM instance
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k
    if seed is not None:
        sampling_params.seed = seed

    # Store args for common benchmark function
    class Args:
        def __init__(self):
            self.seed = seed
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k

    benchmark_args = Args()

    # Load evaluation config
    config = load_benchmark_config(config_path)

    # Load dataset
    samples = load_benchmark_dataset(
        split=split, subject=subject, max_samples=max_samples
    )

    # Model info for saving
    model_info = {
        "model": args.get("model"),
        "split": split,
        "subject": subject,
        "max_samples": max_samples,
        "batch_size": batch_size,
    }

    # Create a generation function that matches the HF interface
    def generate_with_params(prompts: list[str], images: list = None) -> list[str]:
        """
        Generate responses for prompts with associated images.
        Args:
            prompts: List of prompt strings
            images: List of image data (can be None for text-only)
        Returns:
            List of response strings
        """
        # Prepare inputs for vLLM batch inference
        inputs = []
        if images is None:
            images = [None] * len(prompts)

        for prompt, image in zip(prompts, images):
            messages = get_message(prompt, image)
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(
                    f"Warning: Failed to apply chat template,\
                        using original prompt: {e}"
                )
                formatted_prompt = prompt

            input_data = {"prompt": formatted_prompt}
            if image is not None:
                input_data["multi_modal_data"] = {"image": image}
            inputs.append(input_data)

        # Use our pre-configured sampling_params
        outputs = llm.generate(inputs, sampling_params, use_tqdm=False)
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            responses.append(response)
        return responses

    # Run benchmark
    results = run_benchmark(
        samples=samples,
        config=config,
        args=benchmark_args,
        generate_func=generate_with_params,
        batch_size=batch_size,
        subject=subject,
        output_path=output_path,
        model_info=model_info,
    )

    return results


def create_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark vLLM models on MMMU dataset using offline inference",
        conflict_handler="resolve",
    )

    # Add common benchmark arguments first (with default values)
    parser = add_common_benchmark_args(parser, framework="vllm")

    # Add engine args (this will override conflicting arguments with vLLM defaults)
    EngineArgs.add_cli_args(parser)

    return parser


def invoke_main() -> None:
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
