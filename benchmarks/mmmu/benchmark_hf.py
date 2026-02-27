# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
from eval_utils import (
    add_common_benchmark_args,
    get_message,
    load_benchmark_config,
    load_benchmark_dataset,
    run_benchmark,
)
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

from vllm.utils import FlexibleArgumentParser


def load_model_and_processor(model_name: str):
    """Load HuggingFace Vision-Language model and processor"""
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = None
    for auto_class in [AutoModelForImageTextToText]:
        try:
            model = auto_class.from_pretrained(
                model_name, torch_dtype="auto", trust_remote_code=True
            )
            print(f"Successfully loaded model with {auto_class.__name__}")
            break
        except Exception:
            continue

    if model is None:
        raise ValueError(
            f"Could not load model {model_name} with any available auto class"
        )

    model = model.eval().cuda()

    return model, processor


def generate_response(
    model,
    processor,
    prompt: str,
    image,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    do_sample: bool,
    seed: int,
) -> str:
    """Generate response using HuggingFace Vision-Language model"""
    # Set seed for reproducibility
    set_seed(seed)

    messages = get_message(prompt, image)

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    # Extract generated tokens (excluding input tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()


def hf_generate_func(model, processor, generation_params):
    """Create a generation function for HuggingFace VL models
    that matches the common interface"""

    def generate(prompts: list[str], images: Optional[list] = None) -> list[str]:
        """Generate responses using HuggingFace VL model"""
        responses = []
        if images is None:
            images = [None] * len(prompts)

        for prompt, image in zip(prompts, images):
            response = generate_response(
                model,
                processor,
                prompt,
                image,
                max_tokens=generation_params.max_tokens,
                temperature=generation_params.temperature,
                top_p=generation_params.top_p,
                top_k=generation_params.top_k,
                do_sample=generation_params.do_sample,
                seed=generation_params.seed,
            )
            responses.append(response)
        return responses

    return generate


def main(args):
    # Load model and processor
    print(f"Loading model from {args.model}...")
    model, processor = load_model_and_processor(args.model)

    # Load evaluation config
    config = load_benchmark_config(
        args.config_path if hasattr(args, "config_path") else "eval_config.yaml"
    )

    # Load dataset
    samples = load_benchmark_dataset(
        split=args.split, subject=args.subject, max_samples=args.max_samples
    )

    # Create generation function
    generate_func = hf_generate_func(model, processor, args)

    # Model info for saving
    model_info = {
        "model": args.model,
        "split": args.split,
        "subject": args.subject,
        "max_samples": args.max_samples,
    }

    # Run benchmark using common logic
    results = run_benchmark(
        samples=samples,
        config=config,
        args=args,
        generate_func=generate_func,
        batch_size=1,  # HF processes one at a time
        subject=args.subject,
        output_path=args.output_path,
        model_info=model_info,
    )

    return results


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark HuggingFace models on MMMU dataset from HuggingFace Hub"
    )

    # Add common benchmark arguments
    parser = add_common_benchmark_args(parser, framework="hf")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
