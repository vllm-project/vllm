import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams, RequestOutput
from benchmarks.mmlu_template import MMLUTemplate


def sample_requests(
    # dataset_path: str,
    # num_requests: int,
    # tokenizer: PreTrainedTokenizerBase,
    dev_data_path: str,
    test_data_path: str,
    subjects: List[str],
    # dataset_template: str = "mmlu",
    is_analyse: bool = False,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    nums_questions = []
    dataset = []
    labels = []
    template_class = MMLUTemplate
    for subject in subjects:
        test_dataset = pd.read_csv(os.path.join(test_data_path, subject + "_test.csv"), header=None)
        nums_questions.append(len(test_dataset))
        template = template_class(subject, os.path.join(dev_data_path, subject + "_dev.csv"), is_analyse)
        for idx in range(len(test_dataset)):
            prompt = template.getTemplate(test_dataset, idx)
            dataset.append(prompt)
            labels.append(test_dataset.iloc[idx, -1])
    return dataset, labels, nums_questions


def main(args: argparse.Namespace):
    subjects = ["abstract_algebra"]
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_quant_params_path=args.kv_quant_params_path,
    )
    requests, labels, _ = sample_requests(
        args.dev_data_path,
        args.test_data_path,
        subjects,
        args.is_analyse,
    )
    prompt, label = requests[0], labels[0]
    print(f"the correct answer is\n{label}")
    sampling_params = SamplingParams(
            n=args.n,
            temperature=0.0 if args.use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=args.use_beam_search,
            ignore_eos=True,
            max_tokens=args.output_len,
        )
    outputs = llm.generate(prompt, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluation for quantization.")

    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument("--dev-data-path",
                        type=str,
                        default=None,
                        help="path to few-shot dataset")
    parser.add_argument("--test-data-path",
                        type=str,
                        default=None,
                        help="path to test dataset")
    parser.add_argument("--is-analyse",
                        action="store_true")
    parser.add_argument("--output-len",
                        type=int,
                        default=200,
                        help="nums of max token for evaluation outputs")
    parser.add_argument("--kv-cache-dtype",
                        type=str,
                        default="float16")
    parser.add_argument("--kv-quant-params-path",
                        type=str,
                        default=None)
    args = parser.parse_args()
    main(args)
