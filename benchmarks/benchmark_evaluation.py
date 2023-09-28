import argparse
# import asyncio
# import json
import os
# import random
# import time
from typing import List, Tuple, Dict

# import aiohttp
import numpy as np
import pandas as pd
# from transformers import PreTrainedTokenizerBase
# from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm import LLM, SamplingParams, RequestOutput
from mmlu_template import MMLUTemplate

TEMPLATE_REGITRY = {
    "mmlu": MMLUTemplate,
}


def sample_requests(
    # dataset_path: str,
    # num_requests: int,
    # tokenizer: PreTrainedTokenizerBase,
    dev_data_path: str,
    test_data_path: str,
    subjects: List[str],
    dataset_template: str = "mmlu",
    is_analyse: bool = False,
) -> Tuple[List[str], List[str], List[int]]:
    # Load the dataset.
    nums_questions = []
    dataset = []
    labels = []
    template_class = TEMPLATE_REGITRY[dataset_template]
    for subject in subjects:
        test_dataset = pd.read_csv(os.path.join(test_data_path, subject + "_test.csv"), header=None)
        nums_questions.append(len(test_dataset))
        template = template_class(subject, os.path.join(dev_data_path, subject + "_dev.csv"), is_analyse)
        for idx in range(len(test_dataset)):
            prompt = template.getTemplate(test_dataset, idx)
            dataset.append(prompt)
            labels.append(test_dataset.iloc[idx, -1])
    return dataset, labels, nums_questions


def run_vllm(
    requests: List[str],
    output_len: int,
    model: str,
    tokenizer: str,
    kv_cache_dtype: str = "int8",
    kv_quant_params_path: str = None,
    tensor_parallel_size: int = 1,
    seed: int = 0,
    n: int = 1,
    use_beam_search: bool = False,
    trust_remote_code: bool = False,
) -> List[RequestOutput]:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        kv_cache_dtype=kv_cache_dtype,
        kv_quant_params_path=kv_quant_params_path,
    )
    for prompt in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    # FIXME(woosuk): Do use internal method.
    return llm._run_engine(use_tqdm=True)


def evalute(
    request_outputs: List[RequestOutput],
    labels: List[str],
    nums_questions: List[int],
    subjects: List[str],
    dataset_template: str = "mmlu",
) -> Dict[str, float]:
    template_class = TEMPLATE_REGITRY[dataset_template]
    pred = [template_class.findAnswer(r.outputs[0].text) for r in request_outputs]
    ids = np.cumsum(nums_questions)
    lhs = 0
    accs: List[float] = []
    for rhs in ids:
        pred_paritition = np.array(pred[lhs: rhs])
        labels_partition = np.array(labels[lhs: rhs])
        acc = np.mean(pred_paritition == labels_partition)
        accs.append(acc)
    sub2acc = {sub: acc for sub, acc in zip(subjects, accs)}
    return sub2acc


def main(args: argparse.Namespace):
    subjects = [
        "college_computer_science",
    ]
    dataset, labels, nums_questions = sample_requests(
        args.dev_data_path,
        args.test_data_path,
        subjects,
        is_analyse=args.is_analyse
    )
    request_outputs = run_vllm(
        dataset,
        args.output_len,
        args.model,
        args.tokenizer,
        args.kv_cache_dtype,
        args.kv_quant_params_path,
        args.tensor_parallel_size,
        args.seed, args.n,
        args.use_beam_search,
        args.trust_remote_code,
    )
    sub2acc = evalute(
        request_outputs,
        labels,
        nums_questions,
        subjects,
        )
    print(sub2acc)


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
                        default=100,
                        help="nums of max token for evaluation outputs")
    parser.add_argument("--kv-cache-dtype",
                        type=str,
                        default="float16")
    parser.add_argument("--kv-quant-params-path",
                        type=str,
                        default=None)
    args = parser.parse_args()
    main(args)
