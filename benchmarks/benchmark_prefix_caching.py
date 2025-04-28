# SPDX-License-Identifier: Apache-2.0
"""
Benchmark the efficiency of prefix caching.

This script allows you to benchmark the performance of
a model with and without prefix caching using either fixed prompts
or prompts sampled from the ShareGPT dataset.

Fixed example usage:
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 100 \
        --input-length-range 128:256

ShareGPT example usage:
    # This command samples 20 prompts with input lengths
    # between 128 and 256 tokens from the ShareGPT dataset,
    # then replicates each prompt 5 times.
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-prefix-caching \
        --num-prompts 20 \
        --repeat-count 5 \
        --input-length-range 128:256
"""

import dataclasses
import json
import random
import time
from typing import Optional

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501


def test_prefix(llm=None, sampling_params=None, prompts=None):
    start_time = time.time()

    llm.generate(prompts, sampling_params=sampling_params)

    end_time = time.time()
    print(f"cost time {end_time - start_time}")


@dataclasses.dataclass
class Request:
    prompt: str
    prompt_len: int
    output_len: int


def sample_tokens(tokenizer: PreTrainedTokenizerBase,
                  length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)

    # Remove the special tokens.
    return random.choices(
        [v for k, v in vocab.items() if k not in all_special_ids],
        k=length,
    )


def sample_requests_from_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: Optional[int],
) -> list[Request]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    min_len, max_len = input_length_range
    assert min_len >= 0 and max_len >= min_len, "input_length_range too small"

    # Filter out sequences that are too long or too short
    filtered_requests: list[Request] = []

    for i in range(len(dataset)):
        if len(filtered_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(dataset[i][0]).input_ids
        prompt = tokenizer.decode(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (len(completion_token_ids)
                      if fixed_output_len is None else fixed_output_len)
        if min_len <= prompt_len <= max_len:
            filtered_requests.append(Request(prompt, prompt_len, output_len))

    return filtered_requests


def sample_requests_from_random(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: Optional[int],
    prefix_len: int,
) -> list[Request]:

    requests = []
    prefix_token_ids = sample_tokens(tokenizer, prefix_len)
    min_len, max_len = input_length_range

    for i in range(num_requests):
        unique_part_token_ids = sample_tokens(
            tokenizer,
            random.randint(min_len - prefix_len, max_len - prefix_len))
        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)
        assert (min_len <= prompt_len <= max_len
                ), f"prompt_len {prompt_len} out of range {min_len}:{max_len}"
        requests.append(Request(prompt, prompt_len, fixed_output_len))
    return requests


def repeat_and_sort_requests(requests: list[Request],
                             repeat_count: int,
                             sort: bool = False) -> list[str]:
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda x: x[1])
    else:
        random.shuffle(repeated_requests)
    return [req.prompt for req in repeated_requests]


def main(args):
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(':')))
    random.seed(args.seed)
    if args.dataset_path is not None:
        if args.prefix_len > 0:
            raise ValueError("prefix-len is not supported when "
                             "dataset-path is provided.")
        print(f"Start to sample {args.num_prompts} prompts "
              f"from {args.dataset_path}")
        filtered_requests = sample_requests_from_dataset(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
        )
    else:
        print(f"Start to sample {args.num_prompts} prompts from random")
        filtered_requests = sample_requests_from_random(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
            prefix_len=args.prefix_len,
        )

    # Print some helpful stats of the requests.
    print(f"Sampled {len(filtered_requests)} requests.")
    prompt_lens = [req.prompt_len for req in filtered_requests]
    print(f"Average input length: {sum(prompt_lens) / len(prompt_lens)}")
    print(f"P50 input length: {sorted(prompt_lens)[len(prompt_lens) // 2]}")
    print(f"Min Prompt Length: {min(prompt_lens)}")
    print(f"Max Prompt Length: {max(prompt_lens)}")

    engine_args = EngineArgs.from_cli_args(args)

    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=args.output_len,
                                     detokenize=not args.disable_detokenize)

    print("Testing filtered requests")
    prompts = repeat_and_sort_requests(filtered_requests,
                                       repeat_count=args.repeat_count,
                                       sort=args.sort)

    print("------start generating------")
    test_prefix(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=
        'Benchmark the performance with or without automatic prefix caching.')
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument('--output-len', type=int, default=10)
    parser.add_argument('--num-prompts',
                        type=int,
                        required=True,
                        help="Number of the prompts sampled from dataset")
    parser.add_argument('--repeat-count',
                        type=int,
                        default=1,
                        help='Number of times to repeat each prompt')
    parser.add_argument('--sort',
                        action='store_true',
                        help='Sort prompts by input length')
    parser.add_argument('--input-length-range',
                        type=str,
                        required=True,
                        help='Range of input lengths for sampling prompts,'
                        'specified as "min:max" (e.g., "128:256").')
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Specifies the length of a common prefix to be "
        "added to the input prompt. The input-length-range will "
        "subtract this length when filtering prompts. Only used "
        "when dataset-path is not provided.",
    )
    parser.add_argument(
        '--disable-detokenize',
        action='store_true',
        help=("Do not detokenize responses (i.e. do not include "
              "detokenization time in the latency measurement)"),
    )

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
