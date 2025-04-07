# SPDX-License-Identifier: Apache-2.0
"""
Offline benchmark to test the long document QA throughput.

Example usage:
    # This workload samples 8 different prompts with a default input
    # length of 20000 tokens, then replicates each prompt 2 times 
    # in random order.
    python benchmark_long_document_qa_throughput.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-documents 8 \
        --repeat-count 2 

Commandline arguments:
    --num-documents: The number of documents to sample prompts from.

    --document-length: The length of each document in tokens. 
                       (Optional, default: 20000)

    --output-len: The number of tokens to generate for each prompt.
                  (Optional, default: 10)

    --repeat-count: The number of times to repeat each prompt.
                    (Optional, default: 2)

    --repeat-mode: The mode to repeat prompts. The supported modes are:
        - 'random': shuffle the prompts randomly. (Default)
        - 'tile': the entire prompt list is repeated in sequence. (Potentially
                  lowest cache hit)
        - 'interleave': each prompt is repeated consecutively before 
                        moving to the next element. (Highest cache hit)
    
    --shuffle-seed: Random seed when the repeat mode is "random".
                    (Optional, default: 0)

In the meantime, it also supports all the vLLM engine args to initialize the 
LLM engine. You can refer to the `vllm.engine.arg_utils.EngineArgs` for more
details.
"""

import dataclasses
import random
import time

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser


def test_long_document_qa(llm=None, sampling_params=None, prompts=None):
    """
    Test long document QA with the given prompts and sampling parameters.
    Print the time spent in processing all the prompts.

    Args:
        llm: The language model used for generating responses.
        sampling_params: Sampling parameter used to generate the response.
        prompts: A list of prompt strings to be processed by the LLM.
    """
    start_time = time.time()
    llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    print(f"Time to execute all requests: {end_time - start_time:.4f} secs")


def repeat_prompts(prompts, repeat_count, mode: str):
    """
    Repeat each prompt in the list for a specified number of times.
    The order of prompts in the output list depends on the mode.

    Args:
        prompts: A list of prompts to be repeated.
        repeat_count: The number of times each prompt is repeated.
        mode: The mode of repetition. Supported modes are:
            - 'random': Shuffle the prompts randomly after repetition.
            - 'tile': Repeat the entire prompt list in sequence.
              Example: [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
            - 'interleave': Repeat each prompt consecutively before moving to 
              the next. Example: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].

    Returns:
        A list of repeated prompts in the specified order.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    print("Repeat mode: ", mode)
    if mode == 'random':
        repeated_prompts = prompts * repeat_count
        random.shuffle(repeated_prompts)
        return repeated_prompts
    elif mode == 'tile':
        return prompts * repeat_count
    elif mode == 'interleave':
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * repeat_count)
        return repeated_prompts
    else:
        raise ValueError(f"Invalid mode: {mode}, only support "
                         "'random', 'tile', 'interleave'")


def main(args):
    random.seed(args.shuffle_seed)

    # Prepare the prompts:
    # we append the document id at the beginning to avoid any of the document
    # being the prefix of other documents
    prompts = [
        str(i) + ' '.join(['hi'] * args.document_length)
        for i in range(args.num_documents)
    ]

    prompts = repeat_prompts(prompts, args.repeat_count, mode=args.repeat_mode)

    warmup_prompts = [
        "This is warm up request " + str(i) + \
                ' '.join(['hi'] * args.document_length)
        for i in range(args.num_documents)]

    # Create the LLM engine
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)

    print("------warm up------")
    test_long_document_qa(
        llm=llm,
        prompts=warmup_prompts,
        sampling_params=sampling_params,
    )

    print("------start generating------")
    test_long_document_qa(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=
        'Benchmark the performance with or without automatic prefix caching.')

    parser.add_argument(
        '--document-length',
        type=int,
        # Roughly the number of tokens for a system paper,
        # excluding images
        default=20000,
        help='Range of input lengths for sampling prompts,'
        'specified as "min:max" (e.g., "128:256").')

    parser.add_argument('--num-documents',
                        type=int,
                        default=8,
                        help='Range of input lengths for sampling prompts,'
                        'specified as "min:max" (e.g., "128:256").')

    parser.add_argument('--output-len', type=int, default=10)

    parser.add_argument('--repeat-count',
                        type=int,
                        default=2,
                        help='Number of times to repeat each prompt')

    parser.add_argument("--repeat-mode",
                        type=str,
                        default='random',
                        help='The mode to repeat prompts. The supported '
                        'modes are "random", "tile", and "interleave". '
                        'See repeat_prompts() in the source code for details.')

    parser.add_argument("--shuffle-seed",
                        type=int,
                        default=0,
                        help='Random seed when the repeat mode is "random"')

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
