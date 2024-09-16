import cProfile
import pstats

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

# A very long prompt, total number of tokens is about 15k.
LONG_PROMPT = ["You are an expert in large language models, aren't you?"
               ] * 100
LONG_PROMPT = ' '.join(LONG_PROMPT)


def main(args):
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        use_v2_block_manager=args.use_v2_block_manager,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)
    profiler = cProfile.Profile()

    print("------warm up------")
    for i in range(3):
        output = llm.generate(LONG_PROMPT, sampling_params)

    batched_prompts = [
        ('' if args.share_prefix else ('%d' % i)) + LONG_PROMPT 
        for i in range(1000)
    ]
    print("------start generating------")
    profiler.runctx('llm.generate(batched_prompts, sampling_params)',
                    globals(), locals())

    # analyze the runtime of hashing function
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    total_time = 0
    total_calls = 0
    for func in stats.stats:
        # 'hash_of_block' is the hashing function for v1 block manager
        # and 'hash_block_tokens' is for v2.
        if 'hash_of_block' in func[2] or 'hash_block_tokens' in func[2]:
            total_time = stats.stats[func][3]
            total_calls = stats.stats[func][0]
    print(f"Hashing time:\t{total_time:.2f} seconds")
    print(f"Total time:\t{stats.total_tt:.2f} seconds")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Benchmark the performance of hashing function in'
        'automatic prefix caching.')
    parser.add_argument('--model', type=str, default='lmsys/longchat-7b-16k')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--output-len', type=int, default=1)
    parser.add_argument('--enable-prefix-caching',
                        action='store_true',
                        help='enable prefix caching')
    parser.add_argument('--use-v2-block-manager',
                        action='store_true',
                        help='Use BlockSpaceMangerV2')
    parser.add_argument('--share-prefix',
                        action='store_true',
                        help='Share prefix between different prompts.')
    args = parser.parse_args()
    main(args)
