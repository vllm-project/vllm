#!/usr/bin/env python3
"""
This is a script that produces a realistic PPL measurement 
for the quantized KV cache system by processing a sequence of 
non-overlapping patches of the reference text. Generation of the 
consecutive symbols in each patch is governed (forced)
by the reference text.

The initial context size for the system is set by the parameter 
"--context-size".

The number of output symbols to generate starting from a given 
context is set by the parameter "--sample-size". This variable also 
defines the size of the individual patch.

For the N-token reference text that is split into M patches with the 
system's context size C it takes M*preload + (N-C)*generation time.

Quick correctness validation tips:

Running llama-2-7b model 
( 
    ./vllm/benchmarks/measure_ppl2_MC.py 
    --model=/data/models/llama-2-7b-chat-hf 
    --data=./vllm/tests/prompts/wiki.test.raw 
    --context-size=1024 
    --sample-size=512
)
should result in PPL ~ 6.524227946419175

Running llama-2-7b model 
( 
    ./vllm/benchmarks/measure_ppl2_MC.py 
    --model=/data/models/llama-2-7b-chat-hf 
    --data=./vllm/tests/prompts/wiki.test.raw 
    --context-size=1024 
    --sample-size=512
    --patch-size=1
)
should result in PPL ~ PPL=3.8968611189957523

"""

import argparse
import datetime
import math

from transformers import LlamaTokenizer

from vllm import LLM, SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_wikitext2_text(tokenizer):
    with open(args.data) as f:
        test_text = "\n".join(line.strip() for line in f)
        test_enc = tokenizer(test_text)

    return test_enc, test_text


def vllm_init(args):

    llm = LLM(
        model=args.model,
        tokenizer=None,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        #scales_path=args.kv_cache_scales_path
        #   if args.kv_cache_scales_path!='' else None,
        quantization_param_path=args.kv_cache_scales_path
        if args.kv_cache_scales_path != '' else None,
        enforce_eager=args.enforce_eager)

    sampling_params = SamplingParams(n=1,
                                     temperature=0.0,
                                     top_p=1,
                                     use_beam_search=False,
                                     ignore_eos=True,
                                     ppl_measurement=True,
                                     future_context=[],
                                     presence_penalty=0.0)

    return llm, sampling_params


def vllm_predict(CONT, llm, sampl_par):
    result = llm.generate(prompt_token_ids=CONT, sampling_params=sampl_par)
    return result


def main(args: argparse.Namespace):

    logger.info(f"Initialising @ {datetime.datetime.now()}")
    my_ppl = 0.0

    my_tokenizer = LlamaTokenizer.from_pretrained(args.model)
    logger.info("Loaded the tokenizer.")

    logger.info("Initializing the engine.")
    my_llm, my_sampl_par = vllm_init(args)
    logger.info(my_sampl_par)
    logger.info("Initialized the engine.")

    my_test_enc, my_test_text = get_wikitext2_text(my_tokenizer)
    logger.info("Loaded the test data.")

    my_n_samples = args.sample_size

    my_n_patches = math.ceil(
        (len(my_test_enc['input_ids']) - args.context_size - 1) / my_n_samples)
    if args.patch_size is not None:
        my_n_patches = args.patch_size

    num_tokens_generated = 0
    starting_time = datetime.datetime.now()
    logger.info(f"Starting generation @ {starting_time} \
will try to process {my_n_patches} patche(s), \
generating {my_n_samples} tokens in each patch \
from the initial context of {args.context_size} tokens.")
    for c in range(my_n_patches):
        CONTEXT = []
        my_sampl_par.future_context = []
        CONTEXT.append(
            my_test_enc['input_ids'][c * my_n_samples:c * my_n_samples +
                                     args.context_size])
        upper_boundary = min((c + 1) * my_n_samples + args.context_size,
                             len(my_test_enc['input_ids']))
        my_sampl_par.future_context.append(
            my_test_enc['input_ids'][c * my_n_samples +
                                     args.context_size:upper_boundary])
        my_sampl_par.max_tokens = len(my_sampl_par.future_context[0])
        LOGPROBS = vllm_predict(CONTEXT, my_llm, my_sampl_par)
        num_tokens_generated += len(LOGPROBS[0].outputs[0].token_ids)
        my_ppl -= LOGPROBS[0].outputs[0].cumulative_logprob
        logger.info(f"Iteration {c+1} of {my_n_patches} Intermediate \
Estimates:\n\
\tCross-entropy_intermediate={my_ppl/num_tokens_generated}\n\
\tPerplexity_intermediate={math.exp(my_ppl/num_tokens_generated)}")
    ending_time = datetime.datetime.now()
    logger.info(f"Done @ {ending_time} after processing for \
{ending_time-starting_time} generated {num_tokens_generated} tokens.")

    logger.info(f"Integral Cross-Entropy={my_ppl} Average Cross-Entropy=\
{my_ppl/num_tokens_generated} PPL={math.exp(my_ppl/num_tokens_generated)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument(
        '--data',
        type=str,
        default='./wikitext/wikitext-2-v1/test-00000-of-00001.parquet')
    parser.add_argument('--context-size', type=int, default=4096)
    parser.add_argument('--kv-cache-scales-path', type=str, default='')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--sample-size', type=int, default=512)
    parser.add_argument('--patch-size', type=int, default=None)
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=['auto', 'fp8_e5m2', 'fp8'],
        default='auto',
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    args = parser.parse_args()

    main(args)
