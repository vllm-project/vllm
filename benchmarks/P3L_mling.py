#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
*MULTILINGUAL*  Patch-Perplexity (P3L)

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

Running DeepSeek-V2 model
(
    ./vllm/examples/P3L_mling.py
    --model=meta-llama/Llama-2-7b-chat-hf
    --context-size=1024
    --sample-size=512
)

should result in PPL ~ 8.42927

Running DeepSeek-V2 model
(
    ./vllm/examples/P3L_mling.py
    --model=meta-llama/Llama-2-7b-chat-hf
    --context-size=1024
    --sample-size=512
    --patch-size=1
    --lang-script="cmn_Hant"
)
should result in PPL ~ 2.67962

The multi-linguality is implemented through the additional
key "--lang-script", which defaults to English in Latin
scripture ("eng_Latn").

Please refer to

https://confluence.amd.com/display/MLSE/Multi-Lingual+P3L+Test

for the complete set of possible language-scripture choices.

Running the script with multiple batches is possible
by specifying the --batch-size parameter.

"""

import argparse
import dataclasses
import datetime
import json
import math
import os
import tempfile

import pandas
from huggingface_hub import hf_hub_download

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def get_wikitext2_text(tokenizer):
    with tempfile.TemporaryDirectory() as tmpdirname:
        hf_hub_download(
            repo_id="alexei-v-ivanov-amd/wiki",
            repo_type="dataset",
            filename="wiki.test.raw",
            local_dir=tmpdirname,
        )
        with open(os.path.join(tmpdirname, "wiki.test.raw")) as f:
            test_text = "\n".join(line.strip() for line in f)
            test_enc = tokenizer(test_text)

    return test_enc, test_text


def get_flores_plus_text(tokenizer, lng_script):
    hf_hub_download(
        repo_id="alexei-v-ivanov-amd/flores_plus",
        repo_type="dataset",
        filename=lng_script + ".parquet",
        local_dir="./",
    )

    df = pandas.read_parquet("./" + lng_script + ".parquet")
    test_text = "\n\n".join(line.strip() for line in df["text"])
    test_enc = tokenizer(test_text)

    os.remove("./" + lng_script + ".parquet")

    return test_enc, test_text


def vllm_init(args):
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1,
        ignore_eos=True,
        ppl_measurement=True,
        future_context=[],
        prompt_logprobs=1,
        logprobs=1,
        presence_penalty=0.0,
    )

    return llm, sampling_params


def vllm_predict(CONT, llm, sampl_par):
    result = llm.generate(prompt_token_ids=CONT, sampling_params=sampl_par)
    return result


def main(args: argparse.Namespace):
    MESSAGE = f"Initialising @ {datetime.datetime.now()}"
    logger.info(MESSAGE)
    print(MESSAGE)
    my_ppl = 0.0

    logger.info("Initializing the engine.")
    my_llm, my_sampl_par = vllm_init(args)
    my_tokenizer = my_llm.llm_engine.tokenizer.tokenizer
    logger.info(my_sampl_par)
    logger.info("Initialized the engine.")

    my_n_samples = args.sample_size
    my_lang_script = args.lang_script

    if (
        args.context_size + my_n_samples
    ) > my_llm.llm_engine.model_config.max_model_len:
        MESSAGE = (
            ""
            "Error! The total number of tokens:\n"
            f" prefix ({args.context_size}) + "
            f"to be generated ({my_n_samples})"
            f" can't be bigger than the model limit "
            f"({my_llm.llm_engine.model_config.max_model_len})."
        )
        logger.info(MESSAGE)
        print(MESSAGE)
        return

    my_test_enc, my_test_text = get_flores_plus_text(my_tokenizer, my_lang_script)

    logger.info("Loaded the test data.")

    my_n_patches = math.ceil(
        (len(my_test_enc["input_ids"]) - args.context_size - 1) / my_n_samples
    )
    if args.patch_size is not None:
        my_n_patches = args.patch_size

    num_tokens_generated = 0
    starting_time = datetime.datetime.now()
    MESSAGE = (
        f"Starting generation @ {starting_time}\n"
        " Have the test sample of "
        f"{len(my_test_enc['input_ids'])} tokens"
        f" will try to process {my_n_patches} patche(s),"
        f" generating {my_n_samples} tokens in each patch"
        f" from the initial context of {args.context_size} tokens."
    )

    logger.info(MESSAGE)
    print(MESSAGE)

    my_batchsize = args.batch_size

    for c in range(0, my_n_patches, my_batchsize):
        CONTEXT = []
        my_sampl_par.future_context = []
        my_sampl_par.cntr = []

        for b in range(my_batchsize):
            if (c + b) < my_n_patches:
                upper_boundary = min(
                    (c + b + 1) * my_n_samples + args.context_size,
                    len(my_test_enc["input_ids"]),
                )
                CONTEXT.append(
                    my_test_enc["input_ids"][
                        (c + b) * my_n_samples : (c + b) * my_n_samples
                        + args.context_size
                    ]
                )

                my_sampl_par.future_context.append(
                    my_test_enc["input_ids"][
                        (c + b) * my_n_samples + args.context_size : upper_boundary
                    ]
                )

                my_sampl_par.cntr.append(c + b)

        my_sampl_par.max_tokens = max(
            len(my_sampl_par.future_context[b]) for b in range(len(CONTEXT))
        )

        LOGPROBS = vllm_predict(CONTEXT, my_llm, my_sampl_par)
        for b in range(len(CONTEXT)):
            num_tokens_generated += len(LOGPROBS[b].outputs[0].token_ids)
            my_ppl -= LOGPROBS[b].outputs[0].cumulative_logprob

        if num_tokens_generated < my_n_samples * len(CONTEXT):
            MESSAGE = (
                f"Warning: The number of generated tokens is"
                f"less than requested ({num_tokens_generated}"
                f" < {my_n_samples * len(CONTEXT)})."
            )
            logger.info(MESSAGE)
            print(MESSAGE)

        MESSAGE = (
            f"Iterations {c + 1} through {c + len(CONTEXT)}"
            f" of {my_n_patches} Intermediate "
            "Estimates:\n"
            f"\tCross-entropy_intermediate={my_ppl / num_tokens_generated}\n"
            f"\tPerplexity_intermediate="
            f"{math.exp(my_ppl / num_tokens_generated)}"
        )

        logger.info(MESSAGE)
        print(MESSAGE)

    ending_time = datetime.datetime.now()
    MESSAGE = (
        f"Done @ {ending_time} after processing for"
        f" {ending_time - starting_time}"
        f" generated {num_tokens_generated} tokens."
    )

    logger.info(MESSAGE)
    print(MESSAGE)

    MESSAGE = (
        f"\tIntegral Cross-Entropy={my_ppl}\n\tAverage Cross-Entropy="
        f"{my_ppl / num_tokens_generated}"
        f"\n\tPPL={math.exp(my_ppl / num_tokens_generated)}"
    )

    if args.output_json:
        results = {
            "integral_cross_entropy": my_ppl,
            "average_cross_entropy": my_ppl / num_tokens_generated,
            "ppl": math.exp(my_ppl / num_tokens_generated),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

    logger.info(MESSAGE)
    print(MESSAGE)
    return


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Measure the PPPL (P3L) score of a given model."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./wikitext/wikitext-2-v1/test-00000-of-00001.parquet",
    )
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--lang-script", type=str, default="eng_Latn")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    main(args)
