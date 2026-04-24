# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# isort: skip_file
"""
Usage:
VLLM_USE_V1=1 python3 benchmarks/cohere/sanity_test.py SanityTest.test_single_needle
"""

import dataclasses
import glob
import os
import unittest

import vllm.envs as envs
import prompts_template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.inputs import TextPrompt
from vllm.utils import FlexibleArgumentParser


class NeedleTestSuite:
    def __init__(
        self, tokenizer, needle=None, haystack_dir="", retrieval_question=None
    ):
        """
        :param model_path: The model to test.
        :param tokenizer: The tokenizer to encoder and decoder text.
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError(
                "Needle, haystack, and retrieval_question must be provided."
            )
        self.tokenizer = tokenizer
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        # the hay length, read tokens in the haystack_dir until reach context_length,
        self.context_length = 256000
        # insert the needle into 50% place of the whole context_length
        self.depth_percent = 50
        self.final_context_length_buffer = 150

    def get_context_length_in_tokens(self, context):
        return len(self.tokenizer.encode(context, add_special_tokens=False))

    def read_context_files(self):
        context = ""
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        while (self.get_context_length_in_tokens(context)) < self.context_length:
            for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
                with open(file) as f:
                    context += f.read()
        return context

    def encode_and_trim(self, context, context_length):
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) > context_length:
            context = self.tokenizer.decode(tokens[:context_length])
        return context

    def generate_context(self, context_length, depth_percent):
        # Get your haystack dir files loaded into a string
        context = self.read_context_files()

        # # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.tokenizer.encode(self.needle, add_special_tokens=False)
        tokens_context = self.tokenizer.encode(context, add_special_tokens=False)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.tokenizer.encode(".", add_special_tokens=False)

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.tokenizer.decode(tokens_new_context)
        return new_context

    def generate_prompt(self):
        preamble = self.generate_context(self.context_length, self.depth_percent)
        return prompts_template.GEN_PROMPT_TEMPLATE.format(
            preamble="\n" + preamble + "\n", text=self.retrieval_question
        )


def find_text_between(s, prefix, postfix):
    start_index = s.find(prefix)
    if start_index == -1:
        return None  # Prefix not found
    start_index += len(prefix)

    end_index = s.find(postfix, start_index)
    if end_index == -1:
        return None  # Postfix not found

    return s[start_index:end_index]


@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        prompt_len: The length of the prompt in tokens.
    """

    prompt: str
    prompt_len: int


def run_vllm(
    requests: list[SampleRequest],
    engine_args: EngineArgs,
):
    llm = LLM(**dataclasses.asdict(engine_args))

    # Add the requests to the engine.
    prompts: list[TextPrompt] = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(TextPrompt(prompt=request.prompt))
        sampling_params.append(
            SamplingParams(
                n=1,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                ignore_eos=False,
                max_tokens=400,
            )
        )
    return llm.generate(prompts, sampling_params)


class SanityTest(unittest.TestCase):
    def setUp(self):
        envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN = 1
        envs.VLLM_USE_V1 = 1
        # gs://cohere-icebox/tif/hf_export/command3_7b
        # download the model path above
        # TODO: add auto gcs fetching
        model_dir = "/host/engines/hf_20250113_161619_fashionable_economics-FP8-512k/hf_20250113_161619_fashionable_economics-FP8-512k/poseidon"
        parser = FlexibleArgumentParser(description="Sanity test.")
        parser = AsyncEngineArgs.add_cli_args(parser)
        self.args = parser.parse_args(
            [
                "--model",
                model_dir,
                "--tokenizer",
                model_dir,
                "--gpu_memory_utilization",
                "0.95",
                "--tensor_parallel_size",
                "1",  # feel free to change to more tp sizes
                "--max-num-batched-tokens",
                "2048",
                "--tensor-parallel-size",
                "4",
            ]
        )
        # https://github.com/cohere-ai/vllm-cuda/blob/35441ae912c311aaef362d544414faa3614acaa0/vllm/engine/arg_utils.py#L87
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)

    def test_simple_outputs(self):
        input_prompt = prompts_template.GEN_PROMPT_TEMPLATE.format(
            preamble="", text=prompts_template.GENERATIVE_TEST_CASE
        )
        input_len = len(self.tokenizer(input_prompt).input_ids)
        requests = [SampleRequest(prompt=input_prompt, prompt_len=input_len)]
        output = (
            run_vllm(requests, EngineArgs.from_cli_args(self.args))[0].outputs[0].text
        )
        self.assertTrue("200" in output)

    # def test_guided_generation_outputs(self):
    #     # gg is not supported by vllm yet
    #     pass

    def test_single_needle(self):
        input_prompt = NeedleTestSuite(
            tokenizer=self.tokenizer,
            needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
            # the data source is from: https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main/needlehaystack/PaulGrahamEssays
            haystack_dir="needle_test_data",
            retrieval_question="What is the best thing to do in San Francisco?",
        ).generate_prompt()
        input_len = len(self.tokenizer(input_prompt).input_ids)
        requests = [SampleRequest(prompt=input_prompt, prompt_len=input_len)]
        output = (
            run_vllm(requests, EngineArgs.from_cli_args(self.args))[0].outputs[0].text
        )
        print(output)
        self.assertTrue("Dolores Park" in output)
        self.assertTrue("eat a sandwich" in output)


class FlashInferSanityTest(SanityTest):
    def setUp(self):
        # need to install flashinfer first:
        # pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
        envs.VLLM_ATTENTION_BACKEND = "FLASHINFER"
        # kv cache block is 29k ish
        super().setUp()


class FlashInferFP8KV16SanityTest(SanityTest):
    def setUp(self):
        # need to install flashinfer first:
        # pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
        envs.VLLM_ATTENTION_BACKEND = "FLASHINFER"
        super().setUp()
        # Need to download this quantized mode first:
        # gs://cohere-icebox/tif/benchmark_test/trash_7b_q_model
        self.args.model_dir = "/host/engines/7b-FP8/"
        self.args.tokenizer = "/host/engines/7b-FP8/"
        # kv cache block is 64.5k ish
        self.args.quantization = "fp8"
        self.args.kv_cache_dtype = "auto"


class FlashInferFP8KV8SanityTest(SanityTest):
    def setUp(self):
        # need to install flashinfer first:
        # pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
        envs.VLLM_ATTENTION_BACKEND = "FLASHINFER"
        super().setUp()
        # Need to download this quantized mode first:
        # gs://cohere-icebox/tif/benchmark_test/trash_7b_q_model
        self.args.model_dir = "/host/engines/7b-FP8/"
        self.args.tokenizer = "/host/engines/7b-FP8/"
        # kv cache block is 64.5k ish
        self.args.quantization = "fp8"
        self.args.kv_cache_dtype = "fp8"


if __name__ == "__main__":
    unittest.main()
