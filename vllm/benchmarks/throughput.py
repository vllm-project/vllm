# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark offline inference throughput."""

import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from typing import Any

import torch
import uvloop
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import (
    AIMODataset,
    BurstGPTDataset,
    ConversationDataset,
    InstructCoderDataset,
    MultiModalConversationDataset,
    PrefixRepetitionRandomDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.utils.async_utils import merge_async_iterators


def run_vllm(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    do_profile: bool,
    disable_detokenize: bool = False,
) -> tuple[float, list[RequestOutput] | None]:
    from vllm import LLM, SamplingParams

    llm = LLM(**dataclasses.asdict(engine_args))
    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of"
        " prompt_len and expected_output_len for all requests."
    )
    # Add the requests to the engine.
    prompts: list[TextPrompt | TokensPrompt] = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompt = (
            TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"])
            if "prompt_token_ids" in request.prompt
            else TextPrompt(prompt=request.prompt)
        )
        if request.multi_modal_data:
            assert isinstance(request.multi_modal_data, dict)
            prompt["multi_modal_data"] = request.multi_modal_data
        prompts.append(prompt)

        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            )
        )
    lora_requests: list[LoRARequest] | None = None
    if engine_args.enable_lora:
        lora_requests = [request.lora_request for request in requests]

    use_beam_search = False

    outputs = None
    if not use_beam_search:
        start = time.perf_counter()
        if do_profile:
            llm.start_profile()
        outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_requests, use_tqdm=True
        )
        if do_profile:
            llm.stop_profile()
        end = time.perf_counter()
    else:
        assert lora_requests is None, "BeamSearch API does not support LoRA"
        prompts = [request.prompt for request in requests]
        # output_len should be the same for all requests.
        output_len = requests[0].expected_output_len
        for request in requests:
            assert request.expected_output_len == output_len
        start = time.perf_counter()
        if do_profile:
            llm.start_profile()
        llm.beam_search(
            prompts,
            BeamSearchParams(
                beam_width=n,
                max_tokens=output_len,
                ignore_eos=True,
            ),
        )
        if do_profile:
            llm.stop_profile()
        end = time.perf_counter()
    return end - start, outputs


def run_vllm_chat(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    do_profile: bool,
    disable_detokenize: bool = False,
) -> tuple[float, list[RequestOutput]]:
    """
    Run vLLM chat benchmark. This function is recommended ONLY for benchmarking
    multimodal models as it properly handles multimodal inputs and chat
    formatting. For non-multimodal models, use run_vllm() instead.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(**dataclasses.asdict(engine_args))

    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of "
        "prompt_len and expected_output_len for all requests."
    )

    prompts = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(request.prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            )
        )
    start = time.perf_counter()
    if do_profile:
        llm.start_profile()
    outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
    if do_profile:
        llm.stop_profile()
    end = time.perf_counter()
    return end - start, outputs


async def run_vllm_async(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    do_profile: bool,
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
) -> float:
    from vllm import SamplingParams
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args,
    )

    async with build_async_engine_client_from_engine_args(
        engine_args,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
    ) as llm:
        model_config = llm.model_config
        assert all(
            model_config.max_model_len
            >= (request.prompt_len + request.expected_output_len)
            for request in requests
        ), (
            "Please ensure that max_model_len is greater than the sum of"
            " prompt_len and expected_output_len for all requests."
        )

        # Add the requests to the engine.
        prompts: list[TextPrompt | TokensPrompt] = []
        sampling_params: list[SamplingParams] = []
        lora_requests: list[LoRARequest | None] = []
        for request in requests:
            prompt = (
                TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"])
                if "prompt_token_ids" in request.prompt
                else TextPrompt(prompt=request.prompt)
            )

            if request.multi_modal_data:
                assert isinstance(request.multi_modal_data, dict)
                prompt["multi_modal_data"] = request.multi_modal_data

            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                    detokenize=not disable_detokenize,
                )
            )
            prompts.append(prompt)
            lora_requests.append(request.lora_request)

        generators = []
        start = time.perf_counter()
        if do_profile:
            await llm.start_profile()
        for i, (prompt, sp, lr) in enumerate(
            zip(prompts, sampling_params, lora_requests)
        ):
            generator = llm.generate(prompt, sp, lora_request=lr, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        if do_profile:
            await llm.stop_profile()
        end = time.perf_counter()
        return end - start


def run_hf(
    requests: list[SampleRequest],
    model: str,
    tokenizer: TokenizerLike,
    n: int,
    max_batch_size: int,
    trust_remote_code: bool,
    disable_detokenize: bool = False,
) -> float:
    assert isinstance(tokenizer, PreTrainedTokenizerBase), (
        "the hf backend only supports HF tokenizers"
    )
    llm = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.float16, trust_remote_code=trust_remote_code
    )
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: list[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt = requests[i].prompt
        prompt_len = requests[i].prompt_len
        output_len = requests[i].expected_output_len
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            next_prompt_len = requests[i + 1].prompt_len
            next_output_len = requests[i + 1].expected_output_len
            if (
                max(max_prompt_len, next_prompt_len)
                + max(max_output_len, next_output_len)
            ) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=True,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        if not disable_detokenize:
            # Include the decoding time.
            tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={
            k: results[k] for k in ["elapsed_time", "num_requests", "total_num_tokens"]
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def get_requests(args, tokenizer):
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs = {
        "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }

    if args.dataset_name == "random" or (
        args.dataset_path is None
        and args.dataset_name not in {"prefix_repetition", "random-mm", "random-rerank"}
    ):
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
        if args.backend == "vllm-chat":
            sample_kwargs["enable_multimodal_chat"] = True
    elif args.dataset_name == "sonnet":
        assert tokenizer.chat_template or tokenizer.default_chat_template, (
            "Tokenizer/model must have chat template for sonnet dataset."
        )
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset
    elif args.dataset_name == "hf":
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = VisionArenaDataset
            common_kwargs["dataset_subset"] = None
            common_kwargs["dataset_split"] = "train"
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = InstructCoderDataset
            common_kwargs["dataset_split"] = "train"
        elif args.dataset_path in MultiModalConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = MultiModalConversationDataset
            common_kwargs["dataset_subset"] = args.hf_subset
            common_kwargs["dataset_split"] = args.hf_split
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs["dataset_subset"] = args.hf_subset
            common_kwargs["dataset_split"] = args.hf_split
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = AIMODataset
            common_kwargs["dataset_subset"] = None
            common_kwargs["dataset_split"] = "train"
    elif args.dataset_name == "prefix_repetition":
        dataset_cls = PrefixRepetitionRandomDataset
        sample_kwargs["prefix_len"] = args.prefix_repetition_prefix_len
        sample_kwargs["suffix_len"] = args.prefix_repetition_suffix_len
        sample_kwargs["num_prefixes"] = args.prefix_repetition_num_prefixes
        sample_kwargs["output_len"] = args.prefix_repetition_output_len
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    requests = dataset_cls(**common_kwargs).sample(**sample_kwargs)
    requests = filter_requests_for_dp(requests, args.data_parallel_size)
    return requests


def filter_requests_for_dp(requests, data_parallel_size):
    # Note(zhuohan): The way we get data_parallel_rank is hacky and only
    # works for external launcher mode. Should be cleaned up and deprecated
    # in the future with a better vLLM distributed process design.
    if data_parallel_size == 1:
        return requests

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    data_parallel_rank = global_rank // (world_size // data_parallel_size)
    return [
        r
        for i, r in enumerate(requests)
        if i % data_parallel_size == data_parallel_rank
    ]


def validate_args(args):
    """
    Validate command-line arguments.
    """

    # === Deprecation and Defaulting ===
    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next release. "
            "Please use '--dataset-name' and '--dataset-path' instead.",
            stacklevel=2,
        )
        args.dataset_path = args.dataset

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model

    # === Backend Validation ===
    valid_backends = {"vllm", "hf", "mii", "vllm-chat"}
    if args.backend not in valid_backends:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # === Dataset Configuration ===
    if (
        not args.dataset
        and not args.dataset_path
        and args.dataset_name not in {"prefix_repetition"}
    ):
        print("When dataset path is not set, it will default to random dataset")
        args.dataset_name = "random"
        if args.input_len is None:
            raise ValueError("input_len must be provided for a random dataset")

    # === Dataset Name Specific Checks ===
    # --hf-subset and --hf-split: only used
    # when dataset_name is 'hf'
    if args.dataset_name != "hf" and (
        getattr(args, "hf_subset", None) is not None
        or getattr(args, "hf_split", None) is not None
    ):
        warnings.warn(
            "--hf-subset and --hf-split will be ignored \
                since --dataset-name is not 'hf'.",
            stacklevel=2,
        )
    elif args.dataset_name == "hf":
        if args.dataset_path in (
            VisionArenaDataset.SUPPORTED_DATASET_PATHS.keys()
            | MultiModalConversationDataset.SUPPORTED_DATASET_PATHS
            | ConversationDataset.SUPPORTED_DATASET_PATHS
        ):
            assert args.backend == "vllm-chat", (
                f"{args.dataset_path} needs to use vllm-chat as the backend."
            )
        elif args.dataset_path in (
            InstructCoderDataset.SUPPORTED_DATASET_PATHS
            | AIMODataset.SUPPORTED_DATASET_PATHS
        ):
            assert args.backend == "vllm", (
                f"{args.dataset_path} needs to use vllm as the backend."
            )
        else:
            raise ValueError(f"{args.dataset_path} is not supported by hf dataset.")

    # --random-range-ratio: only used when dataset_name is 'random'
    if args.dataset_name != "random" and args.random_range_ratio is not None:
        warnings.warn(
            "--random-range-ratio will be ignored since \
                --dataset-name is not 'random'.",
            stacklevel=2,
        )

    # --prefix-len: only used when dataset_name is 'random', 'sonnet', or not
    # set.
    if (
        args.dataset_name not in {"random", "sonnet", None}
        and args.prefix_len is not None
    ):
        warnings.warn(
            "--prefix-len will be ignored since --dataset-name\
                 is not 'random', 'sonnet', or not set.",
            stacklevel=2,
        )

    # === LoRA Settings ===
    if getattr(args, "enable_lora", False) and args.backend != "vllm":
        raise ValueError("LoRA benchmarking is only supported for vLLM backend")
    if getattr(args, "enable_lora", False) and args.lora_path is None:
        raise ValueError("LoRA path must be provided when enable_lora is True")

    # === Backend-specific Validations ===
    if args.backend == "hf" and args.hf_max_batch_size is None:
        raise ValueError("HF max batch size is required for HF backend")
    if args.backend != "hf" and args.hf_max_batch_size is not None:
        raise ValueError("HF max batch size is only for HF backend.")

    if (
        args.backend in {"hf", "mii"}
        and getattr(args, "quantization", None) is not None
    ):
        raise ValueError("Quantization is only for vLLM backend.")

    if args.backend == "mii" and args.dtype != "auto":
        raise ValueError("dtype must be auto for MII backend.")
    if args.backend == "mii" and args.n != 1:
        raise ValueError("n must be 1 for MII backend.")
    if args.backend == "mii" and args.tokenizer != args.model:
        raise ValueError("Tokenizer must be the same as the model for MII backend.")

    if args.data_parallel_size > 1 and (
        args.distributed_executor_backend != "external_launcher" or args.async_engine
    ):
        # --data-parallel is not supported fully.
        # Old issue: https://github.com/vllm-project/vllm/issues/16222
        # Currently we only support data parallel with external launcher
        # mode (i.e., launch with toruchrun).
        raise ValueError(
            "Data parallel is only supported with external launcher mode "
            "with synchronous engine in offline benchmark, "
            "please use benchmark serving instead"
        )


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "hf", "mii", "vllm-chat"],
        default="vllm",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["sharegpt", "random", "sonnet", "burstgpt", "hf", "prefix_repetition"],
        help="Name of the dataset to benchmark on.",
        default="sharegpt",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in\
            the next release. The dataset is expected to "
        "be a json in form of list[dict[..., conversations: "
        "list[dict[..., value: <prompt_or_response>]]]]",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the "
        "output length from the dataset.",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--hf-max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for HF backend.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the throughput results in JSON format.",
    )
    parser.add_argument(
        "--async-engine",
        action="store_true",
        default=False,
        help="Use vLLM async engine rather than LLM class.",
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        default=False,
        help="Disable decoupled async engine frontend.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize the response (i.e. do not include "
            "detokenization time in the measurement)"
        ),
    )
    # LoRA
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the lora adapters to use. This can be an absolute path, "
        "a relative path, or a Hugging Face model identifier.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before the random "
        "context in a request (default: 0).",
    )
    # random dataset
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for RandomDataset. Must be in the range [0, 1) to define "
        "a symmetric sampling range "
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )

    # hf dtaset
    parser.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset."
    )
    parser.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Use vLLM Profiling. --profiler-config must be provided on the server.",
    )

    # prefix repetition dataset
    prefix_repetition_group = parser.add_argument_group(
        "prefix repetition dataset options"
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-prefix-len",
        type=int,
        default=None,
        help="Number of prefix tokens per request, used only for prefix "
        "repetition dataset.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-suffix-len",
        type=int,
        default=None,
        help="Number of suffix tokens per request, used only for prefix "
        "repetition dataset. Total input length is prefix_len + suffix_len.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-num-prefixes",
        type=int,
        default=None,
        help="Number of prefixes to generate, used only for prefix repetition "
        "dataset. Prompts per prefix is num_requests // num_prefixes.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-output-len",
        type=int,
        default=None,
        help="Number of output tokens per request, used only for prefix "
        "repetition dataset.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)


def main(args: argparse.Namespace):
    validate_args(args)
    if args.seed is None:
        args.seed = 0
    random.seed(args.seed)
    # Sample the requests.
    if (
        args.backend == "hf" or args.backend == "mii"
    ) and args.tokenizer_mode == "auto":
        # mistral_common tokenizer is only supported on vllm and vllm-chat backends;
        # for hf and mii backends, we use hf tokenizer
        args.tokenizer_mode = "hf"
    tokenizer = get_tokenizer(
        args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    requests = get_requests(args, tokenizer)
    is_multi_modal = any(request.multi_modal_data is not None for request in requests)
    request_outputs: list[RequestOutput] | None = None
    if args.backend == "vllm":
        if args.async_engine:
            elapsed_time = uvloop.run(
                run_vllm_async(
                    requests,
                    args.n,
                    AsyncEngineArgs.from_cli_args(args),
                    disable_frontend_multiprocessing=args.disable_frontend_multiprocessing,
                    disable_detokenize=args.disable_detokenize,
                    do_profile=args.profile,
                )
            )
        else:
            elapsed_time, request_outputs = run_vllm(
                requests,
                args.n,
                EngineArgs.from_cli_args(args),
                disable_detokenize=args.disable_detokenize,
                do_profile=args.profile,
            )
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        if args.profile:
            raise NotImplementedError("Profiling not implemented yet for backend='hf'.")
        elapsed_time = run_hf(
            requests,
            args.model,
            tokenizer,
            args.n,
            args.hf_max_batch_size,
            args.trust_remote_code,
            args.disable_detokenize,
        )
    elif args.backend == "vllm-chat":
        elapsed_time, request_outputs = run_vllm_chat(
            requests,
            args.n,
            EngineArgs.from_cli_args(args),
            disable_detokenize=args.disable_detokenize,
            do_profile=args.profile,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if request_outputs:
        # Note: with the vllm and vllm-chat backends,
        # we have request_outputs, which we use to count tokens.
        total_prompt_tokens = 0
        total_output_tokens = 0
        for ro in request_outputs:
            if not isinstance(ro, RequestOutput):
                continue
            total_prompt_tokens += (
                len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
            )
            total_output_tokens += sum(len(o.token_ids) for o in ro.outputs if o)
        total_num_tokens = total_prompt_tokens + total_output_tokens
    else:
        total_num_tokens = sum(r.prompt_len + r.expected_output_len for r in requests)
        total_output_tokens = sum(r.expected_output_len for r in requests)
        total_prompt_tokens = total_num_tokens - total_output_tokens

    if is_multi_modal and args.backend != "vllm-chat":
        print(
            "\033[91mWARNING\033[0m: Multi-modal request with "
            f"{args.backend} backend detected. The "
            "following metrics are not accurate because image tokens are not"
            " counted. See vllm-project/vllm/issues/9778 for details."
        )
        # TODO(vllm-project/vllm/issues/9778): Count multi-modal token length.
        # vllm-chat backend counts the image tokens now

    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
    )
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)
