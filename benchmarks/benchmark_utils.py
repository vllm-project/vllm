# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import warnings
from typing import Any

from benchmark_dataset import (BurstGPTDataset, ConversationDataset,
                               InstructCoderDataset, RandomDataset,
                               SampleRequest, ShareGPTDataset, SonnetDataset,
                               VisionArenaDataset)


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = extra_info["tensor_parallel_size"]

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)


def get_requests(num_requests: int, args: argparse.Namespace,
                 tokenizer: Any) -> list[SampleRequest]:
    """
    Sample the requests for the benchmark.
    """
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs = {
        "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "num_requests": num_requests,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }

    if args.dataset_path is None or args.dataset_name == "random":
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
        if getattr(args, "backend", False) and args.backend == "vllm-chat":
            sample_kwargs["enable_multimodal_chat"] = True
    elif args.dataset_name == "sonnet":
        assert tokenizer.chat_template or tokenizer.default_chat_template, (
            "Tokenizer/model must have chat template for sonnet dataset.")
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset
    elif args.dataset_name == "hf":
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = VisionArenaDataset
            common_kwargs['dataset_subset'] = None
            common_kwargs['dataset_split'] = "train"
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = InstructCoderDataset
            common_kwargs['dataset_split'] = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs['dataset_subset'] = args.hf_subset
            common_kwargs['dataset_split'] = args.hf_split
            sample_kwargs["enable_multimodal_chat"] = True

    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    return dataset_cls(**common_kwargs).sample(**sample_kwargs)


def validate_dataset(args: argparse.Namespace, ):
    """
    Validate the dataset arguments.
    """
    # === Dataset Configuration ===
    if not args.dataset_path:
        print(
            "When dataset path is not set, it will default to random dataset")
        args.dataset_name = 'random'
        if args.input_len is None:
            raise ValueError("input_len must be provided for a random dataset")

    # === Dataset Name Specific Checks ===
    # --hf-subset and --hf-split: only used
    # when dataset_name is 'hf'
    if args.dataset_name != "hf" and (
            getattr(args, "hf_subset", None) is not None
            or getattr(args, "hf_split", None) is not None):
        warnings.warn("--hf-subset and --hf-split will be ignored \
                since --dataset-name is not 'hf'.",
                      stacklevel=2)
    elif args.dataset_name == "hf":
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            assert args.backend == "vllm-chat", "VisionArenaDataset needs to use vllm-chat as the backend."  #noqa: E501
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            assert args.backend == "vllm", "InstructCoder dataset needs to use vllm as the backend."  #noqa: E501
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            assert args.backend == "vllm-chat", "ConversationDataset needs to use vllm-chat as the backend."  #noqa: E501
        else:
            raise ValueError(
                f"{args.dataset_path} is not supported by hf dataset.")

    # --random-range-ratio: only used when dataset_name is 'random'
    if args.dataset_name != 'random' and args.random_range_ratio is not None:
        warnings.warn("--random-range-ratio will be ignored since \
                --dataset-name is not 'random'.",
                      stacklevel=2)

    # --prefix-len: only used when dataset_name is 'random', 'sonnet', or not
    # set.
    if args.dataset_name not in {"random", "sonnet", None
                                 } and args.prefix_len is not None:
        warnings.warn("--prefix-len will be ignored since --dataset-name\
                 is not 'random', 'sonnet', or not set.",
                      stacklevel=2)
