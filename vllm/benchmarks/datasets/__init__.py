# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.benchmarks.datasets.abstractions import BenchmarkDataset, SampleRequest
from vllm.benchmarks.datasets.arg_utils import add_dataset_parser
from vllm.benchmarks.datasets.burstgpt import BurstGPTDataset
from vllm.benchmarks.datasets.custom import CustomDataset, SpecBench
from vllm.benchmarks.datasets.huggingface import (
    AIMODataset,
    ASRDataset,
    BlazeditDataset,
    ConversationDataset,
    HuggingFaceDataset,
    InstructCoderDataset,
    MLPerfDataset,
    MMStarDataset,
    MMVUDataset,
    MTBenchDataset,
    MultiModalConversationDataset,
    NextEditPredictionDataset,
    VisionArenaDataset,
)
from vllm.benchmarks.datasets.prefix_repetition import PrefixRepetitionRandomDataset
from vllm.benchmarks.datasets.random import (
    RandomDataset,
    RandomDatasetForReranking,
    RandomMultiModalDataset,
    TxtSlicesDataset,
)
from vllm.benchmarks.datasets.sharegpt import ShareGPTDataset
from vllm.benchmarks.datasets.sonnet import SonnetDataset
from vllm.tokenizers import TokenizerLike

# Global cache for LoRA tokenizers.
lora_tokenizer_cache: dict[int, TokenizerLike] = {}


def get_samples(args, tokenizer) -> list[SampleRequest]:
    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    dataset: BenchmarkDataset
    if args.dataset_name == "custom":
        dataset = CustomDataset(
            dataset_path=args.dataset_path, disable_shuffle=args.disable_shuffle
        )
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            skip_chat_template=args.skip_chat_template,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )

    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(
            dataset_path=args.dataset_path, disable_shuffle=args.disable_shuffle
        )
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
            )
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        hf_kwargs = {}
        dataset_class: type[BenchmarkDataset]
        if (
            args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in VisionArenaDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif (
            args.dataset_path in MMVUDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MMVUDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MMVUDataset
            args.hf_split = "validation"
            args.hf_subset = None
        elif (
            args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in InstructCoderDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MTBenchDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in MultiModalConversationDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MultiModalConversationDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MultiModalConversationDataset
        elif (
            args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in ConversationDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = ConversationDataset
        elif (
            args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in AIMODataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif (
            args.dataset_path in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS  # noqa: E501
            or args.hf_name in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in ASRDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = ASRDataset
            args.hf_split = "train"
        elif args.dataset_path in BlazeditDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = BlazeditDataset
            args.hf_split = "train"
            hf_kwargs = {
                "min_distance": args.blazedit_min_distance,
                "max_distance": args.blazedit_max_distance,
            }
        elif (
            args.dataset_path in MLPerfDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MLPerfDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MLPerfDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in MMStarDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MMStarDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MMStarDataset
            args.hf_split = "val"
            args.hf_subset = None
        else:
            supported_datasets = set(
                [
                    dataset_name
                    for cls in HuggingFaceDataset.__subclasses__()
                    for dataset_name in cls.SUPPORTED_DATASET_PATHS
                ]
            )
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats."
            )

        if dataset_class.IS_MULTIMODAL and not (
            args.backend in ("openai-chat", "openai-audio")
            or "embeddings-" in args.backend
        ):
            # multi-modal benchmark is only available on OpenAI Chat
            # endpoint-type.
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' backends."
            )
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
            no_stream=args.no_stream,
            hf_name=args.hf_name,
            disable_shuffle=args.disable_shuffle,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
            skip_chat_template=args.skip_chat_template,
            **hf_kwargs,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "spec_bench": lambda: SpecBench(
                dataset_path=args.dataset_path,
                category=args.spec_bench_category,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_len=args.spec_bench_output_len,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.sharegpt_output_len,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "random": lambda: RandomDataset(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
                prefix_len=args.common_prefix_len,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
                request_id_prefix=args.request_id_prefix,
                batchsize=args.random_batch_size,
                no_oversample=args.no_oversample,
            ),
            "random-mm": lambda: RandomMultiModalDataset(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                range_ratio=args.random_range_ratio,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                base_items_per_request=args.random_mm_base_items_per_request,
                limit_mm_per_prompt=args.random_mm_limit_mm_per_prompt,
                num_mm_items_range_ratio=args.random_mm_num_mm_items_range_ratio,
                bucket_config=args.random_mm_bucket_config,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "random-rerank": lambda: RandomDatasetForReranking(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                input_len=args.random_input_len,
                range_ratio=args.random_range_ratio,
                request_id_prefix=args.request_id_prefix,
                batchsize=args.random_batch_size,
                is_reranker=not args.no_reranker,
            ),
            "prefix_repetition": lambda: PrefixRepetitionRandomDataset(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.prefix_repetition_prefix_len,
                suffix_len=args.prefix_repetition_suffix_len,
                num_prefixes=args.prefix_repetition_num_prefixes,
                output_len=args.prefix_repetition_output_len,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "txt-slices": lambda: TxtSlicesDataset(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                input_len=args.txt_slices_input_len,
                output_len=args.txt_slices_output_len,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
        }

        try:
            # Enforce endpoint compatibility for multimodal datasets.
            if args.dataset_name == "random-mm" and args.backend not in ["openai-chat"]:
                raise ValueError(
                    "Multi-modal content (images) is only supported on "
                    "'openai-chat' backend."
                )
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    return input_requests


__all__ = [
    "add_dataset_parser",
    "get_samples",
    "BenchmarkDataset",
    "SampleRequest",
    "CustomDataset",
    "SpecBench",
    "HuggingFaceDataset",
    "InstructCoderDataset",
    "MLPerfDataset",
    "MMStarDataset",
    "MMVUDataset",
    "MTBenchDataset",
    "MultiModalConversationDataset",
    "NextEditPredictionDataset",
    "VisionArenaDataset",
    "PrefixRepetitionRandomDataset",
    "RandomDataset",
    "RandomDatasetForReranking",
    "RandomMultiModalDataset",
    "ShareGPTDataset",
    "SonnetDataset",
    "BurstGPTDataset",
    "CustomDataset",
    "SpecBench",
    "HuggingFaceDataset",
    "InstructCoderDataset",
    "MLPerfDataset",
    "MMStarDataset",
    "MMVUDataset",
    "MTBenchDataset",
    "MultiModalConversationDataset",
    "NextEditPredictionDataset",
    "VisionArenaDataset",
    "TxtSlicesDataset",
]
