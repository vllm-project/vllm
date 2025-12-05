# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import ast
import json
from contextlib import suppress

from vllm.benchmarks.datasets.random import RandomMultiModalDataset

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser  # type: ignore


class _ValidateDatasetArgs(argparse.Action):
    """Argparse action to validate dataset name and path compatibility."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

        # Get current values of both dataset_name and dataset_path
        dataset_name = getattr(namespace, "dataset_name", "random")
        dataset_path = getattr(namespace, "dataset_path", None)

        # Validate the combination
        if dataset_name == "random" and dataset_path is not None:
            parser.error(
                "Cannot use 'random' dataset with --dataset-path. "
                "Please specify the appropriate --dataset-name (e.g., "
                "'sharegpt', 'custom', 'sonnet') for your dataset file: "
                f"{dataset_path}"
            )


def add_dataset_parser(parser: FlexibleArgumentParser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        action=_ValidateDatasetArgs,
        choices=[
            "sharegpt",
            "burstgpt",
            "sonnet",
            "random",
            "random-mm",
            "random-rerank",
            "hf",
            "custom",
            "prefix_repetition",
            "spec_bench",
            "txt-slices",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        action=_ValidateDatasetArgs,
        help="Path to the sharegpt/sonnet dataset, the HF dataset ID if using HF "
        "dataset, or the path/URL to a txt file for the txt-slices dataset.",
    )
    parser.add_argument(
        "--no-oversample",
        action="store_true",
        help="Do not oversample if the dataset has fewer samples than num-prompts.",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Skip applying chat template to prompt for datasets that support it.",
    )
    parser.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Disable shuffling of dataset samples for deterministic ordering.",
    )

    # group for dataset specific arguments
    custom_group = parser.add_argument_group("custom dataset options")
    custom_group.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help="Number of output tokens per request, used only for custom dataset.",
    )

    spec_bench_group = parser.add_argument_group("spec bench dataset options")
    spec_bench_group.add_argument(
        "--spec-bench-output-len",
        type=int,
        default=256,
        help="Num of output tokens per request, used only for spec bench dataset.",
    )
    spec_bench_group.add_argument(
        "--spec-bench-category",
        type=str,
        default=None,
        help="Category for spec bench dataset. If None, use all categories.",
    )

    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )

    blazedit_group = parser.add_argument_group("blazedit dataset options")
    blazedit_group.add_argument(
        "--blazedit-min-distance",
        type=float,
        default=0.0,
        help="Minimum distance for blazedit dataset. Min: 0, Max: 1.0",
    )
    blazedit_group.add_argument(
        "--blazedit-max-distance",
        type=float,
        default=1.0,
        help="Maximum distance for blazedit dataset. Min: 0, Max: 1.0",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, "
        "used for random sampling and txt-slices dataset.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, "
        "used for random sampling and txt-slices dataset.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]."
        ),
    )
    random_group.add_argument(
        "--random-batch-size",
        type=int,
        default=1,
        help=("Batch size for random sampling. Only used for embeddings benchmark."),
    )
    random_group.add_argument(
        "--no-reranker",
        action="store_true",
        help=(
            "Whether the model supports reranking natively."
            " Only used for reranker benchmark."
        ),
    )

    # random multimodal dataset options
    random_mm_group = parser.add_argument_group(
        "random multimodal dataset options extended from random dataset"
    )
    random_mm_group.add_argument(
        "--random-mm-base-items-per-request",
        type=int,
        default=RandomMultiModalDataset.DEFAULT_BASE_ITEMS_PER_REQUEST,
        help=(
            "Base number of multimodal items per request for random-mm. "
            "Actual per-request count is sampled around this base using "
            "--random-mm-num-mm-items-range-ratio."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-num-mm-items-range-ratio",
        type=float,
        default=RandomMultiModalDataset.DEFAULT_NUM_MM_ITEMS_RANGE_RATIO,
        help=(
            "Range ratio r in [0, 1] for sampling items per request. "
            "We sample uniformly from the closed integer range "
            "[floor(n*(1-r)), ceil(n*(1+r))] "
            "where n is the base items per request. "
            "r=0 keeps it fixed; r=1 allows 0 items. The maximum is clamped "
            "to the sum of per-modality limits from "
            "--random-mm-limit-mm-per-prompt. "
            "An error is raised if the computed min exceeds the max."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-limit-mm-per-prompt",
        type=json.loads,
        default=RandomMultiModalDataset.DEFAULT_LIMIT_MM_PER_PROMPT,
        help=(
            "Per-modality hard caps for items attached per request, e.g. "
            '\'{"image": 3, "video": 0}\'. The sampled per-request item '
            "count is clamped to the sum of these limits. When a modality "
            "reaches its cap, its buckets are excluded and probabilities are "
            "renormalized."
            "OBS.: Only image sampling is supported for now."
        ),
    )

    def _parse_mm_bucket_config(v: object) -> dict[tuple[int, int, int], float]:
        # If already a dict (e.g., programmatic call), normalize keys
        def normalize(d: dict) -> dict[tuple[int, int, int], float]:
            out: dict[tuple[int, int, int], float] = {}
            for k, val in d.items():
                key = k
                if isinstance(key, str):
                    with suppress(Exception):
                        key = ast.literal_eval(key)
                if not (
                    isinstance(key, tuple)
                    and len(key) == 3
                    and all(isinstance(x, int) for x in key)
                ):
                    raise ValueError(
                        f"Invalid bucket key {k!r}. Expected tuple (H, W, T)."
                    )
                out[(int(key[0]), int(key[1]), int(key[2]))] = float(val)
            return out

        if isinstance(v, dict):
            return normalize(v)
        if isinstance(v, str):
            # Python literal (supports tuple keys)
            parsed = ast.literal_eval(v)
            if not isinstance(parsed, dict):
                raise ValueError("Bucket config must parse to a dict.")
            return normalize(parsed)
        raise ValueError("Unsupported value for --random-mm-bucket-config.")

    random_mm_group.add_argument(
        "--random-mm-bucket-config",
        type=_parse_mm_bucket_config,
        default=RandomMultiModalDataset.DEFAULT_MM_ITEM_BUCKET_CONFIG,
        help=(
            "The bucket config is a dictionary mapping a multimodal item"
            "sampling configuration to a probability."
            "Currently allows for 2 modalities: images and videos. "
            "An bucket key is a tuple of (height, width, num_frames)"
            "The value is the probability of sampling that specific item. "
            "Example: "
            "--random-mm-bucket-config "
            "{(256, 256, 1): 0.5, (720, 1280, 1): 0.4, (720, 1280, 16): 0.10} "
            "First item: images with resolution 256x256 w.p. 0.5"
            "Second item: images with resolution 720x1280 w.p. 0.4 "
            "Third item: videos with resolution 720x1280 and 16 frames w.p. 0.1"
            "OBS.: If the probabilities do not sum to 1, they are normalized."
            "OBS bis.: Only image sampling is supported for now."
        ),
    )

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-name",
        type=str,
        default=None,
        help=(
            "Name of the dataset on HuggingFace "
            "(e.g., 'lmarena-ai/VisionArena-Chat'). "
            "Specify this if your dataset-path is a local path."
        ),
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    prefix_repetition_group = parser.add_argument_group(
        "prefix repetition dataset options"
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-prefix-len",
        type=int,
        default=256,
        help="Number of prefix tokens per request, used only for prefix "
        "repetition dataset.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-suffix-len",
        type=int,
        default=256,
        help="Number of suffix tokens per request, used only for prefix "
        "repetition dataset. Total input length is prefix_len + suffix_len.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-num-prefixes",
        type=int,
        default=10,
        help="Number of prefixes to generate, used only for prefix repetition "
        "dataset. Prompts per prefix is num_requests // num_prefixes.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for prefix "
        "repetition dataset.",
    )
