# Full content of the fixed file goes here
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
from typing import List, Literal, Optional, Union

from vllm.engine.arg_utils import add_engine_args
from vllm.entrypoints.llm_args import LLMArgs
from vllm.utils import strtobool


class _DeprecateAction(argparse.Action):
    def __init__(self, *args, replace_with: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_with = replace_with

    def format_message(self, msg):
        return f"DeprecationWarning: {msg}"

    def __call__(self, parser, namespace, values, option_string=None):
        if self.replace_with is None:
            msg = (f"Option '{option_string}' is deprecated and will be removed "
                   "in a future release.")
        else:
            msg = (f"Option '{option_string}' is deprecated and will be replaced by "
                   f"'{self.replace_with}' in a future release.")
        parser.error(self.format_message(msg))


def llm_entrypoint():
    parser = argparse.ArgumentParser(
        description="vLLM Python API main demo.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_engine_args(parser)
    llm_group = parser.add_argument_group("LLM", "Arguments for fine-tuning vLLM")
    # We deprecate and hide those arguments to avoid
    # exposing unnecessary arguments in the main CLI.
    llm_group.add_argument(
        "--enable-lora",
        action="store_true",
        help="Enable LoRA. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    llm_group.add_argument(
        "--max-lora-rank",
        type=int,
        default=None,
        help="Max LoRA rank. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    llm_group.add_argument(
        "--lora-extra-vocab-size",
        type=int,
        default=None,
        help="LoRA extra vocab size. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    llm_group.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="LoRA scale. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    llm_group.add_argument(
        "--max-cpu-loras",
        type=int,
        default=None,
        help="Max number of loras to store in CPU. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    # We can deprecate multi modal as well.
    # NOTE The action=argparse.BooleanOptionalAction fails, as the parser try to
    # set a constant that it does not define. This feature is being removed.
    llm_group.add_argument(
        "--enable-multimodal",
        action="store_true",
        help="Enable multimodal. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    llm_group.add_argument(
        "--image-token-id",
        type=int,
        default=None,
        help="Image token id. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    llm_group.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Image size. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    llm_group.add_argument(
        "--num-image_tokens",
        type=int,
        default=None,
        help="Number of image tokens. (Deprecated; will be removed in a future release.)",
        action=_DeprecateAction,
    )
    args = parser.parse_args()

    # Remove deprecated arguments
    for arg in (
        "enable_lora",
        "max_lora_rank",
        "lora_extra_vocab_size",
        "lora_scale",
        "max_cpu_loras",
        "enable_multimodal",
        "image_token_id",
        "image_size",
        "num_image_tokens",
    ):
        if hasattr(args, arg):
            delattr(args, arg)
    args.worker_use_ray = strtobool(os.environ.get("VLLM_USE_RAY", "False"))

    llm_args = LLMArgs.from_cli_args(args)
    from vllm import LLM

    llm = LLM(llm_args)
    prompts = [
        "Please input your prompt here",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    llm.generate(prompts, sampling_params)


if __name__ == "__main__":
    llm_entrypoint()