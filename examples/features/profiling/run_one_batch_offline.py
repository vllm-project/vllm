# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm import LLM, EngineArgs
from vllm.config import ProfilerConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_MAX_TOKENS = 16


def create_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")

    batch_group = parser.add_argument_group("Batch parameters")
    batch_group.add_argument("--batch-size", type=int, default=1)
    batch_group.add_argument("--prompt-size", type=int, default=128)
    batch_group.add_argument("--prompt-prefix", type=str, default="Hello, my name is")

    profile_group = parser.add_argument_group("Profiling parameters")
    profile_group.add_argument(
        "--profile",
        choices=["none", "prefill", "decode", "both"],
        default="none",
    )
    profile_group.add_argument(
        "--profile-dir",
        type=str,
        default="",
        help="Required when --profile is not 'none'.",
    )

    return parser


def _build_prompt(prefix: str, prompt_size: int) -> str:
    if prompt_size <= 0:
        return ""
    if not prefix:
        prefix = " "
    if len(prefix) >= prompt_size:
        return prefix[:prompt_size]
    repeat_count = (prompt_size + len(prefix) - 1) // len(prefix)
    return (prefix * repeat_count)[:prompt_size]


def _build_profiler_config(
    profile: str, profile_dir: str, max_tokens: int
) -> ProfilerConfig | None:
    if profile == "none":
        return None
    if not profile_dir:
        raise ValueError("--profile-dir must be set when profiling is enabled.")
    if profile == "prefill":
        delay_iterations = 0
        max_iterations = 1
    elif profile == "decode":
        delay_iterations = 1
        max_iterations = max(1, max_tokens)
    else:
        delay_iterations = 0
        max_iterations = 0

    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=profile_dir,
        delay_iterations=delay_iterations,
        max_iterations=max_iterations,
    )


def main(args: dict) -> None:
    max_tokens = DEFAULT_MAX_TOKENS
    batch_size = args.pop("batch_size")
    prompt_size = args.pop("prompt_size")
    prompt_prefix = args.pop("prompt_prefix")
    profile = args.pop("profile")
    profile_dir = args.pop("profile_dir")

    profiler_config = _build_profiler_config(profile, profile_dir, max_tokens)
    if profiler_config is not None:
        args["profiler_config"] = profiler_config

    llm = LLM(**args)

    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = max_tokens
    sampling_params.min_tokens = max_tokens
    sampling_params.ignore_eos = True

    prompt = _build_prompt(prompt_prefix, prompt_size)
    prompts = [prompt] * batch_size

    if profile != "none":
        llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    if profile != "none":
        llm.stop_profile()

    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    parser = create_parser()
    main(vars(parser.parse_args()))
