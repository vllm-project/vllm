# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import gc
import math
from argparse import Namespace
from statistics import mean, median, stdev
from typing import Any

import torch
import torch.nn.functional as F
from tabulate import tabulate
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

PROMPTS = [
    "One of the most important things in life is to",
    "The answer to 1 + 1 is",
]

# (model, logprobs_mode) -> (vllm_errs, vllm_prob_errs)
global_results: dict[tuple[str, str], Any] = {}

stat_name_to_func = {
    "max": max,
    "mean": mean,
    "stdev": stdev,
    "median": median,
    "min": min
}


def get_vllm_outputs(
    args: EngineArgs,
    sampling_params: SamplingParams,
) -> list[dict[str, Any]]:
    llm = LLM(**dataclasses.asdict(args))
    outputs = llm.generate(
        PROMPTS,
        sampling_params=sampling_params,
    )
    final_outputs = []
    for output in outputs:
        final_outputs.append({
            "input_ids": output.prompt_token_ids,
            "output_ids": output.outputs[0].token_ids,
            "logprobs": output.outputs[0].logprobs,
        })
    # Release memory for next benchmark
    del llm
    gc.collect()
    return final_outputs


def compare_with_hf(vllm_outputs: list[dict[str, Any]],
                    args: EngineArgs) -> tuple[list[float], list[float]]:
    model_config = args.create_model_config()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_config.dtype,
        device_map="cuda",
        trust_remote_code=True)
    vllm_errs = []
    vllm_prob_errs = []
    eps = 1e-10
    for vllm_output in vllm_outputs:
        token_ids = torch.tensor(
            [*vllm_output["input_ids"], *vllm_output["output_ids"]],
            device="cuda").unsqueeze(0)
        with torch.inference_mode():
            hf_outputs = hf_model(token_ids)
        if "logprobs" in args.logprobs_mode:
            hf_logprobs = F.log_softmax(hf_outputs.logits, dim=-1)
        else:
            hf_logprobs = hf_outputs.logits

        for i in range(len(vllm_output["logprobs"])):
            for key in vllm_output["logprobs"][i]:
                _real_logprobs = hf_logprobs[0, i - 1 +
                                             len(vllm_output["input_ids"])]
                vllm_rel_err = abs((vllm_output["logprobs"][i][key].logprob -
                                    _real_logprobs[key].item()) /
                                   (_real_logprobs[key].item() + eps))
                vllm_errs.append(vllm_rel_err)
                if "logprobs" in args.logprobs_mode:
                    vllm_prob = math.exp(
                        vllm_output["logprobs"][i][key].logprob)
                    real_prob = math.exp(_real_logprobs[key].item())
                    vllm_prob_err = abs(vllm_prob - real_prob)
                    vllm_prob_errs.append(vllm_prob_err)
    # Release memory for next benchmark
    del hf_model
    gc.collect()
    return vllm_errs, vllm_prob_errs


def run_benchmark(args: Namespace, sampling_params: SamplingParams) -> None:
    engine_args = EngineArgs.from_cli_args(args)
    vllm_outputs = get_vllm_outputs(engine_args, sampling_params)
    vllm_errs, vllm_prob_errs = compare_with_hf(vllm_outputs, engine_args)
    global_results[(args.model, args.logprobs_mode)] = (vllm_errs,
                                                        vllm_prob_errs)


def print_results(
    models: list[str],
    logprobs_modes: list[str],
    stats: list[str],
) -> None:
    headers = ["Model"]
    for logprobs_mode in logprobs_modes:
        if "logprobs" in logprobs_mode:
            headers.append(f"{logprobs_mode} (rel err)")
            probs_header = logprobs_mode.replace("logprobs", "probs")
            headers.append(f"{probs_header} (abs err)")
        else:
            headers.append(f"{logprobs_mode} (rel err)")
    for stat in stats:
        stat_func = stat_name_to_func[stat]
        data = []
        print(f"======{stat} stat======")
        data = []
        for model in models:
            row = [model]
            for logprobs_mode in logprobs_modes:
                vllm_errs, vllm_prob_errs = global_results[(model,
                                                            logprobs_mode)]
                row.append(stat_func(vllm_errs))
                if "logprobs" in logprobs_mode:
                    row.append(stat_func(vllm_prob_errs))
            data.append(row)
        table = tabulate(data, headers=headers, tablefmt="grid")
        print(table)


def main():
    parser = FlexibleArgumentParser(description="Benchmark logprobs.")
    parser.add_argument("--models",
                        type=str,
                        required=False,
                        help="Comma separated list of models to benchmark.")
    parser.add_argument("--logprobs-modes",
                        type=str,
                        required=False,
                        help="Comma separated list of logprobs modes.")
    parser.add_argument("--stats",
                        type=str,
                        required=False,
                        default="max,mean,stdev,median,min",
                        help="Comma separated list of stats to print.")
    parser.add_argument("--temperature",
                        type=float,
                        required=False,
                        default=0.7)
    parser.add_argument("--num-logprobs", type=int, required=False, default=2)
    parser.add_argument("--max-tokens", type=int, required=False, default=512)
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    assert args is not None

    # Process sampling params
    temperature = args.temperature
    del args.temperature
    num_logprobs = args.num_logprobs
    del args.num_logprobs
    max_tokens = args.max_tokens
    del args.max_tokens
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=num_logprobs,
    )

    # Process benchmark sweeping params
    models = args.models
    del args.models
    logprobs_modes = args.logprobs_modes
    del args.logprobs_modes
    models = [args.model] if models is None else models.split(",")
    logprobs_modes = ([args.logprobs_mode]
                      if logprobs_modes is None else logprobs_modes.split(","))

    # Process stats arg
    stats = args.stats.split(",")
    del args.stats

    # Run benchmark for each model and logprobs mode
    for model in models:
        for logprobs_mode in logprobs_modes:
            args.model = model
            args.logprobs_mode = logprobs_mode
            print(f"Running benchmark for {model} with {logprobs_mode=}")
            run_benchmark(args, sampling_params)
    print_results(models, logprobs_modes, stats)


if __name__ == "__main__":
    main()
