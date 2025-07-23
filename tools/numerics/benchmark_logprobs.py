# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import math
from statistics import mean, median, stdev

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

PROMPTS = [
    "One of the most important things in life is to",
    "The answer to 1 + 1 is",
]
vllm_logits_processor_outputs = []


def print_error_stats(header: str, errs: list[float]):
    name_to_func = {
        "max": max,
        "mean": mean,
        "stdev": stdev,
        "median": median,
        "min": min
    }
    print(
        f"{header}:", ", ".join(f"{stat_type}={name_to_func[stat_type](errs)}"
                                for stat_type in name_to_func))


def get_vllm_outputs(args: EngineArgs, temperature: float):
    llm = LLM(**dataclasses.asdict(args))
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    def logits_processor_hook(module, input, output):
        assert isinstance(output, torch.Tensor)
        vllm_logits_processor_outputs.append(output.clone())

    model.logits_processor.register_forward_hook(logits_processor_hook)

    outputs = llm.generate(
        PROMPTS,
        sampling_params=SamplingParams(
            max_tokens=512,
            temperature=temperature,
            logprobs=2,
        ),
    )
    final_outputs = []
    for output in outputs:
        assert len(output.outputs[0].token_ids) == len(
            output.outputs[0].logprobs)
        final_outputs.append({
            "input_ids": output.prompt_token_ids,
            "output_ids": output.outputs[0].token_ids,
            "logprobs": output.outputs[0].logprobs,
        })
    return final_outputs


def compare_with_hf(vllm_outputs, args: EngineArgs, temperature: float):
    model_config = args.create_model_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_config.dtype, device_map="cuda")

    vllm_errs = []
    hook_errs = []
    vllm_prob_errs = []
    hook_prob_errs = []
    hook_log_name = "vLLM w/ F.log_softmax"
    for seq_id, output in enumerate(vllm_outputs):
        token_ids = torch.tensor([*output["input_ids"], *output["output_ids"]],
                                 device="cuda").unsqueeze(0)
        with torch.inference_mode():
            hf_outputs = model(token_ids)
        hf_logprobs = F.log_softmax(hf_outputs.logits / temperature, dim=-1)

        for i in range(len(output["logprobs"])):
            hook_logprobs = F.log_softmax(
                vllm_logits_processor_outputs[i][seq_id] / temperature, dim=-1)
            for key in output["logprobs"][i]:
                _real_logprobs = hf_logprobs[0,
                                             i - 1 + len(output["input_ids"])]
                eps = 1e-10
                vllm_rel_err = abs((output["logprobs"][i][key].logprob -
                                    _real_logprobs[key].item()) /
                                   (_real_logprobs[key].item() + eps))
                hook_rel_err = abs(
                    (hook_logprobs[key].item() - _real_logprobs[key].item()) /
                    (_real_logprobs[key].item() + eps))
                vllm_errs.append(vllm_rel_err)
                hook_errs.append(hook_rel_err)

                vllm_prob = math.exp(output["logprobs"][i][key].logprob)
                hook_prob = math.exp(hook_logprobs[key].item())
                real_prob = math.exp(_real_logprobs[key].item())
                vllm_prob_err = abs(vllm_prob - real_prob)
                hook_prob_err = abs(hook_prob - real_prob)
                vllm_prob_errs.append(vllm_prob_err)
                hook_prob_errs.append(hook_prob_err)

                if (vllm_rel_err > 0.1
                        or hook_rel_err > 0.1) and real_prob < 0.9:
                    print(
                        (i, key),
                        output["logprobs"][i][key],
                        "HF logprobs:",
                        hf_logprobs[0, i - 1 +
                                    len(output["input_ids"])][key].item(),
                        f"{hook_log_name} logprobs:",
                        hook_logprobs[key].item(),
                    )
                    print(f"HF Prob: {real_prob}, vLLM: {vllm_prob}"
                          f", {hook_log_name}: {hook_prob}")
    print("===Relative logprobs errors vs HF===")
    print_error_stats("vLLM", vllm_errs)
    print_error_stats(hook_log_name, hook_errs)
    print("===Absolute probs errors vs HF===")
    print_error_stats("vLLM", vllm_prob_errs)
    print_error_stats(hook_log_name, hook_prob_errs)


def main():
    parser = FlexibleArgumentParser(description="Benchmark logprobs.")
    parser.add_argument("--temperature",
                        type=float,
                        required=False,
                        default=0.7)
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    temperature = args.temperature
    del args.temperature
    engine_args = EngineArgs.from_cli_args(args)
    vllm_outputs = get_vllm_outputs(engine_args, temperature)
    compare_with_hf(vllm_outputs, engine_args, temperature)


if __name__ == "__main__":
    main()
