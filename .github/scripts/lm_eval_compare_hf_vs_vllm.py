import argparse
import os
from typing import Dict, List, Tuple

import lm_eval
import lm_eval.models.utils
import numpy as np
import scipy.stats

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def calculate_z_value(res1: Dict, res2: Dict) -> Tuple[float, float]:
    acc1, acc2 = res1["acc,none"], res2["acc,none"]
    st_err1, st_err2 = res1["acc_stderr,none"], res2["acc_stderr,none"]
    Z = (acc1 - acc2) / np.sqrt((st_err1**2) + (st_err2**2))
    # Determining the p-value
    p_value = 2 * scipy.stats.norm.sf(abs(Z))  # two-tailed test
    return Z, p_value


def print_results(data_to_print: List = None,
                  results_dict: Dict = None,
                  alpha: float = None):
    model1_data, model2_data = data_to_print
    for task in model1_data:
        print(f"Task: {task}")
        print(f"HF Accuracy: {model1_data[task]['acc,none']}")
        print(f"vLLM Accuracy: {model2_data[task]['acc,none']}")
        print(f"HF StdErr: {model1_data[task]['acc_stderr,none']}")
        print(f"vLLM StdErr: {model2_data[task]['acc_stderr,none']}")
        z = results_dict[task]["z"]
        p_value = results_dict[task]["p_value"]
        result = "PASS" if p_value > alpha else "FAIL"
        print(f"Z-Score: {z}, P-Value: {p_value}, p > {alpha}: {result}\n")


def check_passing_score(results_dict: Dict = None,
                        alpha: float = None) -> bool:
    for task in results_dict:
        p_value = task["p_value"]
        if p_value <= alpha:
            return False
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_pretrained",
                        default="EleutherAI/pythia-70m",
                        help="name of model to compare as baseline")
    parser.add_argument("--vllm_pretrained",
                        default="EleutherAI/pythia-70m",
                        help="name of model to compare as difference")
    parser.add_argument("--hf_args",
                        help="huggingface model args <arg>=<value>",
                        default="")
    parser.add_argument("--vllm_args",
                        help="vllm model args <arg>=<value>",
                        default="")
    parser.add_argument("--tasks", type=str, default="arc_easy,hellaswag")
    parser.add_argument(
        "--limit",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for two-tailed z-test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=4,
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks = args.tasks.split(",")
    print("Tasks:", tasks)
    hf_args, vllm_args = "," + args.hf_args, "," + args.vllm_args
    results_hf = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.hf_pretrained}" + hf_args,
        tasks=tasks,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch,
    )
    lm_eval.models.utils.clear_torch_cache()
    print("Memory stats cleared")
    results_vllm = lm_eval.simple_evaluate(
        model="vllm",
        model_args=f"pretrained={args.vllm_pretrained}" + vllm_args,
        tasks=tasks,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch,
    )
    all_res = {}
    for task1, task2 in zip(results_hf["results"].items(),
                            results_vllm["results"].items()):
        assert task1[0] == task2[0]
        z, p_value = calculate_z_value(task1[1], task2[1])
        all_res[task1[0]] = {"z": z, "p_value": p_value}
    print_results([results_hf["results"], results_vllm["results"]], all_res,
                  args.alpha)
    if not check_passing_score:
        print("Accuracy test failed!")
        exit(1)
