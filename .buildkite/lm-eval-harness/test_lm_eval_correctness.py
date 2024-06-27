import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy
import pytest
import requests
import yaml

from vllm.utils import cuda_device_count_stateless

if TYPE_CHECKING:
    import lm_eval as lm_eval_t

# requires a particular lm-evaluation-harness
# pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@9516087b81a61d0e220b22cc1b75be76de23bc10
lm_eval: "lm_eval_t" = pytest.importorskip("lm_eval",
                                           reason="lm_eval required")

RTOL = 0.02
TEST_DATA_FILE = os.environ.get(
    "LM_EVAL_TEST_DATA_FILE",
    ".buildkite/lm-eval-harness/small-models.yaml")


def wait_for_server(timeout=900) -> bool:

    def try_connection() -> bool:
        try:
            r = requests.get("http://localhost:8000/health")
            return r.status_code == 200
        except Exception as _:
            return False

    timeout_part = 15  # retry every 15 seconds
    time_waited = 0
    while time_waited <= timeout:
        time.sleep(timeout_part)
        if try_connection():
            return True
        time_waited = time_waited + timeout_part

    return False


def launch_lm_eval(eval_config):
    model_args=f"pretrained={eval_config['model_name']},tensor_parallel_size={cuda_device_count_stateless()}"

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
    )

    return results


def test_lm_eval_correctness():
    eval_config = yaml.safe_load(
        Path(TEST_DATA_FILE).read_text(encoding="utf-8"))

    # Launch eval requests.
    results = launch_lm_eval(eval_config)

    # Confirm scores match ground truth.
    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][
                metric["name"]]
            print(
                f'{task["name"]} | {metric["name"]}: '
                f'ground_truth={ground_truth} | measured={measured_value}')
            assert numpy.isclose(ground_truth, measured_value, rtol=RTOL)
