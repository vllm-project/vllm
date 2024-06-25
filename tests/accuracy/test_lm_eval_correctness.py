import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy
import pytest
import requests
import yaml

if TYPE_CHECKING:
    import lm_eval as lm_eval_t

# requires a particular lm-evaluation-harness
# pip install lm-eval==0.4.2
lm_eval: "lm_eval_t" = pytest.importorskip("lm_eval",
                                           reason="lm_eval required")

RTOL = 0.02
TEST_DATA_FILE = os.environ.get(
    "LM_EVAL_TEST_DATA_FILE",
    ".buildkite/lm-eval-harness/configs/small-models.txt")


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
    os.environ["OPENAI_API_KEY"] = "dummy"
    openai_args = ",".join([
        f"model={eval_config['model_name']}",
        "tokenizer_backend=huggingface",
        "base_url=http://localhost:8000/v1",
    ])

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=openai_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        batch_size=16,
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
    )

    return results


def test_lm_eval_correctness(num_gpus_available):
    eval_config = yaml.safe_load(
        Path(TEST_DATA_FILE).read_text(encoding="utf-8"))

    # Setup server launch.
    server_args = {
        "model": eval_config["model_name"],
        "max-model-len": 4096,
        "tensor-parallel-size": num_gpus_available,
        # TODO (@robertgshaw2): understand why default (mp) does not
        # shut down cleanly (it works, but not clean).
        "distributed-executor-backend": "ray",
        "disable-log-requests": "",
    }

    server_cmd = "python3 -m vllm.entrypoints.openai.api_server " + \
                    " ".join([f"--{k} {v}"
                                for k, v in server_args.items()])

    try:
        # Launch server.
        server_process = subprocess.Popen("exec " + server_cmd, shell=True)
        assert wait_for_server(), "Server did not start up in time."

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

    finally:
        assert server_process is not None
        server_process.terminate()

        # Make sure the server finishes tearing down.
        time.sleep(10.)
