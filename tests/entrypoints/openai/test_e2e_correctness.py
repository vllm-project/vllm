import os
import subprocess
import time
from threading import Thread

import lm_eval
import numpy
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OPENAI_API_KEY"] = "dummy"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


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


_TASK = "gsm8k"
_SUBTASK = "exact_match,strict-match"
_LIMIT = 250
_NUM_FEWSHOT = 5
_EXPECTED_SCORE = 0.756
_BATCH_SIZE = 10
_NUM_CLIENTS = 3


def launch_lm_eval(idx):
    print(
        f"Running lm_eval thread {idx} ... this will take a minute to start.")

    openai_args = ",".join([
        f"model={MODEL_NAME}",
        "tokenizer_backend=huggingface",
        "base_url=http://localhost:8000/v1",
    ])

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=openai_args,
        tasks=[_TASK],
        batch_size=_BATCH_SIZE,
        num_fewshot=_NUM_FEWSHOT,
        limit=_LIMIT,
    )

    measured_value = results["results"][_TASK][_SUBTASK]
    print(f"measured: {measured_value}")

    assert numpy.isclose(_EXPECTED_SCORE, measured_value, rtol=0.05)


def test_lm_eval_correctness():
    # Setup server launch.
    server_args = {
        "model": MODEL_NAME,
        "max-model-len": 4096,
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

        # Launch N eval jobs to simulate a client under load.
        # Confirm we got the right answer.
        ts = [
            Thread(target=launch_lm_eval, args=(idx, ))
            for idx in range(_NUM_CLIENTS)
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60 * 20)
            assert not t.is_alive()

    finally:
        assert server_process is not None
        server_process.terminate()
