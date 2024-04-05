import dataclasses
import os
import subprocess
import sys
import time
from typing import List

import lm_eval
import lm_eval.models.utils
import numpy
import pytest
import ray
import requests
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = "dummy"

MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds
BASE_URL = "http://localhost:8000"


@dataclasses.dataclass
class Metric:
    name: str
    value: str


@dataclasses.dataclass
class Task:
    name: str
    metrics: List[Metric]


@dataclasses.dataclass
class EvalDefinition:
    tasks: List[Task]
    enable_tensor_parallel: bool = False
    extra_args: List[str] = dataclasses.field(default_factory=list)


# Each entry in this dictionary holds a model id as the key and an
# EvalDefinition as a value. The EvalDefinition holds a list of Tasks
# to evaluate the models on, each with their own pre-recorded Metrics
MODEL_TEST_POINTS = [
    # Llama 2 7B: FP16, FP16 sparse, marlin
    ("NousResearch/Llama-2-7b-chat-hf",
     EvalDefinition(tasks=[
         Task("gsm8k",
              metrics=[
                  Metric("exact_match,strict-match", 0.2266868840030326),
                  Metric("exact_match,flexible-extract", 0.22820318423047764)
              ])
     ])),
    ("neuralmagic/Llama-2-7b-pruned50-retrained-ultrachat",
     EvalDefinition(tasks=[
         Task("gsm8k",
              metrics=[
                  Metric("exact_match,strict-match", 0.09855951478392722),
                  Metric("exact_match,flexible-extract", 0.10083396512509477)
              ])
     ],
                    extra_args=["--sparsity", "sparse_w16a16"])),
    ("neuralmagic/llama-2-7b-chat-marlin",
     EvalDefinition(tasks=[
         Task("gsm8k",
              metrics=[
                  Metric("exact_match,strict-match", 0.14101592115238817),
                  Metric("exact_match,flexible-extract", 0.1652767247915087)
              ])
     ],
                    enable_tensor_parallel=False)),
    # Mistral 7B: FP16, FP16 sparse, marlin
    ("teknium/OpenHermes-2.5-Mistral-7B",
     EvalDefinition(tasks=[
         Task("gsm8k",
              metrics=[
                  Metric("exact_match,strict-match", 0.6004548900682335),
                  Metric("exact_match,flexible-extract", 0.6482183472327521)
              ])
     ])),
    ("neuralmagic/OpenHermes-2.5-Mistral-7B-pruned50",
     EvalDefinition(tasks=[
         Task("gsm8k",
              metrics=[
                  Metric("exact_match,strict-match", 0.4935557240333586),
                  Metric("exact_match,flexible-extract", 0.5269143290371494)
              ])
     ],
                    extra_args=["--sparsity", "sparse_w16a16"])),
    ("neuralmagic/OpenHermes-2.5-Mistral-7B-marlin",
     EvalDefinition(tasks=[
         Task("gsm8k",
              metrics=[
                  Metric("exact_match,strict-match", 0.4935557240333586),
                  Metric("exact_match,flexible-extract", 0.5868081880212282)
              ])
     ],
                    enable_tensor_parallel=False)),
    # Phi 2: marlin
    ("neuralmagic/phi-2-super-marlin",
     EvalDefinition(tasks=[
         Task("gsm8k",
              metrics=[
                  Metric("exact_match,strict-match", 0.49962092494313876),
                  Metric("exact_match,flexible-extract", 0.5041698256254739)
              ])
     ],
                    enable_tensor_parallel=False)),
    # Mixtral: FP16
    ("mistralai/Mixtral-8x7B-Instruct-v0.1",
     EvalDefinition(tasks=[
         Task("gsm8k",
              metrics=[
                  Metric("exact_match,strict-match", 0.6550416982562547),
                  Metric("exact_match,flexible-extract", 0.6603487490523123)
              ])
     ],
                    enable_tensor_parallel=True)),
]


@ray.remote(num_gpus=torch.cuda.device_count())
class ServerRunner:

    def __init__(self, args):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            ["python3", "-m", "vllm.entrypoints.openai.api_server"] + args,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server()

    def ready(self):
        return True

    def _wait_for_server(self):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get(f"{BASE_URL}/health").status_code == 200:
                    break
            except Exception as err:
                if self.proc.poll() is not None:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > MAX_SERVER_START_WAIT_S:
                    raise RuntimeError(
                        "Server failed to start in time.") from err

    def __del__(self):
        if hasattr(self, "proc"):
            self.proc.terminate()


class ServerContextManager:

    def __init__(self, args: List[str]):
        self.args = args
        self.server_runner = None

    def __enter__(self):
        ray.init(ignore_reinit_error=True)
        self.server_runner = ServerRunner.remote([*self.args])
        ray.get(self.server_runner.ready.remote())
        return self.server_runner

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.server_runner is not None:
            del self.server_runner
        ray.shutdown()


@pytest.mark.parametrize("model_id, eval_def", MODEL_TEST_POINTS)
def test_lm_eval_correctness(model_id, eval_def):

    vllm_args = ["--model", model_id, "--disable-log-requests"]

    if eval_def.enable_tensor_parallel:
        tp = torch.cuda.device_count()
        print(f"Enabling tensor parallelism with {tp} devices")
        vllm_args += ["--tensor-parallel-size", str(tp)]

    if eval_def.extra_args:
        vllm_args += eval_def.extra_args

    openai_args = ",".join([
        f"model={model_id}", "tokenizer_backend=huggingface",
        f"base_url={BASE_URL}/v1"
    ])

    with ServerContextManager(vllm_args) as _:
        task_names = [t.name for t in eval_def.tasks]
        results = lm_eval.simple_evaluate(model="local-completions",
                                          model_args=openai_args,
                                          tasks=task_names,
                                          batch_size=64)

    # Clear out model memory
    lm_eval.models.utils.clear_torch_cache()

    for task in eval_def.tasks:
        for metric in task.metrics:
            ground_truth = metric.value
            measured_value = results["results"][task.name][metric.name]
            print(
                f"{task.name} {metric.name}:\n"
                f"ground_truth={ground_truth} measured_value={measured_value}")

            # Metrics must be within 1% of the larger of the two values
            # This corresponds to a 99% accuracy threshold
            assert numpy.isclose(ground_truth, measured_value, rtol=0.01)
