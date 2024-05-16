"""We here have test setting as one controller with two worker.
From controller's perspective, its main duty is to maintain unique central
entrance place, a.k.a service URL, and dispatch request to workers.

To accomplish this job, it need to:
    1. make sure controller could correctly receive worker's join request
    2. make sure worker would repeatly sending heart beat signal to controller
    3. make sure remove stale worker if not receive worker's heart beat

And by worker registration, it also could do the job as:
    1. autoscale with new worker of the same serving model join in.
    2. model roll update by gradudtely join new model pod while delete old one.
    3. request load balance by inspect worker's engine pending request len.
"""
import json
import os
import subprocess
import sys
import time

# using Ray for overall ease of process management, parallel requests,
# and debugging.
import ray
import requests
import openai  # use the official client for correctness check
import pytest
from huggingface_hub import snapshot_download
from vllm.utils import get_ip

MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds
# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# technically this needs Mistral-7B-v0.1 as base, but we're not testing
# generation quality here
LORA_NAME = "typeof/zephyr-7b-beta-lora"

# make check fast
CONTROLLER_HEART_BEAT_EXPIRATION = 10
os.environ["VLLM_CONTROLLER_HEART_BEAT_EXPIRATION"] = str(
    CONTROLLER_HEART_BEAT_EXPIRATION)
os.environ["VLLM_WORKER_HEART_BEAT_INTERVAL"] = "5"
host_ip = get_ip()
controller_addr = "{}:8000".format(host_ip)
worker1_addr = "{}:8001".format(host_ip)
worker2_addr = "{}:8002".format(host_ip)

pytestmark = pytest.mark.asyncio


@ray.remote
class Controller:

    def __init__(self, args):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            ["python3", "-m", "vllm.entrypoints.openai.controller"] + args,
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
                if requests.get("http://{}/health".format(
                        controller_addr)).status_code == 200:
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


@ray.remote(num_gpus=1)
class Worker:

    def __init__(self, args, port):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.port = port
        self.proc = subprocess.Popen(
            ["python3", "-m", "vllm.entrypoints.openai.api_server"] + args,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_worker()

    def ready(self):
        return True

    def _wait_for_worker(self):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get("http://{}:{}/health".format(
                        host_ip, self.port)).status_code == 200:
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


@pytest.fixture(scope="module")
def client():
    client = openai.AsyncOpenAI(
        base_url="http://{}/v1".format(controller_addr),
        api_key="token-abc123",
    )

    yield client


@pytest.fixture(scope="session")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="function")
def singleserver(zephyr_lora_files):
    ray.init()
    # start controller first
    controller = Controller.remote(["--host", host_ip])
    ray.get(controller.ready.remote())
    worker = Worker.remote(
        [
            "--model",
            MODEL_NAME,
            # use half precision for speed and memory savings in CI environment
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "512",
            "--host",
            host_ip,
            "--port",
            "8001",
            "--controller",
            controller_addr,
            "--enforce-eager",
            # lora config below
            "--enable-lora",
            "--lora-modules",
            f"zephyr-lora={zephyr_lora_files}",
            f"zephyr-lora2={zephyr_lora_files}",
            "--max-lora-rank",
            "64",
            "--max-cpu-loras",
            "2",
            "--max-num-seqs",
            "128",
        ],
        8001)
    ray.get(worker.ready.remote())

    yield controller
    ray.shutdown()


@pytest.fixture(scope="function")
def doubleserver(zephyr_lora_files):
    ray.init()
    # start controller first
    controller = Controller.remote(["--host", host_ip])
    ray.get(controller.ready.remote())
    worker0 = Worker.remote(
        [
            "--model",
            MODEL_NAME,
            # use half precision for speed and memory savings in CI environment
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "512",
            "--host",
            host_ip,
            "--port",
            "8001",
            "--controller",
            controller_addr,
            "--enforce-eager",
            # lora config below
            "--enable-lora",
            "--lora-modules",
            f"zephyr-lora={zephyr_lora_files}",
            f"zephyr-lora2={zephyr_lora_files}",
            "--max-lora-rank",
            "64",
            "--max-cpu-loras",
            "2",
            "--max-num-seqs",
            "128",
        ],
        8001)
    ray.get(worker0.ready.remote())

    worker1 = Worker.remote(
        [
            "--model",
            MODEL_NAME,
            # use half precision for speed and memory savings in CI environment
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "512",
            "--host",
            host_ip,
            "--port",
            "8002",
            "--controller",
            controller_addr,
            "--enforce-eager",
            # lora config below
            "--enable-lora",
            "--lora-modules",
            f"zephyr-lora={zephyr_lora_files}",
            f"zephyr-lora2={zephyr_lora_files}",
            "--max-lora-rank",
            "64",
            "--max-cpu-loras",
            "2",
            "--max-num-seqs",
            "128",
        ],
        8002)
    ray.get(worker1.ready.remote())

    yield controller
    ray.shutdown()


async def test_check_models(singleserver):
    res = requests.get("http://{}/list_models".format(controller_addr))
    assert res.status_code == 200

    res = json.loads(res.content.decode('utf-8'))
    models = res["models"]
    served_model = models[0]
    lora_models = models[1:]

    assert served_model == MODEL_NAME
    assert lora_models[0] == "zephyr-lora"
    assert lora_models[1] == "zephyr-lora2"


@pytest.mark.parametrize(
    # first test base model, then test loras
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
)
async def test_load_balance(doubleserver, client: openai.AsyncOpenAI,
                            model_name: str):
    res = requests.get("http://{}/list_models".format(controller_addr))
    assert res.status_code == 200

    res = json.loads(res.content.decode('utf-8'))

    res = requests.get("http://{}/list_workers".format(controller_addr))
    assert res.status_code == 200

    res = json.loads(res.content.decode('utf-8'))

    for i in range(5):
        completion = await client.completions.create(
            model=model_name,
            prompt="Hello, my name is",
            max_tokens=5,
            temperature=0.0)

        assert completion.choices[0].text is not None and len(
            completion.choices[0].text) >= 5

    res = requests.get("http://{}/list_workers".format(controller_addr))
    assert res.status_code == 200

    res = json.loads(res.content.decode('utf-8'))
    worker1 = res[worker1_addr]
    worker2 = res[worker2_addr]

    assert abs(worker1["req_cnt"] - worker2["req_cnt"]) <= 1


async def test_join_and_leave(singleserver):
    res = requests.get("http://{}/list_models".format(controller_addr))
    assert res.status_code == 200

    res = json.loads(res.content.decode('utf-8'))

    res = requests.get("http://{}/list_workers".format(controller_addr))
    assert res.status_code == 200
    res = json.loads(res.content.decode('utf-8'))
    assert worker1_addr in res

    worker1 = Worker.remote(
        [
            "--model",
            MODEL_NAME,
            # use half precision for speed and memory savings in CI environment
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "512",
            "--host",
            host_ip,
            "--port",
            "8002",
            "--controller",
            controller_addr,
            "--enforce-eager",
            # lora config below
            "--enable-lora",
            "--lora-modules",
            f"zephyr-lora={zephyr_lora_files}",
            f"zephyr-lora2={zephyr_lora_files}",
            "--max-lora-rank",
            "64",
            "--max-cpu-loras",
            "2",
            "--max-num-seqs",
            "128",
        ],
        8002)
    ray.get(worker1.ready.remote())

    res = requests.get("http://{}/list_workers".format(controller_addr))
    assert res.status_code == 200
    res = json.loads(res.content.decode('utf-8'))

    assert worker1_addr in res
    assert worker2_addr in res

    del worker1
    # make sure controller already take clean up
    time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION * 2)

    res = requests.get("http://{}/list_workers".format(controller_addr))
    assert res.status_code == 200
    res = json.loads(res.content.decode('utf-8'))

    assert worker1_addr in res
    assert worker2_addr not in res
