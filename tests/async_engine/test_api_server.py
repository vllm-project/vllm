import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pytest
import requests


def _query_server(prompt: str, max_tokens: int = 5) -> dict:
    response = requests.post("http://localhost:8000/generate",
                             json={
                                 "prompt": prompt,
                                 "max_tokens": max_tokens,
                                 "temperature": 0,
                                 "ignore_eos": True
                             })
    response.raise_for_status()
    return response.json()


def _query_server_long(prompt: str) -> dict:
    return _query_server(prompt, max_tokens=500)


@pytest.fixture
def api_server(tokenizer_pool_size: int, engine_use_ray: bool,
               worker_use_ray: bool):
    script_path = Path(__file__).parent.joinpath(
        "api_server_async_engine.py").absolute()
    commands = [
        sys.executable, "-u",
        str(script_path), "--model", "facebook/opt-125m", "--host",
        "127.0.0.1", "--tokenizer-pool-size",
        str(tokenizer_pool_size)
    ]
    if engine_use_ray:
        commands.append("--engine-use-ray")
    if worker_use_ray:
        commands.append("--worker-use-ray")
    uvicorn_process = subprocess.Popen(commands)
    yield
    uvicorn_process.terminate()


@pytest.mark.parametrize("tokenizer_pool_size", [0, 2])
@pytest.mark.parametrize("worker_use_ray", [False, True])
@pytest.mark.parametrize("engine_use_ray", [False, True])
def test_api_server(api_server, tokenizer_pool_size: int, worker_use_ray: bool,
                    engine_use_ray: bool):
    """
    Run the API server and test it.

    We run both the server and requests in separate processes.

    We test that the server can handle incoming requests, including
    multiple requests at the same time, and that it can handle requests
    being cancelled without crashing.
    """
    with Pool(32) as pool:
        # Wait until the server is ready
        prompts = ["warm up"] * 1
        result = None
        while not result:
            try:
                for r in pool.map(_query_server, prompts):
                    result = r
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        # Actual tests start here
        # Try with 1 prompt
        for result in pool.map(_query_server, prompts):
            assert result

        num_aborted_requests = requests.get(
            "http://localhost:8000/stats").json()["num_aborted_requests"]
        assert num_aborted_requests == 0

        # Try with 100 prompts
        prompts = ["test prompt"] * 100
        for result in pool.map(_query_server, prompts):
            assert result

    with Pool(32) as pool:
        # Cancel requests
        prompts = ["canceled requests"] * 100
        pool.map_async(_query_server_long, prompts)
        time.sleep(0.01)
        pool.terminate()
        pool.join()

        # check cancellation stats
        # give it some times to update the stats
        time.sleep(1)

        num_aborted_requests = requests.get(
            "http://localhost:8000/stats").json()["num_aborted_requests"]
        assert num_aborted_requests > 0

    # check that server still runs after cancellations
    with Pool(32) as pool:
        # Try with 100 prompts
        prompts = ["test prompt after canceled"] * 100
        for result in pool.map(_query_server, prompts):
            assert result
