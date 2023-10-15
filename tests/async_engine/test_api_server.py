import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pytest
import requests


def _query_server(prompt: str, params=None) -> dict:
    if params is None:
        params = {}

    response = requests.post("http://localhost:8000/generate",
                             json={
                                 "prompt": prompt,
                                 "max_tokens": 100,
                                 "temperature": 0,
                                 "ignore_eos": True,
                                 **params
                             })
    response.raise_for_status()
    return response.json()


def _query_server_with_params(args: list) -> dict:
    return _query_server(*args)


@pytest.fixture
def api_server():
    script_path = Path(__file__).parent.joinpath(
        "api_server_async_engine.py").absolute()
    # pylint: disable=consider-using-with
    uvicorn_process = subprocess.Popen([
        sys.executable, "-u",
        str(script_path), "--model", "facebook/opt-125m"
    ])
    yield
    uvicorn_process.terminate()


# pylint: disable=redefined-outer-name, unused-argument
def test_api_server(api_server):
    """
    Run the API server and test it.

    We run both the server and requests in separate processes.

    We test that the server can handle incoming requests, including
    multiple requests at the same time, and that it can handle requests
    being cancelled without crashing.
    """
    # Run parallel requests
    with Pool(32) as pool:
        # Wait until the server is ready
        prompts = ["Hello world"] * 1
        result = None
        while not result:
            # pylint: disable=bare-except
            try:
                for result in pool.map(_query_server, prompts):
                    break
            except:
                time.sleep(1)

        # Actual tests start here

        # Try with 1 prompt
        for result in pool.map(_query_server, prompts):
            assert result, "Empty result"

        # Test server side validation

        # Null prompts must be refused by the server
        try:
            for response in pool.map(_query_server, [None]):
                assert False, (
                    "A null prompt should result in 400 Bad Request, " +
                    f"but it gives a response: {response!r}")
        except requests.exceptions.HTTPError as e:
            assert e.response.status_code == 400

        # Unknown sampling params must be refused by the server
        try:
            for response in pool.map(_query_server_with_params,
                                     [("Dummy prompt", {
                                         "unknown_param": 1
                                     })]):
                assert False, ("Passing an unknown sampling param should " +
                               "result in 400 Bad Request, " +
                               f"but it gives a response: {response!r}")
        except requests.exceptions.HTTPError as e:
            assert e.response.status_code == 400
            assert "unknown_param" in e.response.content.decode("utf-8")

        # Stats
        stats = requests.get("http://localhost:8000/stats").json()
        num_aborted_requests = stats["num_aborted_requests"]
        assert num_aborted_requests == 0

        # Try with 100 prompts
        prompts = ["Hello world"] * 100
        for result in pool.map(_query_server, prompts):
            assert result

        # Cancel requests
        pool.map_async(_query_server, prompts)
        time.sleep(0.01)
        pool.terminate()
        pool.join()

        # check cancellation stats
        stats = requests.get("http://localhost:8000/stats").json()
        num_aborted_requests = stats["num_aborted_requests"]
        assert num_aborted_requests > 0

    # check that server still runs after cancellations
    with Pool(32) as pool:
        # Try with 100 prompts
        prompts = ["Hello world"] * 100
        for result in pool.map(_query_server, prompts):
            assert result
