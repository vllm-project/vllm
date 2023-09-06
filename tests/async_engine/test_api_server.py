import time
import subprocess
import sys
import pytest
import requests
from multiprocessing import Pool

def _query_server(prompt: str)->dict:
    response = requests.post("http://localhost:8000/generate", json={"prompt": prompt, "max_tokens": 100, "temperature": 0, "ignore_eos": True})
    response.raise_for_status()
    return response.json()

@pytest.fixture
def api_server():
    # See if we can import it
    import vllm.entrypoints.api_server  # pylint: disable=unused-import
    uvicorn_process = subprocess.Popen([sys.executable, "-u", "-m", "vllm.entrypoints.api_server", "--model", "facebook/opt-125m"])
    yield
    uvicorn_process.terminate()

def test_api_server(api_server):
    """
    Run the API server and test it.

    We run both the server and requests in separate processes.

    We test that the server can handle incoming requests, including
    multiple requests at the same time, and that it can handle requests
    being cancelled without crashing.
    """
    with Pool(32) as pool:
        # Wait until the server is ready
        prompts = ["Hello world"] * 1
        result = None
        while not result:
            try:
                for result in pool.map(_query_server, prompts):
                    break
            except:
                time.sleep(1)

        # Actual tests start here
        # Try with 1 prompt
        for result in pool.map(_query_server, prompts):
            assert result

        # Try with 100 prompts
        prompts = ["Hello world"] * 100
        for result in pool.map(_query_server, prompts):
            assert result

        # Cancel requests
        pool.map_async(_query_server, prompts)
        time.sleep(0.01)
        pool.terminate()
        pool.join()

    with Pool(32) as pool:
        # Try with 100 prompts
        prompts = ["Hello world"] * 100
        for result in pool.map(_query_server, prompts):
            assert result