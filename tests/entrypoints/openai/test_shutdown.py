# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import signal
import subprocess
import sys
import time

import openai
import pytest

from vllm.utils.network_utils import get_open_port

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.mark.asyncio
async def test_shutdown_on_engine_failure():
    """Verify that API returns connection error when server process is killed.

    Starts a vLLM server, kills it to simulate a crash, then verifies that
    subsequent API calls fail appropriately.
    """

    port = get_open_port()

    proc = subprocess.Popen(
        [
            # dtype, max-len etc set so that this can run in CI
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL_NAME,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "128",
            "--enforce-eager",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            "0.05",
            "--max-num-seqs",
            "2",
            "--disable-frontend-multiprocessing",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )

    # Wait for server startup
    start_time = time.time()
    client = openai.AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="dummy",
        max_retries=0,
        timeout=10,
    )

    # Poll until server is ready
    while time.time() - start_time < 30:
        try:
            await client.completions.create(
                model=MODEL_NAME, prompt="Hello", max_tokens=1
            )
            break
        except Exception:
            time.sleep(0.5)
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=1)
                pytest.fail(
                    f"Server died during startup. stdout: {stdout}, stderr: {stderr}"
                )
    else:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("Server failed to start in 30 seconds")

    # Kill server to simulate crash
    proc.terminate()
    time.sleep(1)

    # Verify API calls now fail
    with pytest.raises((openai.APIConnectionError, openai.APIStatusError)):
        await client.completions.create(
            model=MODEL_NAME, prompt="This should fail", max_tokens=1
        )

    return_code = proc.wait(timeout=5)
    assert return_code is not None
