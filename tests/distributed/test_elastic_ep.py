# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

import requests

from vllm.transformers_utils.tokenizer import get_tokenizer

from ..utils import RemoteOpenAIServer, _test_completion, multi_gpu_test

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"


def _send_scale_command(server: RemoteOpenAIServer, new_dp_size: int) -> bool:
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling():
    vllm_serve_args = [
        "--trust-remote-code",
        "--disable-log-requests",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "16384",
        "--no-enable-prefix-caching",
        "--enable-expert-parallel",
        "--all2all-backend",
        "pplx",
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        "128",
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--data-parallel-start-rank",
        "0",
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        vllm_serve_args.extend(["--data-parallel-address", leader_address])

    tokenizer = get_tokenizer(MODEL_NAME, trust_remote_code=True)
    prompt = "Hello, my name is"
    token_ids = tokenizer(prompt).input_ids

    # timeout is 20 minutes
    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        client = server.get_client()
        _test_completion(client, MODEL_NAME, prompt, token_ids)

        # Scale up from 2->4
        assert _send_scale_command(server, 4)
        time.sleep(10)
        _test_completion(client, MODEL_NAME, prompt, token_ids)

        # Scale down from 4->2
        assert _send_scale_command(server, 2)
        time.sleep(5)
        _test_completion(client, MODEL_NAME, prompt, token_ids)
