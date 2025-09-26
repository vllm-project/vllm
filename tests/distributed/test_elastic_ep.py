# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

import requests

from ..evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from ..utils import RemoteOpenAIServer, multi_gpu_test

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"

NUM_GSM8K_QUESTIONS = 200
EXPECTED_ACCURACY = 0.65
ACCURACY_TOL = 0.08


def _send_scale_command(server: RemoteOpenAIServer, new_dp_size: int) -> bool:
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _send_warmup_request(server: RemoteOpenAIServer) -> None:
    url = f"http://{server.host}:{server.port}/v1/completions"
    payload = {
        "prompt": "Which mountain is the highest in the Solar System?",
        "max_tokens": 16,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}

    print("Sending warmup request...")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        print("Warmup request completed.")
    except requests.exceptions.RequestException as e:
        print(f"Warmup request failed: {e}")
        raise


def _run_gsm8k_eval(server: RemoteOpenAIServer, stage: str) -> float:
    _send_warmup_request(server)

    assert server.port is not None
    result = evaluate_gsm8k(
        num_questions=NUM_GSM8K_QUESTIONS,
        num_shots=5,
        max_tokens=512,
        host=f"http://{server.host}",
        port=server.port,
        temperature=0.0,
        seed=42,
    )
    accuracy = result["accuracy"]
    print(
        f"[{stage}] GSM8K accuracy: {accuracy:.3f} "
        f"({result['num_questions']} questions)"
    )
    assert accuracy >= EXPECTED_ACCURACY, (
        f"[{stage}] GSM8K accuracy {accuracy:.3f} is below "
        f"expected threshold {EXPECTED_ACCURACY}"
    )
    return accuracy


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
    # os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    if leader_address:
        vllm_serve_args.extend(["--data-parallel-address", leader_address])

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, "Initial (2 GPUs)")

        assert _send_scale_command(server, 4)
        time.sleep(10)
        scale_up_accuracy = _run_gsm8k_eval(server, "After scale up (4 GPUs)")

        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale up accuracy {scale_up_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        assert _send_scale_command(server, 2)
        time.sleep(5)
        scale_down_accuracy = _run_gsm8k_eval(server, "After scale down (2 GPUs)")

        assert scale_down_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale down accuracy {scale_down_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        print("\nAccuracy Summary:")
        print(f"  Initial:    {initial_accuracy:.3f}")
        print(
            f"  Scale up:   {scale_up_accuracy:.3f} "
            f"(diff: {scale_up_accuracy - initial_accuracy:+.3f})"
        )
        print(
            f"  Scale down: {scale_down_accuracy:.3f} "
            f"(diff: {scale_down_accuracy - initial_accuracy:+.3f})"
        )
        print(f"  Tolerance:  {ACCURACY_TOL:.3f}")
