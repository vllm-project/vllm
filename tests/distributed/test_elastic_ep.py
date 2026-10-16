# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from vllm.platforms import current_platform

from ..evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from ..utils import RemoteOpenAIServer, multi_gpu_test


@pytest.fixture
def cleanup_ray_between_tests():
    """Force-stop any lingering Ray processes between tests."""
    subprocess.run(["ray", "stop", "--force"], timeout=30, capture_output=True)
    time.sleep(5)
    yield


MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
MP_MODEL_NAME = "Qwen/Qwen3-30B-A3B"

NUM_GSM8K_QUESTIONS = 256
EXPECTED_ACCURACY = 0.58
ACCURACY_TOL = 0.08
DEFAULT_ALL2ALL_BACKEND = "allgather_reducescatter"


def _send_scale_command(
    server: RemoteOpenAIServer, new_dp_size: int, timeout: int = 300
) -> bool:
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _traffic_loop(
    server: RemoteOpenAIServer,
    model: str,
    dp_rank: int | None,
    ready: threading.Barrier,
    stop: threading.Event,
    finished: threading.Event,
    is_probe: bool = False,
    request_timeout: int = 120,
) -> list[tuple[float, float, int | None, str | None]]:
    url = server.url_for("is_scaling_elastic_ep" if is_probe else "v1/completions")
    payload = {"model": model, "prompt": "Hello", "max_tokens": 4}
    headers = None if dp_rank is None else {"X-data-parallel-rank": str(dp_rank)}
    request_payload = None if is_probe else payload
    responses = []
    is_ready = False
    while not stop.is_set():
        request_start = time.perf_counter()
        phase = None
        try:
            response = requests.post(
                url, json=request_payload, headers=headers, timeout=request_timeout
            )
            status_code = response.status_code
            if is_probe and status_code == 200:
                phase = response.json()["phase"]
        except requests.exceptions.RequestException:
            status_code = None
        responses.append((request_start, time.perf_counter(), status_code, phase))
        if status_code == 200 and (not is_probe or phase in ("idle", "completed")):
            if not is_ready:
                ready.wait(timeout=request_timeout)
                is_ready = True
            if finished.is_set():
                return responses
        time.sleep(0.05)
    return responses


def _downtime(responses: list[tuple[float, float, int | None, str | None]]) -> float:
    committing = [end for _, end, _, phase in responses if phase == "committing"]
    if not committing:
        return 0
    recovered = next(
        end
        for _, end, status, phase in responses
        if status == 200 and phase in ("idle", "completed") and end > committing[-1]
    )
    return recovered - committing[0]


def _scale_with_traffic(
    server: RemoteOpenAIServer,
    source_dp_size: int,
    new_dp_size: int,
    traffic_mode: str,
    model: str = MODEL_NAME,
    request_timeout: int = 120,
    scale_timeout: int = 300,
) -> None:
    traffic_clients: list[int | None] = []
    if traffic_mode == "light":
        traffic_clients = [0]
    elif traffic_mode == "heavy":
        traffic_clients = [None] * source_dp_size
    clients = [(None, True)] + [(rank, False) for rank in traffic_clients]
    ready = threading.Barrier(len(clients) + 1)
    stop = threading.Event()
    finished = threading.Event()

    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = [
            executor.submit(
                _traffic_loop,
                server,
                model,
                rank,
                ready,
                stop,
                finished,
                is_probe,
                request_timeout,
            )
            for rank, is_probe in clients
        ]
        try:
            ready.wait(timeout=request_timeout)
            start_time = time.perf_counter()
            assert _send_scale_command(server, new_dp_size, timeout=scale_timeout)
            scale_seconds = time.perf_counter() - start_time
            finished.set()
            probe_result, *results = [
                future.result(timeout=request_timeout) for future in futures
            ]
        finally:
            stop.set()

    bad_statuses = {
        status
        for responses in [probe_result, *results]
        for _, _, status, _ in responses
        if status not in (200, 503)
    }
    assert not bad_statuses, f"traffic got unexpected statuses {bad_statuses}"
    probe_committing = [
        start for start, _, _, phase in probe_result if phase == "committing"
    ]
    assert probe_committing, "Scaling probe did not observe commit"
    assert not results or any(
        status == 200
        and start_time <= request_start
        and request_end < probe_committing[0]
        for responses in results
        for request_start, request_end, status, _ in responses
    ), "No request completed successfully during preparation"

    print(
        f"[Elastic EP timing][{source_dp_size}->{new_dp_size}]"
        f"[traffic={traffic_mode}] "
        f"scale_seconds={scale_seconds:.3f} "
        f"downtime_seconds={_downtime(probe_result):.3f}"
    )


def _run_gsm8k_eval(
    server: RemoteOpenAIServer,
    stage: str,
    max_concurrency: int | None = None,
    request_timeout_seconds: float = 600,
) -> dict[str, float | int]:
    assert server.port is not None
    result = evaluate_gsm8k(
        num_questions=NUM_GSM8K_QUESTIONS,
        host=f"http://{server.host}",
        port=server.port,
        max_concurrency=max_concurrency,
        request_timeout_seconds=request_timeout_seconds,
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
    return result


def _assert_scaled_accuracy(
    initial: dict[str, float | int], scaled: dict[str, float | int], stage: str
) -> None:
    assert scaled["accuracy"] >= initial["accuracy"] - ACCURACY_TOL, (
        f"{stage} accuracy {scaled['accuracy']:.3f} dropped more than "
        f"{ACCURACY_TOL} below initial accuracy {initial['accuracy']:.3f}"
    )


def _base_serve_args(
    dp_size: int = 2,
    max_dp_size: int = 4,
    enforce_eager: bool = False,
    all2all_backend: str = DEFAULT_ALL2ALL_BACKEND,
    dp_backend: str = "ray",
    redundant_experts: int = 0,
) -> list[str]:
    args = [
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.8" if dp_backend == "ray" else "0.9",
        "--max-model-len",
        "4096",
        "--enable-expert-parallel",
        "--all2all-backend",
        all2all_backend,
        "--enable-elastic-ep",
        "--elastic-ep-max-dp-size",
        str(max_dp_size),
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        str(redundant_experts),
        "--eplb-config.use_async",
        "true" if dp_backend == "ray" else "false",
        "--eplb-config.step_interval",
        "300",
        "--eplb-config.window_size",
        "5",
        "--data-parallel-backend",
        dp_backend,
        "--data-parallel-size",
        str(dp_size),
        "--api-server-count",
        "1",
        "--disable-access-log-for-endpoints",
        "/is_scaling_elastic_ep",
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address and dp_backend == "ray":
        args.extend(["--data-parallel-address", leader_address])
    if enforce_eager:
        args.append("--enforce-eager")

    return args


@pytest.mark.parametrize(
    ("enforce_eager", "traffic_mode", "all2all_backend"),
    [
        pytest.param(True, "none", DEFAULT_ALL2ALL_BACKEND, id="enforce_eager_none"),
        pytest.param(True, "light", DEFAULT_ALL2ALL_BACKEND, id="enforce_eager_light"),
        pytest.param(True, "heavy", DEFAULT_ALL2ALL_BACKEND, id="enforce_eager_heavy"),
        pytest.param(
            False,
            "heavy",
            DEFAULT_ALL2ALL_BACKEND,
            id="cuda_graphs_heavy",
        ),
        pytest.param(False, "heavy", "nixl_ep", id="cuda_graphs_heavy_nixl_ep"),
    ],
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling(
    enforce_eager: bool,
    traffic_mode: str,
    all2all_backend: str,
    cleanup_ray_between_tests,
):
    from vllm.distributed.eplb.eplb_communicator import has_nixl
    from vllm.utils.import_utils import has_nixl_ep

    if not has_nixl():
        pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")
    if current_platform.is_xpu():
        pytest.skip("Elastic EP with NIXL is not supported on XPU")
    if all2all_backend == "nixl_ep" and not has_nixl_ep():
        pytest.skip("NIXL EP is not installed")

    initial_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_INITIAL_DP", "2"))
    target_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_TARGET_DP", "4"))
    assert target_dp_size > initial_dp_size
    vllm_serve_args = _base_serve_args(
        initial_dp_size, target_dp_size, enforce_eager, all2all_backend
    )

    with RemoteOpenAIServer(
        MODEL_NAME,
        vllm_serve_args,
        max_wait_seconds=1200,
    ) as server:
        initial = _run_gsm8k_eval(server, "Initial")

        _scale_with_traffic(server, initial_dp_size, target_dp_size, traffic_mode)
        scaled_target = _run_gsm8k_eval(server, "After scale up")
        _assert_scaled_accuracy(initial, scaled_target, "Scale up")

        _scale_with_traffic(server, target_dp_size, initial_dp_size, traffic_mode)
        scaled_source = _run_gsm8k_eval(server, "After scale down")
        _assert_scaled_accuracy(initial, scaled_source, "Scale down")

        print("\nAccuracy Summary:")
        print(f"  Initial:    {initial['accuracy']:.3f}")
        print(
            f"  Scale up:   {scaled_target['accuracy']:.3f} "
            f"(diff: {scaled_target['accuracy'] - initial['accuracy']:+.3f})"
        )
        print(
            f"  Scale down: {scaled_source['accuracy']:.3f} "
            f"(diff: {scaled_source['accuracy'] - initial['accuracy']:+.3f})"
        )
        print(f"  Tolerance:  {ACCURACY_TOL:.3f}")


def _run_mp_gsm8k_eval(
    server: RemoteOpenAIServer, stage: str, dp_size: int
) -> dict[str, float | int]:
    result = _run_gsm8k_eval(
        server, stage, max_concurrency=32 * dp_size, request_timeout_seconds=1800
    )
    assert result["num_questions"] == NUM_GSM8K_QUESTIONS
    assert result["invalid_rate"] <= 0.05, f"{stage}: {result}"
    assert result["tokens_per_second"] > 0, f"{stage}: {result}"
    assert result["questions_per_second"] > 0, f"{stage}: {result}"
    print(
        f"[{stage}] output_tokens_per_second={result['tokens_per_second']:.3f} "
        f"questions_per_second={result['questions_per_second']:.3f}"
    )
    return result


def _smoke_mp_ranks(server: RemoteOpenAIServer, dp_size: int) -> None:
    for rank in range(dp_size):
        response = requests.post(
            server.url_for("v1/completions"),
            headers={"X-data-parallel-rank": str(rank)},
            json={
                "model": MP_MODEL_NAME,
                "prompt": "What is 2 + 2?",
                "max_tokens": 32,
                "temperature": 0,
            },
            timeout=120,
        )
        response.raise_for_status()
        assert response.json()["choices"][0]["text"], f"Empty response from rank {rank}"


@pytest.mark.parametrize(
    ("enforce_eager", "traffic_mode"),
    [
        pytest.param(True, "none", id="eager_none"),
        pytest.param(True, "light", id="eager_light"),
        pytest.param(True, "heavy", id="eager_heavy"),
        pytest.param(False, "heavy", id="cuda_graphs_heavy"),
    ],
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_mp_scaling(enforce_eager: bool, traffic_mode: str) -> None:
    if not current_platform.is_cuda():
        pytest.skip("Single-node MP Elastic EP integration requires CUDA")

    args = _base_serve_args(
        enforce_eager=enforce_eager,
        dp_backend="mp",
    )
    with RemoteOpenAIServer(MP_MODEL_NAME, args, max_wait_seconds=2400) as server:
        _smoke_mp_ranks(server, 2)
        initial = _run_mp_gsm8k_eval(server, "Initial", 2)

        _scale_with_traffic(
            server,
            2,
            4,
            traffic_mode,
            model=MP_MODEL_NAME,
            request_timeout=360 if not enforce_eager else 120,
            scale_timeout=1200,
        )
        _smoke_mp_ranks(server, 4)
        scaled_target = _run_mp_gsm8k_eval(server, "After scale up", 4)
        _assert_scaled_accuracy(initial, scaled_target, "Scale up")

        _scale_with_traffic(
            server,
            4,
            2,
            traffic_mode,
            model=MP_MODEL_NAME,
            request_timeout=360 if not enforce_eager else 120,
            scale_timeout=1200,
        )
        _smoke_mp_ranks(server, 2)
        scaled_source = _run_mp_gsm8k_eval(server, "After scale down", 2)
        _assert_scaled_accuracy(initial, scaled_source, "Scale down")


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_mp_scaled_vs_cold() -> None:
    if not current_platform.is_cuda():
        pytest.skip("Single-node MP Elastic EP integration requires CUDA")

    args = _base_serve_args(enforce_eager=True, dp_backend="mp")
    with RemoteOpenAIServer(MP_MODEL_NAME, args, max_wait_seconds=2400) as server:
        initial = _run_mp_gsm8k_eval(server, "Initial", 2)
        _scale_with_traffic(
            server, 2, 4, "none", model=MP_MODEL_NAME, scale_timeout=1200
        )
        scaled_target = _run_mp_gsm8k_eval(server, "Scaled DP4", 4)
        _scale_with_traffic(
            server, 4, 2, "none", model=MP_MODEL_NAME, scale_timeout=1200
        )
        scaled_source = _run_mp_gsm8k_eval(server, "Scaled DP2", 2)
        assert scaled_target["accuracy"] >= initial["accuracy"] - ACCURACY_TOL
        assert scaled_source["accuracy"] >= initial["accuracy"] - ACCURACY_TOL

    cold_target_args = _base_serve_args(
        dp_size=4, enforce_eager=True, dp_backend="mp", redundant_experts=128
    )
    with RemoteOpenAIServer(
        MP_MODEL_NAME, cold_target_args, max_wait_seconds=2400
    ) as server:
        cold_target = _run_mp_gsm8k_eval(server, "Cold DP4", 4)

    with RemoteOpenAIServer(MP_MODEL_NAME, args, max_wait_seconds=2400) as server:
        cold_source = _run_mp_gsm8k_eval(server, "Cold DP2", 2)

    for label, scaled, cold in (
        ("DP4", scaled_target, cold_target),
        ("DP2", scaled_source, cold_source),
    ):
        assert scaled["accuracy"] >= cold["accuracy"] - ACCURACY_TOL, label
        for metric in ("questions_per_second", "tokens_per_second"):
            ratio = scaled[metric] / cold[metric]
            print(f"[{label}] scaled/cold {metric}={ratio:.3f}")
            assert ratio >= 0.85, f"{label} scaled/cold {metric}={ratio:.3f}"
