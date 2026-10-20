# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from vllm.platforms import current_platform

from ..evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from ..utils import RemoteOpenAIServer, multi_gpu_test


@pytest.fixture(autouse=True)
def cleanup_ray_between_tests():
    """Force-stop any lingering Ray processes between tests, unless the test
    joins a Ray cluster started before pytest (LEADER_ADDRESS)."""
    if os.environ.get("LEADER_ADDRESS"):
        yield
        return
    subprocess.run(["ray", "stop", "--force"], timeout=30, capture_output=True)
    time.sleep(5)
    yield


# DeepSeek-V2-Lite unless overridden; a larger model sets its own GSM8K
# settings, accuracy bar, serve arguments and start-up wait below.
MODEL_NAME = os.getenv(
    "VLLM_TEST_ELASTIC_EP_MODEL", "deepseek-ai/DeepSeek-V2-Lite-Chat"
)

NUM_GSM8K_QUESTIONS = int(os.getenv("VLLM_TEST_ELASTIC_EP_GSM8K_QUESTIONS", "256"))
# The first and the last stage's question count; the stages between use
# NUM_GSM8K_QUESTIONS.
NUM_GSM8K_QUESTIONS_ENDS = int(
    os.getenv("VLLM_TEST_ELASTIC_EP_GSM8K_QUESTIONS_ENDS", str(NUM_GSM8K_QUESTIONS))
)
NUM_GSM8K_SHOTS = int(os.getenv("VLLM_TEST_ELASTIC_EP_GSM8K_SHOTS", "5"))
EXPECTED_ACCURACY = float(os.getenv("VLLM_TEST_ELASTIC_EP_ACCURACY_MIN", "0.58"))
ACCURACY_TOL = float(os.getenv("VLLM_TEST_ELASTIC_EP_ACCURACY_MAX_DROP", "0.08"))
# Appended to the serve arguments, so a repeated flag such as --max-model-len
# overrides the default (the parser keeps the last value).
EXTRA_SERVE_ARGS = shlex.split(os.getenv("VLLM_TEST_ELASTIC_EP_EXTRA_ARGS", ""))
SERVER_WAIT_S = float(os.getenv("VLLM_TEST_ELASTIC_EP_SERVER_WAIT_S", "1200"))
DEFAULT_ALL2ALL_BACKEND = "allgather_reducescatter"
# The probe's sleep bounds downtime_seconds' resolution; traffic keeps 50 ms,
# so num_503 (traffic requests only) is unchanged.
PROBE_INTERVAL_S = float(os.getenv("VLLM_TEST_ELASTIC_EP_PROBE_INTERVAL_S", "0.05"))


def _send_scale_command(server: RemoteOpenAIServer, new_dp_size: int) -> bool:
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        # A scale-up onto fresh nodes compiles and captures there, past 300 s.
        timeout = float(os.getenv("VLLM_TEST_ELASTIC_EP_SCALE_TIMEOUT_S", "300"))
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _traffic_loop(
    server: RemoteOpenAIServer,
    dp_rank: int | None,
    ready: threading.Barrier,
    stop: threading.Event,
    finished: threading.Event,
    is_probe: bool = False,
) -> list[tuple[float, float, int | None]]:
    url = server.url_for("is_scaling_elastic_ep" if is_probe else "v1/completions")
    payload = {"model": MODEL_NAME, "prompt": "Hello", "max_tokens": 4}
    headers = None if dp_rank is None else {"X-data-parallel-rank": str(dp_rank)}
    request_payload = None if is_probe else payload
    responses = []
    is_ready = False
    while not stop.is_set():
        request_start = time.perf_counter()
        try:
            response = requests.post(
                url, json=request_payload, headers=headers, timeout=120
            )
            status_code = response.status_code
        except requests.exceptions.RequestException:
            status_code = None
        responses.append((request_start, time.perf_counter(), status_code))
        if status_code == 200:
            if not is_ready:
                ready.wait(timeout=120)
                is_ready = True
            if finished.is_set():
                return responses
        time.sleep(PROBE_INTERVAL_S if is_probe else 0.05)
    return responses


def _downtime(responses: list[tuple[float, float, int | None]]) -> float:
    rejected = [end for _, end, status in responses if status == 503]
    if not rejected:
        return 0
    recovered = next(
        end for _, end, status in responses if status == 200 and end > rejected[-1]
    )
    return recovered - rejected[0]


def _scale_with_traffic(
    server: RemoteOpenAIServer,
    source_dp_size: int,
    new_dp_size: int,
    traffic_mode: str,
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
                _traffic_loop, server, rank, ready, stop, finished, is_probe
            )
            for rank, is_probe in clients
        ]
        try:
            ready.wait(timeout=120)
            start_time = time.perf_counter()
            assert _send_scale_command(server, new_dp_size)
            scale_seconds = time.perf_counter() - start_time
            finished.set()
            probe_result, *results = [future.result(timeout=120) for future in futures]
        finally:
            stop.set()

    bad_statuses = {
        status
        for responses in [probe_result, *results]
        for _, _, status in responses
        if status not in (200, 503)
    }
    assert not bad_statuses, f"traffic got unexpected statuses {bad_statuses}"
    probe_503 = [start for start, _, status in probe_result if status == 503]
    assert probe_503, "Scaling probe did not observe commit"
    assert not results or any(
        status == 200 and start_time <= request_start and request_end < probe_503[0]
        for responses in results
        for request_start, request_end, status in responses
    ), "No request completed successfully during preparation"

    print(
        f"[Elastic EP timing][{source_dp_size}->{new_dp_size}]"
        f"[traffic={traffic_mode}] "
        f"scale_seconds={scale_seconds:.3f} "
        f"downtime_seconds={_downtime(probe_result):.3f} "
        f"num_503={sum(s == 503 for r in results for _, _, s in r)} "
        f"probe_interval_s={PROBE_INTERVAL_S}",
        flush=True,
    )


def _run_gsm8k_eval(
    server: RemoteOpenAIServer, stage: str, num_questions: int = NUM_GSM8K_QUESTIONS
) -> float:
    assert server.port is not None
    result = evaluate_gsm8k(
        num_questions=num_questions,
        num_shots=NUM_GSM8K_SHOTS,
        host=f"http://{server.host}",
        port=server.port,
    )
    accuracy = result["accuracy"]
    print(
        f"[{stage}] GSM8K accuracy: {accuracy:.3f} "
        f"({result['num_questions']} questions)",
        flush=True,
    )
    assert accuracy >= EXPECTED_ACCURACY, (
        f"[{stage}] GSM8K accuracy {accuracy:.3f} is below "
        f"expected threshold {EXPECTED_ACCURACY}"
    )
    return accuracy


def _base_serve_args(
    dp_size: int = 2,
    max_dp_size: int = 4,
    enforce_eager: bool = False,
    all2all_backend: str = DEFAULT_ALL2ALL_BACKEND,
) -> list[str]:
    args = [
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.8",
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
        "0",
        "--eplb-config.use_async",
        "true",
        "--eplb-config.step_interval",
        "300",
        "--eplb-config.window_size",
        "5",
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        str(dp_size),
        "--api-server-count",
        "1",
        "--disable-access-log-for-endpoints",
        "/is_scaling_elastic_ep",
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        args.extend(["--data-parallel-address", leader_address])
    # With Ray it defaults to the DP size, more engines than one node holds.
    if dp_local := os.environ.get("VLLM_TEST_ELASTIC_EP_DP_LOCAL"):
        args.extend(["--data-parallel-size-local", dp_local])
    if enforce_eager:
        args.append("--enforce-eager")
    args.extend(EXTRA_SERVE_ARGS)

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
        pytest.param(False, "heavy", "nixl_ep_v2", id="cuda_graphs_heavy_nixl_ep_v2"),
    ],
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling(
    enforce_eager: bool, traffic_mode: str, all2all_backend: str
):
    from vllm.distributed.eplb.eplb_communicator import has_nixl
    from vllm.utils.import_utils import has_nixl_ep, has_nixl_ep_v2

    if not has_nixl():
        pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")
    if current_platform.is_xpu():
        pytest.skip("Elastic EP with NIXL is not supported on XPU")
    if all2all_backend == "nixl_ep" and not has_nixl_ep():
        pytest.skip("NIXL EP is not installed")
    if all2all_backend == "nixl_ep_v2" and not has_nixl_ep_v2():
        pytest.skip("nixl.runtime.ep is not installed")

    initial_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_INITIAL_DP", "2"))
    target_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_TARGET_DP", "4"))
    assert target_dp_size > initial_dp_size
    vllm_serve_args = _base_serve_args(
        initial_dp_size, target_dp_size, enforce_eager, all2all_backend
    )

    with RemoteOpenAIServer(
        MODEL_NAME,
        vllm_serve_args,
        max_wait_seconds=SERVER_WAIT_S,
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, "Initial", NUM_GSM8K_QUESTIONS_ENDS)

        _scale_with_traffic(server, initial_dp_size, target_dp_size, traffic_mode)
        scale_up_accuracy = _run_gsm8k_eval(server, "After scale up")
        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale up accuracy {scale_up_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        _scale_with_traffic(server, target_dp_size, initial_dp_size, traffic_mode)
        scale_down_accuracy = _run_gsm8k_eval(
            server,
            "After scale down",
            NUM_GSM8K_QUESTIONS
            if all2all_backend == "nixl_ep_v2"
            else NUM_GSM8K_QUESTIONS_ENDS,
        )
        assert scale_down_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale down accuracy {scale_down_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        if all2all_backend == "nixl_ep_v2":
            # New processes take over the EP ranks the leavers had.
            _scale_with_traffic(server, initial_dp_size, target_dp_size, traffic_mode)
            rescale_accuracy = _run_gsm8k_eval(
                server, "After second scale up", NUM_GSM8K_QUESTIONS_ENDS
            )
            assert rescale_accuracy >= initial_accuracy - ACCURACY_TOL, (
                f"Second scale up accuracy {rescale_accuracy:.3f} dropped more "
                f"than {ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
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
        print(f"  Tolerance:  {ACCURACY_TOL:.3f}", flush=True)


def _skip_unless_nixl_ep_v2() -> None:
    from vllm.distributed.eplb.eplb_communicator import has_nixl
    from vllm.utils.import_utils import has_nixl_ep_v2

    if not has_nixl():
        pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")
    if not has_nixl_ep_v2():
        pytest.skip("nixl.runtime.ep is not installed")


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling_sequence():
    """nixl_ep_v2 through the sizes in VLLM_TEST_ELASTIC_EP_SIZES (the first
    is the initial size) under heavy traffic, with GSM8K after each scale.

    VLLM_TEST_ELASTIC_EP_MAX_DP is separate from the sizes: with the domain
    size unset it is the nixl_ep_v2 domain size, so 2,3,2,3 under max 4
    runs a partial first domain.
    """
    _skip_unless_nixl_ep_v2()
    sizes = [
        int(size)
        for size in os.getenv("VLLM_TEST_ELASTIC_EP_SIZES", "2,3,2,3").split(",")
    ]
    max_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_MAX_DP", "4"))
    assert len(sizes) >= 2 and max(sizes) <= max_dp_size, (sizes, max_dp_size)
    assert all(a != b for a, b in zip(sizes, sizes[1:])), sizes

    with RemoteOpenAIServer(
        MODEL_NAME,
        _base_serve_args(sizes[0], max_dp_size, all2all_backend="nixl_ep_v2"),
        max_wait_seconds=SERVER_WAIT_S,
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, "Initial", NUM_GSM8K_QUESTIONS_ENDS)
        for i, (source_dp_size, new_dp_size) in enumerate(zip(sizes, sizes[1:])):
            _scale_with_traffic(server, source_dp_size, new_dp_size, "heavy")
            stage = f"After {source_dp_size}->{new_dp_size}"
            last = i == len(sizes) - 2
            accuracy = _run_gsm8k_eval(
                server,
                stage,
                NUM_GSM8K_QUESTIONS_ENDS if last else NUM_GSM8K_QUESTIONS,
            )
            assert accuracy >= initial_accuracy - ACCURACY_TOL, (
                f"[{stage}] accuracy {accuracy:.3f} dropped more than "
                f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
            )


def _scale_until_done(server: RemoteOpenAIServer, new_dp_size: int) -> list[int]:
    """POST a scale, and again on each 503 (another scale is committing),
    until it gets 200; returns every status. A queued POST waits for the
    running scale and then its own, so it gets twice _send_scale_command's
    timeout."""
    statuses: list[int] = []
    deadline = time.monotonic() + 600
    while True:
        response = requests.post(
            server.url_for("scale_elastic_ep"),
            json={"new_data_parallel_size": new_dp_size},
            timeout=600,
        )
        statuses.append(response.status_code)
        if response.status_code != 503 or time.monotonic() > deadline:
            assert response.status_code == 200, (
                f"scale to {new_dp_size}: statuses {statuses}: {response.text}"
            )
            return statuses
        time.sleep(1)


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scale_while_scaling():
    """A scale request during a scale, or right after a scale-down while its
    leavers still tear down, is queued or refused with 503 and then served:
    never 500 or 408.

    Which path a run took shows in the printed statuses: [200] was queued on
    the scale lock, [503, ..., 200] was refused while the first committed.
    """
    _skip_unless_nixl_ep_v2()
    with RemoteOpenAIServer(
        MODEL_NAME,
        _base_serve_args(2, 4, all2all_backend="nixl_ep_v2"),
        max_wait_seconds=SERVER_WAIT_S,
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, "Initial", NUM_GSM8K_QUESTIONS_ENDS)

        # A second POST for the same target while 2->4 runs.
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(_scale_until_done, server, 4)
            time.sleep(10)
            assert not first.done(), "2->4 finished before the second POST"
            second = _scale_until_done(server, 4)
            print(
                f"[Elastic EP concurrent][2->4 twice] first={first.result()} "
                f"second={second}"
            )

        # A scale-up right after the scale-down's 200: the client waits for
        # the leavers' shutdown and Ray for their GPUs.
        _scale_until_done(server, 2)
        print(f"[Elastic EP concurrent][4->2->4] {_scale_until_done(server, 4)}")
        # The current size: logs "already 4, skipping".
        _scale_until_done(server, 4)

        accuracy = _run_gsm8k_eval(
            server, "After concurrent scales", NUM_GSM8K_QUESTIONS_ENDS
        )
        assert accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Accuracy {accuracy:.3f} dropped more than {ACCURACY_TOL} below "
            f"initial accuracy {initial_accuracy:.3f}"
        )


@multi_gpu_test(num_gpus=2)
def test_elastic_ep_rejects_invalid_size():
    """nixl_ep_v2 refuses a size that is not whole NVLink domains or inside
    the first one, before any engine is touched: HTTP 400, serving goes on.

    Defaults: DP 2, max 8, domain 4, DP 5 refused, valid sizes 2,3,4,8 (DP 1
    fails the expert-fit check). Across domains, e.g. DP 8, max 16, domain
    8: VLLM_TEST_ELASTIC_EP_INITIAL_DP, _MAX_DP, VLLM_NIXL_EP_V2_DOMAIN_SIZE,
    VLLM_TEST_ELASTIC_EP_INVALID_DP and _VALID (comma-separated)."""
    _skip_unless_nixl_ep_v2()
    initial_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_INITIAL_DP", "2"))
    max_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_MAX_DP", "8"))
    domain_size = os.getenv("VLLM_NIXL_EP_V2_DOMAIN_SIZE", "4")
    invalid_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_INVALID_DP", "5"))
    valid = os.getenv("VLLM_TEST_ELASTIC_EP_VALID", "2,3,4,8")
    valid_sizes = [int(size) for size in valid.split(",")]

    with RemoteOpenAIServer(
        MODEL_NAME,
        _base_serve_args(initial_dp_size, max_dp_size, all2all_backend="nixl_ep_v2"),
        env_dict={"VLLM_NIXL_EP_V2_DOMAIN_SIZE": domain_size},
        max_wait_seconds=SERVER_WAIT_S,
    ) as server:
        response = requests.post(
            server.url_for("scale_elastic_ep"),
            json={"new_data_parallel_size": invalid_dp_size},
            timeout=300,
        )
        assert response.status_code == 400, response.text
        assert str(valid_sizes) in response.text, response.text
        probe = requests.post(server.url_for("is_scaling_elastic_ep"), timeout=30)
        assert probe.status_code == 200, probe.status_code
        completion = requests.post(
            server.url_for("v1/completions"),
            json={"model": MODEL_NAME, "prompt": "Hello", "max_tokens": 4},
            timeout=120,
        )
        assert completion.status_code == 200, completion.text
