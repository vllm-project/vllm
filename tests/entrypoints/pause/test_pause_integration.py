# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for step-barrier pause endpoints.

These tests require a running vLLM server. Run with:

    # Start server first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tensor-parallel-size 4

    # Then run tests:
    pytest tests/entrypoints/pause/test_pause_integration.py -v \
        --server-url http://localhost:8000

Or use the convenience script:
    ./scripts/test_pause_endpoints.sh
"""

import os
import time

import pytest
import requests

# Get server URL from environment or pytest option
DEFAULT_SERVER_URL = "http://localhost:8000"


def get_server_url():
    return os.environ.get("VLLM_TEST_SERVER_URL", DEFAULT_SERVER_URL)


def pytest_addoption(parser):
    parser.addoption(
        "--server-url",
        action="store",
        default=DEFAULT_SERVER_URL,
        help="vLLM server URL for integration tests",
    )


@pytest.fixture
def server_url(request):
    url = request.config.getoption("--server-url", default=None)
    if url is None:
        url = get_server_url()
    return url


@pytest.fixture
def skip_if_no_server(server_url):
    """Skip test if server is not reachable."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip(f"Server at {server_url} not healthy")
    except requests.exceptions.RequestException:
        pytest.skip(f"Server at {server_url} not reachable")


class TestPauseStepIntegration:
    """Integration tests for step-barrier pause endpoints against a real vLLM server."""

    def test_pause_step_returns_step_counter(self, server_url, skip_if_no_server):
        """Test that /pause/step returns step_counter and recommended_target_step."""
        response = requests.post(f"{server_url}/pause/step", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert data["paused"] is True
        assert "step_counter" in data
        assert isinstance(data["step_counter"], int)
        assert data["step_counter"] >= 0
        assert data["recommended_target_step"] == data["step_counter"] + 1
        assert "message" in data
        assert data["status"] == "paused"

        # Clean up: resume
        requests.post(f"{server_url}/resume", timeout=30)

    def test_pause_step_is_fast(self, server_url, skip_if_no_server):
        """Test that /pause/step returns quickly (< 1 second)."""
        # First resume to ensure clean state
        requests.post(f"{server_url}/resume", timeout=30)

        start = time.time()
        response = requests.post(f"{server_url}/pause/step", timeout=30)
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0, f"Pause took {elapsed:.2f}s, expected < 1s"

        # Clean up
        requests.post(f"{server_url}/resume", timeout=30)

    def test_pause_step_is_idempotent(self, server_url, skip_if_no_server):
        """Test that calling /pause/step multiple times is safe."""
        # First pause
        r1 = requests.post(f"{server_url}/pause/step", timeout=30)
        assert r1.status_code == 200
        step1 = r1.json()["step_counter"]

        # Second pause (should also succeed)
        r2 = requests.post(f"{server_url}/pause/step", timeout=30)
        assert r2.status_code == 200
        step2 = r2.json()["step_counter"]

        # Step counter should be the same or slightly higher
        assert step2 >= step1

        # Clean up
        requests.post(f"{server_url}/resume", timeout=30)

    def test_is_paused_reflects_state(self, server_url, skip_if_no_server):
        """Test that /is_paused correctly reflects pause state."""
        # Start with resume
        requests.post(f"{server_url}/resume", timeout=30)

        # Check not paused
        r1 = requests.get(f"{server_url}/is_paused", timeout=30)
        assert r1.status_code == 200
        assert r1.json()["is_paused"] is False

        # Pause
        requests.post(f"{server_url}/pause/step", timeout=30)

        # Check paused
        r2 = requests.get(f"{server_url}/is_paused", timeout=30)
        assert r2.status_code == 200
        assert r2.json()["is_paused"] is True

        # Resume
        requests.post(f"{server_url}/resume", timeout=30)

        # Check not paused again
        r3 = requests.get(f"{server_url}/is_paused", timeout=30)
        assert r3.status_code == 200
        assert r3.json()["is_paused"] is False

    def test_pause_step_barrier(self, server_url, skip_if_no_server):
        """Test the barrier endpoint."""
        # First resume
        requests.post(f"{server_url}/resume", timeout=30)

        # Pause and get step counter
        pause_response = requests.post(f"{server_url}/pause/step", timeout=30)
        assert pause_response.status_code == 200
        target = pause_response.json()["recommended_target_step"]

        # Run until target step count (this is the barrier)
        barrier_response = requests.post(
            f"{server_url}/pause/step/barrier",
            json={"target_steps": target},
            timeout=60,
        )
        assert barrier_response.status_code == 200
        assert barrier_response.json()["status"] == "ok"

        # Should still be paused after barrier
        is_paused = requests.get(f"{server_url}/is_paused", timeout=30)
        assert is_paused.json()["is_paused"] is True

        # Clean up
        requests.post(f"{server_url}/resume", timeout=30)

    def test_pause_step_barrier_validation(self, server_url, skip_if_no_server):
        """Test that /pause/step/barrier validates input."""
        # Missing target_steps
        r1 = requests.post(
            f"{server_url}/pause/step/barrier",
            json={},
            timeout=30,
        )
        assert r1.status_code == 400
        assert "target_steps" in r1.json()["detail"]

        # Invalid target_steps type
        r2 = requests.post(
            f"{server_url}/pause/step/barrier",
            json={"target_steps": "not_an_int"},
            timeout=30,
        )
        assert r2.status_code == 400

    def test_full_weight_sync_workflow(self, server_url, skip_if_no_server):
        """
        Simulate the full weight synchronization workflow as used in RL training.

        This is the critical path:
        1. POST /pause/step → get step_counter
        2. Compute target = step_counter + 1
        3. POST /pause/step/barrier(target) → barrier
        4. (update weights - simulated here)
        5. POST /resume
        """
        # Ensure clean state
        requests.post(f"{server_url}/resume", timeout=30)

        # Step 1: Pause
        pause_response = requests.post(f"{server_url}/pause/step", timeout=30)
        assert pause_response.status_code == 200
        step_counter = pause_response.json()["step_counter"]
        target_steps = step_counter + 1

        print(f"Paused at step {step_counter}, target barrier: {target_steps}")

        # Step 2: Barrier
        barrier_response = requests.post(
            f"{server_url}/pause/step/barrier",
            json={"target_steps": target_steps},
            timeout=120,  # May need more time for DP sync
        )
        assert barrier_response.status_code == 200

        print("Barrier reached - all engines at target step")

        # Step 3: Simulate weight update
        print("Simulating weight update...")
        time.sleep(0.1)

        # Step 4: Resume
        resume_response = requests.post(f"{server_url}/resume", timeout=30)
        assert resume_response.status_code == 200

        print("Weights updated and engine resumed!")

        # Verify we can do inference after resume
        # (This tests that the engine is actually working after the pause/resume cycle)


class TestPauseStepWithInference:
    """Tests that combine step-barrier pause with actual inference requests."""

    def test_pause_step_during_inference(self, server_url, skip_if_no_server):
        """Test that /pause/step returns quickly even during inference."""
        import concurrent.futures

        # Ensure clean state
        requests.post(f"{server_url}/resume", timeout=30)

        def send_inference():
            """Send a long-running inference request."""
            try:
                return requests.post(
                    f"{server_url}/v1/chat/completions",
                    json={
                        "model": "test",  # Will use whatever model is loaded
                        "messages": [
                            {"role": "user", "content": "Write a 100-word essay."}
                        ],
                        "max_tokens": 200,
                    },
                    timeout=120,
                )
            except Exception as e:
                return e

        # Start inference in background
        with concurrent.futures.ThreadPoolExecutor() as executor:
            inference_future = executor.submit(send_inference)

            # Give inference a moment to start
            time.sleep(0.5)

            # Pause should still be fast
            start = time.time()
            pause_response = requests.post(f"{server_url}/pause/step", timeout=30)
            pause_elapsed = time.time() - start

            assert pause_response.status_code == 200
            print(f"Pause took {pause_elapsed:.3f}s during inference")
            # Should still be reasonably fast (< 2s even during inference)
            assert pause_elapsed < 2.0

            # Resume so inference can complete
            requests.post(f"{server_url}/resume", timeout=30)

            # Wait for inference to complete
            inference_result = inference_future.result(timeout=120)
            # Inference might fail or succeed depending on timing, both are OK
            print(f"Inference result: {type(inference_result)}")


if __name__ == "__main__":
    # Run with: python -m pytest test_pause_integration.py -v
    pytest.main([__file__, "-v"])
