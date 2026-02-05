# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for step-barrier pause endpoints.

This file contains:
- Lightweight unit tests (no engine running).
- Server-backed integration tests (require a running vLLM server).

Run integration tests with:

    # Start server first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tensor-parallel-size 4

    pytest tests/entrypoints/test_pause_step.py -v --server-url http://localhost:8000
"""

import concurrent.futures
import time

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.pause.api_router import attach_router


@pytest.fixture
def app():
    app = FastAPI()
    attach_router(app)
    return app


def test_pause_step_rejects_conflicting_params(app):
    app.state.engine_client = object()
    client = TestClient(app)
    resp = client.post("/pause/step?no_barrier=true&barrier=50")
    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        "Cannot specify both no_barrier=true and barrier=<value>"
    )


def test_pause_step_requires_async_llm(app):
    app.state.engine_client = object()
    client = TestClient(app)
    resp = client.post("/pause/step")
    assert resp.status_code == 501


class TestPauseStepIntegration:
    """Integration tests for step-barrier pause endpoints."""

    def test_pause_step_default_with_barrier(self, server_url, skip_if_no_server):
        """Test that default /pause/step waits for barrier and returns step."""
        # Ensure clean state
        requests.post(f"{server_url}/resume", timeout=30)

        response = requests.post(f"{server_url}/pause/step", timeout=60)
        assert response.status_code == 200

        data = response.json()
        assert data["paused"] is True
        assert "step_counter" in data
        assert isinstance(data["step_counter"], int)
        assert data["step_counter"] >= 0

        # Clean up: resume
        requests.post(f"{server_url}/resume", timeout=30)

    def test_pause_step_no_barrier_is_fast(self, server_url, skip_if_no_server):
        """Test that /pause/step?no_barrier=true returns quickly (< 1 second)."""
        # First resume to ensure clean state
        requests.post(f"{server_url}/resume", timeout=30)

        start = time.time()
        response = requests.post(f"{server_url}/pause/step?no_barrier=true", timeout=30)
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0, f"Pause took {elapsed:.2f}s, expected < 1s"

        data = response.json()
        assert data["paused"] is True
        assert "step_counter" in data

        # Clean up
        requests.post(f"{server_url}/resume", timeout=30)

    def test_pause_step_custom_barrier(self, server_url, skip_if_no_server):
        """Test /pause/step?barrier=<value> waits for custom target."""
        # Ensure clean state
        requests.post(f"{server_url}/resume", timeout=30)

        # First get current step via fast pause
        fast_response = requests.post(
            f"{server_url}/pause/step?no_barrier=true", timeout=30
        )
        assert fast_response.status_code == 200
        current_step = fast_response.json()["step_counter"]

        # Now pause with custom barrier
        target = current_step + 1
        response = requests.post(
            f"{server_url}/pause/step?barrier={target}", timeout=60
        )
        assert response.status_code == 200
        data = response.json()
        assert data["paused"] is True
        assert data["step_counter"] >= target

        # Clean up
        requests.post(f"{server_url}/resume", timeout=30)

    def test_pause_step_is_idempotent(self, server_url, skip_if_no_server):
        """Test that calling /pause/step multiple times is safe."""
        # First pause
        r1 = requests.post(f"{server_url}/pause/step?no_barrier=true", timeout=30)
        assert r1.status_code == 200
        step1 = r1.json()["step_counter"]

        # Second pause (should also succeed)
        r2 = requests.post(f"{server_url}/pause/step?no_barrier=true", timeout=30)
        assert r2.status_code == 200
        step2 = r2.json()["step_counter"]

        # Step counter should be the same or slightly higher
        assert step2 >= step1

        # Clean up
        requests.post(f"{server_url}/resume", timeout=30)

    def test_full_weight_sync_workflow(self, server_url, skip_if_no_server):
        """
        Simulate the full weight synchronization workflow as used in RL training.

        This is the simplified workflow with the new API:
        1. POST /pause/step → pause + wait for barrier
        2. (update weights - simulated here)
        3. POST /resume
        """
        # Ensure clean state
        requests.post(f"{server_url}/resume", timeout=30)

        # Step 1: Pause with default barrier
        pause_response = requests.post(f"{server_url}/pause/step", timeout=60)
        assert pause_response.status_code == 200
        step_counter = pause_response.json()["step_counter"]

        print(f"Paused at step {step_counter} - all engines synced")

        # Step 2: Simulate weight update
        print("Simulating weight update...")
        time.sleep(0.1)

        # Step 3: Resume
        resume_response = requests.post(f"{server_url}/resume", timeout=30)
        assert resume_response.status_code == 200

        print("Weights updated and engine resumed!")

    def test_advanced_workflow_with_custom_barrier(self, server_url, skip_if_no_server):
        """
        Advanced workflow for when user needs control over barrier target.

        This replicates the old 2-call pattern:
        1. POST /pause/step?no_barrier=true → fast pause, get step
        2. Compute target
        3. POST /pause/step?barrier=<target> → wait for barrier
        4. (update weights)
        5. POST /resume
        """
        # Ensure clean state
        requests.post(f"{server_url}/resume", timeout=30)

        # Step 1: Fast pause to get step counter
        fast_response = requests.post(
            f"{server_url}/pause/step?no_barrier=true", timeout=30
        )
        assert fast_response.status_code == 200
        step_counter = fast_response.json()["step_counter"]

        print(f"Fast pause at step {step_counter}")

        # Step 2: Compute target (user's custom logic)
        target = step_counter + 1 if step_counter > 0 else step_counter

        # Step 3: Wait for barrier
        barrier_response = requests.post(
            f"{server_url}/pause/step?barrier={target}", timeout=60
        )
        assert barrier_response.status_code == 200

        print(f"Barrier reached at step {barrier_response.json()['step_counter']}")

        # Step 4: Simulate weight update
        print("Simulating weight update...")
        time.sleep(0.1)

        # Step 5: Resume
        resume_response = requests.post(f"{server_url}/resume", timeout=30)
        assert resume_response.status_code == 200

        print("Weights updated and engine resumed!")


class TestPauseStepWithInference:
    """Tests that combine step-barrier pause with actual inference requests."""

    def test_pause_step_during_inference(self, server_url, skip_if_no_server):
        """Test that /pause/step?no_barrier=true returns quickly during inference."""
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

            # Pause (no_barrier) should still be fast
            start = time.time()
            pause_response = requests.post(
                f"{server_url}/pause/step?no_barrier=true", timeout=30
            )
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
    pytest.main([__file__, "-v"])
