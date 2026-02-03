# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test for the fix in PR #29987: Eagerly abort cancelled final-step requests.

This test verifies that when a request is aborted during its final execution
step (when it would naturally complete), it is properly marked as aborted
rather than being treated as normally completed.

The test uses a dummy KV connector to verify that the connector receives
the correct finish status (FINISHED_ABORTED, not FINISHED_LENGTH_CAPPED).
"""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from vllm import SamplingParams
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

TEXT_PROMPT = "Hello"


class DummyKVConnectorMetadata(KVConnectorMetadata):
    """Dummy metadata for the test connector."""

    def __init__(self):
        self.requests: list = []


class DummyKVConnector(KVConnectorBase_V1):
    """
    Dummy KV connector that captures request finish statuses to a file.
    This is used to verify the fix - without the fix, a request aborted
    during its final step would be captured as FINISHED_LENGTH_CAPPED
    instead of FINISHED_ABORTED.

    The connector runs in a separate process, so we write statuses to a file
    that can be read by the test process.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        # Get the status file path from extra config
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.status_file = extra_config.get("status_file")
        # Log that we were initialized
        if self.status_file:
            try:
                with open(self.status_file, "a") as f:
                    f.write(f"INIT:{role.name}\n")
            except Exception:
                pass

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return (0, False)

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: Any,
        num_external_tokens: int,
    ):
        pass

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return DummyKVConnectorMetadata()

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Capture the request status when finished by writing to a file."""
        if self.status_file:
            try:
                with open(self.status_file, "a") as f:
                    # Write the status name (e.g., "FINISHED_ABORTED")
                    f.write(f"{request.status.name}\n")
            except Exception as e:
                # Log but don't fail - this is just test instrumentation
                print(f"[DummyKVConnector] Failed to write status: {e}")
        return False, None

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self):
        pass


# Register the dummy connector
KVConnectorFactory.register_connector(
    "DummyKVConnector", __name__, DummyKVConnector.__name__
)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.asyncio
async def test_abort_during_final_step(async_scheduling: bool):
    """
    Test that a request aborted during its final execution step is treated as
    aborted rather than completed.

    This test:
    1. Monkeypatches execute_model to wait for a file to be deleted
    2. Configures a dummy KV connector to capture finish statuses
    3. Starts a request with max_tokens=1 (will complete on first decode step)
    4. Aborts the request, then deletes the file to unblock execute_model
    5. Verifies the KV connector received FINISHED_ABORTED not FINISHED_LENGTH_CAPPED

    See https://github.com/vllm-project/vllm/pull/29987.

    Without the fix, the KV connector would see FINISHED_LENGTH_CAPPED because
    update_from_output() would mark the request as completed before processing
    the abort. This causes KV cache blocks to not be freed properly in
    disaggregated prefill scenarios.

    With the fix, _process_aborts_queue() runs before update_from_output(), so the
    abort takes precedence and the KV connector sees FINISHED_ABORTED.
    """

    # Create three temporary files:
    # 1. ready_file: deleted by execute_model to signal it has started
    # 2. block_file: execute_model waits for this to be deleted
    # 3. status_file: KV connector writes finish statuses here
    with tempfile.NamedTemporaryFile(delete=False) as f:
        ready_file = Path(f.name)
    with tempfile.NamedTemporaryFile(delete=False) as f2:
        block_file = Path(f2.name)
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f3:
        status_file = Path(f3.name)

    try:
        # Get the original execute_model method
        from vllm.v1.worker.gpu_worker import Worker

        original_execute_model = Worker.execute_model

        def execute_model_with_wait(self, scheduler_output):
            # Signal that execute_model has been called by deleting ready_file
            if ready_file.exists():
                ready_file.unlink()

            # Wait for the block file to be deleted (triggered from test after abort)
            # This runs in the worker process (after fork), so we poll the filesystem
            while block_file.exists():
                time.sleep(0.01)
            return original_execute_model(self, scheduler_output)

        # Patch execute_model to inject the wait
        # This happens before the worker process is forked, so the patch applies there
        with patch.object(Worker, "execute_model", execute_model_with_wait):
            request_id = "test-abort-final-step"

            # Configure engine with dummy KV connector
            # Pass the status file path so the connector can write to it
            kv_transfer_config = KVTransferConfig(
                kv_connector="DummyKVConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"status_file": str(status_file)},
            )
            engine_args = AsyncEngineArgs(
                model="meta-llama/Llama-3.2-1B-Instruct",
                enforce_eager=True,
                async_scheduling=async_scheduling,
                kv_transfer_config=kv_transfer_config,
            )

            with set_default_torch_num_threads(1):
                engine = AsyncLLM.from_engine_args(engine_args)

            try:
                # Create a request that will complete after just 1 token
                sampling_params = SamplingParams(
                    max_tokens=1,
                    ignore_eos=True,
                    output_kind=RequestOutputKind.DELTA,
                )

                # Start generation in a task
                outputs = []

                async def generate():
                    async for output in engine.generate(
                        request_id=request_id,
                        prompt=TEXT_PROMPT,
                        sampling_params=sampling_params,
                    ):
                        outputs.append(output)

                gen_task = asyncio.create_task(generate())

                # Wait for execute_model to signal it has started (with timeout)
                timeout = 5.0  # 5 second timeout
                start_time = time.time()
                while ready_file.exists():
                    if time.time() - start_time > timeout:
                        raise TimeoutError(
                            "Timeout waiting for execute_model to start. "
                            "The monkeypatch may not be working correctly, "
                            "for example if spawn was used instead of fork."
                        )
                    await asyncio.sleep(0.01)

                # Abort the request while execute_model is blocked
                await engine.abort(request_id)

                # Now unblock execute_model by deleting the file
                # The abort should be processed before the model output
                block_file.unlink()

                # Wait for generation to complete
                await gen_task

                # Give the scheduler a moment to finish cleanup
                await asyncio.sleep(0.1)

                # Verify we got output
                assert len(outputs) > 0, "Should have received at least one output"

                # The final output should have finish_reason="abort"
                final_output = outputs[-1]
                assert final_output.finished, (
                    "Final output should be marked as finished"
                )
                assert final_output.outputs[0].finish_reason == "abort", (
                    f"Expected finish_reason='abort' but got "
                    f"'{final_output.outputs[0].finish_reason}'. "
                )

                with open(status_file) as f4:
                    status_lines = f4.read().strip().split("\n")
                    # Filter for actual finish statuses (not INIT or empty lines)
                    captured_statuses = [
                        line
                        for line in status_lines
                        if line and line.startswith("FINISHED_")
                    ]

                assert len(captured_statuses) >= 1, (
                    f"Expected at least 1 captured finish status, got "
                    f"{len(captured_statuses)}. File content: {status_lines}"
                )

                assert "FINISHED_ABORTED" in captured_statuses, (
                    f"KV connector should see FINISHED_ABORTED but got "
                    f"{captured_statuses}. "
                )

                # Verify cleanup
                assert not engine.output_processor.has_unfinished_requests()

            finally:
                # Shutdown the engine
                engine.shutdown()

    finally:
        # Clean up temporary files if they still exist
        if ready_file.exists():
            ready_file.unlink()
        if block_file.exists():
            block_file.unlink()
        if status_file.exists():
            status_file.unlink()
