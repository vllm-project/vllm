"""
VLLMWorker - AsyncLLM-backed BaseWorker for the simulation benchmarks.

Hooks are installed before importing vLLM.  Stats are recorded in the
EngineCoreProc child and returned with each finished request's output.
"""

import asyncio
import os
import uuid

from vllm_simulator.simulation.benchmark import BaseWorker, GenericRequest
from vllm_simulator.simulation.vllm.startup import init_hook
from vllm_simulator.utils import get_logger

# Environment must be set before vllm import
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

init_hook()

from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

# AsyncLLM.from_engine_args requires AsyncEngineArgs; aliased so callers
# keep using EngineArgs.
EngineArgs = AsyncEngineArgs

logger = get_logger()


def _resolve_prompt(req: GenericRequest):
    """Resolve prompt from GenericRequest (text or token_ids)."""
    from vllm import TokensPrompt

    if req.prompt is not None:
        return req.prompt
    if req.token_ids is not None:
        return TokensPrompt(prompt_token_ids=req.token_ids)
    raise ValueError("Request must have either prompt or token_ids")


class VLLMWorker(BaseWorker):
    """High-level vLLM worker for simulation benchmarks.

    Accepts a vLLM EngineArgs directly; the EngineArgs hook applies the
    simulation-fixed settings (parallelism, executor, async scheduling).
    enforce_eager defaults to True unless set explicitly.
    """

    def __init__(
        self,
        engine_args: EngineArgs,
        name: str = "vllm_worker0",
    ):
        super().__init__(name)

        if engine_args.enforce_eager == AsyncEngineArgs.enforce_eager:
            engine_args.enforce_eager = True

        # Construction is safe outside a running event loop: AsyncLLM defers
        # its output handler to the first generate() call.
        self._llm = AsyncLLM.from_engine_args(engine_args)
        logger.info("[VLLMWorker] Initialized with model=%s", engine_args.model)

    async def async_generate(self, req: GenericRequest) -> dict:
        """Stream a single request through the engine; returns its stats."""
        # Pass simulation metadata (created_time) via extra_args
        extra_args = None
        if req.custom_params:
            sim_meta = {}
            if "created_time" in req.custom_params:
                sim_meta["created_time"] = req.custom_params["created_time"]
            if "queue_start" in req.custom_params:
                sim_meta["queue_start"] = req.custom_params["queue_start"]
            if "total_request" in req.custom_params:
                sim_meta["total_request"] = req.custom_params["total_request"]
            if sim_meta:
                extra_args = {"simulation": sim_meta}

        sp = SamplingParams(
            max_tokens=req.output_length, ignore_eos=True, extra_args=extra_args
        )
        prompt = _resolve_prompt(req)
        # Callers that join stats across engines (PD) pin the request id;
        # it is kept verbatim since request-id randomization is disabled.
        request_id = req.custom_params.get("request_id") or uuid.uuid4().hex
        req.custom_params["request_id"] = request_id

        final_output = None
        async for output in self._llm.generate(prompt, sp, request_id=request_id):
            final_output = output

        params = final_output.kv_transfer_params if final_output else None
        stats = (params or {}).get("simulation")
        if stats is None:
            raise RuntimeError(f"[VLLMWorker] {self.name}: no stats for {request_id}")
        return {**stats, "rid": final_output.request_id}

    async def flush_cache(self):
        """Clear the device prefix cache and the KV connector cache.

        Call between rounds: the reset fails while requests hold blocks.
        """
        if not await self._llm.reset_prefix_cache(reset_connector=True):
            raise RuntimeError(f"[VLLMWorker] {self.name}: cache reset failed")

    def generate(self, req: GenericRequest) -> dict:
        """Generate output for a single request (sync); returns its stats.

        Only safe as the first generate entrypoint on this worker: AsyncLLM
        binds its output handler to the loop of the first generate() call.
        """
        return asyncio.run(self.async_generate(req))

    def shutdown(self):
        """Shutdown the vLLM engine and its background process."""
        self._llm.shutdown()
        logger.info("[VLLMWorker] Shutting down.")
