"""
VLLMWorker - AsyncLLM-backed BaseWorker for the simulation benchmarks.

Hooks are installed before importing vLLM.  Stats are recorded in the
EngineCoreProc child, dumped to VLLM_SIMULATOR_OUTPUT_DIR at each
trigger_simulation() round boundary and read back by get_*_stats().
"""

import asyncio
import json
import os
import uuid

from vllm_simulator.simulation.benchmark import BaseWorker, GenericRequest
from vllm_simulator.utils import get_logger
from vllm_simulator.simulation.vllm.startup import init_hook

# Environment must be set before vllm import
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

# Captured before any worker writes its own per-worker default below.
_USER_OUTPUT_DIR = os.environ.get("VLLM_SIMULATOR_OUTPUT_DIR")

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

        # Inherited by the engine-core child.  An explicit env value wins;
        # otherwise each worker gets its own directory.
        os.environ["VLLM_SIMULATOR_OUTPUT_DIR"] = (
            _USER_OUTPUT_DIR or f"/tmp/vllm_simulator/{name}"
        )
        self.output_dir = os.path.realpath(os.environ["VLLM_SIMULATOR_OUTPUT_DIR"])

        if engine_args.enforce_eager == AsyncEngineArgs.enforce_eager:
            engine_args.enforce_eager = True

        # Construction is safe outside a running event loop: AsyncLLM defers
        # its output handler to the first generate() call.
        self._llm = AsyncLLM.from_engine_args(engine_args)
        logger.info("[VLLMWorker] Initialized with model=%s", engine_args.model)

        # trigger_simulation alternates start/stop profile (round boundaries)
        self._profile_is_start = True

    async def async_generate(self, req: GenericRequest):
        """Stream a single request through the engine; returns the final output."""
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

        final_output = None
        async for output in self._llm.generate(
            prompt, sp, request_id=request_id
        ):
            final_output = output
        return final_output

    async def trigger_simulation(self):
        """Round separator: profile() in the engine-core child dumps
        request/iteration stats to VLLM_SIMULATOR_OUTPUT_DIR and resets
        them for the next round."""
        if self._profile_is_start:
            await self._llm.start_profile()
        else:
            await self._llm.stop_profile()
        self._profile_is_start = not self._profile_is_start

    async def flush_cache(self):
        """Clear the device prefix cache and the KV connector cache.

        Call between rounds: the reset fails while requests hold blocks.
        """
        if not await self._llm.reset_prefix_cache(reset_connector=True):
            raise RuntimeError(f"[VLLMWorker] {self.name}: cache reset failed")

    def _load_jsonl(self, filename: str) -> list[dict]:
        data = []
        file_path = os.path.join(self.output_dir, filename)
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            logger.error(f"The statistics data({file_path}) does not exist.")
        return data

    def get_request_stats(self) -> list[dict]:
        return self._load_jsonl("request.jsonl")

    def get_iteration_stats(self) -> list[dict]:
        return self._load_jsonl("iteration.jsonl")

    def generate(self, req: GenericRequest):
        """Generate output for a single request (sync).

        Only safe as the first generate entrypoint on this worker: AsyncLLM
        binds its output handler to the loop of the first generate() call.
        """
        return asyncio.run(self.async_generate(req))

    def shutdown(self):
        """Shutdown the vLLM engine and its background process."""
        self._llm.shutdown()
        logger.info("[VLLMWorker] Shutting down.")
