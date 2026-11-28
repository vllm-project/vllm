"""
EngineArgs hook: force all parallelism to 1 and disable async scheduling.

The simulated parallelism comes from the simulator config
(VLLM_SIMULATOR_CONFIG_PATH) and only feeds the time predictor.

Async scheduling runs EngineCore.step_with_batch_queue, bypassing the hooked
EngineCore.step that releases parked requests.
"""

from vllm_simulator.hook import BaseHook
from vllm_simulator.utils import get_logger

logger = get_logger()


class C_VLLMEngineArgsHook(BaseHook):
    """Hook EngineArgs to force parallelism to 1 for simulation."""

    HOOK_CLASS_NAME = "EngineArgs"
    HOOK_MODULE_NAME = "vllm.engine.arg_utils"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__

        def wrapped_post_init(self):
            for attr in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "data_parallel_size_local",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            ):
                setattr(self, attr, 1)
            self.enable_expert_parallel = False

            # A spawned mp worker would start without the hooks and load the
            # real model.
            self.distributed_executor_backend = "uni"

            self.async_scheduling = False

            original_post_init(self)
            logger.info(
                "[vLLM Hijack] EngineArgs: forced parallelism to 1 "
                "(tp=%d, pp=%d)",
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
            )

        target.__post_init__ = wrapped_post_init
