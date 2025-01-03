from vllm.v1.executor.ray_executor import RayExecutor
from vllm.v1.executor.ray_utils import ray
from vllm.v1.outputs import ModelRunnerOutput

class RayXPUExecutor(RayExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = "xpu"
        self._dispatch_key = "XPU"

    # FIXME: add XPU parameters.
    # def _init_workers_ray(self, placement_group: "PlacementGroup",
    #                      **ray_remote_kwargs):
    #    pass

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        # FIXME: XPU do not support ray dag now.
        # Only the first worker (with rank 0) returns the execution result.
        # Others return None.
        outputs = [ray.get(worker.execute_model.remote(scheduler_output)) for worker in self.workers]
        output = outputs[0]
        return output
