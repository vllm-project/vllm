from vllm.executor.gpu_executor import GPUExecutor, GPUExecutorAsync


class CustomGPUExecutor(GPUExecutor):

    def execute_model(self, *args, **kwargs):
        # Drop marker to show that this was ran
        with open(".marker", "w"):
            ...
        return super().execute_model(*args, **kwargs)


class CustomGPUExecutorAsync(GPUExecutorAsync):

    async def execute_model_async(self, *args, **kwargs):
        with open(".marker", "w"):
            ...
        return await super().execute_model_async(*args, **kwargs)


def switch_executor():
    from vllm.plugins import set_async_executor_cls, set_executor_cls
    set_executor_cls(CustomGPUExecutor)
    set_async_executor_cls(CustomGPUExecutorAsync)
