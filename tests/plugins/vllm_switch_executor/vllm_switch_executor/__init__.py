from vllm.executor.gpu_executor import GPUExecutor

class CustomGPUExecutor(GPUExecutor):

    def execute_model(self, *args, **kwargs):
        # Drop marker to show that this was ran
        with open(".marker", "w"):
            ...
        return super().execute_model(*args, **kwargs)

def switch_executor():
    from vllm.plugins import set_executor_cls
    set_executor_cls(CustomGPUExecutor)
