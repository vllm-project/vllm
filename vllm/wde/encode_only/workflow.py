from vllm.wde.core.workflow import Workflow


class EncodeOnlyWorkflow(Workflow):
    EngineArgs: str = "vllm.wde.encode_only.arg_utils:EncodeOnlyEngineArgs"
    InputProcessor: str = "vllm.wde.encode_only.processor.input_processor:EncodeOnlyModelInputProcessor"
    RequestProcessor: str = "vllm.wde.encode_only.processor.input_processor:EncodeOnlyModelRequestProcessor"
    OutputProcessor: str = "vllm.wde.encode_only.processor.output_processor:EncodeOnlyModelOutputProcessor"
    ModelInputBuilder: str = "vllm.wde.encode_only.processor.model_input_builder:EncodeOnlyModelInputBuilder"
    Worker: str = "vllm.wde.encode_only.worker.gpu_worker:Worker"
    Executor: str = "vllm.wde.encode_only.executor.gpu_executor"
    Scheduler: str = "vllm.wde.encode_only.scheduler:EncodeOnlyScheduler"
    AttnBackend: str = "vllm.wde.encode_only.layers.attention.selector:AttnBackend"

    @classmethod
    def from_engine(cls, engine: "LLMEngine"):

        workflow = cls()
        if engine.engine_config.scheduler_config.scheduling in ["sync"]:
            workflow.Executor += ":GPUExecutor"
        elif engine.engine_config.scheduler_config.scheduling in [
                "async", "double_buffer"
        ]:
            workflow.Executor += ":GPUAsyncExecutor"

        return workflow
