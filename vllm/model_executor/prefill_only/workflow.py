class Workflow:
    EngineArgs: str
    Scheduler: str
    AttnBackend: str
    attn_type: str
    Tokenizer: str = "vllm.inputs.prefill_only.tokenizer:Tokenizer"
    InputProcessor: str
    RequestProcessor: str
    OutputProcessor: str
    ModelInputBuilder: str
    Executor: str
    Worker: str

    @classmethod
    def from_engine(cls, engine):
        return cls()


class PrefillOnlyWorkflow(Workflow):
    InputProcessor: str = ("vllm.inputs.prefill_only.preprocessor"
                           ":TextInputProcessor")
    RequestProcessor: str = ("vllm.inputs.prefill_only.preprocessor"
                             ":TextRequestProcessor")
    ModelInputBuilder: str = (
        "vllm.model_executor.prefill_only.model_input_builder"
        ":PrefillOnlyModelInputBuilder")
    Worker: str = "vllm.worker.prefill_only_gpu_worker:Worker"
    Executor: str = "vllm.executor.prefill_only_gpu_executor"
    Scheduler: str = "vllm.core.prefill_only_scheduler:PrefillOnlyScheduler"
    AttnBackend: str = "vllm.attention.prefill_only.selector:AttnBackend"

    @classmethod
    def from_engine(cls, engine):
        workflow = cls()

        if engine.engine_config.scheduler_config.scheduling in ["sync"]:
            workflow.Executor += ":GPUExecutor"
        elif engine.engine_config.scheduler_config.scheduling in [
                "async", "double_buffer"
        ]:
            workflow.Executor += ":GPUAsyncExecutor"

        return workflow
