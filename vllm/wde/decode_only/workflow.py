from vllm.wde.prefill_only.workflow import PrefillOnlyWorkflow


class DecodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = "vllm.wde.decode_only.arg_utils:DecodeOnlyEngineArgs"
    attn_type: str = "DECODER"
    OutputProcessor: str

    @classmethod
    def from_engine(cls, engine):
        if engine.engine_config.model_config.output_last_hidden_states:
            workflow = cls()

            workflow.OutputProcessor = (
                "vllm.wde.decode_only.processor."
                "output_processor:"
                "DecodeOnlyHiddenStatesOutputProcessor")

            if engine.engine_config.model_config.enable_bidirectional:
                workflow.attn_type = "ENCODER"
            else:
                workflow.attn_type = "DECODER"

            if engine.engine_config.scheduler_config.scheduling in ["sync"]:
                workflow.Executor += ":GPUExecutor"
            elif engine.engine_config.scheduler_config.scheduling in [
                    "async", "double_buffer"
            ]:
                workflow.Executor += ":GPUAsyncExecutor"

            return workflow
