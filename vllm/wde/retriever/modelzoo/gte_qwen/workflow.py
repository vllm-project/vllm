from vllm.wde.prefill_only.workflow import PrefillOnlyWorkflow


class Qwen2Workflow(PrefillOnlyWorkflow):
    EngineArgs: str = ("vllm.wde.retriever.modelzoo."
                       "gte_qwen.arg_utils:Qwen2EngineArgs")
    attn_type: str = "DECODER"

    @classmethod
    def from_engine(cls, engine):
        workflow = cls()
        if engine.engine_config.model_config.switch_to_gte_Qwen2:
            workflow.ModelInputBuilder = ("vllm.wde.retriever.modelzoo."
                                          "gte_qwen.model_input_builder:"
                                          "GTEQwenModelInputBuilder")
            workflow.OutputProcessor = ("vllm.wde.retriever.modelzoo."
                                        "gte_qwen.output_processor:"
                                        "GTEQwenOutputProcessor")

        elif engine.engine_config.model_config.output_last_hidden_states:
            workflow.OutputProcessor = (
                "vllm.wde.decode_only.processor."
                "output_processor:"
                "DecodeOnlyHiddenStatesOutputProcessor")
        else:
            raise ValueError("Not supported")

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
