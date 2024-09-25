from vllm.wde.decode_only.workflow import DecodeOnlyWorkflow
from vllm.wde.encode_only.workflow import EncodeOnlyWorkflow


class RetrieverEncodeOnlyWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = ("vllm.wde.retriever.processor."
                            "output_processor:RetrieverModelOutputProcessor")


class RetrieverDecodeOnlyWorkflow(DecodeOnlyWorkflow):
    EngineArgs: str = ("vllm.wde.retriever.arg_utils:"
                       "RetrieverDecodeOnlyEngineArgs")
    OutputProcessor: str = ("vllm.wde.decode_only.processor."
                            "output_processor:"
                            "DecodeOnlyHiddenStatesOutputProcessor")

    @classmethod
    def from_engine(cls, engine):
        workflow = cls()

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
