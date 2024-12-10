from vllm.wde.core.workflow import Workflow
from vllm.wde.prefill_only.workflow import PrefillOnlyWorkflow


class DecodeOnlyWorkflow(Workflow):
    EngineArgs: str = "vllm.wde.decode_only.arg_utils:DecodeOnlyEngineArgs"
    attn_type: str = "DECODER"

    @classmethod
    def from_engine(cls, engine):
        if engine.engine_config.model_config.output_last_hidden_states:
            workflow = PrefillOnlyWorkflow.from_engine(engine)

            if engine.engine_config.model_config.enable_bidirectional:
                workflow.attn_type = "ENCODER"
            else:
                workflow.attn_type = "DECODER"

            workflow.OutputProcessor = (
                "vllm.wde.decode_only.processor."
                "output_processor:"
                "DecodeOnlyHiddenStatesOutputProcessor")
            return workflow
        else:
            raise ValueError("Not supported")
