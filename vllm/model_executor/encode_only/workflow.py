from vllm.model_executor.prefill_only.workflow import PrefillOnlyWorkflow


class EncodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = ("vllm.model_executor.encode_only.arg_utils"
                       ":EncodeOnlyEngineArgs")
    OutputProcessor: str = ("vllm.model_executor.encode_only."
                            "output_processor:PrefillOnlyModelOutputProcessor")
    attn_type: str = "ENCODER"
