from vllm.wde.prefill_only.workflow import PrefillOnlyWorkflow


class EncodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = "vllm.wde.encode_only.arg_utils:EncodeOnlyEngineArgs"
    OutputProcessor: str = ("vllm.wde.encode_only.processor."
                            "output_processor:PrefillOnlyModelOutputProcessor")
    attn_type: str = "ENCODER"
