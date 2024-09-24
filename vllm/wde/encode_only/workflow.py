from vllm.wde.prefill_only.workflow import PrefillOnlyWorkflow


class EncodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = "vllm.wde.encode_only.arg_utils:EncodeOnlyEngineArgs"
    attn_type: str = "ENCODER"
