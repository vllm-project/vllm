TASK = "encode_only"
PREFIX = f"vllm.model_executor.{TASK}.modelzoo"
WORKFLOW = "vllm.model_executor.encode_only.workflow:EncodeOnlyWorkflow"

# Architecture -> (module, workflow).
ENCODE_ONLY_MODELS = {
    "BertForMaskedLM": (PREFIX + ".bert:BertForMaskedLM", WORKFLOW),
}
