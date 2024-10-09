TASK = "encode_only"
PREFIX = f"vllm.wde.{TASK}.modelzoo"
WORKFLOW = "vllm.wde.encode_only.workflow:EncodeOnlyWorkflow"

# Architecture -> (module, workflow).
ENCODE_ONLY_MODELS = {
    "BertForMaskedLM": (PREFIX + ".bert:BertForMaskedLM", WORKFLOW),
}
