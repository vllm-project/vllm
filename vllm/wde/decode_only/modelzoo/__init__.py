TASK = "decode_only"
WORKFLOW = "vllm.wde.decode_only.workflow:DecodeOnlyWorkflow"

# Architecture -> (task, module, class, workflow).
DECODE_ONLY_MODELS = {
    "Qwen2ForCausalLM": (TASK, "qwen2", "Qwen2ForCausalLM", WORKFLOW),
}
