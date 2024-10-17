TASK = "decode_only"
WORKFLOW = "vllm.wde.decode_only.workflow:DecodeOnlyWorkflow"

# Architecture -> (task, module, class, workflow).
DECODE_ONLY_MODELS = {
    "LlamaForCausalLM": (TASK, "llama", "LlamaForCausalLM", WORKFLOW),
    "Qwen2ForCausalLM":
    (TASK, "qwen2", "Qwen2ForCausalLM",
     "vllm.wde.retriever.modelzoo.gte_qwen.workflow:Qwen2Workflow"),
}
