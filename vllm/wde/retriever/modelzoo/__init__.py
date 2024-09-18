TASK = "retriever"
WORKFLOW = "vllm.wde.retriever.workflow:RetrieverWorkflow"

# Architecture -> (wde, module, class, workflow).
RETRIEVER_MODELS = {
    "XLMRobertaModel": (TASK, "bge_m3", "BGEM3Model", WORKFLOW),
}
