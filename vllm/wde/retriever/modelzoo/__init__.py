TASK = "retriever"
WORKFLOW = "vllm.wde.retriever.workflow:RetrieverWorkflow"

# Architecture -> (task, module, class, workflow).
RETRIEVER_MODELS = {
    "XLMRobertaModel": (TASK, "bge_m3", "BGEM3Model", WORKFLOW),
    "BertModel": (TASK, "bert_retriever", "BertRetriever", WORKFLOW),
}
