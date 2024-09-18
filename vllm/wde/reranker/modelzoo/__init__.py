TASK = "reranker"
WORKFLOW = "vllm.wde.reranker.workflow:RerankerWorkflow"

# Architecture -> (task, module, class, workflow).
RERANKER_MODELS = {
    "XLMRobertaForSequenceClassification":
    (TASK, "bge_reranker_v2_m3", "BGERerankerV2M3", WORKFLOW),
}
