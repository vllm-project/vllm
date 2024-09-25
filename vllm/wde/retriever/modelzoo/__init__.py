TASK = "retriever"
RETRIEVER_ENCODER_ONLY_WORKFLOW = ("vllm.wde.retriever.workflow:"
                                   "RetrieverEncodeOnlyWorkflow")

# Architecture -> (task, module, class, workflow).
RETRIEVER_ENCODER_ONLY_MODELS = {
    "XLMRobertaModel":
    (TASK, "bge_m3", "BGEM3Model", RETRIEVER_ENCODER_ONLY_WORKFLOW),
    "BertModel":
    (TASK, "bert_retriever", "BertRetriever", RETRIEVER_ENCODER_ONLY_WORKFLOW),
}

RETRIEVER_DECODER_ONLY_WORKFLOW = ("vllm.wde.retriever.workflow:"
                                   "RetrieverDecodeOnlyWorkflow")

# Architecture -> (task, module, class, workflow).
RETRIEVER_DECODER_ONLY_MODELS = {
    "MistralModel": (TASK, "llama_embedding", "LlamaEmbeddingModel",
                     RETRIEVER_DECODER_ONLY_WORKFLOW)
}

RETRIEVER_MODELS = {
    **RETRIEVER_ENCODER_ONLY_MODELS,
    **RETRIEVER_DECODER_ONLY_MODELS
}
