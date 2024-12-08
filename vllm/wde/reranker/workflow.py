from vllm.wde.encode_only.workflow import EncodeOnlyWorkflow


class RerankerWorkflow(EncodeOnlyWorkflow):
    InputProcessor: str = ("vllm.wde.reranker.processor."
                           "input_processor:RerankerInputProcessor")
    RequestProcessor: str = ("vllm.wde.reranker.processor."
                             "input_processor:RerankerRequestProcessor")
    OutputProcessor: str = ("vllm.wde.reranker.processor."
                            "output_processor:RerankerOutputProcessor")
