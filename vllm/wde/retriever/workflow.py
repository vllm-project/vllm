from vllm.wde.encode_only.workflow import EncodeOnlyWorkflow


class RetrieverWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = "vllm.wde.retriever.processor.output_processor:RetrieverModelOutputProcessor"
