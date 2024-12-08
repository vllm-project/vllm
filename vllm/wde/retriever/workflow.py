from vllm.wde.decode_only.workflow import DecodeOnlyWorkflow
from vllm.wde.encode_only.workflow import EncodeOnlyWorkflow


class RetrieverEncodeOnlyWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = ("vllm.wde.retriever.processor."
                            "output_processor:RetrieverModelOutputProcessor")


class RetrieverDecodeOnlyWorkflow(DecodeOnlyWorkflow):
    EngineArgs: str = ("vllm.wde.retriever.arg_utils:"
                       "RetrieverDecodeOnlyEngineArgs")
